"""
Hailey KV tiering hook for EXO on DGX Spark.

Patches KVPrefixCache to:
1. On eviction: serialize all 40 layers to cold storage
   - KVCache/RotatingKVCache (10 layers) → Mac Pro (192.168.50.11:50051, LAN)
   - ArraysCache (30 layers, SSM state) → Mac Air (192.168.10.174:50051, WiFi)
   - hot_capacity=0 → write-through, no redundant DashMap in DGX RAM
2. On cache miss: fetch all 40 layers from cold stores and restore into KVPrefixCache

Data flow (Rust handles compress/decompress transparently):
  insert(token_idx, layer, f32_bytes) → TurboQuant.encode → store on Mac Pro/Air
  get(token_idx, layer) → Promote RPC → TurboQuant.decode → f32_bytes returned

Shape metadata is stored in _kv_shapes / _ssm_shapes dicts so we can reconstruct
tensor arrays from the flat f32 bytes returned by tc.get().

Usage (called from builder.py on DGX startup):
    from exo.worker.engines.mlx.hailey_kv_hook import install_kv_tiering_hook
    install_kv_tiering_hook()
"""

import gc
import logging
import struct
import sys

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def _setup_file_logging():
    fh = logging.FileHandler('/tmp/hailey-dgx.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.DEBUG)


_setup_file_logging()

# ──────────────────────────────────────────────
# Lazy-init clients
# ──────────────────────────────────────────────

_kv_cache = None      # KVCache layers → Mac Pro
_ssm_cache = None     # ArraysCache layers → Mac Air


def _get_kv_cache():
    global _kv_cache
    if _kv_cache is not None:
        return _kv_cache
    try:
        sys.path.insert(0, '/home/pk/work/EXO_PK')
        from exo_pk.cache import get_cache
        _kv_cache = get_cache(
            dim=256,
            hot_capacity=0,                    # write-through — EXO GPU KVPrefixCache IS the hot tier
            cold_node="192.168.50.11:50051",   # Mac Pro (LAN)
            recompute_node="127.0.0.1:50052",
        )
        logger.info("[hailey] KV cache → Mac Pro 192.168.50.11:50051 (write-through)")
    except Exception as e:
        logger.warning(f"[hailey] KV cache init failed: {e}")
        _kv_cache = None
    return _kv_cache


def _get_ssm_cache():
    global _ssm_cache
    if _ssm_cache is not None:
        return _ssm_cache
    try:
        sys.path.insert(0, '/home/pk/work/EXO_PK')
        from exo_pk.cache import get_cache
        _ssm_cache = get_cache(
            dim=256,
            hot_capacity=0,
            cold_node="192.168.10.174:50051",  # Mac Air (WiFi)
            recompute_node="127.0.0.1:50052",
        )
        logger.info("[hailey] SSM cache → Mac Air 192.168.10.174:50051 (write-through)")
    except Exception as e:
        logger.warning(f"[hailey] SSM cache init failed: {e}")
        _ssm_cache = None
    return _ssm_cache


# ──────────────────────────────────────────────
# Shape stores — held in memory alongside the compressed data on Mac Pro/Air.
# Keyed by (token_idx, layer_idx). Needed to reshape f32 bytes returned by tc.get().
# ──────────────────────────────────────────────

_kv_shapes: dict = {}   # (token_idx, layer) → (k_shape, v_shape)
_ssm_shapes: dict = {}  # (token_idx, layer) → [array_shape, ...]
_kv_offsets: dict = {}  # token_idx → actual KVCache.offset at eviction time


# ──────────────────────────────────────────────
# Eviction path
# ──────────────────────────────────────────────

def _evict_to_cold(prompt_tokens: mx.array, cache_list: list, _lru_idx: int):
    kv = _get_kv_cache()
    ssm = _get_ssm_cache()
    if kv is None and ssm is None:
        return

    prompt_key = int(mx.sum(prompt_tokens).item()) ^ len(prompt_tokens)
    token_idx = prompt_key * 1000  # stable, lru_idx-independent

    kv_n = ssm_n = 0
    saved_offset = None
    try:
        from mlx_lm.models.cache import KVCache, RotatingKVCache, ArraysCache
        for layer_idx, entry in enumerate(cache_list):
            if isinstance(entry, (KVCache, RotatingKVCache)):
                if kv is None or entry.keys is None:
                    continue
                k = np.array(entry.keys.astype(mx.float32))
                v = np.array(entry.values.astype(mx.float32))
                _kv_shapes[(token_idx, layer_idx)] = (k.shape, v.shape)
                # Capture the actual valid token count once (same for all KV layers)
                if saved_offset is None:
                    saved_offset = int(entry.offset)
                kv.insert(token_idx=token_idx, layer=layer_idx, data=k.tobytes() + v.tobytes())
                kv_n += 1
            elif isinstance(entry, ArraysCache):
                if ssm is None or not entry.state:
                    continue
                arrays = [np.array(a.astype(mx.float32)) for a in entry.state]
                _ssm_shapes[(token_idx, layer_idx)] = [a.shape for a in arrays]
                ssm.insert(token_idx=token_idx, layer=layer_idx, data=b''.join(a.tobytes() for a in arrays))
                ssm_n += 1
    except Exception as e:
        logger.warning(f"[hailey] evict failed: {e}")

    if saved_offset is not None:
        _kv_offsets[token_idx] = saved_offset
        logger.info(f"[hailey] Evicted kv={kv_n} ssm={ssm_n} layers (key={prompt_key:x}) offset={saved_offset} prompt_len={len(prompt_tokens)}")
    else:
        logger.info(f"[hailey] Evicted kv={kv_n} ssm={ssm_n} layers (key={prompt_key:x})")


# ──────────────────────────────────────────────
# Retrieval path
# ──────────────────────────────────────────────

def _restore_from_cold(prefix_cache, prompt_tokens: mx.array) -> bool:
    """Fetch all 40 layers via 2 parallel batch RPCs, inject into KVPrefixCache."""
    kv = _get_kv_cache()
    ssm = _get_ssm_cache()
    if kv is None and ssm is None:
        return False

    prompt_key = int(mx.sum(prompt_tokens).item()) ^ len(prompt_tokens)
    token_idx = prompt_key * 1000

    kv_layers  = [l for l in range(40) if (token_idx, l) in _kv_shapes]
    ssm_layers = [l for l in range(40) if (token_idx, l) in _ssm_shapes]

    if not kv_layers and not ssm_layers:
        logger.debug(f"[hailey] restore: no shapes for key={prompt_key:x}")
        return False

    # Use the offset saved at eviction time (includes full chat template tokens).
    # Fall back to len(prompt_tokens) only if offset wasn't recorded.
    restore_offset = _kv_offsets.get(token_idx, len(prompt_tokens))
    logger.debug(f"[hailey] restore: key={prompt_key:x} restore_offset={restore_offset} prompt_len={len(prompt_tokens)}")

    # Fire both batch RPCs in parallel — one connection, one round-trip each
    from concurrent.futures import ThreadPoolExecutor

    def fetch_kv():
        if kv is None or not kv_layers:
            return {}
        try:
            items = kv.get_batch(token_idx=token_idx, layers=kv_layers)
            return {layer: bytes(data) for layer, data in items if data}
        except Exception as e:
            logger.warning(f"[hailey] batch kv fetch failed: {e}")
            return {}

    def fetch_ssm():
        if ssm is None or not ssm_layers:
            return {}
        try:
            items = ssm.get_batch(token_idx=token_idx, layers=ssm_layers)
            return {layer: bytes(data) for layer, data in items if data}
        except Exception as e:
            logger.warning(f"[hailey] batch ssm fetch failed: {e}")
            return {}

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_kv  = pool.submit(fetch_kv)
        f_ssm = pool.submit(fetch_ssm)
        kv_results  = f_kv.result()
        ssm_results = f_ssm.result()

    logger.debug(f"[hailey] restore fetch: kv_layers={kv_layers} ssm_layers={ssm_layers} "
                 f"kv_results.keys={sorted(kv_results.keys())} ssm_results.keys={sorted(ssm_results.keys())} "
                 f"kv_sizes={[len(v) for v in kv_results.values()]} ssm_sizes={[len(v) for v in ssm_results.values()]}")

    n_layers = 40
    restored = []

    try:
        from mlx_lm.models.cache import KVCache, ArraysCache
        for layer_idx in range(n_layers):
            if layer_idx in kv_results:
                f32 = kv_results[layer_idx]
                k_shape, v_shape = _kv_shapes[(token_idx, layer_idx)]
                k_n = int(np.prod(k_shape)) * 4
                v_n = int(np.prod(v_shape)) * 4
                if len(f32) >= k_n + v_n:
                    c = KVCache()
                    c.keys   = mx.array(np.frombuffer(f32[:k_n],        dtype=np.float32).reshape(k_shape),  dtype=mx.bfloat16)
                    c.values = mx.array(np.frombuffer(f32[k_n:k_n+v_n], dtype=np.float32).reshape(v_shape), dtype=mx.bfloat16)
                    restored.append(c)
                    continue
            elif layer_idx in ssm_results:
                f32 = ssm_results[layer_idx]
                shapes = _ssm_shapes[(token_idx, layer_idx)]
                arrays, off = [], 0
                for shape in shapes:
                    n = int(np.prod(shape)) * 4
                    arrays.append(mx.array(np.frombuffer(f32[off:off + n], dtype=np.float32).reshape(shape), dtype=mx.bfloat16))
                    off += n
                c = ArraysCache(len(arrays))
                c.state = arrays
                restored.append(c)
                continue
            else:
                logger.debug(f"[hailey] restore: missing layer {layer_idx}")
                return False

    except Exception as e:
        logger.warning(f"[hailey] restore failed: {e}")
        return False

    # Set KVCache.offset to the exact value recorded at eviction time.
    # This is critical: keys.shape[2] is the pre-allocated buffer (padded),
    # while restore_offset is the actual number of valid token positions.
    from mlx_lm.models.cache import KVCache as _KVCache
    for c in restored:
        if isinstance(c, _KVCache) and c.keys is not None and c.offset == 0:
            c.offset = restore_offset

    # Build CacheSnapshot so EXO can resume SSM state (required for has_ssm models)
    from exo.worker.engines.mlx.cache import snapshot_ssm_states
    snapshot = snapshot_ssm_states(restored)
    snapshot.token_count = restore_offset

    # Inject all 40 layers into KVPrefixCache
    prefix_cache._access_counter += 1
    prefix_cache.prompts.append(prompt_tokens)
    prefix_cache.caches.append(restored)
    prefix_cache._snapshots.append([snapshot])   # list with one snapshot at token_count T
    prefix_cache._media_regions.append([])
    prefix_cache._last_used.append(prefix_cache._access_counter)
    prefix_cache.prefill_tps.append(0.0)
    logger.info(f"[hailey] Cold restore success: {n_layers} layers batch (key={prompt_key:x}) offset={restore_offset}")
    return True


# ──────────────────────────────────────────────
# Hook installation
# ──────────────────────────────────────────────

def install_kv_tiering_hook():
    try:
        from exo.worker.engines.mlx.cache import KVPrefixCache
        from exo.worker.engines.mlx import cache as cache_module

        def _evict_if_needed_hailey(self):
            if not self.caches:
                return
            evicted = False
            while self.caches and self.get_memory_used_percentage() > cache_module._MEMORY_THRESHOLD:
                i = self._last_used.index(min(self._last_used))
                _evict_to_cold(self.prompts[i], self.caches[i], i)
                ntok = len(self.prompts[i])
                self.prompts.pop(i)
                self.caches.pop(i)
                self._snapshots.pop(i)
                self._media_regions.pop(i)
                self._last_used.pop(i)
                self.prefill_tps.pop(i)
                evicted = True
                logger.info(f"[hailey] evicted LRU entry ({ntok} tokens)")
            if evicted:
                gc.collect()
                mx.clear_cache()

        KVPrefixCache._evict_if_needed = _evict_if_needed_hailey
        logger.info("[hailey] eviction hook installed")

        _orig = KVPrefixCache.get_kv_cache

        def _get_kv_cache_hailey(self, model, prompt_tokens, media_regions=None):
            result = _orig(self, model, prompt_tokens, media_regions)
            _, remaining, matched_idx, is_exact = result
            prompt_key = int(mx.sum(prompt_tokens).item()) ^ len(prompt_tokens)
            # Full miss = either no match at all OR match provides zero prefix (remaining == ntok)
            full_miss = (matched_idx is None) or (len(remaining) == len(prompt_tokens))
            logger.debug(f"[hailey] get_kv_cache: key={prompt_key:x} ntok={len(prompt_tokens)} "
                         f"matched_idx={matched_idx} is_exact={is_exact} "
                         f"cache_entries={len(self.prompts)} remaining={len(remaining)} full_miss={full_miss}")
            if full_miss and _restore_from_cold(self, prompt_tokens):
                result = _orig(self, model, prompt_tokens, media_regions)
                cache_r, remaining_r, idx_r, exact_r = result
                # mlx_lm BatchGenerator crashes on empty segments (remaining=[]).
                # Fix: trim 1 from KV offsets and include the last prompt token.
                # SSM state has last token applied once from restore; 1-token prefill
                # applies it again (negligible ~1/T error acceptable for TTFT demo).
                if len(remaining_r) == 0 and len(prompt_tokens) > 0:
                    remaining_r = prompt_tokens[-1:]
                    from mlx_lm.models.cache import KVCache as _KVC
                    for c in cache_r:
                        if isinstance(c, _KVC) and hasattr(c, 'trim'):
                            c.trim(1)
                    logger.info(f"[hailey] patched empty remaining → 1-token prefill (key={prompt_key:x})")
                result = (cache_r, remaining_r, idx_r, exact_r)
            return result

        KVPrefixCache.get_kv_cache = _get_kv_cache_hailey
        logger.info("[hailey] retrieval hook installed")

    except Exception as e:
        logger.warning(f"[hailey] hook install failed: {e}")
