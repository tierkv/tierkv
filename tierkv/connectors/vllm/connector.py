"""TierKV vLLM KV Connector — reactive cold tier for evicted blocks.

Launch:
    vllm serve <model> \\
        --kv-connector tierkv.connectors.vllm.connector.TierKVConnector \\
        --kv-connector-extra-config '{"config_path": "/etc/tierkv/tierkv.toml"}' \\
        --enable-prefix-caching \\
        --block-size 16
"""

from __future__ import annotations

import concurrent.futures
import math
import threading
from dataclasses import dataclass, field
from typing import Optional

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
        SupportsHMA,
    )
except ImportError:
    # Allow import without vLLM installed (testing, EXO-only installs)
    KVConnectorBase_V1 = object
    KVConnectorMetadata = object
    KVConnectorRole = None
    SupportsHMA = object

from tierkv.connectors.vllm.block_registry import BlockRegistry
from tierkv.connectors.vllm.request_handler import RequestHandler
from tierkv.connectors.vllm.restore_handler import RestoreHandler
from tierkv.connectors.vllm.config import VllmConnectorConfig
from tierkv.connectors.vllm.context_tracker import ContextTracker
from tierkv.connectors.vault_client import VllmVaultClient


@dataclass
class TierKVMeta:
    """Passed from scheduler to worker each forward step."""
    store_plan: dict[str, list[tuple[str, str, int]]] = field(default_factory=dict)
    restore_plan: dict[str, list[str]] = field(default_factory=dict)
    # req_id → {block_hash_hex → list[int]} — one block_id per HMA group.
    # Index 0 = primary group (full-attn, tensor-format caches, most non-zero IDs).
    # Index 1 = secondary group (GLA/SSM, list-format caches, more zeros).
    # For standard models (single group) both indices hold the same ID.
    restore_block_ids: dict[str, dict[str, list[int]]] = field(default_factory=dict)


class TierKVConnector(KVConnectorBase_V1, SupportsHMA):
    """
    TierKV cold tier for vLLM.

    Reactive eviction model:
    - Stores blocks only when vLLM evicts them (request_finished)
    - Restores blocks on cache miss (start_load_kv)
    - save_kv_layer is a no-op

    Inherits SupportsHMA so vLLM keeps the Hybrid KV Cache Manager (HMA) enabled.
    HMA is required for hybrid models like Qwen3.5 MoE that mix full-attention
    and linear-attention (SSM) layers.
    """

    # Class-level shared registry so the SCHEDULER and WORKER connector instances
    # (both created inside the same EngineCore process) see the same BlockRegistry.
    # Keyed by engine_id from kv_transfer_config (defaults to "default").
    _shared_registries: "dict[str, BlockRegistry]" = {}

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)

        extra = {}
        if hasattr(vllm_config, "kv_transfer_config") and vllm_config.kv_transfer_config:
            extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.cfg = VllmConnectorConfig.from_dict(extra)

        # Actual vLLM block size — may differ from cfg.block_size for hybrid
        # models (HMA) where the Mamba page size forces a larger alignment
        # (e.g. Qwen3.5 MoE uses 1056 tokens/block despite --block-size 16).
        # This is set by vLLM after KV cache initialization, so it reflects
        # the true token count per GPU KV block.
        self._vllm_block_size: int = getattr(
            vllm_config.cache_config, "block_size", self.cfg.block_size
        )

        engine_id = (
            vllm_config.kv_transfer_config.engine_id
            if hasattr(vllm_config, "kv_transfer_config") and vllm_config.kv_transfer_config
            and hasattr(vllm_config.kv_transfer_config, "engine_id")
            else "default"
        )
        self.registry = TierKVConnector._shared_registries.setdefault(
            engine_id, BlockRegistry()
        )
        self.context_tracker = ContextTracker()

        self.kv_client = VllmVaultClient(
            host=self.cfg.kv_cold_host,
            port=self.cfg.kv_cold_port,
        )
        self.ssm_client = VllmVaultClient(
            host=self.cfg.ssm_cold_host,
            port=self.cfg.ssm_cold_port,
        ) if self.cfg.ssm_cold_host else None

        # Pre-warm gRPC connections so the first cold-restore doesn't pay
        # the TCP + HTTP/2 handshake cost (can be 40s on a cold WiFi link).
        self.kv_client.warmup()
        if self.ssm_client:
            self.ssm_client.warmup()

        self.request_handler = RequestHandler(
            registry=self.registry,
            kv_client=self.kv_client,
            ssm_client=self.ssm_client,
            turbo_quant=self.cfg.turbo_quant,
            max_concurrent_stores=self.cfg.max_inflight_stores,
        )
        self.request_handler._group_size = self.cfg.kv_dim
        self.request_handler._layer_type_map = self.cfg.layer_type_map

        self.restore_handler = RestoreHandler(
            registry=self.registry,
            kv_client=self.kv_client,
            ssm_client=self.ssm_client,
            turbo_quant=self.cfg.turbo_quant,
            group_size=self.cfg.kv_dim,
        )

        # Internal callback — updates context_tracker on every successful store.
        # External SDKs attach additional callbacks via register_store_callback().
        self.request_handler._store_callbacks.append(self._on_block_stored)

        self._restore_plans: dict[str, list] = {}
        self._restore_block_ids: dict[str, dict[str, list[int]]] = {}
        self._store_plans: dict[str, list] = {}

        # Worker side: request IDs whose cold-tier store has completed.
        # Populated by free_fn in _worker_execute_store; drained by get_finished().
        # get_finished() → KVConnectorOutput.finished_sending → scheduler frees blocks.
        self._finished_sending: set[str] = set()
        self._finished_sending_lock = threading.Lock()

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.cfg.max_inflight_promotes,
            thread_name_prefix="tierkv-worker",
        )

    # ── Scheduler side ────────────────────────────────────────────────────────

    def get_num_new_matched_tokens(
        self, request, num_computed_tokens: int
    ) -> tuple[Optional[int], bool]:
        """Check cold tier for matching blocks beyond what's already computed.

        Returns (ext_tokens, load_kv_async).
        - ext_tokens=None means "can't schedule yet" → vLLM defers indefinitely.
        - ext_tokens=0 means "no cold-tier match, schedule normally".
        - ext_tokens>0 means "N tokens available in cold tier".
        """
        block_hashes = getattr(request, "block_hashes", None)
        if not block_hashes:
            return 0, False

        matched_tokens, load_plan = self.restore_handler.plan(block_hashes)
        if matched_tokens > num_computed_tokens:
            self._restore_plans[request.request_id] = load_plan
            return matched_tokens - num_computed_tokens, False
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        """Capture per-group GPU block IDs for the external (cold-restore) tokens.

        vLLM calls this after allocating blocks for a request whose
        get_num_new_matched_tokens returned >0.  `blocks` is a KVCacheBlocks
        object; `blocks.get_block_ids()` returns one list[int] per HMA group.

        We keep ALL groups so the worker can use the correct block ID for each
        KV cache type (tensor-format full-attn vs list-format GLA/SSM).
        """
        if num_external_tokens <= 0 or not blocks:
            return
        req_id = getattr(request, "request_id", None)
        load_plan = self._restore_plans.get(req_id) if req_id else None
        if not load_plan:
            return

        num_external_blocks = math.ceil(num_external_tokens / self._vllm_block_size)

        # KVCacheBlocks.get_block_ids() → tuple[list[int], ...] (one per group)
        try:
            all_group_ids: tuple = blocks.get_block_ids()
        except AttributeError:
            try:
                all_group_ids = (
                    [getattr(b, "block_id", None) for b in list(blocks)],
                )
            except TypeError:
                return

        if not all_group_ids:
            return

        # Sort: primary = most non-zero IDs (full-attn), secondary = GLA/SSM.
        _blen = num_external_blocks
        sorted_groups = sorted(
            all_group_ids,
            key=lambda g: sum(
                1 for bid in g[:_blen] if bid is not None and bid != 0
            ),
            reverse=True,
        )
        primary_ids = sorted_groups[0]
        secondary_ids = sorted_groups[1] if len(sorted_groups) > 1 else primary_ids

        hash_to_group_ids: dict[str, list[int]] = {}
        for i, record in enumerate(load_plan):
            if i >= num_external_blocks:
                break
            pid = primary_ids[i] if i < len(primary_ids) else 0
            sid = secondary_ids[i] if i < len(secondary_ids) else pid
            if pid is not None and pid != 0:
                hash_to_group_ids[record.block_hash.hex()] = [pid, sid if sid else pid]

        if hash_to_group_ids:
            self._restore_block_ids[req_id] = hash_to_group_ids

    def request_finished(
        self, request, block_ids: list[int]
    ) -> tuple[bool, Optional[dict]]:
        """
        Eviction intercept (single-group / non-HMA path).
        Returns (True, None) — TierKV owns GPU block free, vLLM waits.
        Returns (False, None) — nothing to store, vLLM frees immediately.
        """
        # Wrap into a single-group tuple so _request_finished_impl can be shared.
        return self._request_finished_impl(request, (list(block_ids),))

    def request_finished_all_groups(
        self,
        request,
        block_ids: "tuple[list[int], ...]",
    ) -> tuple[bool, Optional[dict]]:
        """
        HMA variant of request_finished — called once for all KV cache groups.
        Required to support hybrid models (Qwen3.5 MoE) where vLLM uses the
        Hybrid KV Cache Manager (HMA) with separate groups for full-attention
        and SSM layers.

        Passes ALL per-group block IDs through to the worker so each KV cache
        group is indexed with its own correct physical block ID.
        """
        return self._request_finished_impl(request, block_ids)

    def _request_finished_impl(
        self,
        request,
        all_group_ids: "tuple[list[int], ...]",
    ) -> tuple[bool, Optional[dict]]:
        """
        Shared implementation for request_finished and request_finished_all_groups.

        all_group_ids: one list[int] per HMA group (length 1 for standard models).

        Stores per-group block IDs so _worker_execute_store can use the correct
        physical block ID for each KV cache group (they are independent in HMA).
        """
        block_hashes = getattr(request, "block_hashes", None)
        if not block_hashes:
            return False, None

        if not self.request_handler.should_store(block_hashes):
            return False, None

        block_size = self._vllm_block_size
        total_tokens = getattr(request, "num_tokens", len(block_hashes) * block_size)
        num_tokens_per_block = [
            min(block_size, max(0, total_tokens - i * block_size))
            for i in range(len(block_hashes))
        ]

        context_id = None
        extra_meta = getattr(request, "extra_metadata", None)
        if extra_meta and isinstance(extra_meta, dict):
            context_id = extra_meta.get("tierkv_context_id")

        # Sort groups: primary = most non-zero IDs (full-attn), secondary = rest (GLA/SSM).
        # For standard single-group models, primary == secondary (same list).
        _blen = len(block_hashes)
        sorted_groups = sorted(
            all_group_ids,
            key=lambda g: sum(1 for i in range(min(_blen, len(g))) if g[i] != 0),
            reverse=True,
        )
        primary_ids = sorted_groups[0] if sorted_groups else []
        secondary_ids = sorted_groups[1] if len(sorted_groups) > 1 else primary_ids

        # Build hash → [primary_block_id, secondary_block_id] map.
        # primary = tensor-format caches (full-attn), secondary = list-format (GLA/SSM).
        hash_to_group_ids: dict[bytes, list[int]] = {}
        for i, bh in enumerate(block_hashes):
            if bh is None:
                continue
            pid = primary_ids[i] if i < len(primary_ids) else 0
            sid = secondary_ids[i] if i < len(secondary_ids) else pid
            hash_to_group_ids[bh] = [pid, sid]

        pending = self.request_handler.register_pending(
            block_hashes=block_hashes,
            num_tokens_per_block=num_tokens_per_block,
            context_id=context_id,
        )

        if not pending:
            return False, None

        self._store_plans[request.request_id] = [
            (
                r.block_hash.hex(),
                hash_to_group_ids.get(r.block_hash, [0, 0]),
                r.layer_type,
                r.num_tokens,
            )
            for r in pending
        ]
        return True, None

    def build_connector_meta(self, scheduler_output) -> TierKVMeta:
        """Package store/restore plans into metadata for the worker."""
        store_plan = {}
        restore_plan = {}
        restore_block_ids = {}

        for req in getattr(scheduler_output, "scheduled_new_reqs", []):
            # vLLM v1 uses NewRequestData.req_id, not request_id
            req_id = getattr(req, "req_id", None) or getattr(req, "request_id", None)
            if req_id and req_id in self._restore_plans:
                restore_plan[req_id] = [
                    r.vault_key for r in self._restore_plans.pop(req_id)
                ]
                if req_id in self._restore_block_ids:
                    restore_block_ids[req_id] = self._restore_block_ids.pop(req_id)

        for req_id, plan in list(self._store_plans.items()):
            store_plan[req_id] = plan
            del self._store_plans[req_id]

        return TierKVMeta(
            store_plan=store_plan,
            restore_plan=restore_plan,
            restore_block_ids=restore_block_ids,
        )

    # ── Worker side ───────────────────────────────────────────────────────────

    def start_load_kv(self, forward_context, **kwargs) -> None:
        """Initiate async restore from cold vault."""
        meta: Optional[TierKVMeta] = getattr(self, "_connector_metadata", None)
        if not meta:
            return

        for request_id, vault_keys in meta.restore_plan.items():
            load_plan = [
                r for r in (
                    self.registry.lookup_by_vault_key(k) for k in vault_keys
                ) if r is not None
            ]
            if not load_plan:
                continue
            future = self._executor.submit(self.restore_handler.execute, load_plan)
            if not hasattr(forward_context, "_tierkv_restore"):
                forward_context._tierkv_restore = {}
            forward_context._tierkv_restore[request_id] = future

        for request_id, block_specs in meta.store_plan.items():
            self._executor.submit(
                self._worker_execute_store, request_id, block_specs, forward_context
            )

    def register_kv_caches(self, kv_caches) -> None:
        """Store reference to GPU KV cache tensors (called once by vLLM worker)."""
        self._kv_caches = kv_caches

    def _worker_execute_store(self, request_id: str, block_specs: list, forward_context):
        """
        Worker-side tensor extraction and store.

        Reads KV tensors from vLLM's paged GPU buffer, serializes them,
        and dispatches to request_handler.worker_store for quantization + gRPC.

        block_specs: list of (block_hash_hex, group_ids, layer_type, num_tokens)
            group_ids is list[int]: [primary_block_id, secondary_block_id].
            primary = tensor-format caches (full-attn), secondary = list-format (GLA/SSM).
            For standard single-group models both are equal.

        kv_caches layout:
            vLLM v1: dict[str, [K_tensor, V_tensor]] — each value is a list of
            2 tensors with shape (num_blocks, ...).
            Legacy: dict[str, Tensor] or list[Tensor], shape [2, num_blocks, ...].
        """
        import torch

        tensor_map: dict[bytes, bytes] = {}

        # kv_caches is set via register_kv_caches() (vLLM v1 worker hook).
        # Fall back to forward_context for older vLLM versions.
        kv_caches = getattr(self, "_kv_caches", None) or getattr(forward_context, "kv_caches", None)
        if kv_caches is None:
            for block_hash_hex, _group_ids, _layer_type, _num_tokens in block_specs:
                self.registry.mark_failed(bytes.fromhex(block_hash_hex))
            return

        # vLLM v1 register_kv_caches passes a dict {layer_name: tensor}
        caches = list(kv_caches.values()) if isinstance(kv_caches, dict) else kv_caches

        for block_hash_hex, group_ids, layer_type, num_tokens in block_specs:
            block_hash = bytes.fromhex(block_hash_hex)

            # group_ids: [primary_id, secondary_id]
            # primary = tensor-format (full-attn), secondary = list-format (GLA/SSM)
            if not group_ids or group_ids[0] is None:
                self.registry.mark_failed(block_hash)
                continue
            primary_id = group_ids[0]
            secondary_id = group_ids[1] if len(group_ids) > 1 else primary_id

            try:
                layer_tensors = []
                for kv_cache in caches:
                    if kv_cache is None:
                        continue
                    # vLLM v1 list format: [K_tensor, V_tensor] — GLA/SSM caches.
                    # Use secondary_id (GLA group block IDs).
                    if isinstance(kv_cache, (list, tuple)):
                        if len(kv_cache) < 2:
                            continue
                        k_t, v_t = kv_cache[0], kv_cache[1]
                        if not hasattr(k_t, "shape") or k_t.ndim < 1:
                            continue
                        block_id = secondary_id if secondary_id else primary_id
                        if block_id >= k_t.shape[0]:
                            continue
                        layer_tensors.append(k_t[block_id].contiguous().flatten())
                        layer_tensors.append(v_t[block_id].contiguous().flatten())
                    elif hasattr(kv_cache, "ndim"):
                        # Unified paged KV layout: (2, total_pages, page_size, *per_page_shape)
                        # Full-attn caches — use primary_id.
                        if kv_cache.ndim < 3 or kv_cache.shape[0] != 2:
                            continue
                        block_id = primary_id
                        page_size = kv_cache.shape[2]
                        pages_per_block = max(1, self._vllm_block_size // page_size)
                        page_start = block_id * pages_per_block
                        page_end = page_start + pages_per_block
                        if page_end > kv_cache.shape[1]:
                            continue
                        layer_tensors.append(kv_cache[0, page_start:page_end].contiguous().flatten())
                        layer_tensors.append(kv_cache[1, page_start:page_end].contiguous().flatten())

                if layer_tensors:
                    tensor_map[block_hash] = (
                        torch.cat(layer_tensors, dim=0)
                        .to(dtype=torch.float32).cpu().numpy().tobytes()
                    )
                else:
                    self.registry.mark_failed(block_hash)
            except Exception:
                self.registry.mark_failed(block_hash)

        # Reconstruct BlockRecord objects on the worker side.
        # The worker connector instance has an empty registry (scheduler and worker
        # are separate processes in vLLM v1), so we build records from block_specs.
        import time as _time
        pending_records = []
        for block_hash_hex, group_ids, layer_type, num_tokens in block_specs:
            block_hash = bytes.fromhex(block_hash_hex)
            if block_hash not in tensor_map:
                continue
            from tierkv.connectors.vllm.block_registry import BlockRecord
            client = (
                self.ssm_client
                if layer_type in ("ssm", "mamba", "linear_attention")
                else self.kv_client
            )
            if client is None:
                client = self.kv_client
            pending_records.append(BlockRecord(
                block_hash=block_hash,
                vault_node=client.node_id,
                vault_key="",
                layer_type=layer_type,
                num_tokens=num_tokens,
                size_bytes=0,
                evicted_at=_time.time(),
                tensor_hash=b"",
                status="pending",
                context_id=None,
                position=0,
            ))

        # free_fn: called by RequestHandler._store_all after all stores complete
        # (or fail). Adds request_id to _finished_sending so get_finished() can
        # return it → KVConnectorOutput.finished_sending → scheduler frees blocks.
        def _mark_done(req_id: str = request_id) -> None:
            with self._finished_sending_lock:
                self._finished_sending.add(req_id)

        self.request_handler.worker_store(
            pending_records=pending_records,
            tensor_map=tensor_map,
            free_fn=_mark_done,
        )

    def _resolve_block_id(self, forward_context, block_hash_hex: str) -> Optional[int]:
        """Resolve block_hash to physical block_id via vLLM's block manager."""
        block_manager = getattr(forward_context, "block_manager", None)
        if block_manager is None:
            return None
        hash_to_block = getattr(block_manager, "hash_to_block_id", None)
        if hash_to_block and block_hash_hex in hash_to_block:
            return hash_to_block[block_hash_hex]
        evictor = getattr(block_manager, "evictor", None)
        if evictor:
            evicted = getattr(evictor, "hash_to_block_id", None)
            if evicted and block_hash_hex in evicted:
                return evicted[block_hash_hex]
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until restored KV data is written into the GPU buffer."""
        # Use get_forward_context() — the same object start_load_kv used to
        # attach _tierkv_restore.  self._current_forward_context is never set
        # by vLLM's base class so it would always be None.
        try:
            from vllm.forward_context import get_forward_context as _get_fwd
            forward_context = _get_fwd()
        except Exception:
            forward_context = getattr(self, "_current_forward_context", None)
        if forward_context is None:
            return
        restore_futures = getattr(forward_context, "_tierkv_restore", None)
        if not restore_futures:
            return

        import torch
        import numpy as np

        # Use self._kv_caches (set by register_kv_caches) — vLLM v1 does NOT
        # attach kv_caches to forward_context, so getattr(forward_context, "kv_caches")
        # would always return None.
        kv_caches = getattr(self, "_kv_caches", None) or getattr(forward_context, "kv_caches", None)
        if kv_caches is None:
            # Drain futures to avoid memory leaks; can't write back without cache ref.
            for future in restore_futures.values():
                try:
                    future.result()
                except Exception:
                    pass
            forward_context._tierkv_restore.clear()
            return

        caches = list(kv_caches.values()) if isinstance(kv_caches, dict) else list(kv_caches)

        # Per-group block ID map captured at update_state_after_alloc, shipped via TierKVMeta.
        # Format: {req_id: {block_hash_hex: [primary_id, secondary_id]}}
        # primary = tensor-format caches (full-attn), secondary = list-format (GLA/SSM).
        meta = getattr(self, "_connector_metadata", None)
        meta_block_ids: dict[str, dict[str, list[int]]] = (
            getattr(meta, "restore_block_ids", {}) if meta else {}
        )

        for request_id, future in list(restore_futures.items()):
            try:
                restored: dict[bytes, tuple[bytes, int]] = future.result()
            except ValueError:
                # Hash mismatch — vLLM re-prefills via fault recovery
                continue
            except Exception:
                continue

            req_block_map = meta_block_ids.get(request_id, {})

            for block_hash, (tensor_bytes, num_tokens) in restored.items():
                group_ids = req_block_map.get(block_hash.hex())
                if not group_ids:
                    continue

                primary_id = group_ids[0]
                secondary_id = group_ids[1] if len(group_ids) > 1 else primary_id

                flat = np.frombuffer(tensor_bytes, dtype=np.float32)
                tensor = torch.from_numpy(flat.copy())
                offset = 0

                for kv_cache in caches:
                    if kv_cache is None:
                        continue
                    # vLLM v1 list format: [K_tensor(num_blocks,...), V_tensor(num_blocks,...)]
                    # GLA/SSM caches — use secondary_id (GLA group's block IDs).
                    # K and V may have different shapes (MLA: compressed K vs full V).
                    if isinstance(kv_cache, (list, tuple)):
                        if len(kv_cache) < 2:
                            continue
                        k_t, v_t = kv_cache[0], kv_cache[1]
                        if not hasattr(k_t, "shape") or k_t.ndim < 1:
                            continue
                        block_id = secondary_id if secondary_id else primary_id
                        if block_id >= k_t.shape[0]:
                            continue
                        k_elems = k_t[0].numel()
                        v_elems = v_t[0].numel()
                        k_slice = tensor[offset:offset + k_elems].reshape(
                            k_t.shape[1:]
                        ).to(dtype=k_t.dtype, device=k_t.device)
                        k_t[block_id].copy_(k_slice)
                        offset += k_elems
                        v_slice = tensor[offset:offset + v_elems].reshape(
                            v_t.shape[1:]
                        ).to(dtype=v_t.dtype, device=v_t.device)
                        v_t[block_id].copy_(v_slice)
                        offset += v_elems
                    elif hasattr(kv_cache, "ndim"):
                        # Unified paged KV layout: (2, total_pages, page_size, *per_page_shape)
                        # Full-attn caches — use primary_id (full-attn group's block IDs).
                        if kv_cache.ndim < 3 or kv_cache.shape[0] != 2:
                            continue
                        block_id = primary_id
                        page_size = kv_cache.shape[2]
                        pages_per_block = max(1, self._vllm_block_size // page_size)
                        page_start = block_id * pages_per_block
                        page_end = page_start + pages_per_block
                        if page_end > kv_cache.shape[1]:
                            continue
                        per_kv_elems = pages_per_block * page_size
                        for d in kv_cache.shape[3:]:
                            per_kv_elems *= d
                        per_kv_shape = (pages_per_block,) + tuple(kv_cache.shape[2:])

                        k_slice = tensor[offset:offset + per_kv_elems].reshape(
                            per_kv_shape
                        ).to(dtype=kv_cache.dtype, device=kv_cache.device)
                        kv_cache[0, page_start:page_end].copy_(k_slice)
                        offset += per_kv_elems

                        v_slice = tensor[offset:offset + per_kv_elems].reshape(
                            per_kv_shape
                        ).to(dtype=kv_cache.dtype, device=kv_cache.device)
                        kv_cache[1, page_start:page_end].copy_(v_slice)
                        offset += per_kv_elems

        forward_context._tierkv_restore.clear()

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        """NO-OP — TierKV is reactive, not eager."""
        pass

    def wait_for_save(self) -> None:
        stats = self.registry.stats()
        if stats["pending"] > 32:
            import sys
            print(f"[tierkv] {stats['pending']} blocks pending vault confirmation",
                  file=sys.stderr)

    def get_finished(
        self, finished_req_ids: "set[str]"
    ) -> "tuple[set[str] | None, set[str] | None]":
        """
        Worker-side: return request IDs whose cold-tier store has completed.

        Called by the vLLM model runner after each forward pass.
        Return value is packed into KVConnectorOutput.finished_sending and sent
        to the scheduler, which then calls _free_blocks(request) to return GPU
        blocks to the free pool.

        Returns (sending_done, recving_done).
        """
        with self._finished_sending_lock:
            if not self._finished_sending:
                return None, None
            done = self._finished_sending.copy()
            self._finished_sending.clear()
        return done, None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_block_stored(
        self,
        context_id: Optional[str],
        block_position: Optional[int],
        vault_key: str,
    ) -> None:
        """Fires on every successful cold store — updates context_tracker."""
        self.context_tracker.increment_stored(context_id)

    # ── SDK hooks — attach external logic without forking this repo ───────────

    def get_context_stored_count(self, context_id: str) -> int:
        """Number of blocks stored in cold tier for this context_id."""
        return self.context_tracker.get_stored_count(context_id)

    def register_store_callback(self, callback) -> None:
        """
        Register a callback fired after each successful block store.
        Signature: callback(context_id: str | None, block_pos: int | None, vault_key: str)
        """
        self.request_handler._store_callbacks.append(callback)

    def set_context_priority(self, context_id: str, priority: int) -> None:
        """
        Set eviction priority for a context (0=normal, 1=high, 2=critical).
        Higher-priority contexts are evicted last when KV cache is under pressure.
        """
        self.context_tracker.set_priority(context_id, priority)
