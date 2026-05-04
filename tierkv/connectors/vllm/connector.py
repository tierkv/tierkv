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

        self.registry = BlockRegistry()
        self.context_tracker = ContextTracker()

        self.kv_client = VllmVaultClient(
            host=self.cfg.kv_cold_host,
            port=self.cfg.kv_cold_port,
        )
        self.ssm_client = VllmVaultClient(
            host=self.cfg.ssm_cold_host,
            port=self.cfg.ssm_cold_port,
        ) if self.cfg.ssm_cold_host else None

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
        self._store_plans: dict[str, list] = {}
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
        pass

    def request_finished(
        self, request, block_ids: list[int]
    ) -> tuple[bool, Optional[dict]]:
        """
        Eviction intercept.
        Returns (True, None) — TierKV owns GPU block free, vLLM waits.
        Returns (False, None) — nothing to store, vLLM frees immediately.
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

        # context_id is optional — set via request.extra_metadata["tierkv_context_id"]
        context_id = None
        extra_meta = getattr(request, "extra_metadata", None)
        if extra_meta and isinstance(extra_meta, dict):
            context_id = extra_meta.get("tierkv_context_id")

        # Build a hash→block_id map from the block_ids vLLM passes us.
        # block_ids are the physical GPU block IDs in the same order as block_hashes.
        hash_to_block_id: dict[bytes, int] = {}
        for i, bh in enumerate(block_hashes):
            if bh is not None and i < len(block_ids):
                hash_to_block_id[bh] = block_ids[i]

        pending = self.request_handler.register_pending(
            block_hashes=block_hashes,
            num_tokens_per_block=num_tokens_per_block,
            context_id=context_id,
        )

        if not pending:
            return False, None

        self._store_plans[request.request_id] = [
            (r.block_hash.hex(), hash_to_block_id.get(r.block_hash), r.layer_type, r.num_tokens)
            for r in pending
        ]
        return True, None

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

        Our registry is hash-based so the group structure doesn't affect logic.
        Flatten all group block_ids and delegate to request_finished.
        """
        flat_ids = [bid for group in block_ids for bid in group]
        return self.request_finished(request, flat_ids)

    def build_connector_meta(self, scheduler_output) -> TierKVMeta:
        """Package store/restore plans into metadata for the worker."""
        store_plan = {}
        restore_plan = {}

        for req in getattr(scheduler_output, "scheduled_new_reqs", []):
            req_id = getattr(req, "request_id", None)
            if req_id and req_id in self._restore_plans:
                restore_plan[req_id] = [
                    r.vault_key for r in self._restore_plans.pop(req_id)
                ]

        for req_id, plan in list(self._store_plans.items()):
            store_plan[req_id] = plan
            del self._store_plans[req_id]

        return TierKVMeta(store_plan=store_plan, restore_plan=restore_plan)

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

        block_specs: list of (block_hash_hex, block_id, layer_type, num_tokens)
            block_id is the physical GPU block index, captured in request_finished
            from the block_ids argument vLLM passes (same order as block_hashes).

        kv_caches layout (vLLM PagedAttention):
            dict[str, torch.Tensor] or list[torch.Tensor], one entry per layer.
            Each tensor: [2, num_blocks, block_size, num_heads, head_dim]
            Index 0 = keys, index 1 = values
        """
        import torch

        tensor_map: dict[bytes, bytes] = {}

        # kv_caches is set via register_kv_caches() (vLLM v1 worker hook).
        # Fall back to forward_context for older vLLM versions.
        kv_caches = getattr(self, "_kv_caches", None) or getattr(forward_context, "kv_caches", None)
        if kv_caches is None:
            for block_hash_hex, _block_id, _layer_type, _num_tokens in block_specs:
                self.registry.mark_failed(bytes.fromhex(block_hash_hex))
            return

        # vLLM v1 register_kv_caches passes a dict {layer_name: tensor}
        caches = list(kv_caches.values()) if isinstance(kv_caches, dict) else kv_caches

        for block_hash_hex, block_id, layer_type, num_tokens in block_specs:
            block_hash = bytes.fromhex(block_hash_hex)

            if block_id is None:
                self.registry.mark_failed(block_hash)
                continue

            try:
                layer_tensors = []
                for kv_cache in caches:
                    if kv_cache is None:
                        continue
                    # Only read from full-attention KV caches: shape [2, num_blocks, ...]
                    # SSM/Mamba caches have different shapes — skip them.
                    if kv_cache.ndim < 3 or kv_cache.shape[0] != 2:
                        continue
                    layer_tensors.append(kv_cache[0, block_id, :num_tokens].contiguous())
                    layer_tensors.append(kv_cache[1, block_id, :num_tokens].contiguous())

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
        for block_hash_hex, block_id, layer_type, num_tokens in block_specs:
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

        # free_fn: in vLLM v1 the scheduler handles block reclaim after
        # request_finished returns True; we don't need to free explicitly here.
        self.request_handler.worker_store(
            pending_records=pending_records,
            tensor_map=tensor_map,
            free_fn=lambda: None,
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
        forward_context = getattr(self, "_current_forward_context", None)
        if forward_context is None:
            return
        restore_futures = getattr(forward_context, "_tierkv_restore", None)
        if not restore_futures:
            return

        import torch
        import numpy as np

        kv_caches = getattr(forward_context, "kv_caches", None)
        if kv_caches is None:
            for future in restore_futures.values():
                future.result()
            return

        for request_id, future in list(restore_futures.items()):
            try:
                restored: dict[bytes, bytes] = future.result()
            except ValueError:
                # Hash mismatch — vLLM re-prefills via fault recovery
                continue
            except Exception:
                continue

            for block_hash, tensor_bytes in restored.items():
                record = self.registry.lookup_stored(block_hash)
                if record is None:
                    continue
                block_id = self._resolve_block_id(forward_context, block_hash.hex())
                if block_id is None:
                    continue

                flat = np.frombuffer(tensor_bytes, dtype=np.float32)
                tensor = torch.from_numpy(flat)
                block_size = record.num_tokens
                offset = 0

                for kv_cache in kv_caches:
                    if kv_cache is None:
                        continue
                    # Skip SSM/Mamba caches — only restore to full-attention [2, ...] tensors
                    if kv_cache.ndim < 3 or kv_cache.shape[0] != 2:
                        continue
                    num_heads = kv_cache.shape[-2]
                    head_dim = kv_cache.shape[-1]
                    expected = block_size * num_heads * head_dim

                    k_slice = tensor[offset:offset + expected].reshape(
                        block_size, num_heads, head_dim
                    ).to(dtype=kv_cache.dtype, device=kv_cache.device)
                    kv_cache[0, block_id, :block_size].copy_(k_slice)
                    offset += expected

                    v_slice = tensor[offset:offset + expected].reshape(
                        block_size, num_heads, head_dim
                    ).to(dtype=kv_cache.dtype, device=kv_cache.device)
                    kv_cache[1, block_id, :block_size].copy_(v_slice)
                    offset += expected

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
