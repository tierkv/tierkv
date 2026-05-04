"""TierKV vLLM KV Connector — reactive cold tier for evicted blocks.

Launch:
    vllm serve <model> \\
        --kv-connector tierkv.connectors.vllm.connector.TierKVConnector \\
        --kv-connector-extra-config '{"config_path": "/etc/tierkv/tierkv.toml"}' \\
        --enable-prefix-caching \\
        --block-size 16
"""

from __future__ import annotations

import hashlib
import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )
except ImportError:
    # Allow import without vLLM installed (for testing, EXO-only installs)
    KVConnectorBase_V1 = object
    KVConnectorMetadata = object
    KVConnectorRole = None

from tierkv.connectors.vllm.block_registry import BlockRegistry
from tierkv.connectors.vllm.request_handler import RequestHandler
from tierkv.connectors.vllm.restore_handler import RestoreHandler
from tierkv.connectors.vllm.config import VllmConnectorConfig
from tierkv.connectors.vllm.context_tracker import ContextTracker
from tierkv.connectors.vllm.chain_registry import ChainRegistry
from tierkv.connectors.vllm.audit import AuditEventEmitter, file_sink
from tierkv.connectors.vault_client import VllmVaultClient


@dataclass
class TierKVMeta:
    """Passed from scheduler to worker each forward step."""
    # request_id -> list of (block_hash_hex, layer_type, num_tokens)
    store_plan: dict[str, list[tuple[str, str, int]]] = field(default_factory=dict)
    # request_id -> ordered list of vault_keys to restore
    restore_plan: dict[str, list[str]] = field(default_factory=dict)


class TierKVConnector(KVConnectorBase_V1):
    """
    TierKV cold tier for vLLM.

    Reactive eviction model:
    - Stores blocks only when vLLM evicts them (request_finished)
    - Restores blocks on cache miss (start_load_kv)
    - save_kv_layer is a no-op
    """

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)

        extra = {}
        if hasattr(vllm_config, "kv_transfer_config") and vllm_config.kv_transfer_config:
            extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.cfg = VllmConnectorConfig.from_dict(extra)

        self.registry = BlockRegistry()
        self.context_tracker = ContextTracker()
        self.chain_registry = ChainRegistry()
        self.audit = AuditEventEmitter()

        # Configure audit file sink if config specifies a log path
        audit_log = extra.get("audit_log")
        if audit_log:
            self.audit.add_sink(file_sink(audit_log))

        # Wire chain registry into store callbacks
        self.chain_registry.register_completion_callback(self._on_chain_complete)

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

        # Register internal store callback for audit + chain tracking
        self.request_handler._store_callbacks.append(self._on_block_stored_internal)

        self._restore_plans: dict[str, list] = {}
        self._store_plans: dict[str, list] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.cfg.max_inflight_promotes,
            thread_name_prefix="tierkv-worker",
        )

    # ────────────────────────────────────────────────────────────────────────── #
    # SCHEDULER SIDE                                                              #
    # ────────────────────────────────────────────────────────────────────────── #

    def get_num_new_matched_tokens(
        self, request, num_computed_tokens: int
    ) -> tuple[Optional[int], bool]:
        """Check cold tier for matching blocks beyond what's already computed."""
        block_hashes = getattr(request, "block_hashes", None)
        if not block_hashes:
            return None, False

        matched_tokens, load_plan = self.restore_handler.plan(block_hashes)
        if matched_tokens > num_computed_tokens:
            new_tokens = matched_tokens - num_computed_tokens
            self._restore_plans[request.request_id] = load_plan
            return new_tokens, False
        return None, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        pass

    def request_finished(
        self, request, block_ids: list[int]
    ) -> tuple[bool, Optional[dict]]:
        """
        Eviction intercept.
        Returns (True, None) = TierKV owns the GPU block free, vLLM waits.
        Returns (False, None) = nothing to store.
        """
        block_hashes = getattr(request, "block_hashes", None)
        if not block_hashes:
            return False, None

        if not self.request_handler.should_store(block_hashes):
            return False, None

        # Determine tokens per block
        num_tokens_per_block = []
        block_size = self.cfg.block_size
        total_tokens = getattr(request, "num_tokens", len(block_hashes) * block_size)
        for i in range(len(block_hashes)):
            remaining = total_tokens - i * block_size
            num_tokens_per_block.append(min(block_size, max(0, remaining)))

        # Extract context_id if provided via metadata
        context_id = None
        extra_meta = getattr(request, "extra_metadata", None)
        if extra_meta and isinstance(extra_meta, dict):
            context_id = extra_meta.get("tierkv_context_id")

        pending = self.request_handler.register_pending(
            block_hashes=block_hashes,
            num_tokens_per_block=num_tokens_per_block,
            context_id=context_id,
        )

        if not pending:
            return False, None

        # Record store plan for worker
        self._store_plans[request.request_id] = [
            (r.block_hash.hex(), r.layer_type, r.num_tokens)
            for r in pending
        ]

        return True, None

    def build_connector_meta(self, scheduler_output) -> TierKVMeta:
        """Package store/restore plans into metadata for the worker."""
        store_plan = {}
        restore_plan = {}

        # Gather restore plans for newly scheduled requests
        scheduled = getattr(scheduler_output, "scheduled_new_reqs", [])
        for req in scheduled:
            req_id = getattr(req, "request_id", None)
            if req_id and req_id in self._restore_plans:
                rp = self._restore_plans.pop(req_id)
                restore_plan[req_id] = [r.vault_key for r in rp]

        # Gather store plans
        for req_id, plan in list(self._store_plans.items()):
            store_plan[req_id] = plan
            del self._store_plans[req_id]

        return TierKVMeta(store_plan=store_plan, restore_plan=restore_plan)

    # ────────────────────────────────────────────────────────────────────────── #
    # WORKER SIDE                                                                 #
    # ──────────────────────────────────────────────────��─────────────────────── #

    def start_load_kv(self, forward_context, **kwargs) -> None:
        """Initiate async restore from cold vault."""
        meta: Optional[TierKVMeta] = getattr(self, "_connector_metadata", None)
        if not meta:
            return

        # Execute restores
        for request_id, vault_keys in meta.restore_plan.items():
            load_plan = [
                self.registry.lookup_by_vault_key(k)
                for k in vault_keys
            ]
            load_plan = [r for r in load_plan if r is not None]
            if not load_plan:
                continue
            future = self._executor.submit(self.restore_handler.execute, load_plan)
            if not hasattr(forward_context, "_tierkv_restore"):
                forward_context._tierkv_restore = {}
            forward_context._tierkv_restore[request_id] = future

        # Execute stores (worker side — read tensors from GPU, ship to vault)
        for request_id, block_specs in meta.store_plan.items():
            self._executor.submit(
                self._worker_execute_store, request_id, block_specs, forward_context
            )

    def _worker_execute_store(self, request_id: str, block_specs: list, forward_context):
        """
        Worker-side tensor extraction and store.

        Reads KV tensors from vLLM's paged GPU buffer, serializes them,
        and dispatches to request_handler.worker_store for quantization + gRPC.

        The kv_caches layout (from vLLM's PagedAttention):
            kv_caches: list[torch.Tensor]  — one per layer
            Each tensor shape: [2, num_blocks, block_size, num_heads, head_dim]
            Index 0 = keys, index 1 = values

        Block IDs are resolved from the block manager via forward_context.
        """
        import torch

        tensor_map: dict[bytes, bytes] = {}
        block_ids_to_free: list[int] = []

        # Access kv_caches from forward context
        kv_caches = getattr(forward_context, "kv_caches", None)
        # Block table maps request → list of physical block IDs
        block_table = getattr(forward_context, "block_tables", None)

        for block_hash_hex, layer_type, num_tokens in block_specs:
            block_hash = bytes.fromhex(block_hash_hex)

            if kv_caches is None:
                # Fallback: no GPU access (e.g. testing or CPU mode)
                self.registry.mark_failed(block_hash)
                continue

            # Find the physical block_id for this block_hash
            # vLLM maps block_hash → block_id via its block allocator
            block_id = self._resolve_block_id(forward_context, block_hash_hex)
            if block_id is None:
                self.registry.mark_failed(block_hash)
                continue

            try:
                # Extract KV data across all layers for this block
                # Concatenate all layers' K+V for this block into one tensor
                layer_tensors = []
                for layer_idx, kv_cache in enumerate(kv_caches):
                    if kv_cache is None:
                        continue
                    # kv_cache shape: [2, num_blocks, block_size, num_heads, head_dim]
                    k_block = kv_cache[0, block_id, :num_tokens].contiguous()
                    v_block = kv_cache[1, block_id, :num_tokens].contiguous()
                    layer_tensors.append(k_block)
                    layer_tensors.append(v_block)

                if layer_tensors:
                    combined = torch.cat(layer_tensors, dim=0)
                    # Move to CPU and serialize as contiguous float32 bytes
                    tensor_bytes = combined.to(dtype=torch.float32).cpu().numpy().tobytes()
                    tensor_map[block_hash] = tensor_bytes
                    block_ids_to_free.append(block_id)
                else:
                    self.registry.mark_failed(block_hash)
            except Exception:
                self.registry.mark_failed(block_hash)

        def free_fn():
            # Release GPU blocks back to vLLM's block allocator
            block_manager = getattr(forward_context, "block_manager", None)
            if block_manager and block_ids_to_free:
                try:
                    block_manager.free_blocks(block_ids_to_free)
                except Exception:
                    pass  # Best effort — blocks may already be freed by scheduler

        pending_records = [
            self.registry._records.get(bytes.fromhex(h))
            for h, _, _ in block_specs
        ]
        pending_records = [r for r in pending_records if r is not None]

        self.request_handler.worker_store(
            pending_records=pending_records,
            tensor_map=tensor_map,
            free_fn=free_fn,
        )

    def _resolve_block_id(self, forward_context, block_hash_hex: str) -> Optional[int]:
        """Resolve block_hash to physical block_id via vLLM's block manager."""
        block_manager = getattr(forward_context, "block_manager", None)
        if block_manager is None:
            return None
        # vLLM's block manager maintains hash → block_id mapping for prefix caching
        hash_to_block = getattr(block_manager, "hash_to_block_id", None)
        if hash_to_block and block_hash_hex in hash_to_block:
            return hash_to_block[block_hash_hex]
        # Alternative: check evictor's mapping
        evictor = getattr(block_manager, "evictor", None)
        if evictor:
            evicted = getattr(evictor, "hash_to_block_id", None)
            if evicted and block_hash_hex in evicted:
                return evicted[block_hash_hex]
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until restored KV data is written to the GPU buffer.

        After start_load_kv fires async restores, this method blocks until
        the data is available. The restored tensor bytes are written into
        vLLM's paged KV cache at the allocated block slots.
        """
        forward_context = getattr(self, "_current_forward_context", None)
        if forward_context is None:
            return
        restore_futures = getattr(forward_context, "_tierkv_restore", None)
        if not restore_futures:
            return

        import torch

        kv_caches = getattr(forward_context, "kv_caches", None)
        if kv_caches is None:
            # No GPU buffer — just drain futures to avoid leaks
            for future in restore_futures.values():
                future.result()
            return

        # Wait on all restore futures and write results to GPU
        for request_id, future in list(restore_futures.items()):
            try:
                restored: dict[bytes, bytes] = future.result()
            except ValueError:
                # Hash mismatch — vLLM will re-prefill via its fault recovery
                continue
            except Exception:
                continue

            # Write restored tensors into the allocated KV cache blocks
            for block_hash, tensor_bytes in restored.items():
                record = self.registry.lookup_stored(block_hash)
                if record is None:
                    continue
                block_id = self._resolve_block_id(forward_context, block_hash.hex())
                if block_id is None:
                    continue

                # Reconstruct tensor and write into GPU cache
                import numpy as np
                flat = np.frombuffer(tensor_bytes, dtype=np.float32)
                tensor = torch.from_numpy(flat)

                # Distribute across layers: each layer gets K + V
                num_layers = len(kv_caches)
                block_size = record.num_tokens
                if num_layers == 0:
                    continue

                # Total elements per layer = block_size * num_heads * head_dim * 2 (K+V)
                per_layer = len(flat) // num_layers // 2
                offset = 0
                for layer_idx, kv_cache in enumerate(kv_caches):
                    if kv_cache is None:
                        continue
                    head_dim = kv_cache.shape[-1]
                    num_heads = kv_cache.shape[-2]
                    expected = block_size * num_heads * head_dim

                    # Write K
                    k_slice = tensor[offset:offset + expected].reshape(
                        block_size, num_heads, head_dim
                    ).to(dtype=kv_cache.dtype, device=kv_cache.device)
                    kv_cache[0, block_id, :block_size].copy_(k_slice)
                    offset += expected

                    # Write V
                    v_slice = tensor[offset:offset + expected].reshape(
                        block_size, num_heads, head_dim
                    ).to(dtype=kv_cache.dtype, device=kv_cache.device)
                    kv_cache[1, block_id, :block_size].copy_(v_slice)
                    offset += expected

        # Clear processed futures
        forward_context._tierkv_restore.clear()

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        """NO-OP — TierKV is reactive, not eager."""
        pass

    def wait_for_save(self) -> None:
        """Wait for pending stores if backpressure is high."""
        stats = self.registry.stats()
        if stats["pending"] > 32:
            import sys
            print(
                f"[tierkv] {stats['pending']} blocks pending vault confirmation",
                file=sys.stderr,
            )

    # ────────────────────────────────────────────────────────────────────────── #
    # INTERNAL CALLBACKS                                                          #
    # ────────────────────────────────────────────────────────────────────────── #

    def _on_block_stored_internal(
        self,
        context_id: Optional[str],
        block_position: Optional[int],
        vault_key: str,
    ) -> None:
        """Internal callback wired to request_handler — fires on every successful store."""
        # Update context tracker
        self.context_tracker.increment_stored(context_id)

        # Update chain registry
        self.chain_registry.on_block_stored(context_id, block_position, vault_key)

        # Emit audit event
        block_hash_hex = vault_key.split("-")[1] if "-" in vault_key else ""
        self.audit.emit_block_stored(
            context_id=context_id,
            block_hash_hex=block_hash_hex,
            vault_key=vault_key,
            position=block_position,
        )

    def _on_chain_complete(self, context_id: str, chain) -> None:
        """Called when all blocks for a context are stored."""
        self.audit.emit_chain_complete(
            context_id=context_id,
            chain_hash=chain.chain_hash.hex(),
            total_blocks=chain.expected_blocks,
        )

    # ────────────────────────────────────────────────────────────────────────── #
    # SDK HOOKS — context tracking, chain registry, audit                         #
    # ────────────────────────────────────────────────────────────────────────── #

    def set_context_priority(self, context_id: str, priority: int) -> None:
        """Set priority for GPU budget shaping (0=normal, 1=high, 2=critical)."""
        old = self.context_tracker.get_priority(context_id)
        self.context_tracker.set_priority(context_id, priority)
        self.audit.emit_priority_changed(context_id, old, priority)

    def register_store_callback(self, callback) -> None:
        """Register callback: (context_id: str, block_pos: int, vault_key: str)."""
        self.request_handler._store_callbacks.append(callback)

    def get_context_stored_count(self, context_id: str) -> int:
        """Get count of stored blocks for a context."""
        return self.context_tracker.get_stored_count(context_id)

    # ── Chain Registry API ────────────────────────────────────────────────────

    def create_chain(
        self,
        context_id: str,
        expected_blocks: int,
        source_hash: Optional[str] = None,
        priority: int = 0,
    ) -> None:
        """
        Declare a new block chain for a context.

        Must be called before blocks are stored so the chain registry
        can track completeness. The chain is "complete" when all
        expected_blocks positions have been stored to cold tier.

        Args:
            context_id: SHA-256(source + version) or any unique identifier
            expected_blocks: total blocks in this context's KV cache
            source_hash: optional source identifier for provenance
            priority: 0=normal, 1=high, 2=critical
        """
        self.chain_registry.create_chain(
            context_id=context_id,
            expected_blocks=expected_blocks,
            source_hash=source_hash,
            priority=priority,
        )
        self.audit.emit_chain_created(
            context_id=context_id,
            expected_blocks=expected_blocks,
            source_hash=source_hash,
            priority=priority,
        )

    def is_chain_restorable(self, context_id: str) -> bool:
        """Check if context has a contiguous restorable prefix in cold tier."""
        return self.chain_registry.is_restorable(context_id)

    def get_chain_completeness(self, context_id: str) -> float:
        """Return 0.0–1.0 completeness ratio for a context's block chain."""
        return self.chain_registry.get_completeness(context_id)

    def register_chain_completion_callback(self, callback) -> None:
        """Register callback(context_id, chain_record) fired on 100% completeness."""
        self.chain_registry.register_completion_callback(callback)

    def get_chain_stats(self) -> dict:
        """Get aggregate chain statistics."""
        return self.chain_registry.stats()

    # ── Audit API ─────────────────────────────────────────────────────────────

    def add_audit_sink(self, sink) -> None:
        """Add an event sink: callable(AuditEvent) — file, callback, external."""
        self.audit.add_sink(sink)

    def verify_audit_integrity(self) -> tuple[bool, Optional[int]]:
        """
        Verify the audit log hash chain is intact.
        Returns (is_valid, first_broken_sequence_or_None).
        """
        return self.audit.verify_chain_integrity()

    def get_audit_events(
        self,
        context_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> list:
        """Query audit events with optional filters."""
        from tierkv.connectors.vllm.audit import AuditEventType
        evt = None
        if event_type:
            try:
                evt = AuditEventType(event_type)
            except ValueError:
                pass
        return self.audit.get_events(
            context_id=context_id,
            event_type=evt,
            since=since,
            limit=limit,
        )

    def emit_integrity_violation(
        self,
        context_id: Optional[str],
        block_hash_hex: str,
        expected_hash: str,
        actual_hash: str,
    ) -> None:
        """Manually emit an integrity violation event (called by restore handler)."""
        self.audit.emit_integrity_violation(
            context_id=context_id,
            block_hash_hex=block_hash_hex,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
        )
