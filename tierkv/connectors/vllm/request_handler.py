"""Request handler — eviction intercept via request_finished()."""

from __future__ import annotations

import concurrent.futures
import hashlib
import time
from typing import Optional

from tierkv.connectors.vault_client import VllmVaultClient
from tierkv.connectors.vllm.block_registry import BlockRegistry, BlockRecord


class RequestHandler:
    """
    Handles request_finished() — the eviction intercept.

    Scheduler side:
        1. Decides which blocks to store (should_store filter)
        2. Records pending store intent in BlockRegistry
        3. Passes block_ids to worker via connector metadata

    Worker side (worker_store):
        1. Reads tensors from GPU buffer using block_ids
        2. Quantizes with TurboQuant INT8
        3. Ships to vault over gRPC
        4. Marks blocks as stored in registry
        5. Calls free_fn to release GPU blocks

    free_fn MUST be called even if gRPC fails.
    """

    def __init__(
        self,
        registry: BlockRegistry,
        kv_client: VllmVaultClient,
        ssm_client: Optional[VllmVaultClient],
        turbo_quant: bool = True,
        max_concurrent_stores: int = 8,
    ) -> None:
        self.registry = registry
        self.kv_client = kv_client
        self.ssm_client = ssm_client or kv_client
        self.turbo_quant = turbo_quant
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent_stores,
            thread_name_prefix="tierkv-store",
        )
        self._group_size: int = 128
        self._layer_type_map: dict = {}
        self._store_callbacks: list = []

    def should_store(self, block_hashes: list[Optional[bytes]]) -> bool:
        """
        Decide whether to store these blocks.

        Store if:
        - At least one block has a valid hash
        - Not all blocks already stored in cold tier
        """
        hashed = [h for h in block_hashes if h is not None]
        if not hashed:
            return False
        return not all(
            self.registry.lookup_stored(h) is not None for h in hashed
        )

    def register_pending(
        self,
        block_hashes: list[Optional[bytes]],
        num_tokens_per_block: list[int],
        context_id: Optional[str] = None,
    ) -> list[BlockRecord]:
        """
        Scheduler side: register blocks as pending before worker stores them.
        Returns list of pending records for the worker.
        """
        pending = []
        for i, block_hash in enumerate(block_hashes):
            if block_hash is None:
                continue
            if self.registry.lookup_stored(block_hash) is not None:
                continue

            layer_type = self._layer_type_map.get(i, "full_attention")
            client = (
                self.ssm_client
                if layer_type in ("ssm", "mamba", "linear_attention")
                else self.kv_client
            )
            record = BlockRecord(
                block_hash=block_hash,
                vault_node=client.node_id,
                vault_key="",
                layer_type=layer_type,
                num_tokens=num_tokens_per_block[i] if i < len(num_tokens_per_block) else 0,
                size_bytes=0,
                evicted_at=time.time(),
                tensor_hash=b"",
                status="pending",
                context_id=context_id,
                position=i,
            )
            self.registry.register(record)
            pending.append(record)
        return pending

    def worker_store(
        self,
        pending_records: list[BlockRecord],
        tensor_map: dict[bytes, bytes],
        free_fn,
    ) -> None:
        """
        Worker side: quantize, store, free.
        Dispatched to executor — returns immediately.
        free_fn is called after all stores complete (or fail).
        """
        self._executor.submit(
            self._store_all, pending_records, tensor_map, free_fn
        )

    def _store_all(
        self,
        pending_records: list[BlockRecord],
        tensor_map: dict[bytes, bytes],
        free_fn,
    ) -> None:
        try:
            for record in pending_records:
                tensor_bytes = tensor_map.get(record.block_hash)
                if tensor_bytes is None:
                    self.registry.mark_failed(record.block_hash)
                    continue
                try:
                    self._store_one(record, tensor_bytes)
                except Exception:
                    self.registry.mark_failed(record.block_hash)
        finally:
            free_fn()

    def _store_one(self, record: BlockRecord, tensor_bytes: bytes) -> None:
        # Quantize
        if self.turbo_quant:
            from tierkv_core import TurboQuant
            quant = TurboQuant(self._group_size)
            payload = bytes(quant.encode(tensor_bytes))
            is_quantized = True
        else:
            payload = tensor_bytes
            is_quantized = False

        tensor_hash = hashlib.sha256(payload).digest()

        # Select client
        client = (
            self.ssm_client
            if record.layer_type in ("ssm", "mamba", "linear_attention")
            else self.kv_client
        )

        # gRPC store
        vault_key = client.store_block_sync(
            block_hash_hex=record.block_hash.hex(),
            payload=payload,
            layer_type=record.layer_type,
            num_tokens=record.num_tokens,
            is_quantized=is_quantized,
        )

        # Update record
        record.size_bytes = len(payload)
        record.tensor_hash = tensor_hash
        self.registry.update_vault_key(record.block_hash, vault_key)
        self.registry.mark_stored(record.block_hash)

        # Fire store callbacks
        for cb in self._store_callbacks:
            try:
                cb(record.context_id, record.position, vault_key)
            except Exception:
                pass

    def pending_count(self) -> int:
        return self.registry.stats()["pending"]
