"""Restore handler — cold load via start_load_kv()."""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import time
from typing import Optional

_log = logging.getLogger("tierkv.restore")

from tierkv.connectors.vault_client import VllmVaultClient, VllmBlockData
from tierkv.connectors.vllm.block_registry import BlockRegistry, BlockRecord


class RestoreHandler:
    """
    Handles start_load_kv() — the cold restore path.

    For each incoming request, walks block_hashes in order.
    Finds the longest contiguous prefix of blocks stored in cold tier.
    Fires BatchPromote RPC to fetch them.
    Verifies tensor_hash on each block before returning to vLLM.

    If tensor_hash mismatches: raises ValueError.
    vLLM will re-prefill the affected request.
    """

    # Pending blocks older than this are treated as failed during plan().
    # Prevents a hung gRPC store from blocking prefix restore indefinitely.
    PENDING_TTL_SECONDS: float = 30.0

    # Maximum expected decoded block size.  Sized for a 1056-token HMA block
    # at float32 with the largest head configuration we've seen in production.
    _PREWARM_BYTES: int = 128 * 1024 * 1024  # 128 MiB

    # Number of pre-warmed buffers.  One per possible block in a restore plan —
    # with --num-gpu-blocks-override 40, at most 40 blocks are ever restored at
    # once.  We never cycle (each block in the plan gets its own buffer), so
    # memoryview slices returned to wait_for_layer_load are always valid.
    _POOL_CAPACITY: int = 40

    def __init__(
        self,
        registry: BlockRegistry,
        kv_client: VllmVaultClient,
        ssm_client: Optional[VllmVaultClient],
        turbo_quant: bool = True,
        group_size: int = 128,
    ) -> None:
        self.registry = registry
        self.kv_client = kv_client
        self.ssm_client = ssm_client or kv_client
        self.turbo_quant = turbo_quant
        self._group_size = group_size
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="tierkv-restore",
        )

        # Pre-allocate a pool of decode buffers and pre-touch all their pages at
        # init time, so that page faults happen here rather than during the
        # latency-critical cold restore path.
        #
        # Without this, bytearray(102 MB) inside execute() faults ~25,600 OS
        # pages at ~30 µs each inside vLLM's mmap-heavy process = ~0.77 s per
        # block.  A pool of 8 pre-touched buffers covers all blocks in a typical
        # cold restore batch without reallocation.
        #
        # decode_into accepts a buffer larger than the decoded output (writes to
        # the first decoded_len bytes, ignores the rest), so each pool buffer can
        # be reused across blocks of varying sizes.
        if self.turbo_quant:
            self._buf_pool: list[bytearray] = []
            for _ in range(self._POOL_CAPACITY):
                buf = bytearray(self._PREWARM_BYTES)
                mv = memoryview(buf)
                for offset in range(0, self._PREWARM_BYTES, 4096):
                    mv[offset] = 0
                self._buf_pool.append(buf)
            self._pool_idx: int = 0
        else:
            self._buf_pool = []
            self._pool_idx = 0

    def plan(self, block_hashes: list[bytes]) -> tuple[int, list[BlockRecord]]:
        """
        Walk block_hashes in prefix order.
        Stop at first miss — prefix must be contiguous.
        Only count "stored" blocks — not "pending".

        Stale pending blocks (older than PENDING_TTL_SECONDS) are expired to
        "failed" before the walk so they don't block the prefix indefinitely.

        Returns (matched_token_count, ordered_load_plan).
        """
        self.registry.expire_pending(self.PENDING_TTL_SECONDS)

        load_plan: list[BlockRecord] = []
        for block_hash in block_hashes:
            record = self.registry.lookup_stored(block_hash)
            if record is None:
                break
            load_plan.append(record)

        matched_tokens = sum(r.num_tokens for r in load_plan)
        return matched_tokens, load_plan

    def _decode_one(
        self,
        block_data: VllmBlockData,
        record: BlockRecord,
        pool_buf: Optional[bytearray],
    ) -> tuple[bytes, tuple[object, int]]:
        """
        Decode a single block.  Runs in the thread pool so that multiple
        decode_into calls (which release the GIL) execute concurrently.

        Returns (block_hash, (tensor, num_tokens)).
        Raises ValueError on tensor_hash mismatch.
        """
        # Tamper detection (cheap; SHA-256 also releases GIL in CPython 3.9+)
        actual_hash = hashlib.sha256(block_data.payload).digest()
        if record.tensor_hash and actual_hash != record.tensor_hash:
            raise ValueError(
                f"[tierkv] tensor hash mismatch: {block_data.block_hash_hex[:8]} "
                f"— cold storage may be corrupted. Forcing re-prefill."
            )

        if block_data.is_quantized and self.turbo_quant:
            from tierkv_core import TurboQuant
            quant = TurboQuant(self._group_size)
            decoded_len = quant.decoded_size(len(block_data.payload))
            if pool_buf is not None:
                quant.decode_into(block_data.payload, pool_buf)
                tensor: object = memoryview(pool_buf)[:decoded_len]
            else:
                fresh = bytearray(decoded_len)
                quant.decode_into(block_data.payload, fresh)
                tensor = memoryview(fresh)
        else:
            tensor = block_data.payload

        return record.block_hash, (tensor, record.num_tokens)

    def execute(self, load_plan: list[BlockRecord]) -> dict[bytes, tuple[bytes, int]]:
        """
        Fires BatchPromote RPCs split by vault node, then decodes all blocks
        in parallel using the thread pool (decode_into releases the GIL).
        Returns: block_hash -> (dequantized tensor bytes, num_tokens).

        num_tokens is the token count that was stored for this block (may be
        less than block_size for the last partial block of a request).

        Raises ValueError on tensor_hash mismatch.
        """
        if not load_plan:
            return {}

        # Split by client
        kv_records = [
            r for r in load_plan
            if r.layer_type not in ("ssm", "mamba", "linear_attention")
        ]
        ssm_records = [
            r for r in load_plan
            if r.layer_type in ("ssm", "mamba", "linear_attention")
        ]

        grpc_futures = {}
        if kv_records:
            grpc_futures["kv"] = self._executor.submit(
                self.kv_client.batch_promote_sync,
                [r.vault_key for r in kv_records],
            )
        if ssm_records:
            grpc_futures["ssm"] = self._executor.submit(
                self.ssm_client.batch_promote_sync,
                [r.vault_key for r in ssm_records],
            )

        # Collect (block_data, record) pairs as gRPC results arrive.
        # Pool buffer assignment is single-threaded to avoid index races.
        decode_futures: list[concurrent.futures.Future] = []

        for key, grpc_future in grpc_futures.items():
            blocks: list[VllmBlockData] = grpc_future.result()
            records = kv_records if key == "kv" else ssm_records

            for block_data, record in zip(blocks, records):
                if not block_data.payload:
                    continue

                # Assign a pre-warmed pool buffer (single-threaded).
                if (block_data.is_quantized and self.turbo_quant
                        and self._buf_pool
                        and self._pool_idx < len(self._buf_pool)):
                    pool_buf: Optional[bytearray] = self._buf_pool[self._pool_idx]
                    self._pool_idx += 1
                else:
                    pool_buf = None

                # Dispatch decode to thread pool — GIL is released inside
                # decode_into, so N workers truly run in parallel.
                decode_futures.append(
                    self._executor.submit(self._decode_one, block_data, record, pool_buf)
                )

        results: dict[bytes, tuple[bytes, int]] = {}
        for fut in concurrent.futures.as_completed(decode_futures):
            block_hash, kv = fut.result()  # propagates ValueError on hash mismatch
            results[block_hash] = kv

        return results
