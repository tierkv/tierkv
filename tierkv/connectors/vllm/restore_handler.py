"""Restore handler — cold load via start_load_kv()."""

from __future__ import annotations

import concurrent.futures
import hashlib
from typing import Optional

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
    # Pre-allocating and pre-touching this buffer at init eliminates the page-fault
    # storm that would otherwise occur on the first decode call inside vLLM's
    # large mmap'd process (~84 GB on DGX GB10).
    _PREWARM_BYTES: int = 128 * 1024 * 1024  # 128 MiB

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
        self._pool_size = 8
        if self.turbo_quant:
            self._buf_pool: list[bytearray] = []
            for _ in range(self._pool_size):
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

    def execute(self, load_plan: list[BlockRecord]) -> dict[bytes, tuple[bytes, int]]:
        """
        Fires BatchPromote RPCs split by vault node.
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

        results: dict[bytes, tuple[bytes, int]] = {}

        for key, grpc_future in grpc_futures.items():
            blocks: list[VllmBlockData] = grpc_future.result()
            records = kv_records if key == "kv" else ssm_records

            for block_data, record in zip(blocks, records):
                if not block_data.payload:
                    continue

                # Tamper detection
                actual_hash = hashlib.sha256(block_data.payload).digest()
                if record.tensor_hash and actual_hash != record.tensor_hash:
                    raise ValueError(
                        f"[tierkv] tensor hash mismatch: {block_data.block_hash_hex[:8]} "
                        f"— cold storage may be corrupted. Forcing re-prefill."
                    )

                # Dequantize — decode_into writes directly into a pre-allocated
                # bytearray, eliminating the 3 × ~102 MB intermediate
                # allocations that cause a page-fault storm inside vLLM's
                # large mmap'd process.  decode_into also releases the GIL.
                if block_data.is_quantized and self.turbo_quant:
                    from tierkv_core import TurboQuant
                    quant = TurboQuant(self._group_size)
                    decoded_len = quant.decoded_size(len(block_data.payload))
                    # Pop a pre-warmed buffer from the pool — pages already
                    # faulted in at init, zero page-fault overhead for the write.
                    # decode_into accepts an oversized buffer (writes only the
                    # first decoded_len bytes).  We return a memoryview slice so
                    # np.frombuffer downstream gets a zero-copy view.
                    if self._buf_pool and decoded_len <= self._PREWARM_BYTES:
                        pool_buf = self._buf_pool[self._pool_idx % self._pool_size]
                        self._pool_idx += 1
                        quant.decode_into(block_data.payload, pool_buf)
                        tensor = memoryview(pool_buf)[:decoded_len]
                    else:
                        fresh = bytearray(decoded_len)
                        quant.decode_into(block_data.payload, fresh)
                        tensor = memoryview(fresh)
                else:
                    tensor = block_data.payload

                results[record.block_hash] = (tensor, record.num_tokens)

        return results
