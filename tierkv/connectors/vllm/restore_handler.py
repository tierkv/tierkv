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
                    buf = bytearray(decoded_len)
                    quant.decode_into(block_data.payload, buf)
                    tensor = buf  # bytearray satisfies buffer protocol — np.frombuffer accepts it
                else:
                    tensor = block_data.payload

                results[record.block_hash] = (tensor, record.num_tokens)

        return results
