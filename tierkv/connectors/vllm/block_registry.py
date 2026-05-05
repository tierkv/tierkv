"""Thread-safe block registry: block_hash -> BlockRecord."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Literal, Optional
import time

BlockStatus = Literal["pending", "stored", "failed"]


@dataclass
class BlockRecord:
    block_hash: bytes
    vault_node: str
    vault_key: str
    layer_type: str  # "full_attention" | "sliding_window" | "ssm"
    num_tokens: int
    size_bytes: int
    evicted_at: float
    tensor_hash: bytes
    status: BlockStatus = "pending"
    # SDK hooks (optional, generic)
    context_id: Optional[str] = None
    position: Optional[int] = None


class BlockRegistry:
    """
    Thread-safe index: block_hash -> BlockRecord.

    Written by RequestHandler when blocks are evicted.
    Read by RestoreHandler when new requests arrive.

    The scheduler MUST use lookup_stored() not lookup().
    A "pending" block must be treated as a cold miss.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[bytes, BlockRecord] = {}
        self._vault_index: dict[str, bytes] = {}

    def register(self, record: BlockRecord) -> None:
        with self._lock:
            self._records[record.block_hash] = record
            if record.vault_key:
                self._vault_index[record.vault_key] = record.block_hash

    def mark_stored(self, block_hash: bytes) -> None:
        with self._lock:
            rec = self._records.get(block_hash)
            if rec:
                rec.status = "stored"

    def mark_failed(self, block_hash: bytes) -> None:
        with self._lock:
            rec = self._records.get(block_hash)
            if rec:
                rec.status = "failed"

    def lookup_stored(self, block_hash: bytes) -> Optional[BlockRecord]:
        """Returns record only if status == 'stored'. None otherwise."""
        with self._lock:
            rec = self._records.get(block_hash)
            return rec if rec and rec.status == "stored" else None

    def is_pending(self, block_hash: bytes) -> bool:
        """Returns True if the block is registered but still being stored."""
        with self._lock:
            rec = self._records.get(block_hash)
            return rec is not None and rec.status == "pending"

    def lookup_by_vault_key(self, vault_key: str) -> Optional[BlockRecord]:
        with self._lock:
            block_hash = self._vault_index.get(vault_key)
            if block_hash:
                return self._records.get(block_hash)
            return None

    def lookup_many_stored(self, block_hashes: list[bytes]) -> dict[bytes, BlockRecord]:
        with self._lock:
            return {
                h: self._records[h]
                for h in block_hashes
                if h in self._records and self._records[h].status == "stored"
            }

    def remove(self, block_hash: bytes) -> None:
        with self._lock:
            rec = self._records.pop(block_hash, None)
            if rec and rec.vault_key:
                self._vault_index.pop(rec.vault_key, None)

    def update_vault_key(self, block_hash: bytes, vault_key: str) -> None:
        with self._lock:
            rec = self._records.get(block_hash)
            if rec:
                if rec.vault_key:
                    self._vault_index.pop(rec.vault_key, None)
                rec.vault_key = vault_key
                self._vault_index[vault_key] = block_hash

    def purge_stale(self, max_age_seconds: float = 300.0) -> int:
        """
        Remove records that are no longer useful:
        - failed records older than max_age_seconds
        - stored records older than max_age_seconds (vault TTL cleanup)

        Returns number of records removed.
        """
        cutoff = time.time() - max_age_seconds
        to_remove = []
        with self._lock:
            for block_hash, rec in self._records.items():
                if rec.evicted_at < cutoff and rec.status in ("failed", "stored"):
                    to_remove.append(block_hash)
        for block_hash in to_remove:
            self.remove(block_hash)
        return len(to_remove)

    def expire_pending(self, ttl_seconds: float = 30.0) -> int:
        """
        Mark pending records that have been stuck longer than ttl_seconds as failed.
        Prevents stale pending blocks from blocking prefix restore indefinitely.

        Returns number of records transitioned.
        """
        cutoff = time.time() - ttl_seconds
        expired = 0
        with self._lock:
            for rec in self._records.values():
                if rec.status == "pending" and rec.evicted_at < cutoff:
                    rec.status = "failed"
                    expired += 1
        return expired

    def stats(self) -> dict:
        with self._lock:
            statuses = [r.status for r in self._records.values()]
            return {
                "total": len(statuses),
                "stored": statuses.count("stored"),
                "pending": statuses.count("pending"),
                "failed": statuses.count("failed"),
            }
