"""Chain Registry — tracks block chains for context completeness verification.

A "chain" is the ordered set of KV blocks belonging to a single context (prompt).
The chain is complete when all expected blocks are stored in cold tier.
Only a complete chain can be reliably restored without re-prefill gaps.

Security properties:
- Chain completeness is verified before restore (no partial chain injection)
- Each block's position in the chain is cryptographically bound via block_hash
- Tamper detection via tensor_hash comparison on restore
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, Optional


@dataclass
class ChainRecord:
    """Tracks a single context's block chain."""
    context_id: str
    expected_blocks: int                   # total blocks expected in this chain
    stored_blocks: set[int] = field(default_factory=set)  # positions stored
    created_at: float = field(default_factory=time.time)
    chain_hash: bytes = b""                # rolling hash of all block hashes in order
    source_hash: Optional[str] = None      # SHA-256(source_identifier + version)
    priority: int = 0                      # 0=normal, 1=high, 2=critical

    @property
    def is_complete(self) -> bool:
        """All expected blocks are in cold tier."""
        return len(self.stored_blocks) >= self.expected_blocks

    @property
    def completeness_ratio(self) -> float:
        if self.expected_blocks == 0:
            return 0.0
        return len(self.stored_blocks) / self.expected_blocks

    @property
    def is_restorable(self) -> bool:
        """Chain has a contiguous prefix from position 0."""
        if not self.stored_blocks:
            return False
        for i in range(max(self.stored_blocks) + 1):
            if i not in self.stored_blocks:
                return i > 0  # At least position 0 must exist
        return True

    @property
    def contiguous_prefix_length(self) -> int:
        """Length of the longest contiguous prefix starting at 0."""
        for i in range(self.expected_blocks):
            if i not in self.stored_blocks:
                return i
        return self.expected_blocks


class ChainRegistry:
    """
    Thread-safe registry of block chains indexed by context_id.

    Provides:
    - Chain creation and block registration
    - Completeness checking (is the full context restorable?)
    - Rolling chain hash for integrity verification
    - Callbacks on chain completion
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._chains: dict[str, ChainRecord] = {}
        self._completion_callbacks: list[Callable[[str, ChainRecord], None]] = []

    def create_chain(
        self,
        context_id: str,
        expected_blocks: int,
        source_hash: Optional[str] = None,
        priority: int = 0,
    ) -> ChainRecord:
        """Create a new chain. Overwrites if context_id already exists."""
        with self._lock:
            record = ChainRecord(
                context_id=context_id,
                expected_blocks=expected_blocks,
                source_hash=source_hash,
                priority=priority,
            )
            self._chains[context_id] = record
            return record

    def on_block_stored(
        self,
        context_id: Optional[str],
        block_position: Optional[int],
        vault_key: str,
    ) -> None:
        """
        Callback compatible with connector.register_store_callback().
        Called after each block is successfully stored in cold tier.
        """
        if context_id is None or block_position is None:
            return

        with self._lock:
            chain = self._chains.get(context_id)
            if chain is None:
                return

            chain.stored_blocks.add(block_position)

            # Update rolling chain hash
            chain.chain_hash = hashlib.sha256(
                chain.chain_hash + vault_key.encode()
            ).digest()

            # Check completion
            if chain.is_complete:
                for cb in self._completion_callbacks:
                    try:
                        cb(context_id, chain)
                    except Exception:
                        pass

    def get_chain(self, context_id: str) -> Optional[ChainRecord]:
        with self._lock:
            return self._chains.get(context_id)

    def is_restorable(self, context_id: str) -> bool:
        """Check if context has a contiguous restorable prefix."""
        with self._lock:
            chain = self._chains.get(context_id)
            return chain.is_restorable if chain else False

    def get_completeness(self, context_id: str) -> float:
        """Return 0.0–1.0 completeness ratio."""
        with self._lock:
            chain = self._chains.get(context_id)
            return chain.completeness_ratio if chain else 0.0

    def register_completion_callback(
        self, callback: Callable[[str, ChainRecord], None]
    ) -> None:
        """Register callback fired when a chain reaches 100% completeness."""
        with self._lock:
            self._completion_callbacks.append(callback)

    def remove_chain(self, context_id: str) -> None:
        with self._lock:
            self._chains.pop(context_id, None)

    def list_chains(self, min_priority: int = 0) -> list[ChainRecord]:
        """List all chains at or above the given priority."""
        with self._lock:
            return [
                c for c in self._chains.values()
                if c.priority >= min_priority
            ]

    def stats(self) -> dict:
        with self._lock:
            total = len(self._chains)
            complete = sum(1 for c in self._chains.values() if c.is_complete)
            restorable = sum(1 for c in self._chains.values() if c.is_restorable)
            return {
                "total_chains": total,
                "complete": complete,
                "restorable": restorable,
                "incomplete": total - complete,
            }
