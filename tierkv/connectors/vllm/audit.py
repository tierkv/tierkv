"""Audit Event Emitter — tamper-proof event logging for KV block operations.

Provides cryptographic audit trail for:
- Block store events (eviction to cold)
- Block restore events (promotion from cold)
- Chain completion events
- Integrity violations (hash mismatches)
- Priority escalation events

Each event is hash-chained to the previous, forming a tamper-evident log.
If any event is modified or removed, the chain breaks.

Events are emitted to configurable sinks (file, callback, external service).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Callable, Optional


class AuditEventType(Enum):
    BLOCK_STORED = "block_stored"
    BLOCK_RESTORED = "block_restored"
    BLOCK_STORE_FAILED = "block_store_failed"
    BLOCK_RESTORE_FAILED = "block_restore_failed"
    CHAIN_CREATED = "chain_created"
    CHAIN_COMPLETE = "chain_complete"
    INTEGRITY_VIOLATION = "integrity_violation"
    PRIORITY_CHANGED = "priority_changed"
    CONTEXT_REGISTERED = "context_registered"
    CONTEXT_EVICTED = "context_evicted"


@dataclass
class AuditEvent:
    """A single tamper-evident audit event."""
    sequence: int
    event_type: AuditEventType
    timestamp: float
    context_id: Optional[str]
    block_hash_hex: Optional[str]
    vault_key: Optional[str]
    details: dict[str, Any] = field(default_factory=dict)
    prev_hash: bytes = b""           # hash of previous event (chain link)
    event_hash: bytes = b""          # hash of this event (computed on creation)

    def compute_hash(self) -> bytes:
        """Compute SHA-256 of this event's content + prev_hash."""
        payload = (
            f"{self.sequence}:{self.event_type.value}:{self.timestamp}:"
            f"{self.context_id}:{self.block_hash_hex}:{self.vault_key}:"
            f"{json.dumps(self.details, sort_keys=True)}"
        ).encode()
        return hashlib.sha256(self.prev_hash + payload).digest()

    def to_dict(self) -> dict:
        return {
            "seq": self.sequence,
            "type": self.event_type.value,
            "ts": self.timestamp,
            "context_id": self.context_id,
            "block_hash": self.block_hash_hex,
            "vault_key": self.vault_key,
            "details": self.details,
            "prev_hash": self.prev_hash.hex(),
            "event_hash": self.event_hash.hex(),
        }


class AuditEventEmitter:
    """
    Tamper-proof audit event log with hash chaining.

    Usage:
        emitter = AuditEventEmitter()
        emitter.add_sink(file_sink("/var/log/tierkv/audit.jsonl"))
        emitter.add_sink(lambda e: external_service.send(e.to_dict()))

        # Events are automatically hash-chained
        emitter.emit_block_stored(context_id, block_hash_hex, vault_key, size_bytes)
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._sequence: int = 0
        self._last_hash: bytes = b"\x00" * 32  # genesis hash
        self._sinks: list[Callable[[AuditEvent], None]] = []
        self._events: list[AuditEvent] = []  # in-memory buffer (bounded)
        self._max_buffer: int = 10000

    def add_sink(self, sink: Callable[[AuditEvent], None]) -> None:
        """Add an event sink (file writer, callback, external service)."""
        with self._lock:
            self._sinks.append(sink)

    def _emit(
        self,
        event_type: AuditEventType,
        context_id: Optional[str] = None,
        block_hash_hex: Optional[str] = None,
        vault_key: Optional[str] = None,
        **details,
    ) -> AuditEvent:
        with self._lock:
            self._sequence += 1
            event = AuditEvent(
                sequence=self._sequence,
                event_type=event_type,
                timestamp=time.time(),
                context_id=context_id,
                block_hash_hex=block_hash_hex,
                vault_key=vault_key,
                details=details,
                prev_hash=self._last_hash,
            )
            event.event_hash = event.compute_hash()
            self._last_hash = event.event_hash

            # Buffer (ring)
            self._events.append(event)
            if len(self._events) > self._max_buffer:
                self._events = self._events[-self._max_buffer:]

            # Dispatch to sinks
            for sink in self._sinks:
                try:
                    sink(event)
                except Exception:
                    pass

            return event

    # ── Typed emit methods ────────────────────────────────────────────────────

    def emit_block_stored(
        self,
        context_id: Optional[str],
        block_hash_hex: str,
        vault_key: str,
        size_bytes: int = 0,
        layer_type: str = "",
        position: Optional[int] = None,
    ) -> AuditEvent:
        return self._emit(
            AuditEventType.BLOCK_STORED,
            context_id=context_id,
            block_hash_hex=block_hash_hex,
            vault_key=vault_key,
            size_bytes=size_bytes,
            layer_type=layer_type,
            position=position,
        )

    def emit_block_restored(
        self,
        context_id: Optional[str],
        block_hash_hex: str,
        vault_key: str,
        latency_ms: float = 0,
    ) -> AuditEvent:
        return self._emit(
            AuditEventType.BLOCK_RESTORED,
            context_id=context_id,
            block_hash_hex=block_hash_hex,
            vault_key=vault_key,
            latency_ms=latency_ms,
        )

    def emit_store_failed(
        self,
        context_id: Optional[str],
        block_hash_hex: str,
        error: str = "",
    ) -> AuditEvent:
        return self._emit(
            AuditEventType.BLOCK_STORE_FAILED,
            context_id=context_id,
            block_hash_hex=block_hash_hex,
            error=error,
        )

    def emit_restore_failed(
        self,
        context_id: Optional[str],
        block_hash_hex: str,
        error: str = "",
    ) -> AuditEvent:
        return self._emit(
            AuditEventType.BLOCK_RESTORE_FAILED,
            context_id=context_id,
            block_hash_hex=block_hash_hex,
            error=error,
        )

    def emit_integrity_violation(
        self,
        context_id: Optional[str],
        block_hash_hex: str,
        expected_hash: str = "",
        actual_hash: str = "",
    ) -> AuditEvent:
        return self._emit(
            AuditEventType.INTEGRITY_VIOLATION,
            context_id=context_id,
            block_hash_hex=block_hash_hex,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
        )

    def emit_chain_created(
        self,
        context_id: str,
        expected_blocks: int,
        source_hash: Optional[str] = None,
        priority: int = 0,
    ) -> AuditEvent:
        return self._emit(
            AuditEventType.CHAIN_CREATED,
            context_id=context_id,
            expected_blocks=expected_blocks,
            source_hash=source_hash,
            priority=priority,
        )

    def emit_chain_complete(
        self,
        context_id: str,
        chain_hash: str = "",
        total_blocks: int = 0,
    ) -> AuditEvent:
        return self._emit(
            AuditEventType.CHAIN_COMPLETE,
            context_id=context_id,
            chain_hash=chain_hash,
            total_blocks=total_blocks,
        )

    def emit_priority_changed(
        self,
        context_id: str,
        old_priority: int,
        new_priority: int,
    ) -> AuditEvent:
        return self._emit(
            AuditEventType.PRIORITY_CHANGED,
            context_id=context_id,
            old_priority=old_priority,
            new_priority=new_priority,
        )

    # ── Verification ──────────────────────────────────────────────────────────

    def verify_chain_integrity(self) -> tuple[bool, Optional[int]]:
        """
        Verify the hash chain is intact.
        Returns (is_valid, first_broken_sequence_or_None).
        """
        with self._lock:
            prev_hash = b"\x00" * 32
            for event in self._events:
                expected = event.compute_hash()
                if event.event_hash != expected:
                    return False, event.sequence
                if event.prev_hash != prev_hash:
                    return False, event.sequence
                prev_hash = event.event_hash
            return True, None

    def get_events(
        self,
        context_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query events with optional filters."""
        with self._lock:
            results = self._events
            if context_id:
                results = [e for e in results if e.context_id == context_id]
            if event_type:
                results = [e for e in results if e.event_type == event_type]
            if since:
                results = [e for e in results if e.timestamp >= since]
            return results[-limit:]

    @property
    def sequence_number(self) -> int:
        return self._sequence


def file_sink(path: str) -> Callable[[AuditEvent], None]:
    """Create a file-based audit sink (append JSONL)."""
    def _write(event: AuditEvent) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
    return _write
