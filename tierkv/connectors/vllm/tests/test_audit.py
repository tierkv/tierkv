"""Tests for AuditEventEmitter."""

import json
import tempfile
import pytest

from tierkv.connectors.vllm.audit import (
    AuditEventEmitter,
    AuditEventType,
    AuditEvent,
    file_sink,
)


@pytest.fixture
def emitter():
    return AuditEventEmitter()


class TestAuditEvents:
    def test_emit_block_stored(self, emitter):
        event = emitter.emit_block_stored(
            context_id="ctx-1",
            block_hash_hex="aabb1122",
            vault_key="vk-aabb-1",
            size_bytes=4096,
            layer_type="full_attention",
            position=0,
        )
        assert event.sequence == 1
        assert event.event_type == AuditEventType.BLOCK_STORED
        assert event.context_id == "ctx-1"
        assert event.event_hash != b""

    def test_sequential_numbering(self, emitter):
        emitter.emit_block_stored("a", "h1", "v1")
        emitter.emit_block_stored("a", "h2", "v2")
        e3 = emitter.emit_block_stored("a", "h3", "v3")
        assert e3.sequence == 3

    def test_hash_chaining(self, emitter):
        e1 = emitter.emit_block_stored("a", "h1", "v1")
        e2 = emitter.emit_block_stored("a", "h2", "v2")

        assert e2.prev_hash == e1.event_hash
        assert e1.prev_hash == b"\x00" * 32  # genesis

    def test_verify_chain_integrity_passes(self, emitter):
        for i in range(10):
            emitter.emit_block_stored("ctx", f"h{i}", f"v{i}")

        is_valid, broken_at = emitter.verify_chain_integrity()
        assert is_valid is True
        assert broken_at is None

    def test_verify_chain_detects_tampering(self, emitter):
        emitter.emit_block_stored("ctx", "h1", "v1")
        emitter.emit_block_stored("ctx", "h2", "v2")
        emitter.emit_block_stored("ctx", "h3", "v3")

        # Tamper with event 2
        emitter._events[1].details["size_bytes"] = 999999

        is_valid, broken_at = emitter.verify_chain_integrity()
        assert is_valid is False
        assert broken_at == 2  # event 2 hash no longer matches

    def test_integrity_violation_event(self, emitter):
        event = emitter.emit_integrity_violation(
            context_id="ctx-x",
            block_hash_hex="deadbeef",
            expected_hash="aaa",
            actual_hash="bbb",
        )
        assert event.event_type == AuditEventType.INTEGRITY_VIOLATION
        assert event.details["expected_hash"] == "aaa"
        assert event.details["actual_hash"] == "bbb"

    def test_chain_complete_event(self, emitter):
        event = emitter.emit_chain_complete(
            context_id="ctx-done",
            chain_hash="ff" * 32,
            total_blocks=10,
        )
        assert event.event_type == AuditEventType.CHAIN_COMPLETE
        assert event.details["total_blocks"] == 10

    def test_priority_changed_event(self, emitter):
        event = emitter.emit_priority_changed("ctx-p", old_priority=0, new_priority=2)
        assert event.event_type == AuditEventType.PRIORITY_CHANGED
        assert event.details["old_priority"] == 0
        assert event.details["new_priority"] == 2


class TestAuditQuery:
    def test_filter_by_context(self, emitter):
        emitter.emit_block_stored("ctx-a", "h1", "v1")
        emitter.emit_block_stored("ctx-b", "h2", "v2")
        emitter.emit_block_stored("ctx-a", "h3", "v3")

        events = emitter.get_events(context_id="ctx-a")
        assert len(events) == 2
        assert all(e.context_id == "ctx-a" for e in events)

    def test_filter_by_type(self, emitter):
        emitter.emit_block_stored("a", "h1", "v1")
        emitter.emit_integrity_violation("a", "h2", "exp", "act")
        emitter.emit_block_stored("a", "h3", "v3")

        events = emitter.get_events(event_type=AuditEventType.INTEGRITY_VIOLATION)
        assert len(events) == 1

    def test_limit(self, emitter):
        for i in range(50):
            emitter.emit_block_stored("ctx", f"h{i}", f"v{i}")

        events = emitter.get_events(limit=5)
        assert len(events) == 5


class TestAuditSinks:
    def test_callback_sink(self, emitter):
        received = []
        emitter.add_sink(lambda e: received.append(e))

        emitter.emit_block_stored("ctx", "h1", "v1")
        assert len(received) == 1
        assert received[0].event_type == AuditEventType.BLOCK_STORED

    def test_file_sink(self, emitter):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        emitter.add_sink(file_sink(path))
        emitter.emit_block_stored("ctx", "h1", "v1")
        emitter.emit_chain_complete("ctx", "ff" * 32, 5)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2

        event1 = json.loads(lines[0])
        assert event1["type"] == "block_stored"
        assert event1["context_id"] == "ctx"
        assert len(event1["event_hash"]) == 64  # hex SHA-256

    def test_sink_failure_does_not_crash(self, emitter):
        def bad_sink(event):
            raise RuntimeError("sink broken")

        emitter.add_sink(bad_sink)
        # Should not raise
        event = emitter.emit_block_stored("ctx", "h1", "v1")
        assert event.sequence == 1


class TestEventSerialization:
    def test_to_dict(self, emitter):
        event = emitter.emit_block_stored("ctx", "h1", "v1", size_bytes=1024)
        d = event.to_dict()
        assert d["type"] == "block_stored"
        assert d["context_id"] == "ctx"
        assert d["block_hash"] == "h1"
        assert d["vault_key"] == "v1"
        assert d["details"]["size_bytes"] == 1024
        assert isinstance(d["prev_hash"], str)
        assert isinstance(d["event_hash"], str)
