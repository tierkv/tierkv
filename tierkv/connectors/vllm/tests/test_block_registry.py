"""Tests for BlockRegistry."""

import threading
import time
import pytest

from tierkv.connectors.vllm.block_registry import BlockRegistry, BlockRecord


def _make_record(hash_byte: int, status="pending") -> BlockRecord:
    return BlockRecord(
        block_hash=bytes([hash_byte]) * 32,
        vault_node="node:50051",
        vault_key=f"vk-{hash_byte:02x}",
        layer_type="full_attention",
        num_tokens=16,
        size_bytes=4096,
        evicted_at=time.time(),
        tensor_hash=b"\x00" * 32,
        status=status,
    )


class TestBlockRegistry:
    def test_register_and_lookup_stored(self, registry):
        rec = _make_record(0x01)
        registry.register(rec)

        # Pending record not returned by lookup_stored
        assert registry.lookup_stored(rec.block_hash) is None

        registry.mark_stored(rec.block_hash)
        assert registry.lookup_stored(rec.block_hash) is rec

    def test_mark_failed(self, registry):
        rec = _make_record(0x02)
        registry.register(rec)
        registry.mark_failed(rec.block_hash)

        assert registry.lookup_stored(rec.block_hash) is None
        assert rec.status == "failed"

    def test_lookup_by_vault_key(self, registry):
        rec = _make_record(0x03)
        registry.register(rec)
        registry.mark_stored(rec.block_hash)

        found = registry.lookup_by_vault_key("vk-03")
        assert found is rec

        assert registry.lookup_by_vault_key("nonexistent") is None

    def test_remove(self, registry):
        rec = _make_record(0x04)
        registry.register(rec)
        registry.mark_stored(rec.block_hash)

        registry.remove(rec.block_hash)
        assert registry.lookup_stored(rec.block_hash) is None
        assert registry.lookup_by_vault_key("vk-04") is None

    def test_lookup_many_stored(self, registry):
        r1 = _make_record(0x10)
        r2 = _make_record(0x11)
        r3 = _make_record(0x12)
        registry.register(r1)
        registry.register(r2)
        registry.register(r3)

        registry.mark_stored(r1.block_hash)
        registry.mark_stored(r3.block_hash)
        # r2 stays pending

        result = registry.lookup_many_stored([r1.block_hash, r2.block_hash, r3.block_hash])
        assert r1.block_hash in result
        assert r3.block_hash in result
        assert r2.block_hash not in result

    def test_stats(self, registry):
        for i in range(5):
            registry.register(_make_record(i))

        registry.mark_stored(bytes([0]) * 32)
        registry.mark_stored(bytes([1]) * 32)
        registry.mark_failed(bytes([2]) * 32)

        stats = registry.stats()
        assert stats["total"] == 5
        assert stats["stored"] == 2
        assert stats["pending"] == 2
        assert stats["failed"] == 1

    def test_update_vault_key(self, registry):
        rec = _make_record(0x20)
        registry.register(rec)

        registry.update_vault_key(rec.block_hash, "new-key")
        assert rec.vault_key == "new-key"
        assert registry.lookup_by_vault_key("new-key") is rec
        assert registry.lookup_by_vault_key("vk-20") is None

    def test_thread_safety(self, registry):
        """Concurrent register/lookup should not raise."""
        errors = []

        def writer(start_byte):
            try:
                for i in range(100):
                    rec = _make_record(start_byte + i)
                    rec.block_hash = (start_byte + i).to_bytes(32, "big")
                    rec.vault_key = f"vk-{start_byte}-{i}"
                    registry.register(rec)
                    registry.mark_stored(rec.block_hash)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    registry.stats()
                    registry.lookup_stored(b"\x00" * 32)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(128,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
