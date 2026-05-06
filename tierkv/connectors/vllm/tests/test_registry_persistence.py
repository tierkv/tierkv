"""Tests for PersistentBlockRegistry — SQLite write-behind persistence."""

from __future__ import annotations

import tempfile
import time
import os

import pytest

from tierkv.connectors.vllm.block_registry import BlockRecord
from tierkv.connectors.vllm.registry_persistence import PersistentBlockRegistry


def _make_record(block_hash: bytes, vault_key: str = "vk-1") -> BlockRecord:
    return BlockRecord(
        block_hash=block_hash,
        vault_node="kv:50051",
        vault_key=vault_key,
        layer_type="full_attention",
        num_tokens=16,
        size_bytes=256,
        evicted_at=time.time(),
        tensor_hash=b"\xab" * 32,
        status="stored",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "tierkv" / "registry.db")


class TestPersistentBlockRegistry:

    def test_creates_db_file(self, db_path):
        reg = PersistentBlockRegistry(db_path)
        reg.close()
        assert os.path.exists(db_path)

    def test_mark_stored_persists_to_disk(self, db_path):
        block_hash = b"\x01" * 32
        reg = PersistentBlockRegistry(db_path)
        rec = _make_record(block_hash, "vk-persist-1")
        reg.register(rec)
        reg.mark_stored(block_hash)
        reg.close()  # flushes write queue

        # Re-open: should load from disk
        reg2 = PersistentBlockRegistry(db_path)
        stored = reg2.lookup_stored(block_hash)
        reg2.close()
        assert stored is not None
        assert stored.vault_key == "vk-persist-1"
        assert stored.num_tokens == 16

    def test_remove_deletes_from_disk(self, db_path):
        block_hash = b"\x02" * 32
        reg = PersistentBlockRegistry(db_path)
        rec = _make_record(block_hash, "vk-remove-me")
        reg.register(rec)
        reg.mark_stored(block_hash)
        reg.remove(block_hash)
        reg.close()

        reg2 = PersistentBlockRegistry(db_path)
        stored = reg2.lookup_stored(block_hash)
        reg2.close()
        assert stored is None

    def test_pending_records_not_persisted(self, db_path):
        block_hash = b"\x03" * 32
        reg = PersistentBlockRegistry(db_path)
        rec = _make_record(block_hash, "vk-pending")
        rec.status = "pending"
        reg.register(rec)
        # Do NOT call mark_stored — stays pending
        reg.close()

        reg2 = PersistentBlockRegistry(db_path)
        stored = reg2.lookup_stored(block_hash)
        reg2.close()
        # Pending records are not written to disk
        assert stored is None

    def test_multiple_blocks_survive_restart(self, db_path):
        hashes = [bytes([i]) * 32 for i in range(5)]
        reg = PersistentBlockRegistry(db_path)
        for i, h in enumerate(hashes):
            rec = _make_record(h, f"vk-{i}")
            reg.register(rec)
            reg.mark_stored(h)
        reg.close()

        reg2 = PersistentBlockRegistry(db_path)
        for i, h in enumerate(hashes):
            stored = reg2.lookup_stored(h)
            assert stored is not None, f"block {i} missing after restart"
            assert stored.vault_key == f"vk-{i}"
        reg2.close()

    def test_vault_index_restored_on_load(self, db_path):
        block_hash = b"\x04" * 32
        vault_key = "vk-index-test"
        reg = PersistentBlockRegistry(db_path)
        rec = _make_record(block_hash, vault_key)
        reg.register(rec)
        reg.mark_stored(block_hash)
        reg.close()

        reg2 = PersistentBlockRegistry(db_path)
        found = reg2.lookup_by_vault_key(vault_key)
        reg2.close()
        assert found is not None
        assert found.block_hash == block_hash

    def test_stale_records_pruned_on_load(self, db_path):
        block_hash = b"\x05" * 32
        reg = PersistentBlockRegistry(db_path)
        rec = _make_record(block_hash, "vk-stale")
        # Artificially set evicted_at to 2 days ago
        rec.evicted_at = time.time() - 2 * 86400
        reg.register(rec)
        reg.mark_stored(block_hash)
        reg.close()

        # Load with max_age_seconds=86400 (1 day) — record should be discarded
        reg2 = PersistentBlockRegistry(db_path, max_age_seconds=86400.0)
        stored = reg2.lookup_stored(block_hash)
        reg2.close()
        assert stored is None

    def test_readonly_mode_loads_but_does_not_write(self, db_path):
        block_hash = b"\x06" * 32
        # Write with a read-write registry
        reg = PersistentBlockRegistry(db_path)
        rec = _make_record(block_hash, "vk-ro")
        reg.register(rec)
        reg.mark_stored(block_hash)
        reg.close()

        # Open read-only: should load existing record
        reg_ro = PersistentBlockRegistry(db_path, readonly=True)
        stored = reg_ro.lookup_stored(block_hash)
        assert stored is not None

        # mark_stored on readonly should not crash or write
        new_hash = b"\x07" * 32
        new_rec = _make_record(new_hash, "vk-ro-new")
        reg_ro.register(new_rec)
        reg_ro.mark_stored(new_hash)
        # No writer thread — no error

        # Verify the new block is NOT in the db (readonly didn't write)
        reg2 = PersistentBlockRegistry(db_path)
        assert reg2.lookup_stored(new_hash) is None
        reg2.close()

    def test_in_memory_still_works_while_writing(self, db_path):
        """In-memory reads are immediate even before background write completes."""
        block_hash = b"\x08" * 32
        reg = PersistentBlockRegistry(db_path)
        rec = _make_record(block_hash, "vk-imm")
        reg.register(rec)
        reg.mark_stored(block_hash)
        # Don't close — in-memory should be queryable immediately
        assert reg.lookup_stored(block_hash) is not None
        reg.close()
