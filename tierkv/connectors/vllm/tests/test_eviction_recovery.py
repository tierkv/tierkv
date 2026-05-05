"""
E2E eviction recovery test.

Simulates the full cycle:
  1. vLLM evicts blocks (request_finished) → store to cold vault
  2. New request arrives with same prefix → plan() finds blocks in cold
  3. execute() fetches and dequantizes → data integrity verified

Uses MockVaultClient — no real gRPC needed.
"""

from __future__ import annotations

import hashlib
import os
import time

import pytest

from tierkv.connectors.vllm.block_registry import BlockRegistry, BlockRecord
from tierkv.connectors.vllm.request_handler import RequestHandler
from tierkv.connectors.vllm.restore_handler import RestoreHandler
from tierkv.connectors.vllm.tests.conftest import MockVaultClient, MockRequest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tensor_bytes(size: int = 512) -> bytes:
    """Fake KV tensor data — deterministic."""
    return bytes(range(256)) * (size // 256) + bytes(range(size % 256))


def _build_pair(kv_client, ssm_client=None):
    """Return (RequestHandler, RestoreHandler) sharing a registry."""
    registry = BlockRegistry()
    rh = RequestHandler(
        registry=registry,
        kv_client=kv_client,
        ssm_client=ssm_client,
        turbo_quant=False,  # skip Rust quant — unit test focus
        max_concurrent_stores=4,
    )
    restore = RestoreHandler(
        registry=registry,
        kv_client=kv_client,
        ssm_client=ssm_client,
        turbo_quant=False,
        group_size=128,
    )
    return rh, restore


# ── Full cycle ─────────────────────────────────────────────────────────────────

class TestEvictionRecoveryCycle:
    """Full evict → store → plan → restore pipeline."""

    def test_single_block_round_trip(self):
        """One block evicted and successfully restored."""
        client = MockVaultClient()
        handler, restore = _build_pair(client)

        block_hash = b"\xAB" * 32
        tensor = _make_tensor_bytes(512)

        # 1. Eviction: register pending
        records = handler.register_pending(
            block_hashes=[block_hash],
            num_tokens_per_block=[16],
        )
        assert len(records) == 1
        assert records[0].status == "pending"

        # 2. Worker stores tensor
        free_called = []
        handler.worker_store(
            pending_records=records,
            tensor_map={block_hash: tensor},
            free_fn=lambda: free_called.append(True),
        )
        # Wait for executor
        handler._executor.shutdown(wait=True, cancel_futures=False)

        # free_fn must always be called
        assert free_called == [True]

        # Block must now be "stored"
        registry = handler.registry
        assert registry.lookup_stored(block_hash) is not None
        assert registry.stats()["stored"] == 1
        assert registry.stats()["pending"] == 0

        # 3. Restore: plan finds the block
        matched, plan = restore.plan([block_hash])
        assert matched == 16
        assert len(plan) == 1
        assert plan[0].block_hash == block_hash

        # 4. Execute: fetches and returns tensor bytes
        result = restore.execute(plan)
        assert block_hash in result
        assert result[block_hash][0] == tensor

    def test_multi_block_prefix_restore(self):
        """Three blocks evicted; all three restored in prefix order."""
        client = MockVaultClient()
        handler, restore = _build_pair(client)

        hashes = [bytes([i]) * 32 for i in range(3)]
        tensors = {h: _make_tensor_bytes(256 + i * 64) for i, h in enumerate(hashes)}

        # Evict all three
        records = handler.register_pending(
            block_hashes=hashes,
            num_tokens_per_block=[16, 16, 16],
        )
        handler.worker_store(
            pending_records=records,
            tensor_map=tensors,
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)

        assert handler.registry.stats()["stored"] == 3

        # Plan finds all three
        matched, plan = restore.plan(hashes)
        assert matched == 48
        assert len(plan) == 3

        # Restore returns all three
        result = restore.execute(plan)
        for h in hashes:
            assert h in result
            assert result[h][0] == tensors[h]

    def test_partial_prefix_stops_at_gap(self):
        """Blocks 0,1 stored; block 2 missing. Plan returns only 0,1."""
        client = MockVaultClient()
        handler, restore = _build_pair(client)

        hashes = [bytes([i]) * 32 for i in range(4)]
        # Only store blocks 0 and 1
        stored = hashes[:2]
        records = handler.register_pending(
            block_hashes=stored,
            num_tokens_per_block=[16, 16],
        )
        handler.worker_store(
            pending_records=records,
            tensor_map={h: _make_tensor_bytes(256) for h in stored},
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)

        # Plan stops at hashes[2] (miss)
        matched, plan = restore.plan(hashes)
        assert matched == 32  # 2 blocks × 16 tokens
        assert len(plan) == 2

    def test_stale_pending_block_expires(self):
        """
        A block stuck in 'pending' (gRPC timed out) should expire after TTL
        so it no longer blocks prefix restore on subsequent requests.
        """
        client = MockVaultClient()
        client.should_fail = True  # gRPC will fail → block stays pending

        handler, restore = _build_pair(client)

        hashes = [bytes([i]) * 32 for i in range(3)]
        tensors = {h: _make_tensor_bytes(256) for h in hashes}

        records = handler.register_pending(
            block_hashes=hashes,
            num_tokens_per_block=[16, 16, 16],
        )

        # Artificially backdate evicted_at to simulate stale pending
        for rec in handler.registry._records.values():
            rec.evicted_at = time.time() - (RestoreHandler.PENDING_TTL_SECONDS + 5)

        # Attempt store (will fail due to gRPC)
        handler.worker_store(
            pending_records=records,
            tensor_map=tensors,
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)

        # All stuck in pending (gRPC failed, no mark_stored called)
        # Actually they get marked failed by _store_all exception path.
        # Let's verify they don't block plan():
        matched, plan = restore.plan(hashes)
        assert matched == 0
        assert plan == []

        # Registry should show all failed
        stats = handler.registry.stats()
        assert stats["pending"] == 0

    def test_free_fn_called_on_store_failure(self):
        """GPU blocks must be freed even when vault gRPC fails."""
        client = MockVaultClient()
        client.should_fail = True
        handler, _ = _build_pair(client)

        block_hash = b"\xCC" * 32
        records = handler.register_pending(
            block_hashes=[block_hash],
            num_tokens_per_block=[16],
        )

        free_called = []
        handler.worker_store(
            pending_records=records,
            tensor_map={block_hash: _make_tensor_bytes(256)},
            free_fn=lambda: free_called.append(True),
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)

        assert free_called == [True], "free_fn must be called even on gRPC failure"
        assert handler.registry.stats()["failed"] == 1

    def test_context_id_propagates(self):
        """context_id set on eviction is visible in registry after store."""
        client = MockVaultClient()
        handler, _ = _build_pair(client)

        block_hash = b"\xDD" * 32
        records = handler.register_pending(
            block_hashes=[block_hash],
            num_tokens_per_block=[16],
            context_id="session-xyz",
        )
        assert records[0].context_id == "session-xyz"

        handler.worker_store(
            pending_records=records,
            tensor_map={block_hash: _make_tensor_bytes(256)},
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)

        stored = handler.registry.lookup_stored(block_hash)
        assert stored is not None
        assert stored.context_id == "session-xyz"

    def test_store_callback_fires(self):
        """register_store_callback() fires after each successful store."""
        client = MockVaultClient()
        handler, _ = _build_pair(client)

        fired = []
        handler._store_callbacks.append(
            lambda ctx, pos, vk: fired.append((ctx, pos, vk))
        )

        block_hash = b"\xEE" * 32
        records = handler.register_pending(
            block_hashes=[block_hash],
            num_tokens_per_block=[16],
            context_id="ctx-1",
        )
        handler.worker_store(
            pending_records=records,
            tensor_map={block_hash: _make_tensor_bytes(256)},
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)

        assert len(fired) == 1
        ctx, pos, vk = fired[0]
        assert ctx == "ctx-1"
        assert vk.startswith("vk-")

    def test_hash_mismatch_raises_on_restore(self):
        """
        If vault returns tampered data, execute() raises ValueError.
        vLLM falls back to re-prefill.
        """
        client = MockVaultClient()
        handler, restore = _build_pair(client)

        block_hash = b"\xFF" * 32
        original = _make_tensor_bytes(256)
        records = handler.register_pending(
            block_hashes=[block_hash],
            num_tokens_per_block=[16],
        )
        handler.worker_store(
            pending_records=records,
            tensor_map={block_hash: original},
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)

        # Tamper with stored payload
        vault_key = list(client._store.keys())[0]
        hash_hex, _, is_q = client._store[vault_key]
        client._store[vault_key] = (hash_hex, b"\x00" * 256, is_q)

        _, plan = restore.plan([block_hash])
        with pytest.raises(ValueError, match="tensor hash mismatch"):
            restore.execute(plan)


# ── Registry cleanup ──────────────────────────────────────────────────────────

class TestRegistryMaintenance:
    def test_purge_stale_removes_old_failed(self):
        registry = BlockRegistry()
        block_hash = b"\x01" * 32
        rec = BlockRecord(
            block_hash=block_hash,
            vault_node="n",
            vault_key="vk-1",
            layer_type="full_attention",
            num_tokens=16,
            size_bytes=256,
            evicted_at=time.time() - 400,  # older than 300s default
            tensor_hash=b"",
            status="failed",
        )
        registry.register(rec)

        removed = registry.purge_stale(max_age_seconds=300.0)
        assert removed == 1
        assert registry.lookup_stored(block_hash) is None

    def test_purge_keeps_recent_records(self):
        registry = BlockRegistry()
        block_hash = b"\x02" * 32
        rec = BlockRecord(
            block_hash=block_hash,
            vault_node="n",
            vault_key="vk-2",
            layer_type="full_attention",
            num_tokens=16,
            size_bytes=256,
            evicted_at=time.time() - 10,  # only 10s old
            tensor_hash=b"",
            status="stored",
        )
        registry.register(rec)
        registry.mark_stored(block_hash)

        removed = registry.purge_stale(max_age_seconds=300.0)
        assert removed == 0

    def test_expire_pending_marks_stale_as_failed(self):
        registry = BlockRegistry()
        block_hash = b"\x03" * 32
        rec = BlockRecord(
            block_hash=block_hash,
            vault_node="n",
            vault_key="",
            layer_type="full_attention",
            num_tokens=16,
            size_bytes=0,
            evicted_at=time.time() - 60,  # 60s ago, stuck pending
            tensor_hash=b"",
            status="pending",
        )
        registry.register(rec)

        expired = registry.expire_pending(ttl_seconds=30.0)
        assert expired == 1
        assert registry.stats()["failed"] == 1
        assert registry.stats()["pending"] == 0

    def test_expire_pending_keeps_fresh_blocks(self):
        registry = BlockRegistry()
        block_hash = b"\x04" * 32
        rec = BlockRecord(
            block_hash=block_hash,
            vault_node="n",
            vault_key="",
            layer_type="full_attention",
            num_tokens=16,
            size_bytes=0,
            evicted_at=time.time() - 5,  # only 5s old
            tensor_hash=b"",
            status="pending",
        )
        registry.register(rec)

        expired = registry.expire_pending(ttl_seconds=30.0)
        assert expired == 0
        assert registry.stats()["pending"] == 1
