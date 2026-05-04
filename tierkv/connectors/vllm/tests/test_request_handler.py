"""Tests for RequestHandler."""

import time
import threading
import pytest
from unittest.mock import patch

from tierkv.connectors.vllm.block_registry import BlockRegistry
from tierkv.connectors.vllm.request_handler import RequestHandler
from tierkv.connectors.vllm.tests.conftest import MockVaultClient


@pytest.fixture
def handler(registry, mock_client):
    h = RequestHandler(
        registry=registry,
        kv_client=mock_client,
        ssm_client=None,
        turbo_quant=False,  # Skip TurboQuant in unit tests
        max_concurrent_stores=2,
    )
    h._group_size = 128
    return h


class TestShouldStore:
    def test_returns_false_for_empty_hashes(self, handler):
        assert handler.should_store([]) is False
        assert handler.should_store([None, None]) is False

    def test_returns_true_for_new_blocks(self, handler):
        assert handler.should_store([b"\x01" * 32, b"\x02" * 32]) is True

    def test_returns_false_when_all_already_stored(self, handler, registry):
        h1, h2 = b"\x01" * 32, b"\x02" * 32
        # Pre-populate as stored
        from tierkv.connectors.vllm.block_registry import BlockRecord
        for h in (h1, h2):
            rec = BlockRecord(
                block_hash=h, vault_node="n", vault_key=f"vk-{h.hex()[:4]}",
                layer_type="full_attention", num_tokens=16, size_bytes=100,
                evicted_at=time.time(), tensor_hash=b"", status="stored",
            )
            registry.register(rec)
            registry.mark_stored(h)

        assert handler.should_store([h1, h2]) is False

    def test_returns_true_when_some_not_stored(self, handler, registry):
        h1, h2 = b"\x01" * 32, b"\x02" * 32
        from tierkv.connectors.vllm.block_registry import BlockRecord
        rec = BlockRecord(
            block_hash=h1, vault_node="n", vault_key="vk-1",
            layer_type="full_attention", num_tokens=16, size_bytes=100,
            evicted_at=time.time(), tensor_hash=b"", status="stored",
        )
        registry.register(rec)
        registry.mark_stored(h1)

        assert handler.should_store([h1, h2]) is True


class TestRegisterPending:
    def test_creates_pending_records(self, handler, registry):
        hashes = [b"\x10" * 32, b"\x11" * 32]
        pending = handler.register_pending(hashes, [16, 16])

        assert len(pending) == 2
        assert all(r.status == "pending" for r in pending)
        assert registry.stats()["pending"] == 2

    def test_skips_already_stored(self, handler, registry):
        h1 = b"\x20" * 32
        from tierkv.connectors.vllm.block_registry import BlockRecord
        rec = BlockRecord(
            block_hash=h1, vault_node="n", vault_key="vk-x",
            layer_type="full_attention", num_tokens=16, size_bytes=100,
            evicted_at=time.time(), tensor_hash=b"", status="stored",
        )
        registry.register(rec)
        registry.mark_stored(h1)

        pending = handler.register_pending([h1, b"\x21" * 32], [16, 16])
        assert len(pending) == 1
        assert pending[0].block_hash == b"\x21" * 32

    def test_context_id_propagated(self, handler):
        hashes = [b"\x30" * 32]
        pending = handler.register_pending(hashes, [16], context_id="ctx-abc")
        assert pending[0].context_id == "ctx-abc"


class TestWorkerStore:
    def test_store_marks_stored(self, handler, registry, mock_client):
        hashes = [b"\x40" * 32]
        pending = handler.register_pending(hashes, [16])

        done = threading.Event()
        original_free = lambda: done.set()

        tensor_map = {b"\x40" * 32: b"\x00" * 512}
        handler.worker_store(pending, tensor_map, original_free)

        done.wait(timeout=5)
        assert registry.lookup_stored(b"\x40" * 32) is not None
        assert len(mock_client.store_calls) == 1

    def test_free_fn_always_called_on_failure(self, handler, registry, mock_client):
        mock_client.should_fail = True
        hashes = [b"\x50" * 32]
        pending = handler.register_pending(hashes, [16])

        done = threading.Event()
        handler.worker_store(pending, {b"\x50" * 32: b"\x00" * 512}, done.set)

        done.wait(timeout=5)
        assert done.is_set()  # free_fn called despite failure
        assert registry.lookup_stored(b"\x50" * 32) is None

    def test_free_fn_called_when_tensor_missing(self, handler, registry):
        hashes = [b"\x60" * 32]
        pending = handler.register_pending(hashes, [16])

        done = threading.Event()
        handler.worker_store(pending, {}, done.set)  # empty tensor_map

        done.wait(timeout=5)
        assert done.is_set()
        rec = registry._records.get(b"\x60" * 32)
        assert rec.status == "failed"
