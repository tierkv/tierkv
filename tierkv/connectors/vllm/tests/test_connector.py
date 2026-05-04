"""Tests for TierKVConnector — scheduler-side logic with mocked worker."""

import time
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional

from tierkv.connectors.vllm.block_registry import BlockRegistry, BlockRecord
from tierkv.connectors.vllm.tests.conftest import MockVaultClient, MockRequest


@dataclass
class MockSchedulerOutput:
    scheduled_new_reqs: list = field(default_factory=list)


@pytest.fixture
def connector():
    """Create TierKVConnector with mocked vLLM imports and vault clients."""
    # Mock vLLM imports
    mock_base = MagicMock()
    mock_base.KVConnectorBase_V1 = object
    mock_base.KVConnectorMetadata = object
    mock_base.KVConnectorRole = MagicMock()
    mock_base.KVConnectorRole.SCHEDULER = 0

    with patch.dict("sys.modules", {
        "vllm": MagicMock(),
        "vllm.distributed": MagicMock(),
        "vllm.distributed.kv_transfer": MagicMock(),
        "vllm.distributed.kv_transfer.kv_connector": MagicMock(),
        "vllm.distributed.kv_transfer.kv_connector.v1": MagicMock(),
        "vllm.distributed.kv_transfer.kv_connector.v1.base": mock_base,
    }):
        # Re-import with mocked vLLM
        import importlib
        import tierkv.connectors.vllm.connector as conn_module
        importlib.reload(conn_module)

        # Patch the base class to be a plain object
        original_init = conn_module.TierKVConnector.__init__

        def patched_init(self, *args, **kwargs):
            # Skip super().__init__ since base is mocked
            self.cfg = conn_module.VllmConnectorConfig()
            self.registry = BlockRegistry()
            self.context_tracker = conn_module.ContextTracker()

            client = MockVaultClient()
            self.kv_client = client
            self.ssm_client = None

            from tierkv.connectors.vllm.request_handler import RequestHandler
            from tierkv.connectors.vllm.restore_handler import RestoreHandler

            self.request_handler = RequestHandler(
                registry=self.registry,
                kv_client=client,
                ssm_client=None,
                turbo_quant=False,
                max_concurrent_stores=2,
            )
            self.request_handler._group_size = 128
            self.request_handler._layer_type_map = {}

            self.restore_handler = RestoreHandler(
                registry=self.registry,
                kv_client=client,
                ssm_client=None,
                turbo_quant=False,
                group_size=128,
            )

            self._restore_plans = {}
            self._store_plans = {}
            import concurrent.futures
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="tierkv-test"
            )

        conn_module.TierKVConnector.__init__ = patched_init
        connector = conn_module.TierKVConnector.__new__(conn_module.TierKVConnector)
        patched_init(connector)
        return connector


class TestGetNumNewMatchedTokens:
    def test_returns_none_when_no_cold_hits(self, connector):
        req = MockRequest(block_hashes=[b"\x01" * 32, b"\x02" * 32])
        result, is_async = connector.get_num_new_matched_tokens(req, 0)
        assert result is None
        assert is_async is False

    def test_returns_count_when_hits_exist(self, connector):
        # Pre-store some blocks
        h1, h2 = b"\x01" * 32, b"\x02" * 32
        for h in (h1, h2):
            rec = BlockRecord(
                block_hash=h, vault_node="n", vault_key=f"vk-{h.hex()[:4]}",
                layer_type="full_attention", num_tokens=16, size_bytes=100,
                evicted_at=time.time(), tensor_hash=b"\x00" * 32, status="stored",
            )
            connector.registry.register(rec)
            connector.registry.mark_stored(h)

        req = MockRequest(
            request_id="req-match",
            block_hashes=[h1, h2, b"\x03" * 32],
        )
        result, is_async = connector.get_num_new_matched_tokens(req, 0)
        assert result == 32  # 2 blocks × 16 tokens
        assert "req-match" in connector._restore_plans

    def test_returns_none_when_no_block_hashes(self, connector):
        req = MockRequest(block_hashes=[])
        result, _ = connector.get_num_new_matched_tokens(req, 0)
        assert result is None


class TestRequestFinished:
    def test_returns_false_when_nothing_to_store(self, connector):
        req = MockRequest(block_hashes=[])
        result, meta = connector.request_finished(req, [])
        assert result is False

    def test_returns_true_and_registers_pending(self, connector):
        req = MockRequest(
            request_id="req-evict",
            block_hashes=[b"\x10" * 32, b"\x11" * 32],
            num_tokens=32,
        )
        result, meta = connector.request_finished(req, [0, 1])
        assert result is True
        assert "req-evict" in connector._store_plans
        assert connector.registry.stats()["pending"] == 2

    def test_returns_false_when_all_already_stored(self, connector):
        h1 = b"\x20" * 32
        rec = BlockRecord(
            block_hash=h1, vault_node="n", vault_key="vk-x",
            layer_type="full_attention", num_tokens=16, size_bytes=100,
            evicted_at=time.time(), tensor_hash=b"", status="stored",
        )
        connector.registry.register(rec)
        connector.registry.mark_stored(h1)

        req = MockRequest(block_hashes=[h1], num_tokens=16)
        result, _ = connector.request_finished(req, [0])
        assert result is False


class TestBuildConnectorMeta:
    def test_packages_store_and_restore_plans(self, connector):
        # Set up a restore plan
        rec = BlockRecord(
            block_hash=b"\x30" * 32, vault_node="n", vault_key="vk-30",
            layer_type="full_attention", num_tokens=16, size_bytes=100,
            evicted_at=time.time(), tensor_hash=b"", status="stored",
        )
        connector._restore_plans["req-restore"] = [rec]

        # Set up a store plan
        connector._store_plans["req-store"] = [
            ("aa" * 32, "full_attention", 16)
        ]

        mock_output = MockSchedulerOutput(
            scheduled_new_reqs=[MockRequest(request_id="req-restore")]
        )
        meta = connector.build_connector_meta(mock_output)

        assert "req-restore" in meta.restore_plan
        assert meta.restore_plan["req-restore"] == ["vk-30"]
        assert "req-store" in meta.store_plan


class TestSaveKvLayer:
    def test_is_noop(self, connector):
        # Should not raise
        connector.save_kv_layer("layer_0", None, None)
