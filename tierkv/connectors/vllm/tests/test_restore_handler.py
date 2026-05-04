"""Tests for RestoreHandler."""

import hashlib
import time
import pytest

from tierkv.connectors.vllm.block_registry import BlockRegistry, BlockRecord
from tierkv.connectors.vllm.restore_handler import RestoreHandler
from tierkv.connectors.vllm.tests.conftest import MockVaultClient


def _stored_record(registry, hash_byte, vault_key, payload=b"data"):
    """Helper to create and register a stored record."""
    block_hash = bytes([hash_byte]) * 32
    tensor_hash = hashlib.sha256(payload).digest()
    rec = BlockRecord(
        block_hash=block_hash,
        vault_node="mock:50051",
        vault_key=vault_key,
        layer_type="full_attention",
        num_tokens=16,
        size_bytes=len(payload),
        evicted_at=time.time(),
        tensor_hash=tensor_hash,
        status="stored",
    )
    registry.register(rec)
    registry.mark_stored(block_hash)
    return rec


@pytest.fixture
def restore_handler(registry, mock_client):
    return RestoreHandler(
        registry=registry,
        kv_client=mock_client,
        ssm_client=None,
        turbo_quant=False,
        group_size=128,
    )


class TestPlan:
    def test_returns_zero_for_no_hits(self, restore_handler):
        matched, plan = restore_handler.plan([b"\x01" * 32, b"\x02" * 32])
        assert matched == 0
        assert plan == []

    def test_stops_at_first_miss(self, restore_handler, registry):
        _stored_record(registry, 0x01, "vk-1")
        _stored_record(registry, 0x02, "vk-2")
        # 0x03 not stored

        hashes = [bytes([0x01]) * 32, bytes([0x02]) * 32, bytes([0x03]) * 32]
        matched, plan = restore_handler.plan(hashes)

        assert matched == 32  # 2 blocks × 16 tokens
        assert len(plan) == 2

    def test_skips_pending_blocks(self, restore_handler, registry):
        # First block stored, second pending
        _stored_record(registry, 0x10, "vk-10")

        pending_hash = bytes([0x11]) * 32
        pending_rec = BlockRecord(
            block_hash=pending_hash, vault_node="n", vault_key="vk-11",
            layer_type="full_attention", num_tokens=16, size_bytes=100,
            evicted_at=time.time(), tensor_hash=b"", status="pending",
        )
        registry.register(pending_rec)

        hashes = [bytes([0x10]) * 32, pending_hash, bytes([0x12]) * 32]
        matched, plan = restore_handler.plan(hashes)

        # Stops at pending (treated as miss)
        assert matched == 16
        assert len(plan) == 1


class TestExecute:
    def test_empty_plan_returns_empty(self, restore_handler):
        result = restore_handler.execute([])
        assert result == {}

    def test_promotes_and_returns_data(self, restore_handler, registry, mock_client):
        payload = b"\xAA" * 256
        rec = _stored_record(registry, 0x20, "vk-20", payload=payload)

        # Pre-populate mock store
        mock_client._store["vk-20"] = (rec.block_hash.hex(), payload, False)

        result = restore_handler.execute([rec])
        assert rec.block_hash in result
        assert result[rec.block_hash] == payload
        assert len(mock_client.promote_calls) == 1

    def test_raises_on_hash_mismatch(self, restore_handler, registry, mock_client):
        original_payload = b"\xBB" * 256
        rec = _stored_record(registry, 0x30, "vk-30", payload=original_payload)

        # Store tampered data
        tampered = b"\xCC" * 256
        mock_client._store["vk-30"] = (rec.block_hash.hex(), tampered, False)

        with pytest.raises(ValueError, match="tensor hash mismatch"):
            restore_handler.execute([rec])

    def test_splits_kv_and_ssm(self, restore_handler, registry, mock_client):
        kv_payload = b"\x01" * 128
        ssm_payload = b"\x02" * 128

        kv_rec = _stored_record(registry, 0x40, "vk-40", payload=kv_payload)
        ssm_hash = bytes([0x41]) * 32
        ssm_rec = BlockRecord(
            block_hash=ssm_hash,
            vault_node="mock:50051",
            vault_key="vk-41",
            layer_type="ssm",
            num_tokens=16,
            size_bytes=len(ssm_payload),
            evicted_at=time.time(),
            tensor_hash=hashlib.sha256(ssm_payload).digest(),
            status="stored",
        )
        registry.register(ssm_rec)
        registry.mark_stored(ssm_hash)

        mock_client._store["vk-40"] = (kv_rec.block_hash.hex(), kv_payload, False)
        mock_client._store["vk-41"] = (ssm_hash.hex(), ssm_payload, False)

        result = restore_handler.execute([kv_rec, ssm_rec])
        assert kv_rec.block_hash in result
        assert ssm_hash in result
