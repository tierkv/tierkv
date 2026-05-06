"""
Integration tests: vault_client serialization, SSM store routing, registry lookup.

These tests run without a live gRPC server or vLLM — MockVaultClient covers the
network boundary.
"""

from __future__ import annotations

import hashlib
import time

import pytest

from tierkv.connectors.vllm.block_registry import BlockRegistry, BlockRecord
from tierkv.connectors.vllm.request_handler import RequestHandler
from tierkv.connectors.vllm.restore_handler import RestoreHandler
from tierkv.connectors.vllm.tests.conftest import MockVaultClient
from tierkv.connectors.vault_client import (
    _encode_varint,
    _encode_string,
    _encode_bytes,
    _encode_bool,
    _encode_uint32,
    _decode_fields,
    _deserialize_batch_promote_response,
    _deserialize_store_response,
    _serialize_store_request,
    _serialize_batch_promote_request,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tensor(size: int = 256) -> bytes:
    return bytes(range(256)) * (size // 256) + bytes(range(size % 256))


def _build_pair(kv_client, ssm_client=None, turbo_quant=False):
    registry = BlockRegistry()
    rh = RequestHandler(
        registry=registry,
        kv_client=kv_client,
        ssm_client=ssm_client,
        turbo_quant=turbo_quant,
        max_concurrent_stores=2,
    )
    restore = RestoreHandler(
        registry=registry,
        kv_client=kv_client,
        ssm_client=ssm_client,
        turbo_quant=turbo_quant,
        group_size=128,
    )
    return rh, restore


# ── Protobuf serialization round-trips ────────────────────────────────────────

class TestVaultClientSerialization:
    """Manual proto encode/decode round-trips in vault_client.py."""

    def test_encode_varint_small(self):
        assert _encode_varint(0) == b"\x00"
        assert _encode_varint(1) == b"\x01"
        assert _encode_varint(127) == b"\x7f"

    def test_encode_varint_multibyte(self):
        # 128 encodes as two bytes: 0x80 0x01
        assert _encode_varint(128) == b"\x80\x01"
        assert _encode_varint(300) == b"\xac\x02"

    def test_encode_string_round_trip(self):
        encoded = _encode_string(1, "hello")
        fields = _decode_fields(encoded)
        assert len(fields) == 1
        field_num, wire_type, value = fields[0]
        assert field_num == 1
        assert wire_type == 2
        assert value == b"hello"

    def test_encode_bytes_round_trip(self):
        payload = b"\xde\xad\xbe\xef" * 16
        encoded = _encode_bytes(3, payload)
        fields = _decode_fields(encoded)
        assert len(fields) == 1
        _, _, value = fields[0]
        assert value == payload

    def test_encode_bool_true(self):
        encoded = _encode_bool(4, True)
        fields = _decode_fields(encoded)
        assert len(fields) == 1
        _, wire_type, value = fields[0]
        assert wire_type == 0  # varint
        assert value == 1

    def test_encode_bool_false_is_empty(self):
        # Proto3: default values are not encoded
        assert _encode_bool(4, False) == b""

    def test_encode_uint32_zero_is_empty(self):
        assert _encode_uint32(5, 0) == b""

    def test_encode_uint32_nonzero(self):
        encoded = _encode_uint32(5, 42)
        fields = _decode_fields(encoded)
        _, _, value = fields[0]
        assert value == 42

    def test_store_request_serialize(self):
        req = {
            "block_hash_hex": "deadbeef" * 8,
            "layer_type": "full_attention",
            "payload": _make_tensor(128),
            "is_quantized": True,
            "num_tokens": 16,
        }
        data = _serialize_store_request(req)
        fields = _decode_fields(data)
        field_map = {f: v for f, _, v in fields}
        assert field_map[1] == req["block_hash_hex"].encode()
        assert field_map[2] == req["layer_type"].encode()
        assert field_map[3] == req["payload"]
        # field 4 (is_quantized=True) and field 5 (num_tokens) should be present
        assert 4 in field_map
        assert field_map[5] == 16

    def test_batch_promote_request_serialize_multiple_keys(self):
        keys = ["vk-aabb-1", "vk-ccdd-2", "vk-eeff-3"]
        data = _serialize_batch_promote_request({"vault_keys": keys})
        fields = _decode_fields(data)
        # All fields should be field_num=1 (repeated string)
        assert all(f == 1 for f, _, _ in fields)
        assert len(fields) == len(keys)
        assert [v.decode() for _, _, v in fields] == keys

    def test_batch_promote_request_empty_keys(self):
        data = _serialize_batch_promote_request({"vault_keys": []})
        assert data == b""

    def test_deserialize_store_response_empty(self):
        resp = _deserialize_store_response(b"")
        assert resp == {"vault_key": ""}

    def test_deserialize_batch_promote_response_empty(self):
        resp = _deserialize_batch_promote_response(b"")
        assert resp == {"blocks": []}

    def test_decode_fields_empty(self):
        assert _decode_fields(b"") == []

    def test_store_response_round_trip(self):
        # Manually craft a StoreResponse { vault_key = "vk-test-1" } (field 1, string)
        vault_key = "vk-test-1"
        raw = _encode_string(1, vault_key)
        resp = _deserialize_store_response(raw)
        assert resp["vault_key"] == vault_key

    def test_batch_promote_response_round_trip(self):
        # Craft a VllmBatchPromoteResponse with one block
        # VllmBlockData { block_hash_hex=1, payload=2, is_quantized=3 }
        block_hash = "aabbccdd"
        payload = b"\x01\x02\x03\x04"
        block_bytes = (
            _encode_string(1, block_hash)
            + _encode_bytes(2, payload)
            + _encode_bool(3, True)
        )
        # Wrap in outer message: blocks = field 1, length-delimited
        from tierkv.connectors.vault_client import _encode_varint as ev
        outer = ev((1 << 3) | 2) + ev(len(block_bytes)) + block_bytes
        resp = _deserialize_batch_promote_response(outer)
        assert len(resp["blocks"]) == 1
        assert resp["blocks"][0]["block_hash_hex"] == block_hash
        assert resp["blocks"][0]["payload"] == payload
        assert resp["blocks"][0]["is_quantized"] is True


# ── SSM store routing ─────────────────────────────────────────────────────────

class TestSSMStoreRouting:
    """Blocks with layer_type='ssm' must route to ssm_client, not kv_client."""

    def test_ssm_block_routes_to_ssm_client(self):
        kv = MockVaultClient(node_id="kv:50051")
        ssm = MockVaultClient(node_id="ssm:50052")
        handler, _ = _build_pair(kv, ssm_client=ssm)
        handler._layer_type_map = {0: "linear_attention"}

        ssm_hash = b"\xAA" * 32
        records = handler.register_pending([ssm_hash], [16])
        assert len(records) == 1
        assert records[0].vault_node == "ssm:50052"
        assert records[0].layer_type == "linear_attention"

        handler.worker_store(
            pending_records=records,
            tensor_map={ssm_hash: _make_tensor(256)},
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)

        assert len(ssm.store_calls) == 1
        assert len(kv.store_calls) == 0
        assert ssm.store_calls[0]["layer_type"] == "linear_attention"

    def test_kv_block_routes_to_kv_client(self):
        kv = MockVaultClient(node_id="kv:50051")
        ssm = MockVaultClient(node_id="ssm:50052")
        handler, _ = _build_pair(kv, ssm_client=ssm)
        handler._layer_type_map = {0: "full_attention"}

        kv_hash = b"\xBB" * 32
        records = handler.register_pending([kv_hash], [16])
        assert records[0].vault_node == "kv:50051"

        handler.worker_store(
            pending_records=records,
            tensor_map={kv_hash: _make_tensor(256)},
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)

        assert len(kv.store_calls) == 1
        assert len(ssm.store_calls) == 0

    def test_mixed_blocks_route_independently(self):
        """One KV block and one SSM block in same batch — each goes to correct client."""
        kv = MockVaultClient(node_id="kv:50051")
        ssm = MockVaultClient(node_id="ssm:50052")
        handler, restore = _build_pair(kv, ssm_client=ssm)
        handler._layer_type_map = {0: "full_attention", 1: "linear_attention"}
        restore._group_size = 128

        kv_hash = b"\xCC" * 32
        ssm_hash = b"\xDD" * 32
        records = handler.register_pending([kv_hash, ssm_hash], [16, 16])

        handler.worker_store(
            pending_records=records,
            tensor_map={
                kv_hash: _make_tensor(256),
                ssm_hash: _make_tensor(256),
            },
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)

        assert len(kv.store_calls) == 1
        assert len(ssm.store_calls) == 1
        assert kv.store_calls[0]["layer_type"] == "full_attention"
        assert ssm.store_calls[0]["layer_type"] == "linear_attention"

    def test_ssm_restore_uses_ssm_client(self):
        """execute() splits SSM blocks to ssm_client for BatchPromote."""
        kv = MockVaultClient(node_id="kv:50051")
        ssm = MockVaultClient(node_id="ssm:50052")
        handler, restore = _build_pair(kv, ssm_client=ssm)
        handler._layer_type_map = {0: "linear_attention"}

        ssm_hash = b"\xEE" * 32
        tensor = _make_tensor(256)
        records = handler.register_pending([ssm_hash], [16])
        handler.worker_store(
            pending_records=records,
            tensor_map={ssm_hash: tensor},
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)

        _, plan = restore.plan([ssm_hash])
        assert len(plan) == 1
        assert plan[0].layer_type == "linear_attention"

        result = restore.execute(plan)
        # SSM client should have been called for promote, not KV client
        assert len(ssm.promote_calls) == 1
        assert len(kv.promote_calls) == 0
        assert ssm_hash in result
        assert result[ssm_hash][0] == tensor

    def test_ssm_client_fallback_to_kv_when_none(self):
        """When ssm_client=None, SSM blocks fall back to kv_client."""
        kv = MockVaultClient(node_id="kv:50051")
        handler, _ = _build_pair(kv, ssm_client=None)
        handler._layer_type_map = {0: "linear_attention"}

        ssm_hash = b"\xFF" * 32
        records = handler.register_pending([ssm_hash], [16])
        # With ssm_client=None, fallback is kv_client
        assert records[0].vault_node == "kv:50051"

        handler.worker_store(
            pending_records=records,
            tensor_map={ssm_hash: _make_tensor(256)},
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)
        assert len(kv.store_calls) == 1


# ── Registry lookup semantics ─────────────────────────────────────────────────

class TestRegistryLookupSemantics:
    """Explicit registry state transitions: pending → stored."""

    def test_pending_not_visible_to_lookup_stored(self):
        registry = BlockRegistry()
        block_hash = b"\x01" * 32
        rec = BlockRecord(
            block_hash=block_hash, vault_node="n", vault_key="",
            layer_type="full_attention", num_tokens=16, size_bytes=0,
            evicted_at=time.time(), tensor_hash=b"", status="pending",
        )
        registry.register(rec)
        assert registry.lookup_stored(block_hash) is None
        assert registry.is_pending(block_hash)

    def test_mark_stored_makes_block_visible(self):
        registry = BlockRegistry()
        block_hash = b"\x02" * 32
        rec = BlockRecord(
            block_hash=block_hash, vault_node="n", vault_key="vk-1",
            layer_type="full_attention", num_tokens=16, size_bytes=256,
            evicted_at=time.time(), tensor_hash=b"", status="pending",
        )
        registry.register(rec)
        registry.mark_stored(block_hash)
        stored = registry.lookup_stored(block_hash)
        assert stored is not None
        assert stored.vault_key == "vk-1"

    def test_lookup_by_vault_key(self):
        registry = BlockRegistry()
        block_hash = b"\x03" * 32
        rec = BlockRecord(
            block_hash=block_hash, vault_node="n", vault_key="vk-xyz",
            layer_type="full_attention", num_tokens=16, size_bytes=256,
            evicted_at=time.time(), tensor_hash=b"", status="stored",
        )
        registry.register(rec)
        registry.mark_stored(block_hash)
        found = registry.lookup_by_vault_key("vk-xyz")
        assert found is not None
        assert found.block_hash == block_hash

    def test_update_vault_key_updates_index(self):
        registry = BlockRegistry()
        block_hash = b"\x04" * 32
        rec = BlockRecord(
            block_hash=block_hash, vault_node="n", vault_key="",
            layer_type="full_attention", num_tokens=16, size_bytes=0,
            evicted_at=time.time(), tensor_hash=b"", status="pending",
        )
        registry.register(rec)
        registry.update_vault_key(block_hash, "vk-new")
        registry.mark_stored(block_hash)
        assert registry.lookup_by_vault_key("vk-new") is not None
        assert registry.lookup_by_vault_key("") is None

    def test_mark_failed_hides_from_lookup_stored(self):
        registry = BlockRegistry()
        block_hash = b"\x05" * 32
        rec = BlockRecord(
            block_hash=block_hash, vault_node="n", vault_key="vk-2",
            layer_type="full_attention", num_tokens=16, size_bytes=0,
            evicted_at=time.time(), tensor_hash=b"", status="pending",
        )
        registry.register(rec)
        registry.mark_failed(block_hash)
        assert registry.lookup_stored(block_hash) is None
        assert registry.stats()["failed"] == 1

    def test_remove_cleans_vault_index(self):
        registry = BlockRegistry()
        block_hash = b"\x06" * 32
        rec = BlockRecord(
            block_hash=block_hash, vault_node="n", vault_key="vk-3",
            layer_type="full_attention", num_tokens=16, size_bytes=256,
            evicted_at=time.time(), tensor_hash=b"", status="stored",
        )
        registry.register(rec)
        registry.mark_stored(block_hash)
        registry.remove(block_hash)
        assert registry.lookup_stored(block_hash) is None
        assert registry.lookup_by_vault_key("vk-3") is None
        assert registry.stats()["total"] == 0

    def test_full_store_restore_cycle_through_handler(self):
        """Full cycle: register → worker_store → plan → execute, checking registry at each step."""
        kv = MockVaultClient()
        handler, restore = _build_pair(kv)

        block_hash = b"\x07" * 32
        tensor = _make_tensor(512)

        # Step 1: register_pending — block is pending
        records = handler.register_pending([block_hash], [32])
        assert handler.registry.is_pending(block_hash)
        assert handler.registry.lookup_stored(block_hash) is None

        # Step 2: worker_store — block transitions to stored
        handler.worker_store(
            pending_records=records,
            tensor_map={block_hash: tensor},
            free_fn=lambda: None,
        )
        handler._executor.shutdown(wait=True, cancel_futures=False)
        assert not handler.registry.is_pending(block_hash)
        assert handler.registry.lookup_stored(block_hash) is not None

        # Step 3: plan — finds the stored block
        matched, plan = restore.plan([block_hash])
        assert matched == 32
        assert len(plan) == 1

        # Step 4: execute — fetches from vault and returns tensor
        result = restore.execute(plan)
        assert block_hash in result
        retrieved_tensor, num_tokens = result[block_hash]
        assert retrieved_tensor == tensor
        assert num_tokens == 32
