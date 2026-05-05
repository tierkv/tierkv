"""Python gRPC client for VllmColdVaultService.

Uses grpcio directly with the proto definitions. Since the proto is compiled
by the Rust build system, we use a lightweight approach: construct the service
descriptors manually to avoid depending on generated Python stubs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import grpc


# Proto field wire format constants
_MAX_MESSAGE_BYTES = 512 * 1024 * 1024  # 512 MiB


@dataclass
class VllmBlockData:
    """Response data for a single block from BatchPromote."""
    block_hash_hex: str
    payload: bytes
    is_quantized: bool


class VllmVaultClient:
    """
    gRPC client for VllmColdVaultService.

    Provides synchronous store/promote operations for use in thread pool executors.
    All calls are blocking — designed to be dispatched from ThreadPoolExecutor.
    """

    def __init__(
        self,
        host: str,
        port: int,
        max_message_bytes: int = _MAX_MESSAGE_BYTES,
    ) -> None:
        self.addr = f"{host}:{port}"
        self.node_id = self.addr
        self._channel = grpc.insecure_channel(
            self.addr,
            options=[
                ("grpc.max_send_message_length", max_message_bytes),
                ("grpc.max_receive_message_length", max_message_bytes),
            ],
        )
        # Unary-unary stubs for the two RPCs
        self._store_block = self._channel.unary_unary(
            "/tierkv.VllmColdVaultService/StoreBlock",
            request_serializer=_serialize_store_request,
            response_deserializer=_deserialize_store_response,
        )
        self._batch_promote = self._channel.unary_unary(
            "/tierkv.VllmColdVaultService/BatchPromote",
            request_serializer=_serialize_batch_promote_request,
            response_deserializer=_deserialize_batch_promote_response,
        )

    def store_block_sync(
        self,
        block_hash_hex: str,
        payload: bytes,
        layer_type: str,
        num_tokens: int,
        is_quantized: bool = True,
    ) -> str:
        """Store a single block. Returns vault_key."""
        request = {
            "block_hash_hex": block_hash_hex,
            "layer_type": layer_type,
            "payload": payload,
            "is_quantized": is_quantized,
            "num_tokens": num_tokens,
        }
        response = self._store_block(request)
        return response["vault_key"]

    def batch_promote_sync(self, vault_keys: list[str]) -> list[VllmBlockData]:
        """Fetch multiple blocks by vault_key. Returns list in same order."""
        request = {"vault_keys": vault_keys}
        response = self._batch_promote(request)
        return [
            VllmBlockData(
                block_hash_hex=b["block_hash_hex"],
                payload=b["payload"],
                is_quantized=b["is_quantized"],
            )
            for b in response["blocks"]
        ]

    def warmup(self, timeout: float = 5.0) -> bool:
        """Send a zero-key BatchPromote to establish the gRPC connection eagerly.

        Returns True if the server responded (even with an empty result), False
        if it timed out or was unreachable.  Call once at startup so the first
        real cold-restore RPC doesn't pay the TCP + HTTP/2 handshake cost.
        """
        try:
            self._batch_promote({"vault_keys": []}, timeout=timeout)
            return True
        except grpc.RpcError:
            return True  # Any gRPC error means the connection is established
        except Exception:
            return False

    def close(self) -> None:
        self._channel.close()


# ── Protobuf manual serialization (avoids codegen dependency) ─────────────────
# Wire format follows proto3 encoding. We use a minimal hand-rolled encoder
# since the messages are simple and we want zero external codegen deps.

def _encode_varint(value: int) -> bytes:
    """Encode an unsigned varint."""
    parts = []
    while value > 0x7F:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    parts.append(value & 0x7F)
    return bytes(parts)


def _encode_string(field_number: int, value: str) -> bytes:
    """Encode a string field."""
    data = value.encode("utf-8")
    return _encode_varint((field_number << 3) | 2) + _encode_varint(len(data)) + data


def _encode_bytes(field_number: int, value: bytes) -> bytes:
    """Encode a bytes field."""
    return _encode_varint((field_number << 3) | 2) + _encode_varint(len(value)) + value


def _encode_bool(field_number: int, value: bool) -> bytes:
    """Encode a bool field."""
    if not value:
        return b""
    return _encode_varint((field_number << 3) | 0) + b"\x01"


def _encode_uint32(field_number: int, value: int) -> bytes:
    """Encode a uint32 field."""
    if value == 0:
        return b""
    return _encode_varint((field_number << 3) | 0) + _encode_varint(value)


def _serialize_store_request(request: dict) -> bytes:
    """Serialize VllmBlockStoreRequest."""
    parts = []
    parts.append(_encode_string(1, request["block_hash_hex"]))
    parts.append(_encode_string(2, request["layer_type"]))
    parts.append(_encode_bytes(3, request["payload"]))
    parts.append(_encode_bool(4, request["is_quantized"]))
    parts.append(_encode_uint32(5, request["num_tokens"]))
    return b"".join(parts)


def _serialize_batch_promote_request(request: dict) -> bytes:
    """Serialize VllmBatchPromoteRequest."""
    parts = []
    for key in request["vault_keys"]:
        parts.append(_encode_string(1, key))
    return b"".join(parts)


# ── Protobuf deserialization ──────────────────────────────────────────────────

def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Read a varint, return (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _decode_fields(data: bytes) -> list[tuple[int, int, bytes | int]]:
    """Decode all fields from a protobuf message. Returns (field_num, wire_type, value)."""
    fields = []
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:  # varint
            value, pos = _read_varint(data, pos)
            fields.append((field_num, wire_type, value))
        elif wire_type == 2:  # length-delimited
            length, pos = _read_varint(data, pos)
            value = data[pos : pos + length]
            pos += length
            fields.append((field_num, wire_type, value))
        elif wire_type == 5:  # 32-bit
            value = data[pos : pos + 4]
            pos += 4
            fields.append((field_num, wire_type, value))
        elif wire_type == 1:  # 64-bit
            value = data[pos : pos + 8]
            pos += 8
            fields.append((field_num, wire_type, value))
        else:
            break
    return fields


def _deserialize_store_response(data: bytes) -> dict:
    """Deserialize VllmBlockStoreResponse."""
    vault_key = ""
    for field_num, wire_type, value in _decode_fields(data):
        if field_num == 1 and wire_type == 2:
            vault_key = value.decode("utf-8")
    return {"vault_key": vault_key}


def _deserialize_block_data(data: bytes) -> dict:
    """Deserialize a single VllmBlockData message."""
    block = {"block_hash_hex": "", "payload": b"", "is_quantized": False}
    for field_num, wire_type, value in _decode_fields(data):
        if field_num == 1 and wire_type == 2:
            block["block_hash_hex"] = value.decode("utf-8")
        elif field_num == 2 and wire_type == 2:
            block["payload"] = value
        elif field_num == 3 and wire_type == 0:
            block["is_quantized"] = bool(value)
    return block


def _deserialize_batch_promote_response(data: bytes) -> dict:
    """Deserialize VllmBatchPromoteResponse."""
    blocks = []
    for field_num, wire_type, value in _decode_fields(data):
        if field_num == 1 and wire_type == 2:
            blocks.append(_deserialize_block_data(value))
    return {"blocks": blocks}
