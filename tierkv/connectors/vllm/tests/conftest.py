"""Shared test fixtures for vLLM connector tests."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Optional

from tierkv.connectors.vllm.block_registry import BlockRegistry
from tierkv.connectors.vault_client import VllmBlockData


@dataclass
class MockRequest:
    request_id: str = "req-001"
    block_hashes: list[bytes] = field(default_factory=list)
    num_tokens: int = 64
    extra_metadata: Optional[dict] = None


class MockVaultClient:
    """Mock VllmVaultClient for testing without gRPC."""

    def __init__(self, node_id: str = "mock:50051"):
        self.node_id = node_id
        self._store: dict[str, tuple[str, bytes, bool]] = {}
        self._counter = 0
        self.store_calls: list[dict] = []
        self.promote_calls: list[list[str]] = []
        self.should_fail: bool = False

    def store_block_sync(
        self,
        block_hash_hex: str,
        payload: bytes,
        layer_type: str,
        num_tokens: int,
        is_quantized: bool = True,
    ) -> str:
        if self.should_fail:
            raise ConnectionError("vault unreachable")
        self._counter += 1
        vault_key = f"vk-{block_hash_hex[:8]}-{self._counter}"
        self._store[vault_key] = (block_hash_hex, payload, is_quantized)
        self.store_calls.append({
            "block_hash_hex": block_hash_hex,
            "payload": payload,
            "layer_type": layer_type,
            "num_tokens": num_tokens,
        })
        return vault_key

    def batch_promote_sync(self, vault_keys: list[str]) -> list[VllmBlockData]:
        self.promote_calls.append(vault_keys)
        results = []
        for key in vault_keys:
            if key in self._store:
                hash_hex, payload, is_q = self._store[key]
                results.append(VllmBlockData(
                    block_hash_hex=hash_hex,
                    payload=payload,
                    is_quantized=is_q,
                ))
            else:
                results.append(VllmBlockData(
                    block_hash_hex="",
                    payload=b"",
                    is_quantized=False,
                ))
        return results

    def close(self):
        pass


@pytest.fixture
def registry():
    return BlockRegistry()


@pytest.fixture
def mock_client():
    return MockVaultClient()


@pytest.fixture
def mock_request():
    return MockRequest(
        block_hashes=[b"\x01" * 32, b"\x02" * 32, b"\x03" * 32, b"\x04" * 32],
        num_tokens=64,
    )
