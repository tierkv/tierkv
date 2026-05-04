"""Tests for ChainRegistry."""

import pytest

from tierkv.connectors.vllm.chain_registry import ChainRegistry, ChainRecord


@pytest.fixture
def chain_reg():
    return ChainRegistry()


class TestChainRecord:
    def test_is_complete(self):
        rec = ChainRecord(context_id="ctx-1", expected_blocks=3)
        assert not rec.is_complete
        rec.stored_blocks = {0, 1, 2}
        assert rec.is_complete

    def test_completeness_ratio(self):
        rec = ChainRecord(context_id="ctx-1", expected_blocks=4)
        rec.stored_blocks = {0, 2}
        assert rec.completeness_ratio == 0.5

    def test_is_restorable_contiguous(self):
        rec = ChainRecord(context_id="ctx-1", expected_blocks=5)
        rec.stored_blocks = {0, 1, 2}
        assert rec.is_restorable

    def test_is_restorable_gap(self):
        rec = ChainRecord(context_id="ctx-1", expected_blocks=5)
        rec.stored_blocks = {0, 2, 3}  # gap at 1
        assert rec.is_restorable  # prefix is just [0]

    def test_not_restorable_empty(self):
        rec = ChainRecord(context_id="ctx-1", expected_blocks=5)
        assert not rec.is_restorable

    def test_contiguous_prefix_length(self):
        rec = ChainRecord(context_id="ctx-1", expected_blocks=5)
        rec.stored_blocks = {0, 1, 2, 4}  # gap at 3
        assert rec.contiguous_prefix_length == 3


class TestChainRegistry:
    def test_create_and_get(self, chain_reg):
        chain_reg.create_chain("ctx-a", expected_blocks=10, source_hash="src123")
        chain = chain_reg.get_chain("ctx-a")
        assert chain is not None
        assert chain.expected_blocks == 10
        assert chain.source_hash == "src123"

    def test_on_block_stored(self, chain_reg):
        chain_reg.create_chain("ctx-b", expected_blocks=3)

        chain_reg.on_block_stored("ctx-b", 0, "vk-0")
        chain_reg.on_block_stored("ctx-b", 1, "vk-1")

        chain = chain_reg.get_chain("ctx-b")
        assert chain.stored_blocks == {0, 1}
        assert not chain.is_complete

        chain_reg.on_block_stored("ctx-b", 2, "vk-2")
        assert chain.is_complete

    def test_completion_callback_fires(self, chain_reg):
        completed = []
        chain_reg.register_completion_callback(
            lambda ctx_id, chain: completed.append(ctx_id)
        )
        chain_reg.create_chain("ctx-c", expected_blocks=2)
        chain_reg.on_block_stored("ctx-c", 0, "vk-0")
        assert completed == []
        chain_reg.on_block_stored("ctx-c", 1, "vk-1")
        assert completed == ["ctx-c"]

    def test_ignores_unknown_context(self, chain_reg):
        # Should not raise
        chain_reg.on_block_stored("unknown", 0, "vk-0")
        chain_reg.on_block_stored(None, 0, "vk-0")

    def test_is_restorable(self, chain_reg):
        chain_reg.create_chain("ctx-d", expected_blocks=5)
        assert not chain_reg.is_restorable("ctx-d")
        chain_reg.on_block_stored("ctx-d", 0, "vk-0")
        assert chain_reg.is_restorable("ctx-d")

    def test_get_completeness(self, chain_reg):
        chain_reg.create_chain("ctx-e", expected_blocks=4)
        assert chain_reg.get_completeness("ctx-e") == 0.0
        chain_reg.on_block_stored("ctx-e", 0, "vk-0")
        chain_reg.on_block_stored("ctx-e", 1, "vk-1")
        assert chain_reg.get_completeness("ctx-e") == 0.5

    def test_rolling_chain_hash(self, chain_reg):
        chain_reg.create_chain("ctx-f", expected_blocks=2)
        chain_reg.on_block_stored("ctx-f", 0, "vk-0")
        chain = chain_reg.get_chain("ctx-f")
        hash_after_first = chain.chain_hash
        assert hash_after_first != b""

        chain_reg.on_block_stored("ctx-f", 1, "vk-1")
        assert chain.chain_hash != hash_after_first  # hash evolves

    def test_stats(self, chain_reg):
        chain_reg.create_chain("a", expected_blocks=2)
        chain_reg.create_chain("b", expected_blocks=1)
        chain_reg.on_block_stored("a", 0, "vk-0")
        chain_reg.on_block_stored("a", 1, "vk-1")  # complete
        chain_reg.on_block_stored("b", 0, "vk-b0")  # complete

        stats = chain_reg.stats()
        assert stats["total_chains"] == 2
        assert stats["complete"] == 2

    def test_remove_chain(self, chain_reg):
        chain_reg.create_chain("ctx-g", expected_blocks=5)
        chain_reg.remove_chain("ctx-g")
        assert chain_reg.get_chain("ctx-g") is None

    def test_list_chains_by_priority(self, chain_reg):
        chain_reg.create_chain("low", expected_blocks=1, priority=0)
        chain_reg.create_chain("high", expected_blocks=1, priority=1)
        chain_reg.create_chain("critical", expected_blocks=1, priority=2)

        high_plus = chain_reg.list_chains(min_priority=1)
        assert len(high_plus) == 2
        critical_only = chain_reg.list_chains(min_priority=2)
        assert len(critical_only) == 1
