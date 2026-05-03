"""
Thin Python wrapper around the Rust hailey_core extension.

Usage:
    from exo_pk import get_cache

    cache = get_cache(dim=2048)
    cache.insert(token_idx=0, layer=3, data=kv_bytes)
    kv_bytes = cache.get(token_idx=0, layer=3)
    print(cache.stats())
"""

from hailey_core import TieredKVCache, TurboQuant

_COLD_NODE      = "192.168.50.12:50051"
_RECOMPUTE_NODE = "192.168.50.10:50052"
_HOT_CAPACITY   = 16_384   # token×layer slots in hot tier (~16K tokens)


def get_cache(
    dim: int,
    hot_capacity: int = _HOT_CAPACITY,
    cold_node: str = _COLD_NODE,
    recompute_node: str = _RECOMPUTE_NODE,
) -> TieredKVCache:
    """
    Create and return a TieredKVCache instance.

    Args:
        dim:            KV vector dimension (must be divisible by 8).
                        Typical values: 2048 (Qwen 7B), 4096 (Qwen 72B),
                        7168 (Qwen 235B MLA head dim).
        hot_capacity:   Max token×layer slots held in hot tier.
        cold_node:      Mac Air gRPC address for cold vault.
        recompute_node: DGX Spark gRPC address for KV recompute.

    Returns:
        TieredKVCache ready to call .insert() / .get() / .stats()
    """
    return TieredKVCache(
        hot_capacity=hot_capacity,
        dim=dim,
        cold_addr=cold_node,
        recompute_addr=recompute_node,
    )


__all__ = ["get_cache", "TieredKVCache", "TurboQuant"]
