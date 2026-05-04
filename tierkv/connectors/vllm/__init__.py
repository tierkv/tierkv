def __getattr__(name):
    if name == "TierKVConnector":
        from tierkv.connectors.vllm.connector import TierKVConnector
        return TierKVConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["TierKVConnector"]
