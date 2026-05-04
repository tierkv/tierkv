def get_cache(*args, **kwargs):
    """Lazy wrapper — imports tierkv_core only when actually called."""
    from .cache import get_cache as _get_cache
    return _get_cache(*args, **kwargs)


__all__ = ["get_cache"]
