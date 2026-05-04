"""Generic context tracker for SDK integration hooks."""

from __future__ import annotations

from threading import RLock
from typing import Callable, Optional


class ContextTracker:
    """
    Tracks per-context stored block counts.

    Enables external SDKs to tag blocks with a context_id,
    track stored counts, and receive store callbacks.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._stored_counts: dict[str, int] = {}
        self._priorities: dict[str, int] = {}

    def increment_stored(self, context_id: Optional[str]) -> None:
        if context_id is None:
            return
        with self._lock:
            self._stored_counts[context_id] = self._stored_counts.get(context_id, 0) + 1

    def get_stored_count(self, context_id: str) -> int:
        with self._lock:
            return self._stored_counts.get(context_id, 0)

    def set_priority(self, context_id: str, priority: int) -> None:
        with self._lock:
            self._priorities[context_id] = priority

    def get_priority(self, context_id: str) -> int:
        with self._lock:
            return self._priorities.get(context_id, 0)
