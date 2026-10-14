"""A small in-memory limit on how often one visitor can run the detector."""

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    """Allows at most `limit` calls per `window` seconds for each key (a visitor's address)."""

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._calls: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> float:
        """Record a call for `key`. Returns 0 if allowed, else seconds until it would be."""
        now = self._clock()
        with self._lock:
            calls = self._calls.setdefault(key, deque())
            while calls and calls[0] <= now - self.window:
                calls.popleft()
            if len(calls) >= self.limit:
                return calls[0] + self.window - now
            calls.append(now)
            if len(self._calls) > 10_000:
                self._forget_idle(now)
            return 0.0

    def _forget_idle(self, now: float) -> None:
        idle = [key for key, calls in self._calls.items() if calls[-1] <= now - self.window]
        for key in idle:
            del self._calls[key]
