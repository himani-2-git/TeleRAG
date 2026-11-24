"""A small in-memory sliding-window rate limiter."""

import time
from collections import deque
from collections.abc import Callable, Hashable


class SlidingWindowRateLimiter:
    """Allow at most max_events per key in any window_seconds period.

    Each key (for example a Telegram user id) has its own list of event times. An event is
    allowed if fewer than max_events happened in the last window_seconds.

    Not thread-safe on purpose: the bot calls it from the asyncio event loop, which runs
    one handler step at a time. Counts are lost on restart, which is fine for abuse control.
    """

    def __init__(
        self,
        max_events: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        max_tracked_keys: int = 10_000,
    ):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._clock = clock
        self._max_tracked_keys = max_tracked_keys
        self._events: dict[Hashable, deque[float]] = {}

    def try_acquire(self, key: Hashable) -> float:
        """Record one event for the key if allowed.

        Returns:
            0.0 if the event is allowed (and recorded), otherwise the number of seconds the
            caller must wait before the next event would be allowed. Refused attempts are
            not recorded, so hammering the bot does not extend the wait.
        """
        if self.max_events <= 0:
            return 0.0

        now = self._clock()
        if key not in self._events and len(self._events) >= self._max_tracked_keys:
            self._forget_idle_keys(now)

        events = self._events.setdefault(key, deque())
        window_start = now - self.window_seconds
        while events and events[0] <= window_start:
            events.popleft()

        if len(events) >= self.max_events:
            return events[0] + self.window_seconds - now

        events.append(now)
        return 0.0

    def _forget_idle_keys(self, now: float) -> None:
        """Drop keys with no event inside the window, so memory cannot grow without limit."""
        window_start = now - self.window_seconds
        idle_keys = [key for key, events in self._events.items() if events[-1] <= window_start]
        for key in idle_keys:
            del self._events[key]
