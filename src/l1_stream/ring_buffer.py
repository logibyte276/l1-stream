"""Thread-safe bounded ring buffer for live sensor messages."""

from __future__ import annotations

import threading
from collections import deque
from itertools import islice
from typing import Any, Deque, List, Optional

__all__ = ["RingBuffer"]


class RingBuffer:
    """Holds the most recent ``maxlen`` items, dropping the OLDEST on overflow.

    Backed by ``collections.deque(maxlen=N)``: appending is O(1) and, once
    full, adding a new item automatically discards the oldest one.

    Drop-oldest is the right policy for live sensor data specifically because
    stale readings are worse than missing ones on a moving robot. A five-second-old
    scan does not describe where anything is now; acting on it is worse than
    acting on nothing. The alternative -- block the producer until the consumer
    catches up -- would stall the socket thread and make the kernel drop
    packets anyway, just less visibly.

    Every method takes a lock, because the producer (socket thread) and the
    consumer (your main loop) touch this concurrently.
    """

    __slots__ = ("_buf", "_lock", "_condition", "_total_received", "_total_dropped")

    def __init__(self, maxlen: int):
        if isinstance(maxlen, bool) or not isinstance(maxlen, int):
            raise TypeError(f"maxlen must be an int, got {type(maxlen).__name__}")
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self._buf: Deque[Any] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._total_received = 0
        self._total_dropped = 0

    # --- properties ---

    @property
    def maxlen(self) -> int:
        """Capacity. Never ``None`` -- the constructor requires an int."""
        return int(self._buf.maxlen)  # type: ignore[arg-type]

    @property
    def total_received(self) -> int:
        """How many items have ever been appended (not how many are held now)."""
        with self._lock:
            return self._total_received

    @property
    def total_dropped(self) -> int:
        """How many were evicted because the buffer was full.

        If this climbs steadily your consumer loop is slower than the sensor:
        either enlarge the buffer or accept that you are sampling, not capturing.
        """
        with self._lock:
            return self._total_dropped

    # --- producer side ---

    def append(self, item: Any) -> None:
        """Add an item. Called by the reader thread; you rarely need this."""
        with self._condition:
            if len(self._buf) == self._buf.maxlen:
                self._total_dropped += 1
            self._buf.append(item)
            self._total_received += 1
            self._condition.notify_all()

    # --- consumer side ---

    def latest(self) -> Optional[Any]:
        """Most recent item, or ``None`` if nothing has arrived yet."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def latest_n(self, n: int, allow_partial: bool = True) -> List[Any]:
        """The ``n`` most recent items, oldest-first, as a snapshot list.

        The guardrails differ on purpose:

        * ``n > maxlen`` **always** raises ``ValueError``. No amount of waiting
          can satisfy it, so it is a bug in the calling code. Silently handing
          back fewer would hide the mistake.
        * ``n`` greater than what is buffered *right now* (but <= maxlen)
          returns what is available by default. That happens legitimately all
          the time -- right after ``start()``, during a brief dropout, after
          ``drain()``. Raising there would crash on every startup. Pass
          ``allow_partial=False`` when your maths genuinely needs exactly ``n``
          samples, e.g. a fixed-window filter that short data would corrupt.
        * ``n < 1`` or a non-int always raises; that is a caller bug.
        """
        # bool is a subclass of int, so True would otherwise sneak through as n=1.
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an int, got {type(n).__name__}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if n > self.maxlen:
            raise ValueError(
                f"Asked for the {n} most recent items, but this buffer only holds "
                f"{self.maxlen}. Either lower n, or construct the buffer bigger."
            )

        with self._lock:
            available = len(self._buf)
            if n > available and not allow_partial:
                raise ValueError(
                    f"Asked for exactly {n} items but only {available} have arrived "
                    f"so far (capacity {self.maxlen}). Wait, or use allow_partial=True."
                )
            if n >= available:
                return list(self._buf)
            # islice walks the deque once (O(n)). Indexing it in a loop would be
            # O(n^2), because deque indexing is O(n) toward the middle.
            return list(islice(self._buf, available - n, available))

    def snapshot(self) -> List[Any]:
        """Everything currently buffered, oldest-first. Does not consume."""
        with self._lock:
            return list(self._buf)

    def drain(self) -> List[Any]:
        """Everything currently buffered, oldest-first, AND clear the buffer.

        Use this when you want each item exactly once instead of just the
        newest -- e.g. accumulating every scan rather than sampling one.
        """
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
            return items

    def wait_for_new(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Block until at least one new item arrives, then return the newest.

        Returns ``None`` on timeout. Lower-latency and far cheaper than
        polling :meth:`latest` in a busy loop, because it sleeps until the
        producer actually wakes it.

        Note this returns only the newest item. If several arrive while you
        were away you will not see the ones in between -- use :meth:`drain`
        if you need every one.
        """
        with self._condition:
            count_before = self._total_received
            got_one = self._condition.wait_for(
                lambda: self._total_received != count_before, timeout=timeout
            )
            if not got_one:
                return None
            return self._buf[-1] if self._buf else None

    def clear(self) -> None:
        """Discard everything held. Counters are left alone."""
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"RingBuffer(held={len(self._buf)}/{self._buf.maxlen}, "
                f"received={self._total_received}, dropped={self._total_dropped})"
            )
