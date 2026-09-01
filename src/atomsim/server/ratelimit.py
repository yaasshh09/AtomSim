"""My per-client token bucket for the job endpoints.

My job API is a compute oracle: one POST buys seconds of pinned CPU (1.4 s for
a Hartree-Fock solve, 1.7 s for a 96-cubed isosurface, measured on a 14-core
laptop and slower anywhere you would rent). On a laptop that is nobody's
problem. On a public URL it is my whole attack surface, and it needs no malice
to hurt: a crawler following gallery links would do it by accident.

I use a token bucket rather than a fixed window because my traffic is bursty by
design. Opening the gallery costs nothing metered (the strip is thumbnail GETs,
and I serve those from `lru_cache`d renders rather than job POSTs), but clicking
*through* it fires one job per tile for as long as the reader keeps clicking,
and then they read for a minute. A fixed window either rejects that click-through
or permits a sustained rate high enough to melt the host; a bucket lets me allow
the burst and still cap the average.

I am deliberately not a general-purpose limiter. I hold no distributed state, so
I count per process, which is exactly right for the single-instance deployment I
am pinned to and wrong the moment there are two.
"""

import threading
import time
from collections.abc import Callable

#: My burst. I size it off the widest honest click-storm the UI can actually
#: produce, which is a click-through of a whole shell row:
#:
#:   * the row is n**2 tiles and the n control offers n up to 6, so 36 tiles;
#:   * each tile costs one auto-fired job (the plane in Plane view, or the
#:     isosurface in Cloud view with the surface shown), because invalidation
#:     resets those to `idle` and the view refetches;
#:   * the cloud is a button rather than an effect, so a reader who also presses
#:     Sample on every tile doubles it, to 72;
#:   * plus one Hartree-Fock solve when the model is HF and the atom changed.
#:
#: That is 73, rounded up. I size it this way on purpose: `_build_rate_limiter`
#: claims my default is wider than any honest click-storm, and that claim has to
#: survive the widest row rather than a comfortable one. See
#: `test_default_burst_covers_widest_click_through`.
DEFAULT_CAPACITY = 80
#: Seconds I take to refill an empty bucket, so my sustained rate is
#: capacity/period. I hold it at the same 1/3 job per second my narrower bucket
#: allowed: widening the burst is meant to stop me rejecting real readers, not
#: to raise the average a client can sustain. Earning one token back takes 3 s
#: either way, so my `Retry-After` is unchanged by the resizing.
DEFAULT_PERIOD = 240.0
#: How many clients I track. Bounds my own memory; see `_prune_locked`.
DEFAULT_MAX_CLIENTS = 4096


class TokenBucket:
    """My fractional-token bucket, keyed by client, refilling continuously."""

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        period: float = DEFAULT_PERIOD,
        max_clients: int = DEFAULT_MAX_CLIENTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        self._capacity = float(capacity)
        self._rate = capacity / period  # tokens per second
        self._max_clients = max_clients
        self._clock = clock
        self._lock = threading.Lock()
        #: key -> (tokens remaining, when that count was accurate)
        self._buckets: dict[str, tuple[float, float]] = {}

    def check(self, key: str) -> float | None:
        """I spend a token for `key`. None if allowed, else the seconds to wait.

        I return the wait rather than a bare boolean because that is what lets
        the caller answer with a truthful `Retry-After` instead of an
        unexplained refusal.
        """
        now = self._clock()
        with self._lock:
            tokens, seen = self._buckets.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - seen) * self._rate)
            if tokens < 1.0:
                # I do not bank the elapsed time against a refused request: I
                # store the level I just computed, so a caller that hammers the
                # endpoint cannot reset its own clock.
                self._buckets[key] = (tokens, now)
                return (1.0 - tokens) / self._rate
            self._buckets[key] = (tokens - 1.0, now)
            self._prune_locked(now)
            return None

    def _prune_locked(self, now: float) -> None:
        """I forget clients whose buckets have refilled. Caller holds the lock.

        A client at full capacity is indistinguishable to me from one I have
        never seen, so dropping it loses no information and keeps a table keyed
        by arbitrary remote addresses from growing without limit.
        """
        if len(self._buckets) <= self._max_clients:
            return
        full_after = self._capacity / self._rate
        self._buckets = {
            key: entry
            for key, entry in self._buckets.items()
            if now - entry[1] < full_after
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)
