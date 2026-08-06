"""A per-client token bucket for the job endpoints.

The job API is a compute oracle: one POST buys seconds of pinned CPU (1.4 s for
a Hartree-Fock solve, 1.7 s for a 96-cubed isosurface, measured on a 14-core
laptop and slower anywhere you would rent). On a laptop that is nobody's
problem. On a public URL it is the whole attack surface, and it needs no malice
to hurt: a crawler following gallery links would do it by accident.

The shape is a token bucket rather than a fixed window because the traffic is
bursty by design. A reader opening the gallery fires nine jobs in two seconds
and then reads for a minute. A fixed window either rejects that opening burst
or permits a sustained rate high enough to melt the host; a bucket allows the
burst and still caps the average.

Deliberately not a general-purpose limiter. There is no distributed state, so
it counts per process, which is exactly right for the single-instance
deployment this server is pinned to and wrong the moment there are two.
"""

import threading
import time
from collections.abc import Callable

#: Burst. Sized off the widest honest click-storm: the gallery row for n = 4 is
#: sixteen states, and a reader who clicks every one wants sixteen jobs.
DEFAULT_CAPACITY = 20
#: Seconds to refill an empty bucket, so the sustained rate is capacity/period.
DEFAULT_PERIOD = 60.0
#: Tracked clients. Bounds the limiter's own memory; see `_prune_locked`.
DEFAULT_MAX_CLIENTS = 4096


class TokenBucket:
    """Fractional-token bucket, keyed by client, refilling continuously."""

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
        """Spend a token for `key`. None if allowed, else seconds until one exists.

        Returning the wait rather than a bare boolean is what lets the caller
        answer with a truthful `Retry-After` instead of an unexplained refusal.
        """
        now = self._clock()
        with self._lock:
            tokens, seen = self._buckets.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - seen) * self._rate)
            if tokens < 1.0:
                # Do not bank the elapsed time against a refused request: store
                # the level we just computed so a caller that hammers the
                # endpoint cannot reset its own clock.
                self._buckets[key] = (tokens, now)
                return (1.0 - tokens) / self._rate
            self._buckets[key] = (tokens - 1.0, now)
            self._prune_locked(now)
            return None

    def _prune_locked(self, now: float) -> None:
        """Forget clients whose buckets have refilled. Caller holds the lock.

        A client at full capacity is indistinguishable from one that has never
        been seen, so dropping it loses no information and keeps a table keyed
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
