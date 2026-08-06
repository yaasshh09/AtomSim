"""Minimal in-memory async-job pattern: create -> run (in any thread) -> poll/stream.

Deliberately simple, but not unbounded. A finished job holds its whole result,
and a result is an array: 1.2 MB for the default 100k-point cloud, about 20 MB
for a million-point one once psi is counted alongside the positions. A store
that only ever grew was fine for one person on a laptop for an afternoon and is
a slow memory leak on a host that stays up for weeks, so the store keeps the
most recent `max_jobs` and drops the rest oldest-first.

Only finished jobs are ever evicted. Dropping a RUNNING one would leave its
worker thread writing a result into an object nobody can reach, and the client
watching its websocket would wait for a completion that can no longer be
reported. If nothing has finished, the store exceeds the cap and says so by
simply not evicting, which is the honest failure: too much memory beats a job
that vanishes mid-flight.
"""

import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

#: Retained finished jobs. 16 x 20 MB is the worst case (sixteen consecutive
#: requests at the million-point ceiling); 16 x 1.2 MB is the realistic one.
#: The client fetches a job's data immediately after it reports DONE, so the
#: window in which eviction could beat a fetch is milliseconds wide.
DEFAULT_MAX_JOBS = 16


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


#: A job that will never change again, and so may be evicted.
_FINISHED = (JobStatus.DONE, JobStatus.ERROR)


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    result: Any = None
    error: str | None = None


class JobStore:
    def __init__(
        self,
        max_jobs: int = DEFAULT_MAX_JOBS,
        on_evict: Callable[[str], None] | None = None,
    ) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max_jobs = max_jobs
        #: Called once per evicted id. The server keeps per-job lookup tables
        #: beside the store, and they would leak in step with it otherwise.
        self._on_evict = on_evict

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
            evicted = self._evict_locked()
        # Outside the lock: a callback reaching back into the store would
        # deadlock, and nothing here needs the store to stay frozen.
        for job_id in evicted:
            if self._on_evict is not None:
                self._on_evict(job_id)
        return job

    def _evict_locked(self) -> list[str]:
        """Drop finished jobs, oldest first, until the store fits. Caller holds the lock."""
        evicted: list[str] = []
        while len(self._jobs) > self._max_jobs:
            oldest = next(
                (jid for jid, job in self._jobs.items() if job.status in _FINISHED),
                None,
            )
            if oldest is None:
                break  # everything still in flight; over the cap is the safe side
            del self._jobs[oldest]
            evicted.append(oldest)
        return evicted

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def run(self, job_id: str, fn: Callable[[Callable[[float], None]], Any]) -> None:
        """Execute fn in the calling thread, streaming progress into the job."""
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"unknown job id: {job_id}")
        job.status = JobStatus.RUNNING

        def report(fraction: float) -> None:
            job.progress = min(max(fraction, 0.0), 1.0)

        try:
            job.result = fn(report)
        except Exception as exc:  # honest failure: surface type + message
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = JobStatus.ERROR
        else:
            job.progress = 1.0
            job.status = JobStatus.DONE
