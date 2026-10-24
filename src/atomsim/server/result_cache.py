"""Finished job results, kept so an identical request is answered without recomputing.

Every job here is a pure function of its validated request: sampling takes a
seed, and the solvers, planes and isosurfaces are deterministic. So a cached
result is not an approximation of the fresh one, it is the fresh one, and a hit
changes nothing a reader can see except how long the wait is. That is the whole
justification, and it is why errors are never cached (a timeout or a refusal
says something about that moment, not about the request) and why the key is the
request after validation, defaults filled in, rather than the raw body.

Bounded by bytes, not entries: one million-point cloud weighs as much as two
hundred hydrogen planes, and a count bound would be either useless for the one
or wasteful for the other.
"""

import dataclasses
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

#: Default budget. A 100k-point cloud with its psi is ~3 MB and a 512^2 plane
#: is 2 MB, so this holds the popular states several times over.
DEFAULT_MAX_BYTES = 256 * 1024 * 1024


def nbytes(obj: Any, _seen: set[int] | None = None) -> int:
    """The numpy payload a result holds, walking dataclasses, tuples, lists and dicts."""
    seen = set() if _seen is None else _seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return sum(nbytes(getattr(obj, f.name), seen) for f in dataclasses.fields(obj))
    if isinstance(obj, (tuple, list)):
        return sum(nbytes(x, seen) for x in obj)
    if isinstance(obj, dict):
        return sum(nbytes(v, seen) for v in obj.values())
    return 0


class ResultCache:
    """A thread-safe LRU of finished results, evicted oldest-first past `max_bytes`."""

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._items: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_bytes
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return item[0]

    def put(self, key: str, result: Any) -> None:
        size = nbytes(result)
        if size > self._max:
            return  # one oversized result would evict everything else for nothing
        with self._lock:
            if key in self._items:
                self._bytes -= self._items.pop(key)[1]
            self._items[key] = (result, size)
            self._bytes += size
            while self._bytes > self._max:
                _, (_, dropped) = self._items.popitem(last=False)
                self._bytes -= dropped

    def __len__(self) -> int:
        return len(self._items)

    @property
    def size_bytes(self) -> int:
        return self._bytes
