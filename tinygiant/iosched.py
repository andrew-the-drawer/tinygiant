"""Cold-expert loader: explicit pread with F_NOCACHE into a bounded LRU buffer.

Bypassing the page cache keeps RAM use deterministic (so a 16 GB machine can be
simulated on a bigger one) and lets us control what is in flight and when.
"""

import fcntl
import heapq
import os
import threading
import time
from collections import OrderedDict

import numpy as np

PRIO_DEMAND = 0
PRIO_DRAFT = 5
PRIO_LOOKAHEAD = 10   # + j
PRIO_TOKEN_PRIOR = 30


class _Entry:
    __slots__ = ("key", "size", "buf", "views", "ready", "inflight", "prio",
                 "generation", "last_use", "prefetched", "used")

    def __init__(self, key, size, prio, generation):
        self.key = key
        self.size = size
        self.buf = None
        self.views = None
        self.ready = threading.Event()
        self.inflight = False
        self.prio = prio
        self.generation = generation
        self.last_use = -1
        self.prefetched = prio != PRIO_DEMAND
        self.used = False


class ColdLoader:

    def __init__(self, layer_files, layouts, n_experts, budget_bytes, n_threads=4, nocache=True):
        """layer_files: {layer: path}; layouts: {(layer, expert): (offset, [(key, fmt, size), ...])}"""
        self.layouts = layouts
        self.budget = budget_bytes
        self.fds = {}
        for layer, path in layer_files.items():
            fd = os.open(path, os.O_RDONLY)
            if nocache and hasattr(fcntl, "F_NOCACHE"):
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            self.fds[layer] = fd
        self.ready_mask = np.zeros((max(layer_files) + 1, n_experts), bool)

        self.lock = threading.Condition()
        self.entries = {}
        self.lru = OrderedDict()
        self.resident_bytes = 0
        self.queue = []
        self.seq = 0
        self.generation = 0
        self.token = 0
        self.layer = -1
        self.free_bufs = {}
        self.stop = False

        self.stats = dict(bytes_read=0, n_reads=0, demand_wait_s=0.0, n_demand=0,
                          n_prefetch_hit=0, n_prefetch_wasted=0, n_requests=0,
                          n_evict=0, n_dropped=0, n_overflow=0, read_time_s=0.0)

        self.threads = [threading.Thread(target=self._worker, daemon=True)
                        for _ in range(n_threads)]
        for t in self.threads:
            t.start()

    def close(self):
        with self.lock:
            self.stop = True
            self.lock.notify_all()
        for t in self.threads:
            t.join(timeout=1.0)
        for fd in self.fds.values():
            os.close(fd)

    # --- public API -------------------------------------------------------

    def new_token(self):
        """Start of a forward pass: queued prefetches from older passes are dropped,
        and entries used by the previous pass become evictable."""
        with self.lock:
            self.generation += 1
            self.token += 1
            self.layer = -1

    def begin_layer(self, layer):
        """Experts fetched for earlier layers of this pass become evictable."""
        self.layer = layer

    def has(self, key):
        return bool(self.ready_mask[key])

    def retarget(self, layer, keep):
        """Drop queued (not yet started) prefetches for `layer` whose expert is not in `keep`."""
        with self.lock:
            for key, e in list(self.entries.items()):
                if key[0] == layer and not e.inflight and not e.ready.is_set() \
                        and e.prio != PRIO_DEMAND and key[1] not in keep:
                    self._drop_locked(key, e)

    def request_many(self, keys, prio):
        with self.lock:
            for key in keys:
                self.stats["n_requests"] += 1
                e = self.entries.get(key)
                if e is not None:
                    if e.ready.is_set():
                        self.lru.move_to_end(key)
                    elif prio < e.prio:
                        e.prio = prio
                        e.generation = self.generation
                        self._push_locked(prio, key)
                    continue
                e = _Entry(key, self._size(key), prio, self.generation)
                self.entries[key] = e
                self._push_locked(prio, key)

    def request(self, key, prio):
        with self.lock:
            self.stats["n_requests"] += 1
            e = self.entries.get(key)
            if e is not None:
                if e.ready.is_set():
                    self.lru.move_to_end(key)
                    return e
                if prio < e.prio:
                    e.prio = prio
                    e.generation = self.generation
                    self._push_locked(prio, key)
                return e
            e = _Entry(key, self._size(key), prio, self.generation)
            self.entries[key] = e
            self._push_locked(prio, key)
            return e

    def get(self, key):
        """Blocking fetch. Returns dict of numpy views for the expert."""
        waited = 0.0
        while True:
            e = self.request(key, PRIO_DEMAND)
            if not e.ready.is_set():
                t0 = time.perf_counter()
                e.ready.wait()
                waited += time.perf_counter() - t0
            with self.lock:
                if e.views is None:
                    continue
                if e.prefetched and not e.used:
                    self.stats["n_prefetch_hit"] += 1
                if waited > 0:
                    self.stats["demand_wait_s"] += waited
                    self.stats["n_demand"] += 1
                e.used = True
                e.last_use = (self.token, self.layer)
                self.lru.move_to_end(key)
                return e.views

    def snapshot_stats(self):
        with self.lock:
            s = dict(self.stats)
            s["resident_mb"] = self.resident_bytes / 1024**2
            s["n_resident"] = len(self.lru)
            s["queued"] = len(self.queue)
        return s

    def reset_stats(self):
        with self.lock:
            for k in self.stats:
                self.stats[k] = 0 if isinstance(self.stats[k], int) else 0.0

    # --- internals --------------------------------------------------------

    def _size(self, key):
        return sum(s for _, _, s in self.layouts[key][1])

    def _push_locked(self, prio, key):
        self.seq += 1
        heapq.heappush(self.queue, (prio, self.seq, key))
        self.lock.notify()

    def _alloc(self, size):
        pool = self.free_bufs.get(size)
        if pool:
            return pool.pop()
        return np.empty(size, dtype=np.uint8)

    def _drop_locked(self, key, e):
        del self.entries[key]
        self.stats["n_dropped"] += 1
        e.views = None
        e.ready.set()

    def _evict_locked(self, need):
        protected = (self.token, self.layer)
        while self.resident_bytes + need > self.budget and self.lru:
            victim = None
            for k, e in self.lru.items():
                if e.last_use != protected:
                    victim = k
                    break
            if victim is None:
                return False
            e = self.lru.pop(victim)
            del self.entries[victim]
            self.ready_mask[victim] = False
            self.resident_bytes -= e.size
            self.stats["n_evict"] += 1
            if e.prefetched and not e.used:
                self.stats["n_prefetch_wasted"] += 1
            self.free_bufs.setdefault(e.size, []).append(e.buf)
            e.buf = None
            e.views = None
        return self.resident_bytes + need <= self.budget

    def _next_job_locked(self):
        """Pops the queue until a runnable job is found. Returns (key, entry, buf) or None."""
        while self.queue:
            _, _, key = heapq.heappop(self.queue)
            e = self.entries.get(key)
            if e is None or e.ready.is_set() or e.inflight:
                continue
            if e.prio != PRIO_DEMAND and e.generation != self.generation:
                self._drop_locked(key, e)
                continue
            if not self._evict_locked(e.size):
                if e.prio != PRIO_DEMAND:
                    self._drop_locked(key, e)
                    continue
                self.stats["n_overflow"] += 1
            e.inflight = True
            self.resident_bytes += e.size
            return key, e, self._alloc(e.size)
        return None

    def _worker(self):
        while True:
            with self.lock:
                job = None
                while not self.stop and job is None:
                    job = self._next_job_locked()
                    if job is None and not self.queue:
                        self.lock.wait()
                if self.stop:
                    return
            key, e, buf = job

            layer, expert = key
            offset, parts = self.layouts[key]
            t0 = time.perf_counter()
            n = os.preadv(self.fds[layer], [memoryview(buf)], offset)
            dt = time.perf_counter() - t0

            views = {"_formats": {}}
            pos = 0
            for name, fmt, size in parts:
                views["_formats"][name] = fmt
                if fmt == "float16":
                    views[name] = buf[pos:pos + size].view(np.float16)
                else:
                    views[name] = buf[pos:pos + size]
                pos += size

            with self.lock:
                self.stats["bytes_read"] += n
                self.stats["n_reads"] += 1
                self.stats["read_time_s"] += dt
                e.buf = buf
                e.views = views
                e.inflight = False
                self.lru[key] = e
                self.ready_mask[layer, expert] = True
                e.ready.set()
