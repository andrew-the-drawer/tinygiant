"""Raw SSD read throughput on expert slots that are unlikely to be cached
(low-ranked experts of an expert cache), single-stream and with 8 parallel readers.

  python benchmarks/ssd_probe.py ~/models/gptoss120b_cache
"""

import fcntl
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def main(cache_dir, n_layers=8, experts=range(96, 128)):
    idx = json.load(open(os.path.join(cache_dir, "index.json")))
    jobs = []
    for l in range(n_layers):
        info = idx["layers"][str(l)]
        fd = os.open(os.path.join(cache_dir, info["file"]), os.O_RDONLY)
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        for e in experts:
            off = info["expert_offsets"][str(e)]
            size = sum(off["sizes"].values())
            jobs.append((fd, off["offset"], size))
    total = sum(j[2] for j in jobs)

    def read(job):
        fd, off, size = job
        buf = np.empty(size, np.uint8)
        return os.preadv(fd, [memoryview(buf)], off)

    half = len(jobs) // 2
    t0 = time.perf_counter()
    for j in jobs[:half]:
        read(j)
    t1 = time.perf_counter()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(read, jobs[half:]))
    t2 = time.perf_counter()
    s1, s2 = sum(j[2] for j in jobs[:half]), sum(j[2] for j in jobs[half:])
    print(f"{len(jobs)} reads, {total / 1024**3:.1f} GB")
    print(f"single stream: {s1 / (t1 - t0) / 1024**3:.2f} GB/s")
    print(f"8 threads:     {s2 / (t2 - t1) / 1024**3:.2f} GB/s")


if __name__ == "__main__":
    main(os.path.expanduser(sys.argv[1]))
