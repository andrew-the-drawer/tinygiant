"""Push file-backed pages out of the page cache without sudo.

macOS keeps recently written/read files in the unified buffer cache, and
F_NOCACHE only prevents *new* caching, so a freshly built expert cache is served
from RAM instead of the SSD. Allocating and touching most of physical memory
forces the kernel to drop clean file pages first.
"""

import subprocess
import sys
import time

import numpy as np


def physical_ram():
    return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)


def main():
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
    n = int(physical_ram() * frac)
    t0 = time.perf_counter()
    chunks = []
    step = 1 << 30
    for off in range(0, n, step):
        a = np.empty(min(step, n - off), np.uint8)
        a[::4096] = 1          # touch every page
        chunks.append(a)
    print(f"touched {n / 1024**3:.1f} GB in {time.perf_counter() - t0:.1f}s", flush=True)
    del chunks


if __name__ == "__main__":
    main()
