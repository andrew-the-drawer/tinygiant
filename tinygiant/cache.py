import ctypes
import mmap
from pathlib import Path

import numpy as np

from ._constants import EXPERT_KEYS_ORDER
from .iosched import ColdLoader


class ExpertCache:
    """Pinned experts live in mlock'd mmap views. Everything else goes through
    the ColdLoader (explicit pread, bounded budget) when one is configured, or
    through plain mmap page-cache faults in legacy mode."""

    def __init__(self, cache_dir, index, cold_budget_mb=None, io_threads=4, nocache=True):
        self.cache_dir = Path(cache_dir)
        self.index = index
        self.n_layers = int(index["n_layers"])
        self.n_experts = int(index["n_experts"])
        self.is_q4 = index.get("dtype") == "q4_k"
        self._lib = None

        self._mmaps = {}
        self._fds = {}
        self._views = {}
        self.layouts = {}
        layer_files = {}
        for layer_key, layer_info in index["layers"].items():
            layer_idx = int(layer_key)
            path = self.cache_dir / layer_info["file"]
            layer_files[layer_idx] = str(path)
            fd = open(path, "rb")
            mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
            self._fds[layer_idx] = fd
            self._mmaps[layer_idx] = mm
            formats = layer_info.get("tensor_formats", {})
            for exp_key, exp_info in layer_info["expert_offsets"].items():
                exp_id = int(exp_key)
                offset = exp_info["offset"]
                views = {"_formats": formats}
                parts = []
                pos = offset
                for k in EXPERT_KEYS_ORDER:
                    size = exp_info["sizes"][k]
                    fmt = formats.get(k, "q4_k")
                    if fmt == "float16":
                        views[k] = np.frombuffer(mm, dtype=np.float16, count=size // 2, offset=pos)
                    else:
                        views[k] = np.frombuffer(mm, dtype=np.uint8, count=size, offset=pos)
                    parts.append((k, fmt, size))
                    pos += size
                self._views[(layer_idx, exp_id)] = views
                self.layouts[(layer_idx, exp_id)] = (offset, parts)

        self.expert_bytes = sum(s for _, _, s in self.layouts[(0, 0)][1])
        self.loader = None
        if cold_budget_mb is not None:
            self.loader = ColdLoader(layer_files, self.layouts, self.n_experts,
                                     int(cold_budget_mb * 1024**2), io_threads, nocache)

        self.pinned = set()
        self.pinned_mask = np.zeros((self.n_layers, self.n_experts), bool)
        self.access_counts = {}
        self.reset_stats()

    def close(self):
        if self.loader:
            self.loader.close()

    def set_lib(self, lib):
        self._lib = lib

    def reset_stats(self):
        self.stats = dict(accesses=0, pinned_hits=0, cold_hits=0, misses=0)
        if self.loader:
            self.loader.reset_stats()

    def snapshot_stats(self):
        s = dict(self.stats)
        if self.loader:
            s.update(self.loader.snapshot_stats())
        return s

    # --- access -----------------------------------------------------------

    def new_token(self):
        if self.loader:
            self.loader.new_token()

    def begin_layer(self, layer):
        if self.loader:
            self.loader.begin_layer(layer)

    def get(self, layer, expert):
        key = (layer, expert)
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        self.stats["accesses"] += 1
        if self.pinned_mask[layer, expert]:
            self.stats["pinned_hits"] += 1
            return self._views[key]
        if self.loader is None:
            self.stats["misses"] += 1
            return self._views[key]
        if self.loader.has(key):
            self.stats["cold_hits"] += 1
        else:
            self.stats["misses"] += 1
        return self.loader.get(key)

    def get_many(self, layer, experts):
        """Fetch several experts of one layer; misses are issued together so the
        SSD sees them at full queue depth instead of one at a time."""
        if self.loader is not None:
            missing = [(layer, e) for e in experts
                       if not self.pinned_mask[layer, e] and not self.loader.has((layer, e))]
            if len(missing) > 1:
                self.loader.request_many(missing, 0)
        return [self.get(layer, e) for e in experts]

    def prefetch(self, layer, expert, prio):
        if self.loader is None or self.pinned_mask[layer, expert]:
            return
        self.loader.request((layer, expert), prio)

    def prefetch_many(self, layer, experts, prio):
        if self.loader is None:
            return
        keys = [(layer, int(e)) for e in experts
                if not self.pinned_mask[layer, e] and not self.loader.ready_mask[layer, e]]
        if keys:
            self.loader.request_many(keys, prio)

    def resident_masks(self, l0, l1):
        """Resident mask for layers l0..l1-1 as a [l1-l0, n_experts] array."""
        m = self.pinned_mask[l0:l1]
        if self.loader is None:
            return m
        return m | self.loader.ready_mask[l0:l1]

    def retarget(self, layer, keep):
        if self.loader is not None:
            self.loader.retarget(layer, keep)

    def resident_mask(self, layer):
        m = self.pinned_mask[layer]
        if self.loader is None:
            return m
        return m | self.loader.ready_mask[layer]

    # --- pinning ----------------------------------------------------------

    def _lock_expert(self, layer, exp_id):
        views = self._views[(layer, exp_id)]
        for k, v in views.items():
            if k != "_formats" and isinstance(v, np.ndarray):
                if self._lib:
                    self._lib.tg_mlock(v.ctypes.data, ctypes.c_size_t(v.nbytes))
                else:
                    np.sum(v)
        self.pinned.add((layer, exp_id))
        self.pinned_mask[layer, exp_id] = True

    def pinned_bytes(self):
        return len(self.pinned) * self.expert_bytes

    def pin_hot_experts(self, n_per_layer):
        profile = self.index.get("activation_profile")
        if not profile:
            return 0
        count = 0
        for layer_key in sorted(profile.keys(), key=int):
            for exp_id in profile[layer_key]["expert_ranking"][:n_per_layer]:
                self._lock_expert(int(layer_key), exp_id)
                count += 1
        self.reset_stats()
        return count

    def pin_from_usage(self, n_per_layer, n_layers):
        return self.pin_nonuniform({l: n_per_layer for l in range(n_layers)}, n_layers)

    def pin_nonuniform(self, pins_per_layer, n_layers):
        """Pin a variable number of experts per layer: calibration counts first,
        static activation profile as backfill."""
        per_layer = {}
        for (layer, expert), cnt in self.access_counts.items():
            per_layer.setdefault(layer, []).append((-cnt, expert))
        profile = self.index.get("activation_profile", {})

        count = 0
        for layer in range(n_layers):
            n_pin = pins_per_layer.get(layer, 0)
            if n_pin <= 0:
                continue
            pinned_this_layer = set()
            for _, exp_id in sorted(per_layer.get(layer, []))[:n_pin]:
                self._lock_expert(layer, exp_id)
                pinned_this_layer.add(exp_id)
                count += 1
            remaining = n_pin - len(pinned_this_layer)
            if remaining > 0 and str(layer) in profile:
                for exp_id in profile[str(layer)]["expert_ranking"]:
                    if exp_id not in pinned_this_layer:
                        self._lock_expert(layer, exp_id)
                        pinned_this_layer.add(exp_id)
                        count += 1
                        remaining -= 1
                        if remaining <= 0:
                            break
        self.reset_stats()
        self.access_counts = {}
        return count
