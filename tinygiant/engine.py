import ctypes
import json
import os
import subprocess
import time

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize as gguf_dequantize

from ._constants import (
    EMBED_DIM, EXPERT_INTERMEDIATE, HEAD_DIM, N_EXPERTS, N_EXPERTS_USED,
    N_HEADS, N_KV_HEADS, N_LAYERS, RMS_EPS, VOCAB_SIZE,
)
from ._lib import load_tinygiant_lib
from ._math import build_rope_cache
from .cache import ExpertCache
from .iosched import PRIO_DRAFT, PRIO_LOOKAHEAD, PRIO_TOKEN_PRIOR
from .predict import LookaheadPredictor
from .router import Router, RoutingConfig
from .specdec import SpecStats, dist, generate_spec, sample

GGML_Q4_K = 12
GGML_Q6_K = 14


def _dequant(tensor):
    return gguf_dequantize(tensor.data, tensor.tensor_type).astype(np.float32)


def _default_threads():
    """Largest non-efficiency core cluster. Mixing clusters of different speed
    was measured slower (the barrier waits for the slowest thread)."""
    best = 0
    try:
        for level in range(4):
            r = subprocess.run(["sysctl", "-n", f"hw.perflevel{level}.name",
                                f"hw.perflevel{level}.physicalcpu"],
                               capture_output=True, text=True, timeout=2)
            parts = r.stdout.split()
            if len(parts) < 2:
                break
            if "efficiency" not in parts[0].lower():
                best = max(best, int(parts[1]))
    except Exception:
        pass
    return best if best > 0 else max(1, (os.cpu_count() or 2) // 2)


class NWSEngine:

    def __init__(self, model_path, cache_dir, lib_path=None, threads=None,
                 cold_budget_mb=None, io_threads=8, kv_max=4096, verbose=True):
        self.verbose = verbose
        self._say("TinyGiant Inference Engine")
        self._say("=" * 60)
        t_start = time.perf_counter()

        self.reader = GGUFReader(model_path)
        self.tensor_map = {t.name: t for t in self.reader.tensors}

        with open(os.path.join(cache_dir, "index.json")) as f:
            self.cache_index = json.load(f)
        if self.cache_index.get("dtype") != "q4_k":
            raise RuntimeError("Only the q4 expert cache is supported (rebuild with --format q4)")

        self.lib = load_tinygiant_lib(lib_path)
        if self.lib is None:
            raise RuntimeError("libtinygiant not found: run `make`")
        self.threads = threads or _default_threads()
        self.lib.tg_set_threads(self.threads)

        self.expert_cache = ExpertCache(cache_dir, self.cache_index,
                                        cold_budget_mb=cold_budget_mb, io_threads=io_threads)
        self.expert_cache.set_lib(self.lib)
        mode = f"cold budget {cold_budget_mb} MB, {io_threads} I/O threads" if cold_budget_mb else "mmap (page cache)"
        self._say(f"Experts: {N_LAYERS} layers x {N_EXPERTS} [{mode}], {self.threads} compute threads")

        self.embd_tensor = self.tensor_map["token_embd.weight"]

        t = self.tensor_map["output.weight"]
        self.output_fmt = int(t.tensor_type)
        if self.output_fmt in (GGML_Q4_K, GGML_Q6_K):
            self.output_w = t.data.reshape(-1).view(np.uint8).copy()
            self.lib.tg_mlock(self.output_w.ctypes.data, ctypes.c_size_t(self.output_w.nbytes))
        else:
            self.output_w = _dequant(t)
        self.output_norm = _dequant(self.tensor_map["output_norm.weight"])

        self.attn_norms, self.ffn_norms, self.routers, self.qk_norms = [], [], [], []
        for i in range(N_LAYERS):
            self.attn_norms.append(_dequant(self.tensor_map[f"blk.{i}.attn_norm.weight"]))
            self.ffn_norms.append(_dequant(self.tensor_map[f"blk.{i}.ffn_norm.weight"]))
            self.routers.append(np.ascontiguousarray(_dequant(self.tensor_map[f"blk.{i}.ffn_gate_inp.weight"])))
            self.qk_norms.append((_dequant(self.tensor_map[f"blk.{i}.attn_q_norm.weight"]),
                                  _dequant(self.tensor_map[f"blk.{i}.attn_k_norm.weight"])))

        self.attn_weights = []
        attn_bytes = 0
        for i in range(N_LAYERS):
            tq = self.tensor_map[f"blk.{i}.attn_q.weight"]
            tk = self.tensor_map[f"blk.{i}.attn_k.weight"]
            tv = self.tensor_map[f"blk.{i}.attn_v.weight"]
            to = self.tensor_map[f"blk.{i}.attn_output.weight"]
            w = {
                "q": tq.data.reshape(-1).view(np.uint8).copy(),
                "k": tk.data.reshape(-1).view(np.uint8).copy(),
                "v": gguf_dequantize(tv.data, tv.tensor_type).astype(np.float16).copy(),
                "o": to.data.reshape(-1).view(np.uint8).copy(),
            }
            for v in w.values():
                self.lib.tg_mlock(v.ctypes.data, ctypes.c_size_t(v.nbytes))
                attn_bytes += v.nbytes
            self.attn_weights.append(w)
        self._say(f"Attention weights: {attn_bytes / 1024**2:.0f} MB mlock'd; "
                  f"output head {self.output_w.nbytes / 1024**2:.0f} MB")

        self.rope_cos, self.rope_sin = build_rope_cache(kv_max)
        self.kv_max = kv_max
        self.kv_len = 0
        self.kv_k = [np.zeros((N_KV_HEADS, kv_max, HEAD_DIM), np.float32) for _ in range(N_LAYERS)]
        self.kv_v = [np.zeros((N_KV_HEADS, kv_max, HEAD_DIM), np.float32) for _ in range(N_LAYERS)]

        self.router = Router(self.routers, N_EXPERTS_USED)
        self.lookahead = LookaheadPredictor(self.routers, self.ffn_norms, RMS_EPS)
        self.routing = RoutingConfig()
        self.lookahead_depth = 0
        self.lookahead_extra = 0
        self.token_prior = None
        self.token_prior_n = N_EXPERTS_USED
        self.draft_prefetch = True
        self.trace = None
        self.reset_timers()

        self._say(f"Engine ready in {time.perf_counter() - t_start:.1f}s")

    def _say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def close(self):
        self.expert_cache.close()

    # --- config -----------------------------------------------------------

    def set_threads(self, n):
        self.threads = n
        self.lib.tg_set_threads(n)

    def reset_timers(self):
        self.timers = dict(attn=0.0, route=0.0, prefetch=0.0, gather=0.0, moe=0.0, head=0.0)
        self.n_rows = 0
        self.n_passes = 0

    def stats(self):
        s = dict(self.timers)
        s["rows"] = self.n_rows
        s["passes"] = self.n_passes
        s.update(self.expert_cache.snapshot_stats())
        s.update({f"router_{k}": v for k, v in self.router.stats.items()})
        return s

    def reset_kv(self):
        self.kv_len = 0

    # --- weights ------------------------------------------------------------

    def embed(self, token_id):
        row = self.embd_tensor.data[token_id:token_id + 1]
        return gguf_dequantize(row, self.embd_tensor.tensor_type).reshape(-1).astype(np.float32)

    # --- calibration / pinning --------------------------------------------

    def calibrate(self, prompt_tokens, n_tokens=10):
        logits = self.prefill(prompt_tokens)
        next_token = int(np.argmax(logits))
        for step in range(n_tokens):
            logits = self.forward_rows([next_token], len(prompt_tokens) + step)[0]
            next_token = int(np.argmax(logits))
        self.reset_kv()

    def pin_experts(self, n_per_layer, calibrate_tokens=None, prompt_tokens=None, pins_per_layer=None):
        if calibrate_tokens and prompt_tokens:
            self._say(f"Calibrating ({calibrate_tokens} tokens)...")
            t0 = time.perf_counter()
            self.calibrate(prompt_tokens, n_tokens=calibrate_tokens)
            self._say(f"  {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        if pins_per_layer is None:
            pins_per_layer = {l: n_per_layer for l in range(N_LAYERS)}
        count = self.expert_cache.pin_nonuniform(pins_per_layer, N_LAYERS)
        gb = self.expert_cache.pinned_bytes() / 1024**3
        self._say(f"Pinned {count} experts ({gb:.2f} GB) in {time.perf_counter() - t0:.1f}s")
        self.reset_timers()
        return count

    # --- forward ------------------------------------------------------------

    def forward_one_token(self, token_id, pos, mode="exact"):
        return self.forward_rows([token_id], pos, mode)[0]

    def prefill(self, tokens, chunk=8):
        logits = None
        for i in range(0, len(tokens), chunk):
            logits = self.forward_rows(tokens[i:i + chunk], i)
        return logits[-1]

    def _prefetch_token_prior(self, token_ids):
        ec = self.expert_cache
        for l in range(N_LAYERS):
            cand = set()
            for tok in token_ids:
                cand.update(int(e) for e in self.token_prior.predict(l, tok, self.token_prior_n))
            if cand:
                ec.prefetch_many(l, sorted(cand), PRIO_TOKEN_PRIOR + l)

    def forward_rows(self, token_ids, start_pos, mode="exact"):
        """Runs k tokens at positions start_pos..start_pos+k-1 through the model.
        Returns logits [k, vocab]. KV entries for those positions are (re)written."""
        lib, ec = self.lib, self.expert_cache
        k = len(token_ids)
        D = EMBED_DIM
        X = np.ascontiguousarray(np.stack([self.embed(t) for t in token_ids]), dtype=np.float32)
        normed = np.empty(D, np.float32)
        attn_out = np.empty(D, np.float32)
        moe_out = np.empty((k, D), np.float32)
        tm = self.timers
        draft = mode == "draft"
        tr = self.trace if (self.trace is not None and k == 1 and not draft) else None
        need_resident = draft or self.routing.bias != 0.0 or tr is not None

        ec.new_token()
        requested = np.zeros((N_LAYERS, N_EXPERTS), bool)
        if self.token_prior is not None and not draft:
            t0 = time.perf_counter()
            self._prefetch_token_prior(token_ids)
            tm["prefetch"] += time.perf_counter() - t0
        if tr:
            tr.begin_token(token_ids[0], start_pos)

        for i in range(N_LAYERS):
            t0 = time.perf_counter()
            ec.begin_layer(i)
            aw = self.attn_weights[i]
            q_norm_w, k_norm_w = self.qk_norms[i]
            for r in range(k):
                x = X[r]
                pos = start_pos + r
                lib.tg_rms_norm(x.ctypes.data, self.attn_norms[i].ctypes.data,
                                normed.ctypes.data, D, ctypes.c_float(RMS_EPS))
                lib.tg_attention_decode_q4(
                    aw["q"].ctypes.data, aw["k"].ctypes.data, aw["v"].ctypes.data, aw["o"].ctypes.data,
                    q_norm_w.ctypes.data, k_norm_w.ctypes.data,
                    self.kv_k[i].ctypes.data, self.kv_v[i].ctypes.data, pos, self.kv_max,
                    self.rope_cos.ctypes.data, self.rope_sin.ctypes.data,
                    normed.ctypes.data, attn_out.ctypes.data,
                    D, N_HEADS, N_KV_HEADS, HEAD_DIM, pos)
                x += attn_out
            t1 = time.perf_counter()
            tm["attn"] += t1 - t0

            rr = np.sqrt(np.mean(X * X, axis=1, keepdims=True) + RMS_EPS)
            N = np.ascontiguousarray((X / rr) * self.ffn_norms[i], dtype=np.float32)
            resident = ec.resident_mask(i) if need_resident else None
            union = {}
            for r in range(k):
                logits = self.router.logits(i, N[r])
                idx, w = self.router.select(logits, resident, self.routing, restrict=draft)
                if draft and self.draft_prefetch:
                    for e in self.router._topk(logits):
                        if not resident[e]:
                            ec.prefetch(i, int(e), PRIO_DRAFT)
                if tr:
                    tr.record_layer(i, logits, idx, resident, X[0])
                for e, wt in zip(idx, w):
                    union.setdefault(int(e), {})[r] = float(wt)
            t2 = time.perf_counter()
            tm["route"] += t2 - t1

            if self.lookahead_depth and not draft and i + 1 < N_LAYERS:
                l0, l1 = i + 1, min(N_LAYERS, i + 1 + self.lookahead_depth)
                cand = self.lookahead.candidates(l0, l1, X, N_EXPERTS_USED + self.lookahead_extra)
                ec.retarget(l0, set(np.flatnonzero(cand[0]).tolist()))
                new = cand & ~ec.resident_masks(l0, l1) & ~requested[l0:l1]
                requested[l0:l1] |= new
                for j in range(l1 - l0):
                    experts = np.flatnonzero(new[j])
                    if experts.size:
                        ec.loader.request_many([(l0 + j, int(e)) for e in experts],
                                               PRIO_LOOKAHEAD + j + 1)
            t3 = time.perf_counter()
            tm["prefetch"] += t3 - t2

            experts = sorted(union)
            n = len(experts)
            row_w = np.zeros((n, k), np.float32)
            for ei, e in enumerate(experts):
                for r, wt in union[e].items():
                    row_w[ei, r] = wt
            views = ec.get_many(i, experts)
            t4 = time.perf_counter()
            tm["gather"] += t4 - t3

            gate_ptrs = (ctypes.c_void_p * n)(*[v["gate"].ctypes.data for v in views])
            up_ptrs = (ctypes.c_void_p * n)(*[v["up"].ctypes.data for v in views])
            down_ptrs = (ctypes.c_void_p * n)(*[v["down"].ctypes.data for v in views])
            fmts = (ctypes.c_int * n)(*[_down_fmt(v) for v in views])
            moe_out.fill(0)
            lib.tg_moe_forward_rows(ctypes.addressof(gate_ptrs), ctypes.addressof(up_ptrs),
                                    ctypes.addressof(down_ptrs), ctypes.addressof(fmts), n,
                                    row_w.ctypes.data, N.ctypes.data, moe_out.ctypes.data, k,
                                    D, EXPERT_INTERMEDIATE)
            X += moe_out
            tm["moe"] += time.perf_counter() - t4

        if tr:
            tr.end_token()

        t0 = time.perf_counter()
        logits = np.empty((k, VOCAB_SIZE), np.float32)
        out = np.empty(D, np.float32)
        for r in range(k):
            lib.tg_rms_norm(X[r].ctypes.data, self.output_norm.ctypes.data,
                            out.ctypes.data, D, ctypes.c_float(RMS_EPS))
            if self.output_fmt == GGML_Q6_K:
                lib.tg_matmul_q6k(self.output_w.ctypes.data, out.ctypes.data,
                                  logits[r].ctypes.data, VOCAB_SIZE, D)
            elif self.output_fmt == GGML_Q4_K:
                lib.tg_matmul_q4k(self.output_w.ctypes.data, out.ctypes.data,
                                  logits[r].ctypes.data, VOCAB_SIZE, D)
            else:
                logits[r] = self.output_w @ out
        tm["head"] += time.perf_counter() - t0

        self.kv_len = start_pos + k
        self.n_rows += k
        self.n_passes += 1
        return logits

    # --- generation ---------------------------------------------------------

    def decode(self, logits, pos, n_tokens, temperature=0.7, top_p=0.9, spec_k=0,
               rng=None, spec_stats=None):
        """Yields n_tokens tokens given the exact logits for position pos."""
        rng = rng or np.random.default_rng(42)
        if spec_k > 0:
            yield from generate_spec(self, logits, pos, n_tokens, spec_k, temperature, top_p,
                                     rng, spec_stats or SpecStats())
            return
        for step in range(n_tokens):
            tok = sample(dist(logits, temperature, top_p), rng)
            yield tok
            if step + 1 < n_tokens:
                logits = self.forward_rows([tok], pos + step)[0]

    def generate_stream(self, prompt_tokens, n_tokens=128, temperature=0.7, top_p=0.9,
                        spec_k=0, seed=42, spec_stats=None):
        logits = self.prefill(list(prompt_tokens))
        yield from self.decode(logits, len(prompt_tokens), n_tokens, temperature, top_p,
                               spec_k, np.random.default_rng(seed), spec_stats)

    def generate(self, prompt_tokens, n_tokens=10, temperature=0.7, top_p=0.9, spec_k=0, seed=42):
        self._say(f"\nGenerating {n_tokens} tokens from {len(prompt_tokens)}-token prompt")
        t0 = time.perf_counter()
        logits = self.prefill(list(prompt_tokens))
        t_prefill = time.perf_counter() - t0
        self._say(f"Prefill: {len(prompt_tokens)} tokens in {t_prefill:.2f}s "
                  f"({len(prompt_tokens) / t_prefill:.1f} tok/s)")

        spec_stats = SpecStats()
        t0 = time.perf_counter()
        generated = list(self.decode(logits, len(prompt_tokens), n_tokens, temperature, top_p,
                                     spec_k, np.random.default_rng(seed), spec_stats))
        t_decode = time.perf_counter() - t0
        self._say(f"Decode: {len(generated)} tokens in {t_decode:.2f}s "
                  f"({len(generated) / t_decode:.2f} tok/s)")
        if spec_k > 0:
            self._say(f"Spec: {spec_stats.as_dict()}")
        self.last_run = dict(prefill_s=t_prefill, decode_s=t_decode, n_generated=len(generated),
                             **spec_stats.as_dict())
        return generated


def _down_fmt(views):
    fmt = views.get("_formats", {}).get("down", "q4_k")
    return 1 if fmt == "float16" else 2 if fmt == "q6_k" else 0
