import ctypes
import json
import os
import subprocess
import time

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize as gguf_dequantize

from ._lib import load_tinygiant_lib
from .cache import ExpertCache
from .iosched import PRIO_DRAFT, PRIO_LOOKAHEAD, PRIO_TOKEN_PRIOR
from .predict import LookaheadPredictor
from .router import Router, RoutingConfig
from .spec import ModelSpec
from .specdec import SpecStats, dist, generate_spec, sample

# ggml tensor type -> libtinygiant format code
FMT_CODES = {12: 0, 1: 1, 14: 2, 39: 3, 8: 4}
FMT_NAMES = {"q4_k": 0, "float16": 1, "q6_k": 2, "mxfp4": 3, "q8_0": 4}
ACT_CODES = {"silu": 0, "gptoss": 1}


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
        self.spec = S = ModelSpec.from_gguf(self.reader)
        self._say(f"Model: {S.arch}, {S.n_layers} layers, {S.n_experts}x{S.n_experts_used} experts, "
                  f"d={S.embed_dim}, heads {S.n_heads}/{S.n_kv_heads}x{S.head_dim}, "
                  f"window {S.sliding_window or '-'}, act {S.activation}")

        with open(os.path.join(cache_dir, "index.json")) as f:
            self.cache_index = json.load(f)
        if self.cache_index.get("dtype") not in ("q4_k", "quant"):
            raise RuntimeError("Only the quantized expert cache is supported (rebuild with --format q4)")

        self.lib = load_tinygiant_lib(lib_path)
        if self.lib is None:
            raise RuntimeError("libtinygiant not found: run `make`")
        self.threads = threads or _default_threads()
        self.lib.tg_set_threads(self.threads)

        self.expert_cache = ExpertCache(cache_dir, self.cache_index,
                                        cold_budget_mb=cold_budget_mb, io_threads=io_threads)
        self.expert_cache.set_lib(self.lib)
        mode = f"cold budget {cold_budget_mb} MB, {io_threads} I/O threads" if cold_budget_mb else "mmap (page cache)"
        self._say(f"Experts: {mode}, {self.threads} compute threads")

        tm = self.tensor_map
        self.embd_tensor = tm["token_embd.weight"]
        self.output_w, self.output_fmt = self._weight(tm["output.weight"])
        self.output_norm = _dequant(tm["output_norm.weight"])
        self._mlock(self.output_w)

        self.attn_norms, self.ffn_norms, self.routers, self.router_bias = [], [], [], []
        self.qk_norms, self.attn, self.attn_bias, self.sinks, self.expert_bias = [], [], [], [], []
        attn_bytes = 0
        for i in range(S.n_layers):
            p = f"blk.{i}."
            self.attn_norms.append(_dequant(tm[p + "attn_norm.weight"]))
            self.ffn_norms.append(_dequant(tm[p + f"{S.ffn_norm_name}.weight"]))
            self.routers.append(np.ascontiguousarray(_dequant(tm[p + "ffn_gate_inp.weight"])))
            self.router_bias.append(_dequant(tm[p + "ffn_gate_inp.bias"]) if S.router_bias else None)
            self.qk_norms.append((_dequant(tm[p + "attn_q_norm.weight"]),
                                  _dequant(tm[p + "attn_k_norm.weight"])) if S.qk_norm else (None, None))
            w = {}
            for name, key in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output")):
                w[name] = self._weight(tm[p + key + ".weight"])
                self._mlock(w[name][0])
                attn_bytes += w[name][0].nbytes
            self.attn.append(w)
            self.attn_bias.append({name: _dequant(tm[p + key + ".bias"]) if S.attn_bias else None
                                   for name, key in (("q", "attn_q"), ("k", "attn_k"),
                                                     ("v", "attn_v"), ("o", "attn_output"))})
            self.sinks.append(_dequant(tm[p + "attn_sinks.weight"]) if S.attn_sinks else None)
            if S.expert_bias:
                self.expert_bias.append({
                    k: np.ascontiguousarray(_dequant(tm[p + f"ffn_{k}_exps.bias"]).reshape(S.n_experts, -1))
                    for k in ("gate", "up", "down")})
            else:
                self.expert_bias.append(None)
        self._say(f"Attention weights: {attn_bytes / 1024**2:.0f} MB mlock'd; "
                  f"output head {self.output_w.nbytes / 1024**2:.0f} MB")

        self.rope_cos, self.rope_sin = S.rope_tables(kv_max)
        self.kv_max = kv_max
        self.kv_len = 0
        self.kv_k = [np.zeros((S.n_kv_heads, kv_max, S.head_dim), np.float32) for _ in range(S.n_layers)]
        self.kv_v = [np.zeros((S.n_kv_heads, kv_max, S.head_dim), np.float32) for _ in range(S.n_layers)]

        biases = self.router_bias if S.router_bias else None
        self.router = Router(self.routers, S.n_experts_used, biases)
        self.lookahead = LookaheadPredictor(self.routers, self.ffn_norms, S.rms_eps, biases)
        self.routing = RoutingConfig()
        self.lookahead_depth = 0
        self.lookahead_extra = 0
        self.token_prior = None
        self.token_prior_n = S.n_experts_used
        self.draft_prefetch = True
        self.trace = None
        self.act = ACT_CODES[S.activation]
        self.reset_timers()

        self._say(f"Engine ready in {time.perf_counter() - t_start:.1f}s")

    def _say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _weight(self, t):
        """Raw quantized bytes + format code, or an f16 copy for unsupported types."""
        code = FMT_CODES.get(int(t.tensor_type))
        if code is not None and code != 1:
            return t.data.reshape(-1).view(np.uint8).copy(), code
        return gguf_dequantize(t.data, t.tensor_type).astype(np.float16).copy(), 1

    def _mlock(self, arr):
        self.lib.tg_mlock(arr.ctypes.data, ctypes.c_size_t(arr.nbytes))

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
            pins_per_layer = {l: n_per_layer for l in range(self.spec.n_layers)}
        count = self.expert_cache.pin_nonuniform(pins_per_layer, self.spec.n_layers)
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
        for l in range(self.spec.n_layers):
            cand = set()
            for tok in token_ids:
                cand.update(int(e) for e in self.token_prior.predict(l, tok, self.token_prior_n))
            if cand:
                ec.prefetch_many(l, sorted(cand), PRIO_TOKEN_PRIOR + l)

    @staticmethod
    def _ptr(arr):
        return arr.ctypes.data if arr is not None else None

    def forward_rows(self, token_ids, start_pos, mode="exact"):
        """Runs k tokens at positions start_pos..start_pos+k-1 through the model.
        Returns logits [k, vocab]. KV entries for those positions are (re)written."""
        lib, ec, S = self.lib, self.expert_cache, self.spec
        k = len(token_ids)
        D, L, K = S.embed_dim, S.n_layers, S.n_experts_used
        X = np.ascontiguousarray(np.stack([self.embed(t) for t in token_ids]), dtype=np.float32)
        normed = np.empty(D, np.float32)
        attn_out = np.empty(D, np.float32)
        moe_out = np.empty((k, D), np.float32)
        tm = self.timers
        draft = mode == "draft"
        tr = self.trace if (self.trace is not None and k == 1 and not draft) else None
        need_resident = draft or self.routing.bias != 0.0 or tr is not None
        eps = ctypes.c_float(S.rms_eps)

        ec.new_token()
        requested = np.zeros((L, S.n_experts), bool)
        if self.token_prior is not None and not draft:
            t0 = time.perf_counter()
            self._prefetch_token_prior(token_ids)
            tm["prefetch"] += time.perf_counter() - t0
        if tr:
            tr.begin_token(token_ids[0], start_pos)

        for i in range(L):
            t0 = time.perf_counter()
            ec.begin_layer(i)
            w, b = self.attn[i], self.attn_bias[i]
            q_norm_w, k_norm_w = self.qk_norms[i]
            window = S.sliding_window if S.is_swa_layer(i) else 0
            for r in range(k):
                x = X[r]
                pos = start_pos + r
                lib.tg_rms_norm(x.ctypes.data, self.attn_norms[i].ctypes.data,
                                normed.ctypes.data, D, eps)
                lib.tg_attention_v2(
                    w["q"][1], w["k"][1], w["v"][1], w["o"][1],
                    w["q"][0].ctypes.data, w["k"][0].ctypes.data, w["v"][0].ctypes.data, w["o"][0].ctypes.data,
                    self._ptr(b["q"]), self._ptr(b["k"]), self._ptr(b["v"]), self._ptr(b["o"]),
                    self._ptr(q_norm_w), self._ptr(k_norm_w), self._ptr(self.sinks[i]),
                    self.kv_k[i].ctypes.data, self.kv_v[i].ctypes.data, pos, self.kv_max, window,
                    self.rope_cos.ctypes.data, self.rope_sin.ctypes.data,
                    normed.ctypes.data, attn_out.ctypes.data,
                    D, S.n_heads, S.n_kv_heads, S.head_dim, pos, eps)
                x += attn_out
            t1 = time.perf_counter()
            tm["attn"] += t1 - t0

            rr = np.sqrt(np.mean(X * X, axis=1, keepdims=True) + S.rms_eps)
            N = np.ascontiguousarray((X / rr) * self.ffn_norms[i], dtype=np.float32)
            resident = ec.resident_mask(i) if need_resident else None
            union = {}
            for r in range(k):
                logits = self.router.logits(i, N[r])
                idx, wts = self.router.select(logits, resident, self.routing, restrict=draft)
                if draft and self.draft_prefetch:
                    for e in self.router._topk(logits):
                        if not resident[e]:
                            ec.prefetch(i, int(e), PRIO_DRAFT)
                if tr:
                    tr.record_layer(i, logits, idx, resident, X[0])
                for e, wt in zip(idx, wts):
                    union.setdefault(int(e), {})[r] = float(wt)
            t2 = time.perf_counter()
            tm["route"] += t2 - t1

            if self.lookahead_depth and not draft and i + 1 < L:
                l0, l1 = i + 1, min(L, i + 1 + self.lookahead_depth)
                cand = self.lookahead.candidates(l0, l1, X, K + self.lookahead_extra)
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
            fmts = (ctypes.c_int * n)(*[FMT_NAMES[v["_formats"].get("down", "q4_k")] for v in views])
            gate_fmt = FMT_NAMES[views[0]["_formats"].get("gate", "q4_k")]
            eb = self.expert_bias[i]
            if eb is not None:
                bptr = {key: (ctypes.c_void_p * n)(*[eb[key].ctypes.data + e * eb[key].strides[0] for e in experts])
                        for key in ("gate", "up", "down")}
                gb, ub, db = ctypes.addressof(bptr["gate"]), ctypes.addressof(bptr["up"]), ctypes.addressof(bptr["down"])
            else:
                gb = ub = db = None
            moe_out.fill(0)
            lib.tg_moe_forward_rows_v2(ctypes.addressof(gate_ptrs), ctypes.addressof(up_ptrs),
                                       ctypes.addressof(down_ptrs), gate_fmt, ctypes.addressof(fmts),
                                       gb, ub, db, self.act, n,
                                       row_w.ctypes.data, N.ctypes.data, moe_out.ctypes.data, k,
                                       D, S.expert_inter)
            X += moe_out
            tm["moe"] += time.perf_counter() - t4

        if tr:
            tr.end_token()

        t0 = time.perf_counter()
        logits = np.empty((k, S.vocab_size), np.float32)
        out = np.empty(D, np.float32)
        for r in range(k):
            lib.tg_rms_norm(X[r].ctypes.data, self.output_norm.ctypes.data, out.ctypes.data, D, eps)
            lib.tg_matmul(self.output_fmt, self.output_w.ctypes.data, None, out.ctypes.data,
                          logits[r].ctypes.data, S.vocab_size, D)
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
