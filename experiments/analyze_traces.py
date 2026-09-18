"""Offline analysis of routing traces.

Answers, from recorded traces alone:
  1. Multi-layer lookahead: recall of layer l+j's experts when predicted from the
     hidden state at layer l (with k+extra candidates), for j = 1..J.
  2. Cross-token predictors: previous-token overlap vs token-id prior.
  3. Pinned-only routing: how often restricted top-k equals the full top-k, for
     several pin budgets (proxy for self-speculative draft agreement).
  4. Cache-aware bias: fraction of slots swapped and gate-mass displaced vs beta.

  python experiments/analyze_traces.py experiments/traces/*.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize as gguf_dequantize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinygiant.predict import TokenPrior
from tinygiant.spec import ModelSpec

N_EXPERTS = N_EXPERTS_USED = N_LAYERS = K = None
RMS_EPS = 1e-6


def topk(v, k):
    idx = np.argpartition(v, -k, axis=-1)[..., -k:]
    return idx


def load_model_bits(model_path):
    global N_EXPERTS, N_EXPERTS_USED, N_LAYERS, K, RMS_EPS
    r = GGUFReader(model_path)
    spec = ModelSpec.from_gguf(r)
    N_EXPERTS, N_EXPERTS_USED, N_LAYERS, RMS_EPS = spec.n_experts, spec.n_experts_used, spec.n_layers, spec.rms_eps
    K = N_EXPERTS_USED
    tm = {t.name: t for t in r.tensors}
    routers = np.stack([gguf_dequantize(tm[f"blk.{i}.ffn_gate_inp.weight"].data,
                                        tm[f"blk.{i}.ffn_gate_inp.weight"].tensor_type).astype(np.float32)
                        for i in range(N_LAYERS)])
    norms = np.stack([gguf_dequantize(tm[f"blk.{i}.{spec.ffn_norm_name}.weight"].data,
                                      tm[f"blk.{i}.{spec.ffn_norm_name}.weight"].tensor_type).astype(np.float32)
                      for i in range(N_LAYERS)])
    if spec.router_bias:
        bias = np.stack([gguf_dequantize(tm[f"blk.{i}.ffn_gate_inp.bias"].data,
                                         tm[f"blk.{i}.ffn_gate_inp.bias"].tensor_type).astype(np.float32)
                         for i in range(N_LAYERS)])
        return routers, norms, bias
    return routers, norms, None


def lookahead_recall(traces, routers, norms, bias, max_j, extra_list):
    """recall[j][extra] = mean fraction of true top-k at layer l+j found in
    top-(k+extra) predicted from hidden state at layer l."""
    res = {}
    for j in range(1, max_j + 1):
        hits = {x: [] for x in extra_list}
        for d in traces:
            H = d["hidden_in"].astype(np.float32)   # [T, L, D] residual after attention at layer l
            S = d["selected"]                        # [T, L, K]
            T = H.shape[0]
            for l in range(N_LAYERS - j):
                t = l + j
                h = H[:, l]
                rr = np.sqrt((h * h).mean(axis=1, keepdims=True) + RMS_EPS)
                logits = (h / rr * norms[t]) @ routers[t].T   # [T, E]
                if bias is not None:
                    logits = logits + bias[t]
                order = np.argsort(logits, axis=1)[:, ::-1]
                truth = S[:, t]
                for x in extra_list:
                    pred = order[:, :K + x]
                    m = np.array([len(set(pred[i]) & set(truth[i])) / K for i in range(T)])
                    hits[x].append(m.mean())
        res[j] = {x: float(np.mean(hits[x])) for x in extra_list}
    return res


def prev_token_overlap(traces):
    ov = []
    for d in traces:
        S = d["selected"]
        for t in range(1, S.shape[0]):
            ov.append(np.mean([len(set(S[t, l]) & set(S[t - 1, l])) / K for l in range(N_LAYERS)]))
    return float(np.mean(ov))


def token_prior_eval(train, test, n_list):
    prior = TokenPrior.from_traces(train, N_EXPERTS, k_keep=16, min_count=1)
    res = {}
    for n in n_list:
        rec, cov = [], []
        for p in test:
            d = np.load(p)
            for t in range(d["selected"].shape[0]):
                tok = int(d["tokens"][t])
                for l in range(N_LAYERS):
                    pred = prior.predict(l, tok, n)
                    truth = set(d["selected"][t, l].tolist())
                    cov.append(len(pred) > 0)
                    if len(pred):
                        rec.append(len(set(pred.tolist()) & truth) / K)
        res[n] = dict(recall_when_known=float(np.mean(rec)) if rec else 0.0,
                      coverage=float(np.mean(cov)))
    return res


def pinned_agreement(traces, pin_counts):
    """Pin the top-N experts per layer by frequency in the traces (an optimistic
    oracle-ish pin set), then compare restricted vs full top-k selection."""
    counts = np.zeros((N_LAYERS, N_EXPERTS), np.int64)
    for d in traces:
        S = d["selected"]
        for l in range(N_LAYERS):
            np.add.at(counts[l], S[:, l].ravel(), 1)
    res = {}
    for n_pin in pin_counts:
        pinned = np.zeros((N_LAYERS, N_EXPERTS), bool)
        for l in range(N_LAYERS):
            pinned[l, np.argsort(counts[l])[::-1][:n_pin]] = True
        same, jacc, hit = [], [], []
        for d in traces:
            L_ = d["logits"]   # [T, L, E]
            for t in range(L_.shape[0]):
                for l in range(N_LAYERS):
                    lg = L_[t, l]
                    full = set(topk(lg, K).tolist())
                    masked = np.where(pinned[l], lg, -np.inf)
                    rest = set(topk(masked, K).tolist())
                    same.append(full == rest)
                    jacc.append(len(full & rest) / len(full | rest))
                    hit.append(len([e for e in full if pinned[l, e]]) / K)
        res[n_pin] = dict(layer_exact=float(np.mean(same)), jaccard=float(np.mean(jacc)),
                          pinned_hit=float(np.mean(hit)),
                          token_all_layers_exact=float(np.mean(np.array(same).reshape(-1, N_LAYERS).all(axis=1))))
    return res


def bias_sweep(traces, pin_counts, betas):
    counts = np.zeros((N_LAYERS, N_EXPERTS), np.int64)
    for d in traces:
        S = d["selected"]
        for l in range(N_LAYERS):
            np.add.at(counts[l], S[:, l].ravel(), 1)
    res = {}
    for n_pin in pin_counts:
        pinned = np.zeros((N_LAYERS, N_EXPERTS), bool)
        for l in range(N_LAYERS):
            pinned[l, np.argsort(counts[l])[::-1][:n_pin]] = True
        res[n_pin] = {}
        for beta in betas:
            swapped, mass, hit = [], [], []
            for d in traces:
                L_ = d["logits"]
                for t in range(L_.shape[0]):
                    for l in range(N_LAYERS):
                        lg = L_[t, l]
                        full = topk(lg, K)
                        biased = topk(lg + beta * pinned[l], K)
                        p = np.exp(lg - lg.max()); p /= p.sum()
                        fs, bs = set(full.tolist()), set(biased.tolist())
                        swapped.append(len(fs - bs) / K)
                        mass.append(p[list(fs - bs)].sum() - p[list(bs - fs)].sum())
                        hit.append(np.mean(pinned[l, biased]))
            res[n_pin][beta] = dict(slots_swapped=float(np.mean(swapped)),
                                    displaced_mass=float(np.mean(mass)),
                                    pinned_hit=float(np.mean(hit)))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces", nargs="+")
    p.add_argument("--model", default=str(Path.home() / "models/Qwen3-30B-A3B-Q4_K_M.gguf"))
    p.add_argument("--max-j", type=int, default=8)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "trace_analysis.json"))
    args = p.parse_args()

    routers, norms, bias = load_model_bits(args.model)
    paths = sorted(args.traces)
    traces = [np.load(p) for p in paths]
    n_tokens = sum(d["tokens"].shape[0] for d in traces)
    print(f"{len(traces)} traces, {n_tokens} tokens")
    out = {}

    print("\n1. Lookahead recall (predict layer l+j from hidden state at layer l)")
    extras = [0, 2, 4, 8]
    la = lookahead_recall(traces, routers, norms, bias, args.max_j, extras)
    print("   j   " + "  ".join(f"k+{x:<2d}" for x in extras))
    for j, r in la.items():
        print(f"   {j:<3d} " + "  ".join(f"{r[x]:.2f}" for x in extras))
    out["lookahead_recall"] = {str(j): {str(x): v for x, v in r.items()} for j, r in la.items()}

    print("\n2. Cross-token predictors")
    ov = prev_token_overlap(traces)
    print(f"   previous-token overlap: {ov:.2f}")
    half = len(paths) // 2 or 1
    tp = token_prior_eval(paths[:half], paths[half:] or paths[:half], [4, 8, 16])
    for n, r in tp.items():
        print(f"   token-id prior top-{n}: recall {r['recall_when_known']:.2f} (coverage {r['coverage']:.0%})")
    out["prev_token_overlap"] = ov
    out["token_prior"] = {str(k): v for k, v in tp.items()}

    print("\n3. Pinned-only (draft) routing agreement vs pin budget")
    pins = [16, 32, 48, 64, 96]
    pa = pinned_agreement(traces, pins)
    for n, r in pa.items():
        print(f"   pin {n:3d}/layer: pinned-hit {r['pinned_hit']:.2f}  layer-exact {r['layer_exact']:.2f}  "
              f"jaccard {r['jaccard']:.2f}  token-all-layers-exact {r['token_all_layers_exact']:.2f}")
    out["pinned_agreement"] = {str(k): v for k, v in pa.items()}

    print("\n4. Cache-aware bias sweep (selection only)")
    betas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    bs = bias_sweep(traces, [32, 64], betas)
    for n, per in bs.items():
        print(f"   pin {n}/layer:")
        for b, r in per.items():
            print(f"     beta {b:<4} hit {r['pinned_hit']:.2f}  swapped {r['slots_swapped']:.3f}  displaced mass {r['displaced_mass']:.4f}")
    out["bias_sweep"] = {str(n): {str(b): r for b, r in per.items()} for n, per in bs.items()}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
