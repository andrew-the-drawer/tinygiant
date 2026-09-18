"""Quality cost of cache-aware routing bias: teacher-forced KL vs unbiased routing.

Feeds a fixed token sequence through the engine under several RoutingConfigs
and compares next-token distributions position by position.

  python experiments/quality_kl.py --pin 16 --betas 0.5,1,2
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinygiant._tokenizer import load_tokenizer
from tinygiant.engine import NWSEngine
from tinygiant.router import RoutingConfig

TEXT = (
    "Mixture-of-experts models route each token to a small subset of expert networks. "
    "Because only a few experts fire per token, the active parameter count is far smaller than "
    "the total, which makes it possible to stream dormant experts from disk. The router computes "
    "a score for every expert and selects the top-k; the gate weights are the softmax of those "
    "scores. When an expert is not resident in memory, the system must either wait for it or "
    "substitute a resident expert with a slightly lower score. The question is how much quality "
    "that substitution costs.\n\n"
    "def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n"
    "        elif items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
    "    return -1\n"
)


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def teacher_force(engine, tokens, chunk=8):
    engine.reset_kv()
    out = []
    for i in range(0, len(tokens), chunk):
        out.append(engine.forward_rows(tokens[i:i + chunk], i))
    return np.concatenate(out)[:-1]   # distribution for tokens[1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-30B-A3B-Q4_K_M.gguf"))
    p.add_argument("--cache", default=os.path.expanduser("~/models/nws_q4_cache"))
    p.add_argument("--pin", type=int, default=16)
    p.add_argument("--cold-mb", type=float, default=512)
    p.add_argument("--betas", default="0.25,0.5,1,2")
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--margin", type=float, default=None)
    args = p.parse_args()

    tok = load_tokenizer(args.model)
    ids = tok.encode(TEXT)
    engine = NWSEngine(args.model, args.cache, cold_budget_mb=args.cold_mb, verbose=False)
    engine.pin_experts(args.pin, calibrate_tokens=10, prompt_tokens=ids[:16])

    engine.routing = RoutingConfig()
    base = log_softmax(teacher_force(engine, ids))
    base_p = np.exp(base)
    targets = np.array(ids[1:])
    base_nll = -base[np.arange(len(targets)), targets].mean()
    print(f"{len(ids)} tokens; baseline NLL {base_nll:.4f} (ppl {np.exp(base_nll):.2f})")
    print(f"{'beta':>6} {'KL(base||b)':>12} {'top1 agree':>11} {'NLL':>8} {'ppl':>8} {'pin hit':>8} {'swapped':>8}")

    for beta in [float(b) for b in args.betas.split(",")]:
        engine.routing = RoutingConfig(bias=beta, tau=args.tau, margin=args.margin)
        engine.expert_cache.reset_stats()
        for k in engine.router.stats:
            engine.router.stats[k] = 0
        lb = log_softmax(teacher_force(engine, ids))
        kl = (base_p * (base - lb)).sum(axis=-1).mean()
        agree = (lb.argmax(-1) == base.argmax(-1)).mean()
        nll = -lb[np.arange(len(targets)), targets].mean()
        st = engine.stats()
        acc = max(1, st["accesses"])
        swapped = st["router_swapped"] / max(1, st["router_slots"])
        print(f"{beta:6.2f} {kl:12.5f} {agree:11.3f} {nll:8.4f} {np.exp(nll):8.2f} "
              f"{st['pinned_hits'] / acc:8.3f} {swapped:8.3f}")
    engine.close()


if __name__ == "__main__":
    main()
