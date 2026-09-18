"""Compare the TinyGiant engine against llama.cpp reference logits/tokens.

  python experiments/llamacpp_reference.py --model M.gguf --out /tmp/ref.npz
  python experiments/validate_vs_llamacpp.py --model M.gguf --cache CACHE --ref /tmp/ref.npz
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinygiant.engine import NWSEngine


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--pin", type=int, default=None, help="experts pinned per layer (default: all)")
    p.add_argument("--cold-mb", type=float, default=4096)
    args = p.parse_args()

    ref = np.load(args.ref)
    ids = [int(t) for t in ref["ids"]]
    engine = NWSEngine(os.path.expanduser(args.model), os.path.expanduser(args.cache),
                       cold_budget_mb=args.cold_mb)
    S = engine.spec
    engine.pin_experts(args.pin if args.pin is not None else S.n_experts)

    engine.reset_kv()
    ours = np.concatenate([engine.forward_rows(ids[i:i + 8], i) for i in range(0, len(ids), 8)])
    ref_l = ref["logits"]
    lo, lr = log_softmax(ours), log_softmax(ref_l)
    kl = (np.exp(lr) * (lr - lo)).sum(axis=-1)
    agree = (lo.argmax(-1) == lr.argmax(-1))
    print(f"prompt positions: {len(ids)}")
    print(f"  KL(ref||ours) per position: mean {kl.mean():.4f}, max {kl.max():.4f}")
    print(f"  top-1 agreement: {agree.mean():.2%}   ours argmax {lo.argmax(-1).tolist()}")
    print(f"                                   ref  argmax {lr.argmax(-1).tolist()}")

    gen = list(engine.decode(ours[-1], len(ids), len(ref["gen"]), temperature=0.0))
    ref_gen = ref["gen"].tolist()
    match = 0
    for a, b in zip(gen, ref_gen):
        if a != b:
            break
        match += 1
    print(f"greedy continuation: {match}/{len(ref_gen)} tokens match before first divergence")
    print("  ours:", gen)
    print("  ref: ", ref_gen)
    try:
        from tinygiant._tokenizer import load_tokenizer
        tok = load_tokenizer(os.path.expanduser(args.model))
        print("  ours text:", repr(tok.decode(gen)))
        print("  ref  text:", repr(tok.decode(ref_gen)))
    except Exception as e:
        print("  (no tokenizer for text:", e, ")")
    st = engine.stats()
    n = st["rows"]
    print("timers ms/row:", {k: round(v / n * 1000, 1) for k, v in engine.timers.items()})
    engine.close()


if __name__ == "__main__":
    main()
