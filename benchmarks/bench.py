"""Benchmark runner: one engine configuration over a prompt set, JSONL out.

Example:
  python benchmarks/bench.py --pin 32 --cold-mb 1024 --lookahead 4 --extra 4 --label la4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinygiant._tokenizer import load_tokenizer
from tinygiant.engine import NWSEngine
from tinygiant.predict import TokenPrior
from tinygiant.router import RoutingConfig
from tinygiant.specdec import SpecStats

PROMPTS = {
    "coding": "Write a Python function that implements binary search on a sorted list and returns the index of the target element.",
    "reasoning": "There are three boxes. One contains only apples, one contains only oranges, and one contains both. The labels on all three boxes are wrong. You can pick one fruit from one box. How do you determine what's in each box?",
    "creative": "Write a short story about an astronaut who discovers music coming from an empty planet.",
    "math": "What is the sum of the first 100 positive integers? Show your reasoning step by step.",
    "factual": "Describe the key differences between TCP and UDP protocols, including when you would choose each one.",
}

CALIB_PROMPT = "Explain how mRNA vaccines work, including the role of spike proteins and the immune response."


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-30B-A3B-Q4_K_M.gguf"))
    p.add_argument("--cache", default=os.path.expanduser("~/models/nws_q4_cache"))
    p.add_argument("--pin", type=int, default=32)
    p.add_argument("--calibrate", type=int, default=10)
    p.add_argument("--cold-mb", type=float, default=1024)
    p.add_argument("--legacy-mmap", action="store_true", help="page-cache mode (no cold budget)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--io-threads", type=int, default=8)
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--prompts", default="all")
    p.add_argument("--lookahead", type=int, default=0)
    p.add_argument("--extra", type=int, default=0)
    p.add_argument("--bias", type=float, default=0.0)
    p.add_argument("--margin", type=float, default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--protect-top", type=int, default=0)
    p.add_argument("--token-prior", default=None)
    p.add_argument("--token-prior-n", type=int, default=8)
    p.add_argument("--spec", type=int, default=0)
    p.add_argument("--no-draft-prefetch", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="run")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.jsonl"))
    p.add_argument("--quiet", action="store_true")
    return p


def configure(engine, args):
    engine.routing = RoutingConfig(bias=args.bias, margin=args.margin, tau=args.tau,
                                   protect_top=args.protect_top)
    engine.lookahead_depth = args.lookahead
    engine.lookahead_extra = args.extra
    engine.draft_prefetch = not args.no_draft_prefetch
    engine.token_prior_n = args.token_prior_n
    if args.token_prior:
        engine.token_prior = TokenPrior.load(args.token_prior, min_count=1)


def run_prompt(engine, tok, name, text, args):
    ids = tok.encode(text)
    engine.reset_kv()
    t0 = time.perf_counter()
    logits = engine.prefill(ids)
    t_prefill = time.perf_counter() - t0

    engine.reset_timers()
    engine.expert_cache.reset_stats()
    for k in engine.router.stats:
        engine.router.stats[k] = 0
    spec = SpecStats()
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    gen = list(engine.decode(logits, len(ids), args.tokens, args.temperature, args.top_p,
                             args.spec, rng, spec))
    t_dec = time.perf_counter() - t0
    st = engine.stats()
    n = len(gen)
    acc = max(1, st["accesses"])
    row = dict(
        label=args.label, prompt=name, n_tokens=n,
        tok_s=n / t_dec, prefill_tok_s=len(ids) / t_prefill,
        pinned_hit=st["pinned_hits"] / acc, cold_hit=st["cold_hits"] / acc, miss=st["misses"] / acc,
        wait_ms_tok=st.get("demand_wait_s", 0.0) / n * 1000,
        mb_tok=st.get("bytes_read", 0) / n / 1024**2,
        reads_tok=st.get("n_reads", 0) / n,
        prefetch_hit=st.get("n_prefetch_hit", 0), prefetch_wasted=st.get("n_prefetch_wasted", 0),
        prefetch_dropped=st.get("n_dropped", 0),
        router_swapped=st["router_swapped"], router_slots=st["router_slots"],
        passes=st["passes"], rows=st["rows"],
        text=tok.decode(gen)[:160],
        **{f"ms_{k}": v / n * 1000 for k, v in engine.timers.items()},
        **spec.as_dict(),
    )
    return row


def main():
    args = build_parser().parse_args()
    tok = load_tokenizer(args.model)
    engine = NWSEngine(args.model, args.cache, threads=args.threads,
                       cold_budget_mb=None if args.legacy_mmap else args.cold_mb,
                       io_threads=args.io_threads, verbose=not args.quiet)
    calib = tok.encode(CALIB_PROMPT)
    engine.pin_experts(args.pin, calibrate_tokens=args.calibrate, prompt_tokens=calib)
    configure(engine, args)

    names = list(PROMPTS) if args.prompts == "all" else args.prompts.split(",")
    rows = []
    for name in names:
        row = run_prompt(engine, tok, name, PROMPTS[name], args)
        rows.append(row)
        print(f"[{args.label}] {name:10s} {row['tok_s']:5.2f} tok/s  pin {row['pinned_hit']:.0%} "
              f"cold {row['cold_hit']:.0%} miss {row['miss']:.0%}  wait {row['wait_ms_tok']:.1f} ms "
              f"read {row['mb_tok']:.0f} MB/tok"
              + (f"  acc {row['spec_accept_rate']:.0%} {row['spec_tokens_per_round']:.2f} tok/round" if args.spec else "")
              + f"  | {row['text'][:60]!r}", flush=True)

    with open(args.out, "a") as f:
        for r in rows:
            r["config"] = {k: v for k, v in vars(args).items() if k not in ("out", "quiet")}
            f.write(json.dumps(r) + "\n")
    mean = np.mean([r["tok_s"] for r in rows])
    print(f"[{args.label}] mean {mean:.2f} tok/s over {len(rows)} prompts")
    engine.close()


if __name__ == "__main__":
    main()
