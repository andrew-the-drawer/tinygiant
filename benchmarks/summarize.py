"""Summarize a results JSONL into a markdown table (mean over prompts per label).

Compute time (moe+attn+head+route) varies run to run on this machine because of
an endpoint-security scanner competing for cores, so besides raw tok/s the table
reports a "fair" tok/s that combines each run's measured I/O-side cost
(demand wait + prefetch overhead) with the regime's median compute time.
"""

import json
import sys
from collections import OrderedDict

import numpy as np

COMPUTE = ("ms_moe", "ms_attn", "ms_head", "ms_route")
IO = ("ms_gather", "ms_prefetch")


def regime_of(label):
    return label.split("_", 1)[0]


def main(path):
    groups = OrderedDict()
    for line in open(path):
        r = json.loads(line)
        r["compute_ms"] = sum(r.get(k, 0) for k in COMPUTE)
        r["io_ms"] = sum(r.get(k, 0) for k in IO)
        r["total_ms"] = 1000.0 / r["tok_s"]
        groups.setdefault(r["label"], []).append(r)

    ref = {}
    for label, rows in groups.items():
        ref.setdefault(regime_of(label), []).extend(r["compute_ms"] for r in rows if not r.get("spec_rounds"))
    ref = {k: float(np.median(v)) for k, v in ref.items()}

    cols = [("tok_s", "tok/s", "{:.1f}"), ("fair", "fair tok/s", "{:.1f}"),
            ("compute_ms", "compute ms", "{:.1f}"), ("io_ms", "io ms", "{:.1f}"),
            ("wait_ms_tok", "wait ms", "{:.1f}"), ("ms_prefetch", "pf ms", "{:.1f}"),
            ("pinned_hit", "pin%", "{:.0%}"), ("miss", "miss%", "{:.0%}"),
            ("reads_tok", "reads/tok", "{:.0f}"), ("mb_tok", "MB/tok", "{:.0f}"),
            ("spec_accept_rate", "accept", "{:.0%}"), ("spec_tokens_per_round", "tok/round", "{:.2f}"),
            ("swap_frac", "swapped", "{:.0%}")]
    print(f"Reference compute ms per regime: {ref}\n")
    print("| label | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|" + "|".join("---:" for _ in cols) + "|")
    for label, rows in groups.items():
        reg = regime_of(label)
        for r in rows:
            r["swap_frac"] = r.get("router_swapped", 0) / max(1, r.get("router_slots", 0))
            other = r["total_ms"] - r["compute_ms"] - r["io_ms"]
            if r.get("spec_rounds"):
                # spec runs: scale compute by the same factor the regime median implies
                scale = ref[reg] / max(1e-6, r["compute_ms"] / max(1, r["rows"] / r["n_tokens"]))
                r["fair"] = 1000.0 / (r["compute_ms"] * scale + r["io_ms"] + other)
            else:
                r["fair"] = 1000.0 / (ref[reg] + r["io_ms"] + other)
        vals = []
        for key, _, fmt in cols:
            if key.startswith("spec") and not rows[0].get("spec_rounds"):
                vals.append("")
                continue
            vals.append(fmt.format(np.mean([r.get(key, 0) or 0 for r in rows])))
        print(f"| {label} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results_matrix.jsonl")
