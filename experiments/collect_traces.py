"""Record routing traces (router logits, selections, hidden states) for offline analysis.

  python experiments/collect_traces.py --tokens 64 --out experiments/traces
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinygiant._constants import EMBED_DIM, N_EXPERTS, N_EXPERTS_USED, N_LAYERS
from tinygiant._tokenizer import load_tokenizer
from tinygiant.engine import NWSEngine
from tinygiant.trace import TraceRecorder

PROMPTS = {
    "coding": "Write a Python function that finds the longest common subsequence of two strings using dynamic programming.",
    "reasoning": "A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river in a boat that can only carry him and one item. How does he get everything across?",
    "creative": "Write a short poem about the feeling of discovering something beautiful in an unexpected place.",
    "math": "Prove that the square root of 2 is irrational using proof by contradiction.",
    "factual": "Explain how mRNA vaccines work, including the role of spike proteins and the immune response.",
    "dialog": "User: I'm planning a trip to Japan in April. What should I know?\nAssistant:",
    "history": "The fall of the Western Roman Empire in 476 AD is often attributed to",
    "science": "Photosynthesis converts light energy into chemical energy through a series of reactions that",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-30B-A3B-Q4_K_M.gguf"))
    p.add_argument("--cache", default=os.path.expanduser("~/models/nws_q4_cache"))
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "traces"))
    p.add_argument("--prompts", default="all")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.model)
    engine = NWSEngine(args.model, args.cache, cold_budget_mb=4096, io_threads=4)

    names = list(PROMPTS) if args.prompts == "all" else args.prompts.split(",")
    for name in names:
        ids = tok.encode(PROMPTS[name])
        engine.reset_kv()
        rec = TraceRecorder(N_LAYERS, N_EXPERTS, N_EXPERTS_USED, EMBED_DIM, keep_hidden=True)
        engine.trace = rec
        # single-row prefill so every prompt token is traced too
        logits = None
        for i, t in enumerate(ids):
            logits = engine.forward_rows([t], i)[0]
        gen = list(engine.decode(logits, len(ids), args.tokens, args.temperature, 0.9))
        engine.trace = None
        path = rec.save(out / f"{name}.npz")
        print(f"{name}: {len(ids)} prompt + {len(gen)} generated tokens -> {path}")
        print("  ", repr(tok.decode(gen)[:120]))
    engine.close()


if __name__ == "__main__":
    main()
