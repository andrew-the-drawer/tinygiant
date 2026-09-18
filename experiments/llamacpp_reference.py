"""Ground truth from llama.cpp: per-position logits for a prompt + greedy continuation.

  python experiments/llamacpp_reference.py --model ~/models/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf --out /tmp/ref20b.npz
"""

import argparse
import os
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The key insight about mixture-of-experts models is that")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--gpu-layers", type=int, default=-1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from llama_cpp import Llama
    llm = Llama(model_path=os.path.expanduser(args.model), n_ctx=512, n_gpu_layers=args.gpu_layers,
                logits_all=True, verbose=False)
    ids = llm.tokenize(args.prompt.encode(), add_bos=True, special=False)
    print("prompt ids:", ids)

    t0 = time.perf_counter()
    llm.eval(ids)
    t_prefill = time.perf_counter() - t0
    logits = np.array(llm.scores[:len(ids)], dtype=np.float32).copy()   # [T, vocab]

    gen = []
    t0 = time.perf_counter()
    for _ in range(args.tokens):
        nxt = int(np.argmax(llm.scores[llm.n_tokens - 1]))
        gen.append(nxt)
        llm.eval([nxt])
    t_dec = time.perf_counter() - t0
    gen_logits = np.array(llm.scores[len(ids):llm.n_tokens], dtype=np.float32).copy()

    text = llm.detokenize(gen).decode("utf-8", errors="replace")
    print(f"prefill {len(ids)/t_prefill:.1f} tok/s, decode {args.tokens/t_dec:.1f} tok/s")
    print("greedy:", gen)
    print(repr(text))
    np.savez(args.out, ids=np.array(ids), logits=logits, gen=np.array(gen), gen_logits=gen_logits)
    print("saved", args.out)


if __name__ == "__main__":
    main()
