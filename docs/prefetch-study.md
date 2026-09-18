# Prefetching, routing bias and self-speculation for SSD-streamed MoE

**Date:** 2026-09-18
**Hardware:** Apple M5 Max, 36 GB, internal NVMe (~5 GB/s at queue depth 1, ~12 GB/s with 8 parallel reads)
**Model:** Qwen3-30B-A3B Q4_K_M, expert-contiguous Q4 cache (16.35 GB, 48 layers x 128 experts, top-8)

## Question

The end goal is a ~100B MoE (gpt-oss-120b class: 36 layers x 128 experts, top-4, ~61 GB of experts)
on a 36 GB Mac, and ideally on a 16 GB one. Four techniques were proposed to hide or avoid the SSD
reads for experts that are not pinned in RAM:

1. multi-layer lookahead prefetch (predict layer l+j's experts from the hidden state at layer l),
2. cache-aware routing bias (prefer resident experts when scores are close),
3. batched verification / self-speculative decoding (draft with resident experts only, verify k tokens as one batch),
4. cross-token prefetch from a token-id prior.

Because a 100B model's *fraction* of pinnable experts is what matters, the 30B model was run in two
regimes that reproduce those fractions:

| regime | pinned experts / layer | cold buffer | emulates |
|---|---:|---:|---|
| r16 | 16 / 128 (13%, 2.2 GB) | 512 MB | gpt-oss-120b on a 16 GB Mac (~11% pinnable) |
| r36 | 56 / 128 (44%, 7.3 GB) | 2 GB | gpt-oss-120b on a 36 GB Mac (~43% pinnable) |

Cold experts are read with `pread` + `F_NOCACHE` into a bounded buffer, so the page cache cannot
silently turn the 36 GB machine into an all-in-RAM run.

## Engine changes (all regimes benefit)

| change | effect |
|---|---|
| Persistent spin-then-sleep thread pool; MoE split into (expert x row-chunk) tasks; attention Q/K/V projections fused into one dispatch, heads parallel | 13 -> 22-27 tok/s at 6 threads (single thread: 6.6) |
| Single-row fast path (the 4-row kernel was doing 4x the work for one token) | part of the above |
| Vectorized Q6_K unpack, per-sub-block Q8 sums (2 fewer `vdot` per chunk) | kernel-level |
| A layer's cold misses issued together (queue depth 8) instead of one blocking read at a time | r16 baseline 6.3 -> 12.3 tok/s |
| Output head kept as Q6_K and run through the threaded kernel; embedding rows dequantized on demand | -2.5 GB RAM |
| 4-row batched expert kernel (weights decoded once per 4 input rows) + `forward_rows` | batched prefill and verification; bit-identical to sequential |

Compute per token with everything resident is ~30 ms (MoE 13-16, attention 8-9, head 3, routing 2),
i.e. a ceiling of ~31-34 tok/s on 6 threads. Using the 12 "Performance" cores as well was slower
in the first test (E-core-class threads drag the barrier); see the controlled rerun below.

## Offline trace analysis (662 tokens, 8 prompts)

| predictor | result |
|---|---|
| Lookahead: router of layer l+j applied to the residual at layer l, recall of the true top-8 | j=1: 0.88 (k+0), 0.97 (k+4); j=4: 0.78 / 0.90; j=8: 0.72 / 0.84 |
| Previous token's experts at the same layer | 0.44 overlap |
| Token-id prior (per layer, from traces) | 0.34 recall at top-8, 38% coverage |
| Restricted-to-pinned routing == full routing (per layer) | pin 16: 0%, 32: 4%, 48: 22%, 64: 51%, 96: 91% |

Lookahead is accurate enough to be useful several layers ahead; the token-id prior is not.

## Benchmark matrix

5 prompts x 48 greedy tokens per configuration. `wait` is the time the compute thread spent
blocked on demand reads, `pf` the Python cost of issuing predictions. Because a background
security scanner on this machine intermittently steals P-core time, compute times varied up to 2x
between runs; the *fair* column recomputes tok/s with each regime's median compute time and the
run's own I/O-side costs (wait + pf), which is what the techniques actually change.

### r16 (bandwidth-bound: ~800 MB of expert reads per token)

| config | fair tok/s | wait ms | pf ms | reads/tok | MB/tok | note |
|---|---:|---:|---:|---:|---:|---|
| base | 11.4 | 42.8 | 0 | 287 | 781 | |
| la1 | 14.3 | 20.7 | 5.2 | 330 | 900 | |
| la2 | 13.6 | 21.8 | 8.8 | 362 | 987 | |
| la2x2 | 14.8 | 12.6 | 11.2 | 432 | 1178 | best |
| la4 | 12.9 | 17.0 | 16.4 | 417 | 1137 | |
| la8x4 | 10.1 | 25.7 | 31.0 | 648 | 1763 | wasted reads saturate SSD |
| token prior | 10.3 | 51.5 | 0.5 | 389 | 1057 | |
| bias 0.5 | 11.1 | 45.6 | 0 | 235 | 640 | |
| bias 1.0 | 12.7 | 34.5 | 0 | 161 | 439 | ppl +7.6% |
| bias 2.0 | 19.3 | 8.7 | 0 | 29 | 78 | ppl +37%: unusable |
| spec k=4 | 4.3 | 62.8 | 0 | 883 | 2406 | 32% acceptance |
| spec k=4, no draft prefetch | 5.6 | 49.9 | 0 | 291 | 794 | 49% acceptance |

### r36 (latency-bound: ~26 cold reads per token)

| config | fair tok/s | wait ms | pf ms | reads/tok | note |
|---|---:|---:|---:|---:|---|
| base | 17.7 | 11.4 | 0 | 26 | |
| la1 / la2 / la2x2 | 17.6 / 17.3 / 17.8 | 4.9 / 3.9 / 3.2 | 6.5 / 8.6 / 8.0 | 35-56 | wait removed, overhead added (see rerun) |
| token prior | 17.6 | 12.1 | 0.3 | 48 | no wait reduction |
| bias 0.5 | 20.7 | 3.5 | 0 | 9 | ppl unchanged |
| bias 1.0 | 21.7 | 1.3 | 0 | 3 | ppl +2% |
| spec k=4 | 10.7 | 3.7 | 0 | 36 | 86% acceptance, 4.4 tok/round, 1.4x compute per token |
| spec k=6 | 10.5 | 3.4 | 0 | 36 | 89% acceptance, 6.2 tok/round |

## Quality cost of routing bias (teacher-forced, 193 tokens, KL vs unbiased routing)

| pins/layer | beta | KL | top-1 agreement | perplexity (base 6.60) | slots swapped |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.25 | 0.012 | 94.8% | 6.54 | 7% |
| 16 | 0.5 | 0.025 | 92.7% | 6.72 | 14% |
| 16 | 1.0 | 0.052 | 89.1% | 7.10 | 28% |
| 16 | 2.0 | 0.335 | 76.6% | 9.01 | 49% |
| 56 | 0.5 | 0.014 | 94.8% | 6.56 | 9% |
| 56 | 1.0 | 0.025 | 91.7% | 6.75 | 13% |
| 56 | 2.0 | 0.036 | 90.1% | 6.59 | 13% |

The gate-weight guard at tau=0.1 never triggered (identical numbers): displaced experts rarely
carry more than 10% of the gate mass, so a useful guard needs a much lower threshold or the
logit-margin rule. With many pins the best resident alternative is usually close in score and even
beta=2 is nearly free; with few pins the substitutes are poor and quality collapses.

## Controlled rerun

Key configurations interleaved and repeated twice (the scanner was active during one of the two
r16 repetitions, hence the value of the fair column). Lookahead prediction is vectorized here
(one matmul over all target layers per source layer), which cut its overhead from ~8 to ~4 ms.

| config | r16 fair tok/s | r16 wait+pf ms | r16 MB/tok | r36 fair tok/s | r36 wait+pf ms | r36 reads/tok |
|---|---:|---:|---:|---:|---:|---:|
| base | 14.6 | 36.4 + 0 | 781 | 24.1 | 10.7 + 0 | 26 |
| la1 | 17.7 | 22.2 + 3.7 | 900 | 25.2 | 4.5 + 3.5 | 35 |
| la2x2 | 19.6 | 14.7 + 5.2 | 1170 | 25.4 | 3.4 + 4.3 | 56 |
| la4x2 | 18.8 | 14.7 + 7.3 | 1344 | 24.3 | 4.1 + 5.4 | 73 |
| bias 0.5 | 15.0 | 35.5 + 0 | 640 | 28.1 | 3.6 + 0 | 9 |
| bias 1.0 | 17.4 | 25.4 + 0 | 439 | 30.1 | 1.3 + 0 | 3 |
| la2x2 + bias 0.5 | **20.8** | 12.0 + 5.3 | 1107 | 26.9 | 0.9 + 4.5 | 54 |
| la2x2 + bias 1.0 | 21.2 | 10.5 + 5.4 | 1087 | 27.1 | 0.4 + 4.5 | 50 |

Compute threads (r36 base, same conditions): 6 threads 27.0 ms compute, 12 threads 21.9 ms,
18 threads 25.1 ms. The engine now defaults to the largest non-efficiency core cluster (12 on this M5 Max).

**Page-cache caveat (found later).** `F_NOCACHE` prevents new caching but does not evict pages
that are already cached, and the 16 GB expert cache had been written shortly before these runs, so
part of the "cold" traffic above was served from RAM (implied read rates of 10-23 GB/s exceed
the SSD). Re-measured after evicting the page cache (`benchmarks/evict_cache.py`), the r16
regime reads at a realistic ~7.7 GB/s:

| config (evicted, r16) | tok/s | wait ms | MB/tok |
|---|---:|---:|---:|
| base | 10.8 | 67 | 715 |
| la2x2 + bias 0.5 | 12.4 | 53 | 982 |

The gain shrinks from +42% to +15%: when the SSD really is the limit, lookahead's extra reads
cost real bandwidth. The r36 regime (26 cold reads per token) is far less affected. All
gpt-oss-120b numbers below are taken with the cache evicted before every run.

Recommended defaults: lookahead J=2 with 2 extra candidates plus beta=0.5 when less than ~25% of
experts are pinned; beta=0.5 alone (lookahead off) when more than ~40% are pinned.

## Verdicts

- **Lookahead prefetch: keep for low pin budgets.** J=2 with 2 extra candidates gives +34% in the
  bandwidth-bound regime and stacks with bias (+42% together); deeper prediction turns into wasted
  bandwidth. At 44% pinned it removes nearly all wait but its ~4 ms of prediction overhead is
  larger than the wait it hides once bias is on, so it nets nothing there.
- **Cache-aware bias: keep, with beta <= 0.5 by default.** Free at 44% pinned (+17%), and the right knob
  when a user prefers speed over exactness at low pin budgets (beta=1: +19% for +8% perplexity).
  Selection bias only; gate weights stay unbiased.
- **Token-id prior: drop.** Prefetches issued at token start have plenty of lead time but the
  predictions are too weak; every wrong prefetch costs bandwidth.
- **Self-speculative decoding: drop for this hardware/model.** It needs high acceptance (many pins)
  *and* large I/O wait (few pins) at the same time. The draft passes cost full compute; at r36 they
  hide 11 ms of wait for ~1.4x the compute. A cheap external draft model would change the calculus,
  but no small model shares gpt-oss's tokenizer.
- **Biggest single wins were engine-level:** fine-grained threading (2x) and issuing a layer's misses
  in parallel (2x in the bandwidth-bound regime).

## Implications for gpt-oss-120b

At 44% pinned (36 GB) the model should be compute-bound with lookahead + beta=0.5 bias, which on this
CPU path means roughly the warm ceiling for 5.1B active parameters (~15-20 tok/s at 6 threads).
At 11% pinned (16 GB) the SSD is the limit: ~1.9 GB of expert reads per token at 0% hit;
lookahead-1 and bias are the levers, and expected speed is 2-4 tok/s.
