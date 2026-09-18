# Serving a 100B MoE from SSD on a Mac: algorithms, benchmarks, explanations

**Date:** 2026-09-18
**Machine:** Apple M5 Max, 36 GB unified memory, internal NVMe (measured: 11 GB/s single-stream,
17.8 GB/s with 8 parallel readers, on never-cached data), macOS.
**Models:** Qwen3-30B-A3B Q4_K_M (development), gpt-oss-20b MXFP4 (validation), gpt-oss-120b MXFP4 (target).
**Code:** `tinygiant/` (engine), `csrc/libtinygiant.c` (kernels), `benchmarks/`, `experiments/`.

## 1. Problem

A Mixture-of-Experts model only touches a few experts per token, so most of its weights are
idle at any moment. gpt-oss-120b has 36 layers x 128 experts of 13.2 MB (MXFP4) = 61 GB of
experts, but a token uses 4 per layer: 1.9 GB of expert weights per token. The non-expert weights
(attention Q8_0, output head, norms, routers) are ~1.6 GB. So on a 36 GB Mac the model fits
only if experts stream from SSD, and the question is how fast that can be.

Two budgets were studied, because the pinnable *fraction* of experts is what determines the regime:

| budget | pinned experts per layer | pinned RAM | cold buffer | emulates |
|---|---:|---:|---:|---|
| m36 | 32 / 128 (25%) | 14.2 GB | 1.5 GB | this 36 GB Mac (46/layer = 20 GB made the OS swap) |
| m16 | 12 / 128 (9%) | 5.3 GB | 0.5 GB | a 16 GB Mac |

During development the same regimes were emulated on Qwen3-30B-A3B (48 layers x 128 experts of
2.7 MB, top-8) as r36 = 56/128 pinned and r16 = 16/128 pinned; those results are in
[prefetch-study.md](prefetch-study.md) and summarized here where they add something.

## 2. The engine

### 2.1 Storage layout and the two tiers

`tinygiant-relayout` rewrites the GGUF's interleaved expert tensors into one file per layer where
each expert's gate/up/down matrices are contiguous (`layer_NNN.q4.bin`, raw quantized blocks; an
`index.json` records offsets and formats). One expert is one `pread`.

* **Pinned tier.** The most frequently used experts per layer (10-token calibration counts,
  backfilled from a static activation profile) are `mmap`'d and `mlock`'d. They never leave RAM.
* **Cold tier (`iosched.ColdLoader`).** Everything else is read with `preadv` from file descriptors
  opened with `F_NOCACHE`, into a bounded buffer with LRU eviction. A priority queue feeds 8 I/O
  threads (demand reads first, then prefetches by predicted distance). Entries used by the layer
  currently computing cannot be evicted; a pass boundary drops prefetches queued for an earlier
  pass. Bypassing the page cache keeps RAM use deterministic, which is what makes a 16 GB machine
  emulable on a 36 GB one.

A layer's cold misses are issued together before the first blocking wait, so the SSD sees queue
depth = number of misses instead of 1. On Qwen r16 this alone went from 6.3 to 12.3 tok/s.

### 2.2 Kernels (`csrc/libtinygiant.c`)

All matmuls are "fused": weights stay in their quantized blocks, the activation vector is
quantized once to Q8 (256-wide blocks with per-32 sub-block sums), and ARM `vdotq_s32` does the
int8 dot products. No dequantized copy of any weight ever exists in RAM.

* Formats: Q4_K, Q6_K (Qwen experts), MXFP4 (gpt-oss experts: 1 e8m0 exponent + 16 bytes of e2m1
  nibbles per 32 values, decoded with a 16-entry `vqtbl1q` table lookup), Q8_0 (gpt-oss attention
  and output head), f16.
* 4-row variants decode each weight block once and dot it against 4 input rows. They power batched
  prefill and batched verification; a 1-row fast path avoids doing 4x the work for single-token
  decode.
* MoE as two task phases over (expert x row-chunk): gate/up + activation, then down; per-expert
  biases and the gpt-oss activation (`clamp(gate, max 7) * sigmoid(1.702 gate) * (clamp(up, +-7) + 1)`)
  are handled in-kernel.
* Attention: Q/K/V projections in one dispatch, per-head tasks, generic over weight format, with
  optional QK RMS-norm (Qwen), projection biases, attention sinks (an extra logit per head that
  joins the softmax denominator and contributes no value) and a sliding window (gpt-oss: 128 on
  even layers). RoPE tables include YaRN scaling for gpt-oss (factor 32, mscale 0.1 ln 32 + 1).
* A persistent pthread pool whose workers spin ~1 ms before sleeping; ~200 dispatches per token
  cost nothing measurable. The largest non-efficiency core cluster is used (12 on this chip:
  6 threads 27 ms/token compute, 12 threads 22 ms, 18 threads 25 ms).

Result: Qwen3-30B single thread 6.6 tok/s -> 31.5 tok/s; gpt-oss-20b all-pinned 62 tok/s on CPU
(llama.cpp: 79 tok/s on Metal, 33 tok/s on CPU).

### 2.3 Batched rows

`forward_rows(tokens, start_pos)` runs k tokens per pass: attention row by row (each row appends
its K/V before the next attends), then one MoE call over the union of the rows' experts, each
expert loaded once. It is bit-identical to k sequential passes and serves prefill (chunks of 8)
and speculative verification.

## 3. Algorithms

### 3.1 Multi-layer lookahead prefetch

Each layer adds a small update to the residual stream, so the residual at layer *l* is a good
proxy for the input of layer *l+j*. At layer *l*, after attention, the engine applies the routers
of layers *l+1..l+J* to the current residual (RMS-norm weight folded into the router so all J
targets are one matmul against the normalized residual; router biases added for gpt-oss), takes the
top *k+extra* per target layer, and requests the non-resident ones at priority *j*. Layer *l+1*'s
prediction supersedes earlier ones for that target (queued-but-unstarted requests not in the new
set are dropped); each expert is requested at most once per pass. Offline, from 662 traced Qwen
tokens, recall of the true top-8 was 0.88 one layer ahead (0.97 with 4 extra candidates) and still
0.72 / 0.84 eight layers ahead.

### 3.2 Cache-aware routing bias

Selection: top-k of `logits + beta * resident_mask`; gate weights: softmax of the *unbiased* logits
of the selected experts (the same split DeepSeek uses for load balancing). Optional guards: never
displace the top-`protect_top`, only swap when the logit gap is below a margin, never displace an
expert whose gate probability exceeds tau. The tau=0.1 guard never triggered in practice (the k-th
expert rarely holds 10% of the mass); the margin rule is the useful one if a guard is wanted.

### 3.3 Self-speculative decoding (batched verification)

Draft = the same model with routing restricted to resident experts (zero I/O). Draft k tokens one
at a time, then verify `[t0, d1..dk]` in one batched pass with exact routing (each cold expert
loaded once for all k+1 rows, all misses issued at once). Greedy: accept while argmax matches;
sampling: the standard rejection rule with the residual distribution. The verify pass rewrites the
draft's KV entries; `kv_len` is rolled back to the accepted prefix. The draft pass also computes the
unrestricted routing, so it can prefetch the verify pass's misses ("draft prefetch").

### 3.4 Token-id prior (cross-token prefetch)

A per-layer table token id -> most frequent experts, built from traces, queried when the next
token is sampled so reads for deep layers have a whole pass of lead time.

## 4. Methodology and the three confounds

* 3-5 prompts x 32-48 greedy tokens per configuration; per-run counters for pinned hits, cold-buffer
  hits, demand misses, bytes read, time blocked on demand reads ("wait"), Python prefetch overhead
  ("pf"), and per-phase compute.
* **Endpoint-security scanner.** `wdavdaemon` intermittently takes a core and inflated compute times
  by up to 2x. Runs were interleaved and repeated; `summarize.py` also reports a "fair" tok/s using
  the regime's median compute time with each run's own I/O costs.
* **Page cache.** `F_NOCACHE` does not evict pages already cached, and pages pinned by a previous
  run stay cached after it exits. Qwen's 16 GB cache is cached almost entirely after a few runs;
  the first r16 numbers were inflated (14.6 -> 20.8 tok/s became 10.8 -> 12.4 once evicted).
  `benchmarks/evict_cache.py` (allocate and touch most of RAM) is run before measurements; for the
  120b it was run before each m16 config. The 57 GB gpt-oss cache cannot be cached as a whole.
* **Swap.** Pinning 46 experts/layer (20 GB) plus the cold buffer on a 36 GB machine with an IDE
  open sent 11 GB to swap and made MoE compute look 3-5x slower. The valid budget is 32/layer.

Validation: the gpt-oss-20b engine differs from llama.cpp (Metal) by mean KL 0.009 per position
with 91% top-1 agreement and 6 matching greedy tokens; llama.cpp's own Metal and CPU backends
differ by KL 0.004 and diverge at greedy token 4, so the port is within backend noise. The 120b
cannot be run in llama.cpp on this machine (killed at 63 GB), so its check is indirect: same code
paths as the 20b, coherent output, and perplexity 4.39 vs the 20b's 4.66 on the same text.

## 5. Results: gpt-oss-120b

tok/s over 3 prompts x 32 tokens, greedy; "wait" = ms per token blocked on demand reads;
"reads" = SSD reads per token (13.2 MB each). m36 rows from the rerun on a settled machine.

### m36: 32 experts/layer pinned (14.2 GB), 1.5 GB cold buffer

| config | tok/s | wait ms | pf ms | pinned hit | miss | reads/tok | MB/tok | slots swapped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 9.1 | 73 | 0 | 44% | 30% | 55 | 694 | |
| lookahead 1 | 11.1 | 47 | 4 | 44% | 22% | 70 | 884 | |
| lookahead 2 | 11.5 | 41 | 6 | 44% | 15% | 84 | 1058 | |
| lookahead 2, +2 candidates | 7.5 | 63 | 12 | 44% | 16% | 145 | 1822 | |
| **bias 0.5** | **16.9** | 30 | 0 | 64% | 13% | 21 | 260 | 27% |
| lookahead 1 + bias 0.5 | 13.9 | 23 | 4 | 61% | 11% | 56 | 710 | 23% |
| lookahead 2 + bias 0.5 | 13.8 | 21 | 6 | 58% | 7% | 71 | 890 | 19% |

### m16: 12 experts/layer pinned (5.3 GB), 0.5 GB cold buffer, page cache evicted before each run

| config | tok/s | wait ms | pf ms | pinned hit | miss | reads/tok | MB/tok | slots swapped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 10.5 | 63 | 0 | 21% | 52% | 110 | 1381 | |
| lookahead 1 | 12.5 | 41 | 4 | 21% | 32% | 128 | 1617 | |
| lookahead 2 | 11.4 | 41 | 6 | 21% | 18% | 142 | 1796 | |
| lookahead 2, +2 candidates | 10.8 | 39 | 7 | 21% | 24% | 198 | 2499 | |
| bias 0.5 | 13.2 | 49 | 0 | 51% | 39% | 68 | 858 | 29% |
| lookahead 1 + bias 0.5 | 13.1 | 31 | 5 | 45% | 29% | 113 | 1421 | 26% |
| **lookahead 2 + bias 0.5** | **15.4** | 19 | 6 | 37% | 14% | 128 | 1618 | 23% |

### Quality cost of the bias on gpt-oss-120b (teacher-forced, 194 tokens, KL vs unbiased routing)

| pins/layer | beta | KL | top-1 agreement | perplexity (base 4.39) | slots swapped |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.25 | 0.033 | 92.7% | 4.41 | 13% |
| 32 | 0.5 | 0.054 | 91.2% | 4.45 | 23% |
| 32 | 1.0 | 0.148 | 88.6% | 4.88 | 37% |
| 12 | 0.25 | 0.027 | 91.7% | 4.36 | 11% |
| 12 | 0.5 | 0.053 | 91.2% | 4.36 | 21% |

beta = 0.5 costs about 1% perplexity; beta = 1 costs 11% and is not worth it. (On Qwen3-30B the
picture was the same: beta 0.5 free at 44% pinned, +2% ppl at 13% pinned; beta 2 collapsed
quality, +37% ppl.)

### Summary

| | 36 GB Mac | 16 GB Mac (emulated) |
|---|---:|---:|
| baseline | 9.1 tok/s | 10.5 tok/s |
| best exact-output config | 11.5 (lookahead 2) | 12.5 (lookahead 1) |
| best with bias 0.5 (~1% ppl) | **16.9** (bias alone) | **15.4** (lookahead 2 + bias) |

## 6. Why the numbers come out this way

**Two regimes.** With 32 pins the 120b misses ~1.5 experts per layer. Each layer then waits for
about one 13 MB read at single-stream speed (73 ms / 55 reads = 1.3 ms, i.e. ~10 GB/s): the run is
*latency-bound* and the SSD is mostly idle. With 12 pins it misses ~3 per layer; those are issued
together, the SSD runs at ~15 GB/s and the run is *bandwidth-bound* (1.4 GB/token), which is why
the 16 GB budget is not slower than the 36 GB one. The remedies differ:

* Latency-bound: overlap. Lookahead by one or two layers keeps reads in flight while the previous
  layer computes (+22-27%), and bias removes the miss altogether (+87%).
* Bandwidth-bound: bytes. Every wasted prefetch is 13 MB the demand reads needed. Extra candidates
  (+2 on top-4 is +50% bytes) always lost; lookahead depth 1-2 with no extra candidates is the
  sweet spot; bias reduces bytes directly and stacks with lookahead (+47%).

**Why bias is so effective here.** gpt-oss routes top-4 of 128 and the fourth expert is usually a
close call; nudging it to a resident neighbour changes 20-30% of slots but only ~1% of perplexity,
and cuts cold reads from 55 to 21 per token. It converts the pinned set from a cache into a
soft constraint.

**Why lookahead on top of bias loses at m36.** Once bias has removed most misses, the remaining
wait is ~30 ms while lookahead adds 4-6 ms of prediction work per token and reads 2-3x more
bytes (the candidates it fetches are exactly the near-tie experts the bias would have swapped
away). At m16 the residual wait is larger, so the combination still wins.

**Why the Python overhead matters.** Per token the engine spends ~2 ms routing and ~4-6 ms
predicting in numpy; at 15 tok/s that is 10% of the budget. Moving routing and prediction into
the C library is the next cheap win.

**What did not work.** Self-speculative decoding: draft passes cost full compute, and high
acceptance needs many pins, which is exactly when there is little wait to hide (on Qwen: 86-89%
acceptance at 44% pinned but -40% throughput; 32% acceptance at 13% pinned). The token-id prior:
34% offline recall, so its prefetches were mostly wasted bandwidth. Both remain implemented behind
flags (`--spec`, `--token-prior`).

**What the SSD can do.** The probe (`benchmarks/ssd_probe.py`) reads never-touched experts with
`F_NOCACHE`: 11.1 GB/s single stream, 17.8 GB/s with 8 threads. At 13.2 MB per expert and 4 per
layer, the floor with nothing pinned is ~1.9 GB/token, i.e. ~9 tok/s if fully overlapped; the
measured numbers sit between that floor and the ~40 tok/s compute ceiling.

## 7. Recommended defaults

* gpt-oss-120b on a 36 GB Mac: pin 32/layer, cold buffer 1.5 GB, `--bias 0.5`, no lookahead
  (16.9 tok/s); use `--lookahead 2` instead of bias when exact routing is required (11.5 tok/s).
* On a 16 GB Mac: pin 12/layer, cold buffer 0.5 GB, `--lookahead 2 --bias 0.5` (15.4 tok/s
  emulated; a real M1's slower SSD and 4 P-cores will land lower).
* Never pin more than ~40% of RAM; the OS swapping costs more than any expert hit rate gains.
* Evict the page cache before any benchmark that claims to measure the SSD.

## 8. What is next

1. Routing, prediction and the prefetch bookkeeping in C (saves ~6-8 ms/token).
2. Metal for attention and the pinned experts: compute would drop below the I/O floor, making the
   I/O scheduler the whole story.
3. A real 16 GB M1 run (different SSD and core budget).
4. EAGLE3 draft heads ship with the gpt-oss GGUFs; an external draft that costs ~nothing could
   revive batched verification where the self-draft could not.
