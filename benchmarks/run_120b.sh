#!/bin/bash
# gpt-oss-120b on this 36 GB Mac and a 16 GB-Mac emulation.
#
# Budgets leave the OS headroom: 13.2 MB per expert, so 32 pins/layer is 14.2 GB
# (plus ~2.5 GB of attention/head/KV and the cold buffer). Pinning 46/layer
# (20 GB) made the machine swap and invalidated the run. The page cache is
# evicted once, gently, before the matrix; per-run eviction at 85% of RAM also
# pushed the OS into swap. The 57 GB cache cannot fit in RAM anyway and cold
# reads use F_NOCACHE.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=${M:-$HOME/models/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4.gguf}
C=${C:-$HOME/models/gptoss120b_cache}
OUT=${OUT:-benchmarks/results_120b.jsonl}
TOKENS=${TOKENS:-32}

run() {
  local regime=$1; shift
  local label=$1; shift
  if [ "$regime" = m16 ]; then
    PIN=12; COLD=512
    # small footprint: safe to evict before every run (pages pinned by earlier
    # runs would otherwise serve "cold" reads from RAM)
    $PY benchmarks/evict_cache.py 0.6 > /dev/null; sleep 5
  else
    PIN=32; COLD=1536
  fi
  $PY -u benchmarks/bench.py --model "$M" --cache "$C" --pin $PIN --cold-mb $COLD --tokens $TOKENS \
      --prompts coding,factual,reasoning --quiet --label "${regime}_${label}" --out "$OUT" "$@" \
      2>&1 | grep -E "mean |Traceback|error:"
}

REGIMES=${REGIMES:-"m36 m16"}
$PY benchmarks/evict_cache.py 0.6 > /dev/null
sleep 20
for regime in $REGIMES; do
  run $regime base
  run $regime la1            --lookahead 1
  run $regime la2            --lookahead 2
  run $regime la2x2          --lookahead 2 --extra 2
  run $regime bias0.5        --bias 0.5
  run $regime la1_bias0.5    --lookahead 1 --bias 0.5
  run $regime la2_bias0.5    --lookahead 2 --bias 0.5
done
