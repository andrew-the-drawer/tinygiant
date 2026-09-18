#!/bin/bash
# gpt-oss-120b on this 36 GB Mac (real budget) and a 16 GB-Mac emulation.
# Page cache is evicted before every run so cold reads really come from the SSD.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=${M:-$HOME/models/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4-00001-of-00003.gguf}
C=${C:-$HOME/models/gptoss120b_cache}
OUT=${OUT:-benchmarks/results_120b.jsonl}
TOKENS=${TOKENS:-32}

run() {
  local regime=$1; shift
  local label=$1; shift
  # 13.2 MB per expert: pin 46/layer ~ 22 GB (36 GB Mac), pin 12/layer ~ 5.7 GB (16 GB Mac)
  if [ "$regime" = m16 ]; then PIN=12; COLD=512; else PIN=46; COLD=2048; fi
  $PY benchmarks/evict_cache.py 0.85 > /dev/null
  $PY -u benchmarks/bench.py --model "$M" --cache "$C" --pin $PIN --cold-mb $COLD --tokens $TOKENS \
      --prompts coding,factual,reasoning --quiet --label "${regime}_${label}" --out "$OUT" "$@" \
      2>&1 | grep -E "mean |Traceback|error:"
}

for regime in m36 m16; do
  run $regime base
  run $regime la2x2          --lookahead 2 --extra 2
  run $regime bias0.5        --bias 0.5
  run $regime la2x2_bias0.5  --lookahead 2 --extra 2 --bias 0.5
done
