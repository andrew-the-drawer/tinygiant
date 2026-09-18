#!/bin/bash
# Interleaved, repeated runs of the key configurations so that run-to-run
# compute variance (background scanner) is spread evenly across configs.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=${OUT:-benchmarks/results_controlled.jsonl}
TOKENS=${TOKENS:-48}
REPS=${REPS:-2}

run() {
  local regime=$1; shift
  local label=$1; shift
  if [ "$regime" = r16 ]; then PIN=16; COLD=512; else PIN=56; COLD=2048; fi
  $PY -u benchmarks/bench.py --pin $PIN --cold-mb $COLD --tokens $TOKENS --quiet \
      --label "${regime}_${label}" --out "$OUT" "$@" 2>&1 | grep -E "mean |Traceback|error:"
}

for rep in $(seq $REPS); do
  for regime in r16 r36; do
    run $regime base
    run $regime la1            --lookahead 1
    run $regime la2x2          --lookahead 2 --extra 2
    run $regime la4x2          --lookahead 4 --extra 2
    run $regime bias0.5        --bias 0.5
    run $regime bias1          --bias 1.0
    run $regime la2x2_bias0.5  --lookahead 2 --extra 2 --bias 0.5
    run $regime la2x2_bias1    --lookahead 2 --extra 2 --bias 1.0
  done
done
# compute-thread check under the same conditions
for t in 6 12 18; do
  run r36 threads$t --threads $t
done
