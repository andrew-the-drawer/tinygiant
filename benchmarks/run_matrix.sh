#!/bin/bash
# Benchmark matrix over two memory regimes:
#   r16: ~13% of experts pinned (16/128) + 512 MB cold buffer  (a 100B model on a 16 GB Mac)
#   r36: ~44% pinned (56/128) + 2 GB cold buffer                (a 100B model on a 36 GB Mac)
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=${OUT:-benchmarks/results_matrix.jsonl}
TOKENS=${TOKENS:-48}
PRIOR=experiments/token_prior.npz

run() {
  local regime=$1; shift
  local label=$1; shift
  if [ "$regime" = r16 ]; then PIN=16; COLD=512; else PIN=56; COLD=2048; fi
  $PY -u benchmarks/bench.py --pin $PIN --cold-mb $COLD --tokens $TOKENS --quiet \
      --label "${regime}_${label}" --out "$OUT" "$@" 2>&1 | grep -E "^\["
}

for regime in r16 r36; do
  run $regime base
  run $regime base_io4      --io-threads 4
  run $regime base_io16     --io-threads 16
  run $regime la1           --lookahead 1
  run $regime la2           --lookahead 2
  run $regime la2x2         --lookahead 2 --extra 2
  run $regime la4           --lookahead 4
  run $regime la4x4         --lookahead 4 --extra 4
  run $regime la8x4         --lookahead 8 --extra 4
  [ -f $PRIOR ] && run $regime prior     --token-prior $PRIOR
  [ -f $PRIOR ] && run $regime la4_prior --lookahead 4 --extra 2 --token-prior $PRIOR
  run $regime bias0.5       --bias 0.5
  run $regime bias1         --bias 1.0
  run $regime bias1_tau0.1  --bias 1.0 --tau 0.1
  run $regime bias2_tau0.1  --bias 2.0 --tau 0.1
  run $regime spec4         --spec 4
  run $regime spec6         --spec 6
  run $regime spec4_nodp    --spec 4 --no-draft-prefetch
  run $regime spec4_la4     --spec 4 --lookahead 4 --extra 2
  [ -f $PRIOR ] && run $regime spec4_la4_prior --spec 4 --lookahead 4 --extra 2 --token-prior $PRIOR
done
