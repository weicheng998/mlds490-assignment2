#!/usr/bin/env bash
# 500-episode test rollouts for the three MsPacman runs on ONE GPU, plus the random-policy baseline
# and the sticky-action determinism test.   usage: eval_mspacman.sh <gpu>
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${1:-0}
PY=${PY:-python}
$PY -m dqn.evaluate --env mspacman --run mspacman --random
for r in mspacman_s0 mspacman_s1 mspacman_s0_dqn; do
  [ -f checkpoints/$r/best.pt ] || continue
  $PY -m dqn.evaluate --env mspacman --run $r --which best --record 100
  $PY -m dqn.evaluate --env mspacman --run $r --which final --record 100
done
$PY -m dqn.validate --tests sticky --run mspacman_s0
echo "MSPACMAN EVAL DONE"
