#!/usr/bin/env bash
# CartPole-v1: 5 seeds of DQN on ONE GPU, then 500-episode test rollouts of best.pt and final.pt
# (with Q traces for the calibration plot) and the random-policy baseline.
# usage: run_cartpole.sh <gpu>     Resumable: finished training runs are skipped by the resume logic.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${1:-0} OMP_NUM_THREADS=1
PY=${PY:-python}
for s in 0 1 2 3 4; do
  $PY -m dqn.train --env cartpole --seed $s > logs/cartpole_s$s.log 2>&1 &
done
wait
for s in 0 1 2 3 4; do
  $PY -m dqn.evaluate --env cartpole --run cartpole_s$s --which best --record 100
  $PY -m dqn.evaluate --env cartpole --run cartpole_s$s --which final --record 100
done
$PY -m dqn.evaluate --env cartpole --run cartpole --random
echo "CARTPOLE ALL DONE"
