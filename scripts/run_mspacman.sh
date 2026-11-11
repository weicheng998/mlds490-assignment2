#!/usr/bin/env bash
# ALE/MsPacman-v5, one DQN run on ONE GPU.  usage: run_mspacman.sh <gpu> <seed> <double 0|1> [hours] [steps]
# Resumes from checkpoints/<run>/ if restarted with the same arguments.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:?gpu index}; SEED=${2:?seed}; DOUBLE=${3:?0|1}; H=${4:-17.5}; STEPS=${5:-15000000}
export CUDA_VISIBLE_DEVICES=$GPU
PY=${PY:-python}
TAG=""; [ "$DOUBLE" = "0" ] && TAG="--tag dqn"
$PY -m dqn.train --env mspacman --seed "$SEED" --double "$DOUBLE" $TAG \
    --total-steps "$STEPS" --time-limit-hours "$H"
echo "MSPACMAN seed=$SEED double=$DOUBLE DONE"
