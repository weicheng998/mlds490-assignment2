#!/usr/bin/env bash
# Waits until the three MsPacman training runs have finished, then runs the test rollouts.  usage: after_mspacman.sh <gpu>
cd "$(dirname "$0")/.."
until [ "$(cat logs/mspacman_s0.log logs/mspacman_s1.log logs/mspacman_s0_dqn.log | grep -c 'MSPACMAN seed=.* DONE')" -ge 3 ]; do sleep 120; done
bash scripts/eval_mspacman.sh "${1:-0}"
