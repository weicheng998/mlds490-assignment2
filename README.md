# MLDS 490 — Assignment 2: Deep Q-Network

Author: Weicheng (GitHub `weicheng998`) · Repository: https://github.com/weicheng998/mlds490-assignment2

DQN with a neural network, a replay buffer and a target network on

| Game | Env ID | γ | Network | Code |
|---|---|---|---|---|
| CartPole | `CartPole-v1` | 0.95 | MLP 4-128-128-2 | `dqn/` |
| Ms. Pac-Man | `ALE/MsPacman-v5` (9 actions) | 0.99 | CNN on 4 stacked 88×80 frames (handout `preprocess_observation`) | `dqn/` |

The report (`report/report.pdf`) contains all figures, edge cases, validation results and discussion.

## Setup

```bash
conda activate myenv            # deepdish; PyTorch with CUDA (cu126)
pip install -r requirements.txt # gymnasium[atari]==1.0.0, ale-py==0.10.1, numpy, scipy, matplotlib
```

Training and evaluation **require CUDA** and raise an error at startup otherwise. `--allow-cpu` exists only for
local smoke tests and the CPU-only validation scripts. Every run uses one GPU, selected with `CUDA_VISIBLE_DEVICES`.

## Training and evaluation

```bash
mkdir -p logs
# CartPole-v1: 5 seeds on GPU 0, then 500-episode test rollouts of best.pt and final.pt + random baseline
nohup bash scripts/run_cartpole.sh 0 > logs/cartpole.log 2>&1 &

# MsPacman-v5: one run per GPU.   args: <gpu> <seed> <double 0|1> [time limit hours] [agent steps]
nohup bash scripts/run_mspacman.sh 1 0 1 > logs/mspacman_s0.log     2>&1 &   # Double DQN, seed 0 (main run)
nohup bash scripts/run_mspacman.sh 2 1 1 > logs/mspacman_s1.log     2>&1 &   # Double DQN, seed 1
nohup bash scripts/run_mspacman.sh 3 0 0 > logs/mspacman_s0_dqn.log 2>&1 &   # vanilla DQN target, seed 0
# after training: 500 test episodes each, with Q traces for the calibration plots
bash scripts/eval_mspacman.sh 1

python -m dqn.plot       # every figure -> results/figures/, every reported number -> results/summary.json
bash report/build.sh     # report/report.md + summary.json -> report/report.pdf (pandoc + xelatex)
```

Single runs: `python -m dqn.train --env {cartpole,mspacman} --seed 0 [--double 0|1 --loss huber|mse --total-steps N ...]`
and `python -m dqn.evaluate --env mspacman --run mspacman_s0 --which {best,final} [--record 100] [--random]`.

* **Resume.** Checkpoints (`checkpoints/<run>/ckpt.pt` + `buffer.npz`) are written atomically (temp file, then rename).
  Re-running the same command resumes; the CSV logs are truncated to the checkpoint so they have no duplicate rows.
* **Logs** are appended while training runs (`results/<env>/<run>.csv`, `_heldout.csv`, `_eval.csv`), so
  `python -m dqn.plot` gives valid plots at any time.
* **Seeds.** Training episodes use seeds below 1e8, periodic validation uses 2e9+i, the final 500-episode test uses
  1e9+i. `best.pt` is chosen on the validation seeds only.

## Design in brief

* **Target.** `y = r + γ·(1 − cut)·Q_target(s′, argmax_a Q_online(s′, a))` (Double DQN; `--double 0` gives the
  vanilla max). Loss: Huber. Optimizer: Adam with gradient-norm clipping at 10. Hard target sync.
* **Terminal vs truncation.** `cut = 1` only for a real termination (pole fell / cart left the track; in MsPacman a
  lost life or game over). A time-limit truncation (CartPole's 500 steps, ALE's 108,000 frames) keeps `cut = 0`, and
  the true next observation is stored, so the target still bootstraps.
* **Replay buffer.** Stores single frames and rebuilds 4-frame stacks at sampling time; a stack never contains a frame
  from another episode; sampling is uniform over all complete transitions.
* **MsPacman rewards.** The agent learns from `sign(r)` (clipped reward). All reported scores and reward plots are
  the raw game score. Q-values for MsPacman are therefore in clipped-reward units.
* **"Max Q".** Logged per training episode as the maximum over the states visited in that episode of
  `max_a Q(s,a)`, and separately as the mean of `max_a Q` over a fixed set of held-out states.

## Validation

```bash
python -m dqn.validate --allow-cpu                      # trace, replay, overfit, resume, shuttle (exact Q*), controls  (~6 min CPU)
python -m dqn.validate --allow-cpu --tests pacman_env   # MsPacman preprocessing, lives, rewards, determinism
python -m dqn.validate --tests sticky --run mspacman_s0 # needs a trained checkpoint
python -m dqn.ablate --group ablation --seeds 5 --jobs 9 --allow-cpu   # 7 variants x 5 seeds on CartPole
python -m dqn.ablate --group gamma    --seeds 3 --jobs 9 --allow-cpu   # gamma in {0.5, 0.8, 0.9, 0.95, 0.99}
```

Each test prints its prediction and the measured numbers to `results/validation/<test>.txt`. What each test does and
why it is evidence of correctness is described in the report.

## Layout

```
dqn/common.py       device check, handout preprocessing, seeded vector env, frame stacker, Q-networks
dqn/replay.py       frame-stacking uniform replay buffer
dqn/agent.py        DQN agent: TD target, terminal mask, gradient step, target sync
dqn/train.py        training loop, schedules, logging, periodic validation, checkpoint / resume
dqn/evaluate.py     500-episode test rollouts, random baseline, Q traces
dqn/validate.py     validation experiments      dqn/ablate.py   multi-seed ablations and gamma sweep
dqn/toy.py          tabular MDP environment with exact Q*   dqn/shuttle_mdp.py  value iteration (from Assignment 1)
dqn/plot.py         figures + results/summary.json
scripts/            deepdish launch scripts     results/  logs (CSV/JSON/NPY) and figures     report/  report source + PDF
viz.py              shared plot style
```
