"""Multi-seed CartPole experiments: component ablations and the gamma sweep.

Every variant is the default CartPole configuration with exactly one thing changed, run for
several seeds through the normal training script (so the same code path as the main runs).

  python -m dqn.ablate --group ablation --seeds 5 --jobs 10            # on deepdish (one GPU)
  python -m dqn.ablate --group gamma --seeds 3 --jobs 8 --allow-cpu    # or on CPU

Logs go to results/ablation/, the table and figure are produced by `python -m dqn.plot`.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

GROUPS = {
    "ablation": {
        "full": [],
        "no_target_net": ["--use-target", "0"],                  # target computed with the online net
        "no_replay": ["--buffer", "129", "--learning-starts", "129"],   # buffer holds only one batch
        "no_terminal_mask": ["--mask-terminal", "0"],            # bootstrap even after the pole fell
        "trunc_as_terminal": ["--trunc-as-terminal", "1"],       # treat the 500-step limit as failure
        "mse_loss": ["--loss", "mse"],
        "vanilla_dqn": ["--double", "0"],                        # max over target net instead of Double DQN
    },
    # Follow-up experiments on the CartPole instability (see report).
    "extra": {
        "target_clip_20": ["--target-max", "20"],                # clip targets at the true ceiling 1/(1-gamma)
        "polyak_tau005": ["--tau", "0.005"],                     # soft target updates
        "dueling": ["--dueling", "1"],
    },
    "gamma": {f"gamma_{g}": ["--gamma", str(g)] for g in (0.5, 0.8, 0.9, 0.95, 0.99)},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=list(GROUPS), required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--steps", type=int, default=None, help="override total steps (default: main-run setting)")
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.makedirs("logs", exist_ok=True)
    jobs = []
    for name, extra in GROUPS[a.group].items():
        for s in range(a.seeds):
            cmd = [sys.executable, "-m", "dqn.train", "--env", "cartpole", "--seed", str(s), "--run-name",
                   f"{name}_s{s}", "--results-dir", "results/ablation", "--ckpt-dir", "checkpoints/ablation",
                   "--save-buffer", "0", "--eval-episodes", "30"] + extra
            if a.steps:
                cmd += ["--total-steps", str(a.steps)]
            if a.allow_cpu:
                cmd.append("--allow-cpu")
            jobs.append((f"{name}_s{s}", cmd))

    def run(job):
        name, cmd = job
        with open(f"logs/ablate_{name}.log", "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f"{name}: exit {rc}", flush=True)
        return rc

    with ThreadPoolExecutor(a.jobs) as ex:
        rcs = list(ex.map(run, jobs))
    print(f"ABLATE {a.group} DONE, failures: {sum(r != 0 for r in rcs)}")


if __name__ == "__main__":
    main()
