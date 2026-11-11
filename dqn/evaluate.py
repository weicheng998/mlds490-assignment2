"""Final evaluation: roll out N (default 500) greedy episodes with a trained Q-network.

Test seeds are 1e9 + i. They are disjoint from the training episode seeds (< 1e8) and from
the validation seeds (2e9 + i) used during training to pick best.pt, so nothing leaks.

  python -m dqn.evaluate --env cartpole --run cartpole_s0 --which best
  python -m dqn.evaluate --env mspacman --run mspacman_s0 --which final --record 100
  python -m dqn.evaluate --env mspacman --run mspacman_s0 --random        # random-policy baseline

Outputs in results/<env>/:  <run>_test_<which>.npy (raw scores), .json (statistics) and,
with --record K, <run>_trace_<which>.npz (per-step max_a Q, training reward, cut flag for the
first K episodes; used by the calibration plot).
"""
import argparse
import json
import os

import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.common import ENV_SPECS, TEST_SEED0, build_qnet, get_device
from dqn.train import rollout


def load_agent(env, run, which, ckpt_dir, device):
    with open(os.path.join(ckpt_dir, run, "config.json")) as f:
        cfg = json.load(f)
    spec = ENV_SPECS[env]
    agent = DQNAgent(build_qnet(env, dueling=bool(cfg.get("dueling", 0))), spec["n_actions"], cfg["gamma"], cfg["lr"], device)
    agent.online.load_state_dict(torch.load(os.path.join(ckpt_dir, run, f"{which}.pt"), map_location=device))
    agent.online.eval()
    return agent, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "mspacman"], required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--which", choices=["best", "final"], default="best")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--eps", type=float, default=0.0, help="evaluation epsilon (0 = greedy)")
    p.add_argument("--record", type=int, default=0, help="save per-step Q/reward traces of the first K episodes")
    p.add_argument("--random", action="store_true", help="uniform-random policy baseline (no network)")
    p.add_argument("--sticky", type=float, default=None, help="MsPacman only: override repeat_action_probability")
    p.add_argument("--seed0", type=int, default=TEST_SEED0)
    p.add_argument("--suffix", default="")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    device = get_device(a.allow_cpu)
    res_dir = a.results_dir or f"results/{a.env}"
    os.makedirs(res_dir, exist_ok=True)
    spec = ENV_SPECS[a.env]
    atari = a.env == "mspacman"
    kw = {"repeat_action_probability": a.sticky} if (atari and a.sticky is not None) else None
    n_envs = a.n_envs or (16 if atari else 10)
    seeds = [a.seed0 + i for i in range(a.n)]

    if a.random:
        agent, cfg, which = None, {"life_cut": int(atari), "reward_mode": "clip" if atari else "raw"}, "random"
    else:
        agent, cfg = load_agent(a.env, a.run, a.which, a.ckpt_dir, device)
        which = a.which
    res = rollout(agent, a.env, seeds, spec, n_envs=n_envs, workers=atari, eps=a.eps, record=a.record,
                  life_cut=bool(cfg["life_cut"]), reward_mode=cfg["reward_mode"],
                  reward_scale=cfg.get("reward_scale", 1.0), env_kwargs=kw, random_policy=a.random,
                  rng=np.random.default_rng(a.seed0))
    R, L = res["returns"], res["lengths"]
    tag = f"{a.run}_test_{which}{a.suffix}"
    np.save(os.path.join(res_dir, tag + ".npy"), R)
    stats = {"run": a.run, "which": which, "n": int(a.n), "eps": a.eps, "seed0": a.seed0, "sticky": a.sticky,
             "mean": float(R.mean()), "std": float(R.std(ddof=1)), "median": float(np.median(R)),
             "min": float(R.min()), "max": float(R.max()), "mean_len": float(L.mean()),
             "n_truncated": int(res["truncated"].sum())}
    if a.env == "cartpole":
        stats["frac_500"] = float((R >= 500).mean())
        # Why did the failed episodes end? CartPole terminates when |x| > 2.4 or |theta| > 12 degrees.
        fo = np.array([o for o, t in zip(res["last_obs"], res["truncated"]) if not t]).reshape(-1, 4)
        stats["n_failed"] = int(len(fo))
        stats["failed_by_cart_position"] = int((np.abs(fo[:, 0]) > 2.4).sum())
        stats["failed_by_pole_angle"] = int((np.abs(fo[:, 2]) > 12 * np.pi / 180).sum())
    with open(os.path.join(res_dir, tag + ".json"), "w") as f:
        json.dump(stats, f, indent=1)
    if a.record:
        tr = res["traces"]
        np.savez_compressed(
            os.path.join(res_dir, f"{a.run}_trace_{which}{a.suffix}.npz"),
            q=np.concatenate([tr[k]["q"] for k in tr]), r=np.concatenate([tr[k]["r"] for k in tr]),
            cut=np.concatenate([tr[k]["cut"] for k in tr]), a=np.concatenate([tr[k]["a"] for k in tr]),
            ep=np.concatenate([np.full(len(tr[k]["q"]), k) for k in tr]),
            truncated=res["truncated"][:len(tr)], gamma=cfg["gamma"])
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
