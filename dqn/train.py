"""DQN training (replay buffer + target network) for CartPole-v1 and ALE/MsPacman-v5.

  python -m dqn.train --env cartpole --seed 0
  CUDA_VISIBLE_DEVICES=1 python -m dqn.train --env mspacman --seed 0 --time-limit-hours 16

Logs (appended as training runs, so plots can be made at any time):
  results/<env>/<run>.csv           one row per finished training episode
  results/<env>/<run>_heldout.csv   mean max_a Q on a fixed set of held-out states
  results/<env>/<run>_eval.csv      periodic greedy evaluation on validation seeds
Checkpoints (atomic) in checkpoints/<run>/: ckpt.pt + buffer.npz (resume), best.pt, final.pt.
Re-running the same command resumes from the last checkpoint.
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.common import (ENV_SPECS, HELD_SEED0, VAL_SEED0, FrameStacker, VecEnv,
                        atomic_torch_save, build_qnet, get_device, seed_everything)
from dqn.replay import ReplayBuffer

DEFAULTS = {
    "cartpole": dict(
        gamma=0.95, lr=1e-3, lr_end=1e-5, batch=128, buffer=200_000, n_envs=1, train_freq=1,
        target_sync=500, learning_starts=1_000, total_steps=200_000,
        eps_start=1.0, eps_mid=0.05, eps_mid_step=10_000, eps_end=0.01, eps_end_step=50_000,
        reward_mode="raw", life_cut=0, adam_eps=1e-8, grad_clip=10.0,
        eval_every=5_000, eval_episodes=100, eval_envs=10, heldout_every=500, heldout_n=1_000,
        ckpt_every_min=5.0),
    "mspacman": dict(
        gamma=0.99, lr=1e-4, lr_end=2.5e-5, batch=32, buffer=500_000, n_envs=8, train_freq=4,
        target_sync=2_500, learning_starts=50_000, total_steps=10_000_000,
        eps_start=1.0, eps_mid=0.1, eps_mid_step=1_000_000, eps_end=0.01, eps_end_step=4_000_000,
        reward_mode="clip", life_cut=1, adam_eps=1e-5, grad_clip=10.0,
        eval_every=250_000, eval_episodes=30, eval_envs=10, heldout_every=10_000, heldout_n=2_048,
        ckpt_every_min=30.0),
}

EP_COLS = ["episode", "env", "step", "updates", "return", "length", "max_q", "mean_q",
           "heldout_q", "epsilon", "lr", "loss", "truncated", "wall_s"]
HELD_COLS = ["step", "updates", "episode", "heldout_q"]
EVAL_COLS = ["step", "updates", "episode", "mean", "std", "min", "max", "n", "is_best"]


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "mspacman"], required=True)
    p.add_argument("--seed", type=int, default=0)
    for k, v in DEFAULTS["cartpole"].items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v))
    p.add_argument("--double", type=int, default=1, help="1 = Double DQN target")
    p.add_argument("--dueling", type=int, default=0, help="1 = dueling head (MLP only)")
    p.add_argument("--loss", choices=["huber", "mse"], default="huber")
    p.add_argument("--use-target", type=int, default=1, help="0 = no target network (ablation)")
    p.add_argument("--mask-terminal", type=int, default=1, help="0 = bootstrap through termination (ablation)")
    p.add_argument("--trunc-as-terminal", type=int, default=0, help="1 = treat time-limit truncation as terminal (ablation)")
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--target-max", type=float, default=None, help="clip TD targets at this known return ceiling (experiment)")
    p.add_argument("--tau", type=float, default=0.0, help=">0: Polyak target update every step instead of hard sync")
    p.add_argument("--time-limit-hours", type=float, default=1e9)
    p.add_argument("--tag", default="", help="suffix for the run name")
    p.add_argument("--run-name", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--save-buffer", type=int, default=1, help="0 = do not checkpoint the replay buffer")
    p.add_argument("--allow-cpu", action="store_true", help="smoke tests / CPU validation only")
    return p


def finalize_args(a):
    for k, v in DEFAULTS[a.env].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    if a.run_name is None:
        a.run_name = f"{a.env}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    if a.results_dir is None:
        a.results_dir = f"results/{a.env}"
    return a


# ------------------------------------------------------------- schedules ----
def epsilon_at(a, step):
    """Piecewise linear: eps_start -> eps_mid at eps_mid_step -> eps_end at eps_end_step."""
    if step < a.eps_mid_step:
        return a.eps_start + (a.eps_mid - a.eps_start) * step / max(1, a.eps_mid_step)
    if step < a.eps_end_step:
        f = (step - a.eps_mid_step) / max(1, a.eps_end_step - a.eps_mid_step)
        return a.eps_mid + (a.eps_end - a.eps_mid) * f
    return a.eps_end


def lr_at(a, step):
    """Linear decay from lr to lr_end over total_steps."""
    return a.lr + (a.lr_end - a.lr) * min(1.0, step / max(1, a.total_steps))


def shape_reward(r, mode, scale=1.0):
    """Reward the agent learns from. 'clip' = sign(r) in {-1,0,1}; 'raw' = r * scale."""
    return float(np.sign(r)) if mode == "clip" else r * scale


# ------------------------------------------------------- rollouts / eval ----
def rollout(agent, env_name, seeds, spec, n_envs=10, workers=False, eps=0.0, record=0,
            life_cut=False, reward_mode="raw", reward_scale=1.0, env_kwargs=None,
            random_policy=False, vec=None, rng=None):
    """Play one episode per seed. No learning, nothing is written to the replay buffer.

    Each seed is bound to a fixed slot before the rollout starts, so the set of episodes
    does not depend on which ones happen to finish first. For the first `record` seeds the
    per-step max_a Q, shaped reward and cut flag are kept (for the calibration plot)."""
    rng = rng or np.random.default_rng(12345)
    n = min(n_envs, len(seeds))
    per_slot = [("list", list(seeds[j::n])) for j in range(n)]
    own = vec is None
    if own:
        vec = VecEnv(env_name, per_slot, workers=workers, env_kwargs=env_kwargs)
        frames = vec.reset()
    else:
        frames = vec.reset(per_slot + [("list", [])] * (vec.n - n))[:n]
    st = FrameStacker(n, spec["stack"], spec["obs_shape"], spec["obs_dtype"])
    for i, f in enumerate(frames):
        st.reset(i, f)
    seed_idx = [list(range(j, len(seeds), n)) for j in range(n)]   # which seeds each slot plays
    pos = [0] * n
    active = [True] * n
    R = np.zeros(len(seeds)); L = np.zeros(len(seeds), dtype=np.int64)
    TR = np.zeros(len(seeds), dtype=bool)
    last_obs = [None] * len(seeds)                                 # final observation of each episode
    traces = {k: {"q": [], "r": [], "cut": [], "a": []} for k in range(min(record, len(seeds)))}
    A = spec["n_actions"]
    while any(active):
        if random_policy:
            qmax = np.zeros(n); acts = rng.integers(A, size=n)
        else:
            q = agent.q_values(st.s)
            qmax = q.max(1).values.cpu().numpy(); acts = q.argmax(1).cpu().numpy()
            if eps > 0:
                rnd = rng.random(n) < eps
                acts = np.where(rnd, rng.integers(A, size=n), acts)
        full = list(acts) + [0] * (vec.n - n)
        out = vec.step(full)
        for i in range(n):
            if not active[i]:
                continue
            frame, r, term, trunc, life_lost, nxt = out[i]
            k = seed_idx[i][pos[i]]
            R[k] += r; L[k] += 1
            if k in traces:
                tr = traces[k]
                tr["q"].append(float(qmax[i])); tr["a"].append(int(acts[i]))
                tr["r"].append(shape_reward(r, reward_mode, reward_scale))
                tr["cut"].append(bool(term or (life_cut and life_lost)))
            if term or trunc:
                TR[k] = trunc and not term
                last_obs[k] = frame if frame.ndim == 1 else None
                pos[i] += 1
                if nxt is None:
                    active[i] = False
                else:
                    st.reset(i, nxt)
            else:
                st.push(i, frame)
    if own:
        vec.close()
    return {"returns": R, "lengths": L, "truncated": TR, "traces": traces, "last_obs": last_obs}


def collect_heldout(env_name, spec, n, env_kwargs=None, n_episodes=None):
    """Fixed set of states from random-policy episodes (fixed seeds, never used for training)."""
    n_episodes = n_episodes or (150 if spec["stack"] == 1 else 12)
    rng = np.random.default_rng(HELD_SEED0)
    vec = VecEnv(env_name, [("list", [HELD_SEED0 + k for k in range(n_episodes)])], env_kwargs=env_kwargs)
    st = FrameStacker(1, spec["stack"], spec["obs_shape"], spec["obs_dtype"])
    st.reset(0, vec.reset()[0])
    states = []
    while True:
        states.append(st.s[0].copy())
        frame, r, term, trunc, _, nxt = vec.step([rng.integers(spec["n_actions"])])[0]
        if term or trunc:
            if nxt is None:
                break
            st.reset(0, nxt)
        else:
            st.push(0, frame)
    vec.close()
    states = np.stack(states)
    if len(states) > n:
        states = states[np.linspace(0, len(states) - 1, n).astype(int)]
    return states


@torch.no_grad()
def heldout_q(agent, states, chunk=1024):
    """Mean over the fixed states of max_a Q(s, a)."""
    vals = [agent.q_values(states[i:i + chunk]).max(1).values for i in range(0, len(states), chunk)]
    return float(torch.cat(vals).mean())


# ------------------------------------------------------------------ csv ----
def _open_csv(path, cols, keep=None):
    """Create the CSV, or on resume drop rows logged after the checkpoint (keep(row) -> bool)."""
    if keep is not None and os.path.exists(path):
        with open(path) as f:
            rows = list(csv.reader(f))
        rows = [rows[0]] + [r for r in rows[1:] if keep(dict(zip(rows[0], r)))]
    else:
        rows = [cols]
    with open(path + ".tmp", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    os.replace(path + ".tmp", path)


def _append(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ----------------------------------------------------------------- train ----
def train(a, env_name=None, spec=None, env_kwargs=None, quiet=False):
    """Runs training described by namespace `a`. env_name/spec override a.env (toy MDPs in
    the validation suite pass a callable environment factory). Returns a summary dict."""
    env_name = env_name or a.env
    spec = spec or ENV_SPECS[a.env]
    device = get_device(a.allow_cpu)
    atari = env_name == "mspacman"
    run_dir = os.path.join(a.ckpt_dir, a.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(a.results_dir, exist_ok=True)
    ckpt_path, buf_path = os.path.join(run_dir, "ckpt.pt"), os.path.join(run_dir, "buffer.npz")
    ep_csv = os.path.join(a.results_dir, f"{a.run_name}.csv")
    held_csv = os.path.join(a.results_dir, f"{a.run_name}_heldout.csv")
    eval_csv = os.path.join(a.results_dir, f"{a.run_name}_eval.csv")

    seed_everything(a.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    agent = DQNAgent(build_qnet(env_name, spec, bool(a.dueling)), spec["n_actions"], a.gamma, a.lr, device,
                     double=bool(a.double), loss=a.loss, use_target=bool(a.use_target),
                     mask_terminal=bool(a.mask_terminal), grad_clip=a.grad_clip, adam_eps=a.adam_eps,
                     target_max=a.target_max)
    N = a.n_envs
    buf = ReplayBuffer(N, a.buffer, spec["obs_shape"], spec["obs_dtype"], spec["stack"], seed=a.seed)
    rng = np.random.default_rng(a.seed + 777)                       # exploration randomness
    held = collect_heldout(env_name, spec, a.heldout_n, env_kwargs)

    step, episode, wall_prev, best_val, last_held = 0, 0, 0.0, -np.inf, float("nan")
    ep_count = np.zeros(N, dtype=np.int64)                          # episodes started per slot
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_state(ck["agent"])
        step, episode, wall_prev, best_val = ck["step"], ck["episode"], ck["wall"], ck["best_val"]
        ep_count = ck["ep_count"]
        rng.bit_generator.state = ck["rng"]
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        if os.path.exists(buf_path):
            buf.load(buf_path)
            buf.break_episodes()        # running episodes were lost; never sample across the gap
        _open_csv(ep_csv, EP_COLS, keep=lambda r: int(r["episode"]) < episode)
        _open_csv(held_csv, HELD_COLS, keep=lambda r: int(r["step"]) <= step)
        _open_csv(eval_csv, EVAL_COLS, keep=lambda r: int(r["step"]) <= step)
        print(f"[resume] {a.run_name}: step={step} episode={episode} buffer={len(buf)}", flush=True)
    else:
        _open_csv(ep_csv, EP_COLS); _open_csv(held_csv, HELD_COLS); _open_csv(eval_csv, EVAL_COLS)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in vars(a).items()}, f, indent=1, default=str)
    if not quiet:
        print(f"[config] {vars(a)} device={device}", flush=True)

    # Unique, reproducible seed for every training episode (all < 1e8, disjoint from eval seeds).
    seed_base = lambda i: a.seed * 1_000_003 + i * 1_000_000 + int(ep_count[i])
    vec = VecEnv(env_name, [("count", seed_base(i)) for i in range(N)], workers=atari, env_kwargs=env_kwargs)
    eval_vec = None
    if a.eval_every > 0:
        eval_vec = VecEnv(env_name, [("list", [])] * a.eval_envs, workers=atari, env_kwargs=env_kwargs)
    st = FrameStacker(N, spec["stack"], spec["obs_shape"], spec["obs_dtype"])
    for i, f in enumerate(vec.reset()):
        st.reset(i, f)
    ep_count += 1
    ep_ret = np.zeros(N); ep_len = np.zeros(N, dtype=np.int64); ep_t = np.zeros(N, dtype=np.int64)
    ep_maxq = np.full(N, -np.inf); ep_sumq = np.zeros(N)
    losses, credit, step0, healthy = [], 0.0, step, True
    t0 = time.time(); last_ckpt = t0; last_print = t0
    A = spec["n_actions"]
    val_seeds = [VAL_SEED0 + k for k in range(a.eval_episodes)]

    def save():
        if a.save_buffer:
            buf.save(buf_path)
        atomic_torch_save({"agent": agent.state(), "step": step, "episode": episode,
                           "wall": wall_prev + time.time() - t0, "best_val": best_val,
                           "ep_count": ep_count - 1,     # running episodes restart with the same seeds
                           "rng": rng.bit_generator.state, "py_rng": random.getstate(),
                           "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                           "args": vars(a)}, ckpt_path)
        atomic_torch_save(agent.online.state_dict(), os.path.join(run_dir, "final.pt"))

    def crossed(every):
        return every > 0 and step // every != (step - N) // every

    try:
        while step < a.total_steps:
            if time.time() - t0 > a.time_limit_hours * 3600:
                print("[stop] time limit reached", flush=True)
                break
            eps = epsilon_at(a, step)
            q = agent.q_values(st.s)                                    # (N, A), online network
            qmax = q.max(1).values.cpu().numpy()
            greedy = q.argmax(1).cpu().numpy()
            explore = rng.random(N) < eps                               # epsilon-greedy
            actions = np.where(explore, rng.integers(A, size=N), greedy)
            out = vec.step(actions)
            for i, (frame, r, term, trunc, life_lost, nxt) in enumerate(out):
                # cut = 1 only for a real termination (or a lost life in MsPacman when life_cut=1).
                # A time-limit truncation keeps cut = 0 so the target still bootstraps from s'.
                cut = term or (bool(a.life_cut) and life_lost) or (bool(a.trunc_as_terminal) and trunc)
                buf.add(i, st.s[i, -1], actions[i], shape_reward(r, a.reward_mode, a.reward_scale), cut, ep_t[i])
                ep_ret[i] += r; ep_len[i] += 1
                ep_maxq[i] = max(ep_maxq[i], qmax[i]); ep_sumq[i] += qmax[i]
                if term or trunc:
                    buf.add_final(i, frame, ep_t[i] + 1)                # true s' of the last transition
                    _append(ep_csv, [episode, i, step + N, agent.updates, ep_ret[i], ep_len[i],
                                     round(float(ep_maxq[i]), 5), round(ep_sumq[i] / ep_len[i], 5),
                                     round(last_held, 5), round(eps, 5), f"{lr_at(a, step):.3e}",
                                     round(float(np.mean(losses[-200:])), 6) if losses else "",
                                     int(trunc and not term), round(wall_prev + time.time() - t0, 1)])
                    episode += 1
                    ep_count[i] += 1
                    st.reset(i, nxt)
                    ep_ret[i] = 0; ep_len[i] = 0; ep_t[i] = 0; ep_maxq[i] = -np.inf; ep_sumq[i] = 0
                else:
                    st.push(i, frame)
                    ep_t[i] += 1
            step += N

            if len(buf) >= a.learning_starts:
                credit += N / a.train_freq                              # 1 update per train_freq env steps
                while credit >= 1.0:
                    credit -= 1.0
                    agent.set_lr(lr_at(a, step))
                    loss, _, _ = agent.update(buf.sample(a.batch, device))
                    if not np.isfinite(loss):
                        healthy = False                                 # keep the last good checkpoint
                        raise FloatingPointError(f"non-finite loss at step {step}")
                    losses.append(loss)
                    if a.tau > 0:
                        agent.soft_update(a.tau)                        # Polyak target update
                    elif agent.updates % a.target_sync == 0:
                        agent.sync_target()                             # hard target-network update
                del losses[:-1000]

            if crossed(a.heldout_every):
                last_held = heldout_q(agent, held)
                _append(held_csv, [step, agent.updates, episode, round(last_held, 5)])
            if crossed(a.eval_every):
                res = rollout(agent, env_name, val_seeds, spec, n_envs=a.eval_envs, vec=eval_vec)
                m = float(res["returns"].mean())
                is_best = m >= best_val                                 # ties -> keep the later one
                if is_best:
                    best_val = m
                    atomic_torch_save(agent.online.state_dict(), os.path.join(run_dir, "best.pt"))
                _append(eval_csv, [step, agent.updates, episode, round(m, 3),
                                   round(float(res["returns"].std(ddof=1)), 3), res["returns"].min(),
                                   res["returns"].max(), len(val_seeds), int(is_best)])
                if not quiet:
                    print(f"[eval] step {step} val mean {m:.1f} best {best_val:.1f}", flush=True)
            now = time.time()
            if not quiet and now - last_print > 60:
                last_print = now
                print(f"step {step:9d} ep {episode:6d} eps {eps:.3f} heldQ {last_held:.3f} "
                      f"loss {np.mean(losses[-200:]) if losses else float('nan'):.4f} "
                      f"sps {(step - step0) / (now - t0):.0f} "
                      f"t {(wall_prev + now - t0) / 3600:.2f}h", flush=True)
            if now - last_ckpt > a.ckpt_every_min * 60:
                save()
                last_ckpt = time.time()
    finally:
        if healthy:
            save()
        vec.close()
        if eval_vec is not None:
            eval_vec.close()
        if not quiet:
            print(f"[done] {a.run_name} steps={step} episodes={episode}", flush=True)
    return {"agent": agent, "step": step, "episode": episode, "best_val": best_val,
            "heldout": held, "spec": spec, "run_dir": run_dir}


def main():
    train(finalize_args(build_parser().parse_args()))


if __name__ == "__main__":
    main()
