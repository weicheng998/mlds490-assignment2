"""Shared pieces for DQN on CartPole-v1 and ALE/MsPacman-v5.

  * get_device        CUDA is required; --allow-cpu only for smoke tests / CPU validation scripts.
  * preprocess_observation   the handout function, unchanged.
  * EnvSlot / VecEnv  N environments stepped in lock-step (worker processes for Atari).
                      Every episode gets its own explicit seed; evaluation seeds never overlap training seeds.
  * QMLP / QCNN       the two Q-networks.
"""
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn

# Seed ranges. Training episode seeds are < 1e8, so the three sets are disjoint.
TEST_SEED0 = 1_000_000_000   # final 500-episode evaluation
VAL_SEED0 = 2_000_000_000    # periodic evaluation during training / checkpoint selection
HELD_SEED0 = 3_000_000_000   # random-policy episodes that produce the fixed held-out states


# ---------------------------------------------------------------- device ----
def get_device(allow_cpu=False):
    """CUDA is required. `allow_cpu` exists only for local smoke tests and CPU validation."""
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"[warn] {torch.cuda.device_count()} GPUs visible; using cuda:0 only. "
                  "Set CUDA_VISIBLE_DEVICES to a single card.")
        return torch.device("cuda:0")
    if allow_cpu:
        print("[warn] CUDA unavailable; running on CPU because --allow-cpu was given.")
        return torch.device("cpu")
    raise RuntimeError("CUDA is not available. Training/evaluation requires a GPU "
                       "(use --allow-cpu only for smoke tests and CPU validation scripts).")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------ preprocess ----
mspacman_color = 210 + 164 + 74


def preprocess_observation(obs):
    """Handout preprocessing: 210x160x3 uint8 -> 88x80x1 int8 in [-128, 127]."""
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to greyscale
    img[img == mspacman_color] = 0  # Improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    return img.reshape(88, 80, 1)


# ------------------------------------------------------------------ envs ----
ENV_SPECS = {
    # obs_shape is the shape of ONE stored frame; `stack` frames form the network input.
    "cartpole": dict(obs_shape=(4,), obs_dtype=np.float32, n_actions=2, stack=1),
    "mspacman": dict(obs_shape=(88, 80), obs_dtype=np.int8, n_actions=9, stack=4),
}


def make_env(env_name, **kw):
    """env_name: 'cartpole' | 'mspacman' | a zero-argument callable returning a gymnasium-style env."""
    if callable(env_name):
        return env_name()
    import gymnasium as gym
    if env_name == "cartpole":
        return gym.make("CartPole-v1", **kw)
    if env_name == "mspacman":
        import ale_py
        gym.register_envs(ale_py)
        # v5 defaults: frameskip=4, repeat_action_probability=0.25 (sticky actions),
        # full_action_space=False (9 actions), truncation after 108,000 emulator frames.
        env = gym.make("ALE/MsPacman-v5", **kw)
        assert env.action_space.n == 9
        return env
    raise ValueError(env_name)


class EnvSlot:
    """One environment with per-episode seeding, preprocessing, life tracking and auto-reset.

    seeds: ("count", base) -> base, base+1, ... forever (training)
           ("list", [...]) -> exactly these episodes, then the slot goes idle (evaluation)
    """

    def __init__(self, env_name, seeds, env_kwargs=None):
        self.env = make_env(env_name, **(env_kwargs or {}))
        self.atari = env_name == "mspacman"
        self.set_seeds(seeds)
        self.lives = 0
        self.idle = False

    def set_seeds(self, seeds):
        kind, val = seeds
        self.seed_iter = itertools.count(val) if kind == "count" else iter(list(val))
        self.idle = False

    def _proc(self, obs):
        if self.atari:
            return preprocess_observation(obs).reshape(88, 80)
        return np.asarray(obs, dtype=np.float32)

    def reset(self):
        seed = next(self.seed_iter, None)
        if seed is None:
            self.idle = True
            return None
        obs, info = self.env.reset(seed=int(seed))
        self.lives = info.get("lives", 0)
        return self._proc(obs)

    def step(self, action):
        """Returns (frame, reward, terminated, truncated, life_lost, next_first_frame).

        frame is always the observation that FOLLOWS the action (the true s', also at the
        end of an episode). next_first_frame is the first frame of the next episode when
        this one ended (None otherwise, or None if the seed list is exhausted)."""
        if self.idle:
            return None
        obs, r, term, trunc, info = self.env.step(int(action))
        lives = info.get("lives", 0)
        life_lost = lives < self.lives
        self.lives = lives
        nxt = self.reset() if (term or trunc) else None
        return self._proc(obs), float(r), bool(term), bool(trunc), bool(life_lost), nxt


def _worker(conn, env_name, seeds, env_kwargs):
    torch.set_num_threads(1)
    slot = EnvSlot(env_name, seeds, env_kwargs)
    while True:
        cmd, arg = conn.recv()
        if cmd == "step":
            conn.send(slot.step(arg))
        elif cmd == "reset":
            if arg is not None:
                slot.set_seeds(arg)
            conn.send(slot.reset())
        elif cmd == "close":
            conn.close()
            return


class VecEnv:
    """N EnvSlots stepped together. workers=True puts each one in its own process."""

    def __init__(self, env_name, seeds_per_env, workers=False, env_kwargs=None):
        self.n = len(seeds_per_env)
        self.workers = workers
        if workers:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self.conns, self.procs = [], []
            for s in seeds_per_env:
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, env_name, s, env_kwargs), daemon=True)
                p.start()
                child.close()
                self.conns.append(parent)
                self.procs.append(p)
        else:
            self.slots = [EnvSlot(env_name, s, env_kwargs) for s in seeds_per_env]

    def reset(self, seeds_per_env=None):
        sp = seeds_per_env or [None] * self.n
        if self.workers:
            for c, s in zip(self.conns, sp):
                c.send(("reset", s))
            return [c.recv() for c in self.conns]
        out = []
        for slot, s in zip(self.slots, sp):
            if s is not None:
                slot.set_seeds(s)
            out.append(slot.reset())
        return out

    def step(self, actions):
        if self.workers:
            for c, a in zip(self.conns, actions):
                c.send(("step", a))
            return [c.recv() for c in self.conns]
        return [slot.step(a) for slot, a in zip(self.slots, actions)]

    def close(self):
        if self.workers:
            for c in self.conns:
                try:
                    c.send(("close", None))
                except Exception:
                    pass
            for p in self.procs:
                p.join(timeout=5)
        else:
            for s in self.slots:
                s.env.close()


class FrameStacker:
    """Keeps the current stacked state of N envs: (N, stack, *obs_shape).

    At reset the first frame is repeated `stack` times, so no frame from the previous
    episode is ever part of a state. The replay buffer rebuilds stacks with the same rule."""

    def __init__(self, n, stack, obs_shape, dtype):
        self.s = np.zeros((n, stack) + tuple(obs_shape), dtype=dtype)

    def reset(self, i, frame):
        self.s[i] = frame

    def push(self, i, frame):
        self.s[i, :-1] = self.s[i, 1:]
        self.s[i, -1] = frame


# -------------------------------------------------------------- networks ----
class QMLP(nn.Module):
    """CartPole Q-network: stack*obs_dim -> 128 -> 128 -> n_actions.

    dueling=True splits the last layer into V(s) and A(s,a): Q = V + A - mean_a A."""

    def __init__(self, in_dim, n_actions, hidden=128, dueling=False):
        super().__init__()
        self.dueling = dueling
        out = n_actions + 1 if dueling else n_actions
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out))

    def forward(self, x):
        y = self.net(x.float().flatten(1))
        if self.dueling:
            v, adv = y[:, :1], y[:, 1:]
            return v + adv - adv.mean(1, keepdim=True)
        return y


class QCNN(nn.Module):
    """MsPacman Q-network (Mnih et al. 2015 layout) on 4 stacked 88x80 int8 frames."""

    def __init__(self, stack, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(stack, 32, 8, stride=4), nn.ReLU(),   # 88x80 -> 21x19
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),      # -> 9x8
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),      # -> 7x6
            nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64 * 7 * 6, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        # frames are stored as int8 in [-128, 127]; scale to [-1, 1) here
        return self.head(self.conv(x.float() / 128.0))


def build_qnet(env_name, spec=None, dueling=False):
    spec = spec or ENV_SPECS[env_name]
    if len(spec["obs_shape"]) == 2:
        return QCNN(spec["stack"], spec["n_actions"])
    return QMLP(spec["stack"] * int(np.prod(spec["obs_shape"])), spec["n_actions"],
                hidden=spec.get("hidden", 128), dueling=dueling)
