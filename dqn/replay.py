"""Uniform replay buffer that stores single frames and rebuilds frame stacks on sampling.

Layout: one circular array per environment. Slot i of env e holds
    frame[e,i]  the newest frame of state s_t          act[e,i], rew[e,i]  a_t, r_t
    cut[e,i]    1 if the bootstrap must be cut after this transition (real termination)
    tin[e,i]    min(t, stack-1): how many earlier frames of the SAME episode exist
    valid[e,i]  False for the extra slot that only holds the last frame of an episode
The frame after the last action of an episode (terminated OR truncated) is written to an
extra slot with valid=False. Therefore the next state of slot i is always the stack ending
at slot i+1, and it is the true s' even when the episode was truncated by a time limit.

A stack never reaches back past the first frame of its episode: missing older frames are
replaced by the first frame (same rule as common.FrameStacker).
"""
import json
import os

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, n_envs, capacity, obs_shape, obs_dtype, stack, seed=0):
        self.n, self.stack = n_envs, stack
        self.cap = max(capacity // n_envs, stack + 2)       # slots per env
        self.frame = np.zeros((n_envs, self.cap) + tuple(obs_shape), dtype=obs_dtype)
        self.act = np.zeros((n_envs, self.cap), dtype=np.int64)
        self.rew = np.zeros((n_envs, self.cap), dtype=np.float32)
        self.cut = np.zeros((n_envs, self.cap), dtype=np.float32)
        self.tin = np.zeros((n_envs, self.cap), dtype=np.int64)
        self.valid = np.zeros((n_envs, self.cap), dtype=bool)
        self.head = np.zeros(n_envs, dtype=np.int64)        # next slot to write
        self.size = np.zeros(n_envs, dtype=np.int64)        # slots written (<= cap)
        self.rng = np.random.default_rng(seed)

    def _write(self, e, frame, a, r, cut, t, valid):
        i = self.head[e]
        self.frame[e, i] = frame
        self.act[e, i], self.rew[e, i], self.cut[e, i] = a, r, cut
        self.tin[e, i] = min(t, self.stack - 1)
        self.valid[e, i] = valid
        self.head[e] = (i + 1) % self.cap
        self.size[e] = min(self.size[e] + 1, self.cap)

    def add(self, e, frame, action, reward, cut, t):
        """Transition taken from the state whose newest frame is `frame`, at episode step t."""
        self._write(e, frame, action, reward, float(cut), t, True)

    def add_final(self, e, frame, t):
        """Last observation of an episode (only ever used as a next state)."""
        self._write(e, frame, 0, 0.0, 0.0, t, False)

    def break_episodes(self):
        """After a restart the running episodes are lost: their newest transition has no
        stored next frame, so it must not be sampled."""
        for e in range(self.n):
            if self.size[e] > 0:
                self.valid[e, (self.head[e] - 1) % self.cap] = False

    def __len__(self):
        return int(self.size.sum())

    def sampleable(self, e, i):
        """Mask of (env, slot) pairs that are complete, uncorrupted transitions."""
        age = (self.head[e] - 1 - i) % self.cap            # 0 = newest written slot
        ok = self.valid[e, i] & (age >= 1) & (age < self.size[e])   # slot i and i+1 are written
        # when the ring is full, the oldest frames needed by the stack must not be overwritten
        full = self.size[e] == self.cap
        ok &= ~full | (age + self.tin[e, i] <= self.cap - 1)
        return ok

    def sample_indices(self, batch):
        """Uniform over all sampleable transitions (rejection sampling on the flat index)."""
        es, is_ = [], []
        need = batch
        while need > 0:
            flat = self.rng.integers(0, self.n * self.cap, size=2 * need + 8)
            e, i = flat // self.cap, flat % self.cap
            ok = self.sampleable(e, i)
            es.append(e[ok][:need])
            is_.append(i[ok][:need])
            need -= len(es[-1])
        return np.concatenate(es), np.concatenate(is_)

    def stacked(self, e, i):
        """State whose newest frame is slot (e, i): shape (B, stack, *obs_shape)."""
        out = np.empty((len(e), self.stack) + self.frame.shape[2:], dtype=self.frame.dtype)
        for k in range(self.stack):                         # k frames back in time
            back = np.minimum(k, self.tin[e, i])            # never before the episode start
            out[:, self.stack - 1 - k] = self.frame[e, (i - back) % self.cap]
        return out

    def sample(self, batch, device):
        e, i = self.sample_indices(batch)
        s = self.stacked(e, i)
        s2 = self.stacked(e, (i + 1) % self.cap)
        t = lambda x: torch.as_tensor(x, device=device)
        return t(s), t(self.act[e, i]), t(self.rew[e, i]), t(s2), t(self.cut[e, i])

    # ---- persistence (atomic: write to *.tmp, then rename) ----
    _FIELDS = ("frame", "act", "rew", "cut", "tin", "valid", "head", "size")

    def save(self, path):
        tmp = path + ".tmp.npz"
        np.savez(tmp, rng=np.array([json.dumps(self.rng.bit_generator.state)]),
                 **{k: getattr(self, k) for k in self._FIELDS})
        os.replace(tmp, path)

    def load(self, path):
        with np.load(path) as z:
            for k in self._FIELDS:
                getattr(self, k)[...] = z[k]
            self.rng.bit_generator.state = json.loads(str(z["rng"][0]))
