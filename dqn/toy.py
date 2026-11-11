"""Small MDPs with known answers, used by the validation suite (dqn/validate.py).

ShuttleEnv wraps the tabular shuttle-dispatch MDP from Assignment 1 as an environment:
one-hot state, uniformly random start state, NO terminal states, and a time-limit truncation
after `horizon` steps. Its exact Q* comes from value iteration, so a DQN trained on it can
be compared with the true answer. reward='const' / 'zero' replace the reward by 1 / 0.
"""
import numpy as np

from dqn.shuttle_mdp import ShuttleMDP


def small_shuttle(gamma=0.95):
    return ShuttleMDP(K=5, cf=20.0, ch=2.0, smax=20, gamma=gamma)


class ShuttleEnv:
    def __init__(self, mdp, horizon=20, reward="mdp"):
        self.mdp, self.horizon, self.reward = mdp, horizon, reward
        self.rng = np.random.default_rng(0)

    def _obs(self):
        o = np.zeros(self.mdp.S, dtype=np.float32)
        o[self.s] = 1.0
        return o

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.s = int(self.rng.integers(self.mdp.S))
        self.t = 0
        return self._obs(), {}

    def step(self, a):
        r = {"mdp": self.mdp.R[a, self.s], "const": 1.0, "zero": 0.0}[self.reward]
        self.s = int(self.rng.choice(self.mdp.S, p=self.mdp.P[a, self.s]))
        self.t += 1
        return self._obs(), float(r), False, self.t >= self.horizon, {}   # never terminates

    def close(self):
        pass


def shuttle_spec(mdp):
    return dict(obs_shape=(mdp.S,), obs_dtype=np.float32, n_actions=2, stack=1, hidden=128)
