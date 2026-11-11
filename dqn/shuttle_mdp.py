"""(Copied unchanged from my Assignment 1, mdp/shuttle.py; used here only as an exact Q* reference.)

Shuttle dispatch MDP: model + enumeration (finite horizon), value iteration, policy iteration.

Modeling assumptions (also stated in README / report):
  * State s = number of customers waiting at the start of a period, s in {0, ..., smax=200}.
  * Actions: a=0 hold, a=1 dispatch. Dispatching is allowed at any s (also s=0) and
    carries min(s, K) customers; it costs cf.
  * Holding cost ch per customer still waiting AFTER the decision: ch * (s - a*min(s,K)).
  * Then A ~ Unif{1..5} customers arrive; next state = min(s_after + A, smax).
    Arrivals that would push the queue above smax are turned away (mass lumped at smax).
  * Reward = -cost, discount gamma = 0.95. Finite horizon: terminal value V_T = 0.
"""
import numpy as np

HOLD, DISPATCH = 0, 1


class ShuttleMDP:
    def __init__(self, K=15, cf=100.0, ch=2.0, arrivals=None, smax=200, gamma=0.95):
        self.K, self.cf, self.ch, self.smax, self.gamma = K, float(cf), float(ch), smax, gamma
        self.arrivals = arrivals if arrivals is not None else {a: 0.2 for a in range(1, 6)}
        assert abs(sum(self.arrivals.values()) - 1) < 1e-12
        self.S = smax + 1
        self.P, self.R = self._build()

    def after_decision(self, s, a):
        return s - min(s, self.K) if a == DISPATCH else s

    def cost(self, s, a):
        return (self.cf if a == DISPATCH else 0.0) + self.ch * self.after_decision(s, a)

    def _build(self):
        S = self.S
        P = np.zeros((2, S, S))
        R = np.zeros((2, S))
        for s in range(S):
            for a in (HOLD, DISPATCH):
                left = self.after_decision(s, a)
                R[a, s] = -self.cost(s, a)
                for arr, p in self.arrivals.items():
                    P[a, s, min(left + arr, self.smax)] += p
        return P, R

    # ------------------------------------------------------------ helpers --
    def q_values(self, V):
        return self.R + self.gamma * self.P @ V          # shape (2, S)

    def greedy(self, V, tol=1e-9):
        """Argmax with ties (within tol) broken toward HOLD."""
        Q = self.q_values(V)
        return (Q[DISPATCH] > Q[HOLD] + tol).astype(int)

    def evaluate(self, pi):
        """Exact policy evaluation: solve (I - gamma P_pi) V = R_pi."""
        idx = np.arange(self.S)
        P_pi = self.P[pi, idx]
        R_pi = self.R[pi, idx]
        return np.linalg.solve(np.eye(self.S) - self.gamma * P_pi, R_pi)

    # -------------------------------------------------------- algorithms --
    def enumerate(self, T=500):
        """Backward induction over t = T-1..0 with V_T = 0.
        Returns V (T+1, S) with V[t] the optimal value-to-go at time t, and pi (T, S)."""
        V = np.zeros((T + 1, self.S))
        pi = np.zeros((T, self.S), dtype=int)
        for t in range(T - 1, -1, -1):
            Q = self.q_values(V[t + 1])
            pi[t] = (Q[DISPATCH] > Q[HOLD] + 1e-9).astype(int)
            V[t] = Q.max(0)
        return V, pi

    def value_iteration(self, eps=1e-8, max_iter=100000):
        """Stops when ||V_{k+1}-V_k||_inf < eps (1-gamma) / (2 gamma), which guarantees the
        greedy policy is eps-optimal. Returns V, pi, residual history."""
        V = np.zeros(self.S)
        thresh = eps * (1 - self.gamma) / (2 * self.gamma)
        hist = []
        for _ in range(max_iter):
            V_new = self.q_values(V).max(0)
            res = np.abs(V_new - V).max()
            hist.append(res)
            V = V_new
            if res < thresh:
                break
        return V, self.greedy(V), np.array(hist)

    def policy_iteration(self, pi0=None, max_iter=1000):
        """Howard policy iteration with exact evaluation. Starts from always-hold."""
        pi = np.zeros(self.S, dtype=int) if pi0 is None else pi0.copy()
        history = []
        for _ in range(max_iter):
            V = self.evaluate(pi)
            history.append((pi.copy(), V.copy()))
            Q = self.q_values(V)
            # Only switch when strictly better, so ties cannot cause cycling.
            better = Q[1 - pi, np.arange(self.S)] > Q[pi, np.arange(self.S)] + 1e-9
            if not better.any():
                return V, pi, history
            pi = np.where(better, 1 - pi, pi)
        raise RuntimeError("policy iteration did not converge")


def threshold_of(pi):
    """Smallest s with dispatch, and whether the policy is a pure threshold policy."""
    d = np.nonzero(pi)[0]
    if len(d) == 0:
        return None, True
    tau = int(d[0])
    return tau, bool(pi[tau:].all())


def simulate(K, cf, ch, arrivals, smax, gamma, policy_fn, s0, horizon, n, rng):
    """Independent Monte Carlo simulator (does NOT use the P / R matrices).
    Returns the discounted total reward of n episodes from s0."""
    vals, probs = np.array(list(arrivals.keys())), np.array(list(arrivals.values()))
    totals = np.zeros(n)
    for i in range(n):
        s, g, disc = s0, 0.0, 1.0
        A = rng.choice(vals, size=horizon, p=probs)
        for t in range(horizon):
            a = policy_fn(s)
            carried = min(s, K) if a else 0
            left = s - carried
            g -= disc * ((cf if a else 0.0) + ch * left)
            s = min(left + A[t], smax)
            disc *= gamma
        totals[i] = g
    return totals
