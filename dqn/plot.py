"""All report figures and results/summary.json, generated from the CSV / npy / json logs.

  python -m dqn.plot                      # everything that has data
  python -m dqn.plot --cartpole-run cartpole_s0 --mspacman-run mspacman_s0

Works on partial logs: a run that is still training gives valid (shorter) plots.
"""
import argparse
import glob
import json
import os

import numpy as np

import viz
from viz import AQUA, BLUE, GRAY, INK2, ORANGE, YELLOW, plt

RES = "results"


# ----------------------------------------------------------------- utils ----
def load_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        head = f.readline().strip().split(",")
        rows = [l.strip().split(",") for l in f if l.strip()]
    rows = [r for r in rows if len(r) == len(head)]          # ignore a half-written last line
    if not rows:
        return None
    out = {}
    for j, h in enumerate(head):
        col = [r[j] for r in rows]
        out[h] = np.array([float(c) if c != "" else np.nan for c in col])
    return out


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def moving_avg(x, w=100):
    """Mean of the last w values (fewer at the start)."""
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(i - w, 0)
    return (c[i] - c[lo]) / (i - lo)


def save(fig, name):
    path = os.path.join(RES, "figures", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


NAMES = {"cartpole": "CartPole-v1", "mspacman": "ALE/MsPacman-v5"}
QUNIT = {"cartpole": "Q (reward units; 1 per step)", "mspacman": "Q (clipped-reward units)"}


# ------------------------------------------------ required plots (i)-(iii) ----
def fig_maxq(env, run, ceiling=None):
    d = load_csv(f"{RES}/{env}/{run}.csv"); h = load_csv(f"{RES}/{env}/{run}_heldout.csv")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ep = d["episode"]
    ax.plot(ep, d["max_q"], ".", ms=1.5, color=BLUE, alpha=0.35, rasterized=True)
    ax.plot([], [], ".", color=BLUE, label=r"per-episode max over visited states of $\max_a Q(s,a)$")
    ax.plot(ep, moving_avg(d["max_q"]), color=BLUE, lw=1.8, label="its 100-episode moving average")
    if h is not None:
        ax.plot(h["episode"], h["heldout_q"], color=ORANGE, lw=1.8,
                label=r"mean $\max_a Q$ on fixed held-out states")
    if ceiling:
        ax.axhline(ceiling, color=INK2, ls="--", lw=1)
        ax.text(ep[-1], ceiling, f" 1/(1-γ) = {ceiling:g}", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("training episode"); ax.set_ylabel(QUNIT[env])
    ax.set_title(f"{NAMES[env]}: maximum Q-value vs training episode ({run})")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, f"fig_{run}_maxq.png")


def fig_reward(env, run):
    d = load_csv(f"{RES}/{env}/{run}.csv")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(d["episode"], d["return"], color=BLUE, lw=0.5, alpha=0.35, label="episode reward (raw game score)" if env == "mspacman" else "episode reward")
    ax.plot(d["episode"], moving_avg(d["return"]), color=ORANGE, lw=2, label="moving average of the last 100 episodes")
    ax.set_xlabel("training episode"); ax.set_ylabel("episode reward")
    ax.set_title(f"{NAMES[env]}: training episode reward ({run}, ε-greedy behaviour policy)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, f"fig_{run}_reward.png")


def fig_hist(env, run, whiches=("best", "final")):
    data = [(w, np.load(f"{RES}/{env}/{run}_test_{w}.npy")) for w in whiches
            if os.path.exists(f"{RES}/{env}/{run}_test_{w}.npy")]
    rnd = f"{RES}/{env}/{env}_test_random.npy"
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(4.6 * len(data), 3.4), squeeze=False)
    label = {"best": "validation-selected checkpoint (best.pt)", "final": "last checkpoint (final.pt)"}
    for ax, (w, R) in zip(axes[0], data):
        bins = np.linspace(0, 500, 26) if env == "cartpole" else 30
        if os.path.exists(rnd) and env == "mspacman":
            Rr = np.load(rnd)
            bins = np.linspace(0, max(R.max(), Rr.max()), 36)
            ax.hist(Rr, bins=bins, color=GRAY, alpha=0.6, label=f"random policy: {Rr.mean():.0f} ± {Rr.std(ddof=1):.0f}")
        ax.hist(R, bins=bins, color=BLUE, alpha=0.85, label=f"trained: {R.mean():.1f} ± {R.std(ddof=1):.1f}")
        ax.axvline(R.mean(), color=ORANGE, lw=1.5)
        ax.set_xlabel("episode reward"); ax.set_ylabel("episodes (of 500)")
        ax.set_title(f"{label[w]}\nmean {R.mean():.1f}, std {R.std(ddof=1):.1f}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"{NAMES[env]}: 500 greedy evaluation episodes on unseen seeds ({run})", fontsize=10, y=1.08)
    save(fig, f"fig_{run}_hist.png")


# --------------------------------------------------------- multi-run plots ----
def fig_runs(env, runs, labels, name, xkey="episode"):
    """Moving-average training reward and periodic greedy validation score for several runs."""
    cols = [BLUE, ORANGE, AQUA, YELLOW, GRAY, "#7a4fd6"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    any_data = False
    for k, (run, lab) in enumerate(zip(runs, labels)):
        d = load_csv(f"{RES}/{env}/{run}.csv"); e = load_csv(f"{RES}/{env}/{run}_eval.csv"); h = load_csv(f"{RES}/{env}/{run}_heldout.csv")
        if d is None:
            continue
        any_data = True
        axes[0].plot(d[xkey], moving_avg(d["return"]), color=cols[k % 6], lw=1.5, label=lab)
        if e is not None:
            axes[1].plot(e[xkey], e["mean"], "-o", ms=2.5, color=cols[k % 6], lw=1.2, label=lab)
        if h is not None:
            axes[2].plot(h[xkey], h["heldout_q"], color=cols[k % 6], lw=1.5, label=lab)
    if not any_data:
        plt.close(fig); return
    axes[0].set_title("training reward, 100-episode moving average"); axes[0].set_ylabel("episode reward")
    axes[1].set_title("greedy validation score (unseen seeds)"); axes[1].set_ylabel("mean episode reward")
    axes[2].set_title(r"mean $\max_a Q$ on fixed held-out states"); axes[2].set_ylabel(QUNIT[env])
    for ax in axes:
        ax.set_xlabel("training episode" if xkey == "episode" else "environment steps (agent decisions)")
        ax.legend(fontsize=7)
    fig.suptitle(NAMES[env], y=1.02, fontsize=11)
    save(fig, name)


# ------------------------------------------- non-standard: Q calibration ----
def realized_returns(tr):
    """Discounted return actually obtained after each step, in the units the agent was trained on.
    The return restarts where the bootstrap was cut (termination / life loss). Steps whose return
    is cut off by a time-limit truncation less than `tail` steps later are dropped (unknown tail)."""
    gamma = float(tr["gamma"]); tail = int(np.ceil(np.log(0.01) / np.log(gamma)))
    Q, G = [], []
    for k in np.unique(tr["ep"]):
        m = tr["ep"] == k
        q, r, cut = tr["q"][m], tr["r"][m], tr["cut"][m]
        g = np.zeros(len(r)); acc = 0.0
        for t in range(len(r) - 1, -1, -1):
            acc = r[t] + gamma * (0.0 if cut[t] else acc)
            g[t] = acc
        keep = np.ones(len(r), dtype=bool)
        if not cut[-1]:                      # episode ended by truncation: tail of the return unknown
            last_cut = np.where(cut)[0].max() if cut.any() else -1
            keep[max(last_cut + 1, len(r) - tail):] = False
        Q.append(q[keep]); G.append(g[keep])
    return np.concatenate(Q), np.concatenate(G)


def calib_stats(q, g):
    return {"n": int(len(q)), "mean_Q": float(q.mean()), "mean_G": float(g.mean()), "bias": float((q - g).mean()),
            "mae": float(np.abs(q - g).mean()), "corr": float(np.corrcoef(q, g)[0, 1]) if g.std() > 0 else float("nan"),
            "frac_over": float((q > g).mean())}


def fig_calibration(items, name, summary):
    """items: list of (title, env, run, which). Predicted max_a Q(s_t) vs realized discounted return."""
    items = [it for it in items if os.path.exists(f"{RES}/{it[1]}/{it[2]}_trace_{it[3]}.npz")]
    if not items:
        return
    fig, axes = plt.subplots(1, len(items), figsize=(5.2 * len(items), 3.9), squeeze=False, layout="constrained")
    for ax, (title, env, run, which) in zip(axes[0], items):
        tr = np.load(f"{RES}/{env}/{run}_trace_{which}.npz")
        q, g = realized_returns(tr)
        s = calib_stats(q, g); summary[f"calibration_{run}_{which}"] = s
        lo, hi = min(q.min(), g.min()), max(q.max(), g.max())
        hb = ax.hexbin(g, q, gridsize=45, bins="log", cmap="Blues", extent=(lo, hi, lo, hi), mincnt=1, linewidths=0)
        ax.plot([lo, hi], [lo, hi], color=INK2, ls="--", lw=1, label="perfect calibration")
        edges = np.quantile(g, np.linspace(0, 1, 13)); edges = np.unique(edges)
        if len(edges) > 2:
            idx = np.clip(np.digitize(g, edges[1:-1]), 0, len(edges) - 2)
            xm = [g[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()]
            ym = [q[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()]
            ax.plot(xm, ym, "-o", color=ORANGE, ms=3, lw=1.5, label="mean predicted Q per return bin")
        ax.set_xlabel("realized discounted return $G_t$"); ax.set_ylabel(r"predicted $\max_a Q(s_t,a)$")
        ax.set_title(f"{title}\nbias Q−G = {s['bias']:+.2f}, MAE = {s['mae']:.2f}, n = {s['n']:,}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        fig.colorbar(hb, ax=ax, label="states (log scale)", fraction=0.046, pad=0.03)
    save(fig, name)


# ------------------------------- non-standard: Q aligned to the next failure ----
def aligned_to_cut(tr, K):
    """Rows = cut events (pole fell / life lost); column k = predicted max_a Q and realized return
    k steps BEFORE that event (NaN when the episode or the previous life started less than k steps before)."""
    gamma = float(tr["gamma"])
    Qa, Ga = [], []
    for e in np.unique(tr["ep"]):
        m = tr["ep"] == e
        q, r, cut = tr["q"][m], tr["r"][m], tr["cut"][m]
        g = np.zeros(len(r)); acc = 0.0
        for t in range(len(r) - 1, -1, -1):
            acc = r[t] + gamma * (0.0 if cut[t] else acc); g[t] = acc
        prev = -1
        for c in np.where(cut)[0]:
            rowq = np.full(K, np.nan); rowg = np.full(K, np.nan)
            n = min(K, c - prev)
            rowq[:n] = q[c - np.arange(n)]; rowg[:n] = g[c - np.arange(n)]
            Qa.append(rowq); Ga.append(rowg); prev = c
    return (np.array(Qa), np.array(Ga)) if Qa else (np.zeros((0, K)), np.zeros((0, K)))


def fig_aligned(items, name, summary):
    """items: (title, env, run, which, K, event name)."""
    items = [it for it in items if os.path.exists(f"{RES}/{it[1]}/{it[2]}_trace_{it[3]}.npz")]
    panels = []
    for title, env, run, which, K, ev in items:
        tr = np.load(f"{RES}/{env}/{run}_trace_{which}.npz")
        Qa, Ga = aligned_to_cut(tr, K)
        if len(Qa) >= 5:
            panels.append((title, env, run, which, K, ev, Qa, Ga, float(tr["gamma"]), float(tr["q"].mean())))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.7), squeeze=False)
    for ax, (title, env, run, which, K, ev, Qa, Ga, gamma, qbar) in zip(axes[0], panels):
        k = np.arange(K)
        ax.fill_between(k, np.nanpercentile(Qa, 25, axis=0), np.nanpercentile(Qa, 75, axis=0), color=BLUE, alpha=0.2, lw=0)
        ax.plot(k, np.nanmean(Qa, axis=0), color=BLUE, label=r"predicted $\max_a Q(s_t,a)$: mean and interquartile band")
        ax.plot(k, np.nanmean(Ga, axis=0), color=ORANGE, lw=1.5, label="realized discounted return (mean)")
        if env == "cartpole":
            ax.plot(k, (1 - gamma ** (k + 1)) / (1 - gamma), color=INK2, ls="--", lw=1, label=r"exact: $(1-\gamma^{k+1})/(1-\gamma)$")
        ax.invert_xaxis()
        ax.set_xlabel(f"k = steps before the {ev}"); ax.set_ylabel(QUNIT[env])
        ax.set_title(f"{title}\n{len(Qa)} events", fontsize=9); ax.legend(fontsize=7)
        d = np.nanmean(Qa, axis=0) - np.nanmean(Ga, axis=0)
        summary[f"aligned_{run}_{which}"] = {
            "events": int(len(Qa)), "q_at_k": {str(j): float(np.nanmean(Qa[:, j])) for j in (0, 1, 2, 5, 10, 20, 40, K - 1) if j < K},
            "g_at_k": {str(j): float(np.nanmean(Ga[:, j])) for j in (0, 1, 2, 5, 10, 20, 40, K - 1) if j < K},
            "mean_abs_gap_first10": float(np.abs(d[:10]).mean()), "mean_abs_gap_all": float(np.nanmean(np.abs(d))),
            "mean_q_all_states": qbar}
    save(fig, name)


# ----------------------------------------------- ablations and gamma sweep ----
def run_metrics(prefix, ceiling):
    d = load_csv(prefix + ".csv"); e = load_csv(prefix + "_eval.csv"); h = load_csv(prefix + "_heldout.csv")
    if d is None or e is None or h is None:
        return None
    late = e["step"] > 0.8 * e["step"].max()
    return {"train_last100": float(d["return"][-100:].mean()), "val_final": float(e["mean"][-1]),
            "val_late_mean": float(e["mean"][late].mean()), "val_best": float(e["mean"].max()),
            "frac_evals_solved": float((e["mean"] >= 475).mean()),
            "heldout_q_final": float(h["heldout_q"][-1]), "heldout_q_max": float(np.nanmax(h["heldout_q"])),
            "max_q_peak": float(np.nanmax(d["max_q"])), "frac_ep_over_ceiling": float((d["max_q"] > ceiling).mean()),
            "steps": float(e["step"].max())}


def collect_group(names, ceilings):
    out = {}
    for n in names:
        rows = []
        for f in sorted(glob.glob(f"{RES}/ablation/{n}_s*_eval.csv")):
            m = run_metrics(f[:-len("_eval.csv")], ceilings[n])
            if m:
                rows.append(m)
        if rows:
            out[n] = {"n_seeds": len(rows), "per_seed": rows,
                      **{k + "_mean": float(np.mean([r[k] for r in rows])) for k in rows[0]},
                      **{k + "_std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0 for k in rows[0]}}
    return out


def fig_ablation(summary):
    from dqn.ablate import GROUPS
    names = list(GROUPS["ablation"]) + list(GROUPS["extra"])
    g = collect_group(names, {n: 20.0 for n in names})
    if not g:
        return
    from scipy import stats
    for n in g:                                  # Welch test of each variant against the full agent (per-seed late validation mean)
        if n != "full" and "full" in g:
            a_, b_ = [r["val_late_mean"] for r in g[n]["per_seed"]], [r["val_late_mean"] for r in g["full"]["per_seed"]]
            g[n]["p_vs_full"] = float(stats.ttest_ind(a_, b_, equal_var=False).pvalue)
    summary["ablation"] = g
    names = [n for n in names if n in g]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    spec = [("val_late_mean", "greedy validation reward,\nmean over the last 20% of training", None),
            ("heldout_q_final", r"final mean $\max_a Q$ on held-out states (log scale)", 20.0),
            ("max_q_peak", "largest Q on any visited state (log scale)", 20.0)]
    rng = np.random.default_rng(0)
    for ax, (key, title, line) in zip(axes, spec):
        for j, n in enumerate(names):
            v = np.array([r[key] for r in g[n]["per_seed"]])
            ax.barh(j, v.mean(), color=BLUE if n == "full" else GRAY, alpha=0.75, height=0.6)
            ax.plot(v, j + rng.uniform(-0.15, 0.15, len(v)), "o", color=ORANGE, ms=3.5, mec="white", mew=0.4)
        if line:
            ax.axvline(line, color=INK2, ls="--", lw=1)
        if key != "val_late_mean":
            ax.set_xscale("log")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names if ax is axes[0] else [])
        ax.invert_yaxis(); ax.set_title(title, fontsize=9); ax.grid(axis="y", visible=False)
    fig.suptitle("CartPole-v1 ablations: bars = mean over seeds, dots = individual seeds, dashed = true ceiling 1/(1-γ) = 20",
                 fontsize=10, y=1.03)
    save(fig, "fig_ablation.png")


def fig_gamma(summary):
    gammas = [0.5, 0.8, 0.9, 0.95, 0.99]
    names = [f"gamma_{g}" for g in gammas]
    g = collect_group(names, {n: 1 / (1 - gm) for n, gm in zip(names, gammas)})
    if not g:
        return
    for n, gm in zip(names, gammas):
        if n in g:
            g[n]["ceiling"] = 1 / (1 - gm)
    summary["gamma_sweep"] = g
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    xs = [1 / (1 - gm) for n, gm in zip(names, gammas) if n in g]
    for n, gm in zip(names, gammas):
        if n not in g:
            continue
        c = 1 / (1 - gm)
        v = np.array([r["heldout_q_final"] for r in g[n]["per_seed"]]); w = np.array([r["max_q_peak"] for r in g[n]["per_seed"]])
        axes[0].plot([c] * len(v), v, "o", color=BLUE, ms=4, alpha=0.8); axes[0].plot([c] * len(w), w, "^", color=ORANGE, ms=4, alpha=0.8)
        r = np.array([x["val_late_mean"] for x in g[n]["per_seed"]])
        axes[1].plot([gm] * len(r), r, "o", color=BLUE, ms=4, alpha=0.8)
    axes[0].plot([], [], "o", color=BLUE, label=r"final mean $\max_a Q$, held-out states")
    axes[0].plot([], [], "^", color=ORANGE, label="largest Q on any visited state during training")
    axes[0].plot([min(xs), max(xs)], [min(xs), max(xs)], ls="--", color=INK2, lw=1, label="ceiling: Q = 1/(1-γ)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("predicted ceiling 1/(1-γ)"); axes[0].set_ylabel("learned Q"); axes[0].legend(fontsize=7)
    axes[0].set_title("Learned Q scale follows the predicted ceiling", fontsize=9)
    axes[1].set_xlabel("discount factor γ"); axes[1].set_ylabel("greedy validation reward (late mean)")
    axes[1].set_title("Control performance vs γ (one dot per seed)", fontsize=9)
    save(fig, "fig_gamma_sweep.png")


# ---------------------------------------------------------------- summary ----
def test_stats(env, run, summary):
    from scipy import stats
    out = {}
    for w in ("best", "final"):
        j = load_json(f"{RES}/{env}/{run}_test_{w}.json")
        if j:
            out[w] = j
    rnd_path = f"{RES}/{env}/{env}_test_random.npy"
    if os.path.exists(rnd_path):
        Rr = np.load(rnd_path)
        out["random"] = {**(load_json(f"{RES}/{env}/{env}_test_random.json") or {}),
                         "mean": float(Rr.mean()), "std": float(Rr.std(ddof=1)), "n": int(len(Rr))}
        for w in ("best", "final"):
            p = f"{RES}/{env}/{run}_test_{w}.npy"
            if os.path.exists(p):
                R = np.load(p)
                t = stats.ttest_ind(R, Rr, equal_var=False); u = stats.mannwhitneyu(R, Rr, alternative="greater")
                out[f"{w}_vs_random"] = {"welch_t": float(t.statistic), "welch_p": float(t.pvalue), "mwu_p": float(u.pvalue),
                                         "diff": float(R.mean() - Rr.mean()),
                                         "ci95": [float(R.mean() - Rr.mean() - 1.96 * np.sqrt(R.var(ddof=1) / len(R) + Rr.var(ddof=1) / len(Rr))),
                                                  float(R.mean() - Rr.mean() + 1.96 * np.sqrt(R.var(ddof=1) / len(R) + Rr.var(ddof=1) / len(Rr)))],
                                         "prob_superiority": float(u.statistic / (len(R) * len(Rr)))}
    d = load_csv(f"{RES}/{env}/{run}.csv"); h = load_csv(f"{RES}/{env}/{run}_heldout.csv"); e = load_csv(f"{RES}/{env}/{run}_eval.csv")
    if d is not None:
        ret = d["return"]; n = len(ret)
        tr = {"episodes": int(n), "steps": int(d["step"][-1]), "wall_hours": float(d["wall_s"][-1] / 3600),
              "first100_mean": float(ret[:100].mean()), "last100_mean": float(ret[-100:].mean()),
              "best_ma100": float(moving_avg(ret).max()), "max_q_peak": float(np.nanmax(d["max_q"])),
              "max_q_last100_mean": float(d["max_q"][-100:].mean()), "n_truncated": int(d["truncated"].sum())}
        if n >= 200:
            t = stats.ttest_ind(ret[-100:], ret[:100], equal_var=False)
            tr["last_vs_first_welch_p"] = float(t.pvalue)
        for k in (1000, 2000, 4000, 6000):
            if n >= k:
                tr[f"ma100_at_{k}"] = float(ret[k - 100:k].mean())
        if env == "cartpole":
            tr["frac_ep_over_20"] = float((d["max_q"] > 20).mean())
            ma = moving_avg(ret); hit = np.where(ma >= 475)[0]
            tr["first_ep_ma100_ge_475"] = int(hit[0]) if len(hit) else None
            tr["first_ep_500"] = int(np.where(ret >= 500)[0][0]) if (ret >= 500).any() else None
        if h is not None:
            tr["heldout_q_final"] = float(h["heldout_q"][-1]); tr["heldout_q_max"] = float(np.nanmax(h["heldout_q"]))
        if e is not None:
            tr["val_best"] = float(e["mean"].max()); tr["val_final"] = float(e["mean"][-1])
            tr["val_best_step"] = int(e["step"][np.where(e["is_best"] == 1)[0][-1]])
            if env == "cartpole":
                tr["frac_evals_500"] = float((e["mean"] >= 500).mean())
        out["training"] = tr
    summary[run] = out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cartpole-run", default="cartpole_s0")
    p.add_argument("--mspacman-run", default="mspacman_s0")
    a = p.parse_args()
    summary = {}
    cp_runs = sorted(os.path.basename(f)[:-9] for f in glob.glob(f"{RES}/cartpole/cartpole_s*_eval.csv"))
    pm_runs = sorted(os.path.basename(f)[:-9] for f in glob.glob(f"{RES}/mspacman/mspacman_s*_eval.csv"))
    for env, run, ceil in (("cartpole", a.cartpole_run, 20.0), ("mspacman", a.mspacman_run, None)):
        fig_maxq(env, run, ceil); fig_reward(env, run); fig_hist(env, run)
    for r in pm_runs:
        if r != a.mspacman_run:
            fig_maxq("mspacman", r); fig_reward("mspacman", r); fig_hist("mspacman", r)
    for r in cp_runs:
        test_stats("cartpole", r, summary)
    for r in pm_runs:
        test_stats("mspacman", r, summary)
    fig_runs("cartpole", cp_runs, [r.replace("cartpole_", "seed ").replace("s", "", 1) if False else r for r in cp_runs], "fig_cartpole_seeds.png", xkey="step")
    pm_label = {"mspacman_s0": "Double DQN, seed 0", "mspacman_s1": "Double DQN, seed 1", "mspacman_s0_dqn": "vanilla DQN, seed 0"}
    fig_runs("mspacman", pm_runs, [pm_label.get(r, r) for r in pm_runs], "fig_mspacman_runs.png")
    fig_calibration([("CartPole-v1, best.pt (never fails in these episodes)", "cartpole", a.cartpole_run, "best"),
                     ("CartPole-v1, final.pt", "cartpole", a.cartpole_run, "final")], "fig_calibration_cartpole.png", summary)
    fig_calibration([("MsPacman, Double DQN (best.pt)", "mspacman", "mspacman_s0", "best"),
                     ("MsPacman, vanilla DQN (best.pt)", "mspacman", "mspacman_s0_dqn", "best")], "fig_calibration_mspacman.png", summary)
    # CartPole: final.pt is the checkpoint that has failures to align to (best.pt never fails). MsPacman: best.pt.
    fig_aligned([("CartPole-v1 (final.pt)", "cartpole", a.cartpole_run, "final", 60, "pole falls / cart leaves the track")],
                "fig_aligned_final.png", summary)
    fig_aligned([("MsPacman, Double DQN (best.pt)", "mspacman", "mspacman_s0", "best", 60, "life is lost")],
                "fig_aligned_best.png", summary)
    for r in cp_runs:                 # per-seed: what does final.pt predict right before it fails?
        f = f"{RES}/cartpole/{r}_trace_final.npz"
        if os.path.exists(f):
            Qa, Ga = aligned_to_cut(np.load(f), 60)
            summary[r]["edge"] = {"events": int(len(Qa)),
                                  **({f"q_k{k}": float(np.nanmean(Qa[:, k])) for k in (0, 10, 20)} if len(Qa) else {}),
                                  **({f"g_k{k}": float(np.nanmean(Ga[:, k])) for k in (0, 10, 20)} if len(Qa) else {})}
    fig_ablation(summary)
    fig_gamma(summary)
    for f in sorted(glob.glob(f"{RES}/validation/*.json")):
        summary.setdefault("validation", {})[os.path.basename(f)[:-5]] = load_json(f)
    sh = summary.get("validation", {}).get("shuttle")
    if sh:                                   # per-arm averages over seeds for the report table
        agg = {}
        for r in sh["runs"]:
            agg.setdefault(f"bug{r['bug']}_{r.get('loss', 'huber')}", []).append(r)
        sh["agg"] = {k: {m: float(np.mean([r[m] for r in v])) for m in
                         ("mean_Q", "mae_vs_Qstar", "max_vs_Qstar", "mae_vs_Qbug", "rel_mae_vs_Qstar", "rel_mae_vs_Qbug", "policy_agree")}
                     for k, v in agg.items()}
    with open(f"{RES}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", f"{RES}/summary.json")


if __name__ == "__main__":
    main()
