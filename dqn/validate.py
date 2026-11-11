"""Validation experiments for the DQN implementation (CPU is fine: pass --allow-cpu).

Each test states a prediction, runs the real training/agent/buffer code, and compares the
outcome with a number that does not come from that code (closed form, value iteration,
an independent numpy computation, or a chi-square reference distribution).

  python -m dqn.validate --allow-cpu                       # trace replay overfit resume shuttle controls
  python -m dqn.validate --allow-cpu --tests pacman_env    # needs ale-py (preprocessing / lives / rewards)
  python -m dqn.validate --allow-cpu --tests sticky --run mspacman_s0   # needs a trained checkpoint

Output: results/validation/<test>.txt and <test>.json
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.common import ENV_SPECS, FrameStacker, VecEnv, build_qnet, get_device, make_env, preprocess_observation
from dqn.replay import ReplayBuffer
from dqn.train import build_parser, finalize_args, rollout, train

OUT = "results/validation"
LINES = []


def say(*x):
    s = " ".join(str(v) for v in x)
    print(s, flush=True)
    LINES.append(s)


def finish(name, result):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    LINES.clear()


def make_args(tmp, **over):
    """Training arguments = CartPole defaults + overrides, writing into a temp directory."""
    a = finalize_args(build_parser().parse_args(["--env", "cartpole", "--allow-cpu"]))
    a.ckpt_dir, a.results_dir = os.path.join(tmp, "ck"), os.path.join(tmp, "res")
    for k, v in over.items():
        setattr(a, k, v)
    return a


# ------------------------------------------------------------------------------------
def test_trace(dev):
    """First 10 CartPole steps by hand: the agent's TD targets vs an independent numpy computation."""
    say("== trace: first 10 steps of CartPole-v1, random actions, then until the pole falls ==")
    say("Prediction: y = r + gamma * Q_target(s', argmax_a Q_online(s', a)) for non-terminal steps,")
    say("            y = r = 1 exactly for the terminal step; numpy forward pass must agree to ~1e-6.")
    torch.manual_seed(0)
    spec, gamma = ENV_SPECS["cartpole"], 0.95
    agent = DQNAgent(build_qnet("cartpole"), 2, gamma, 1e-3, dev)
    # make the target network differ from the online one, so mixing them up would be visible
    with torch.no_grad():
        for p in agent.target.parameters():
            p.add_(0.05 * torch.randn_like(p))
    buf = ReplayBuffer(1, 1000, spec["obs_shape"], spec["obs_dtype"], 1, seed=0)
    env = make_env("cartpole")
    rng = np.random.default_rng(0)
    s, _ = env.reset(seed=0)
    rows, t, done = [], 0, False
    while not done:
        a = int(rng.integers(2))
        s2, r, term, trunc, _ = env.step(a)
        buf.add(0, s.astype(np.float32), a, r, term, t)
        rows.append((t, s.copy(), a, r, term, s2.copy()))
        done = term or trunc
        if done:
            buf.add_final(0, s2.astype(np.float32), t + 1)
        s, t = s2, t + 1

    def np_forward(net, x):       # independent forward pass: weights copied out, plain numpy
        W = [(l.weight.detach().cpu().numpy().astype(np.float64), l.bias.detach().cpu().numpy().astype(np.float64))
             for l in net.net if hasattr(l, "weight")]
        h = x.astype(np.float64)
        for k, (w, b) in enumerate(W):
            h = h @ w.T + b
            if k < len(W) - 1:
                h = np.maximum(h, 0)
        return h

    e = np.zeros(len(rows), dtype=np.int64); i = np.arange(len(rows))
    assert buf.sampleable(e, i).all()
    S = torch.as_tensor(buf.stacked(e, i), device=dev); S2 = torch.as_tensor(buf.stacked(e, i + 1), device=dev)
    y_agent = agent.td_target(torch.as_tensor(buf.rew[e, i], device=dev), S2,
                              torch.as_tensor(buf.cut[e, i], device=dev)).cpu().numpy()
    say(f"{'t':>2} {'state (x, x_dot, th, th_dot)':<34} a  r cut {'Q_on(s,.)':<17} {'Q_on(s2,.)':<17} "
        f"{'Q_tg(s2,.)':<17} {'y_numpy':>8} {'y_agent':>8}")
    maxdiff, show = 0.0, list(range(min(10, len(rows)))) + ([len(rows) - 1] if len(rows) > 10 else [])
    for k, (t, s, a, r, term, s2) in enumerate(rows):
        assert np.allclose(buf.stacked(e[k:k + 1], i[k:k + 1] + 1)[0, 0], s2), "stored next state != true s'"
        q_on, q_on2, q_tg2 = np_forward(agent.online, s), np_forward(agent.online, s2), np_forward(agent.target, s2)
        y_np = r + gamma * (0.0 if term else q_tg2[np.argmax(q_on2)])
        maxdiff = max(maxdiff, abs(y_np - y_agent[k]))
        if k in show:
            f = lambda v: "[" + " ".join(f"{u:+.3f}" for u in v) + "]"
            say(f"{t:>2} {f(s):<34} {a}  {r:.0f}  {int(term)}  {f(q_on):<17} {f(q_on2):<17} {f(q_tg2):<17} "
                f"{y_np:8.5f} {y_agent[k]:8.5f}")
    say(f"episode length {len(rows)}; max |y_numpy - y_agent| over all {len(rows)} transitions = {maxdiff:.2e}")
    term_y = float(y_agent[-1])
    say(f"terminal transition target = {term_y:.6f} (predicted exactly 1.0)")
    # target-network behaviour around one gradient step and one sync
    tg_before = [p.clone() for p in agent.target.parameters()]
    on_before = [p.clone() for p in agent.online.parameters()]
    batch = (S, torch.as_tensor(buf.act[e, i], device=dev), torch.as_tensor(buf.rew[e, i], device=dev), S2,
             torch.as_tensor(buf.cut[e, i], device=dev))
    agent.update(batch)
    tg_moved = max(float((p - q).detach().abs().max()) for p, q in zip(agent.target.parameters(), tg_before))  # noqa
    on_moved = max(float((p - q).detach().abs().max()) for p, q in zip(agent.online.parameters(), on_before))
    agent.sync_target()
    gap = max(float((p - q).detach().abs().max()) for p, q in zip(agent.target.parameters(), agent.online.parameters()))
    say(f"after one gradient step: online weights moved by {on_moved:.2e}, target weights moved by {tg_moved:.1e} (predicted 0)")
    say(f"after sync_target(): max |target - online| = {gap:.1e} (predicted 0)")
    ok = maxdiff < 1e-4 and abs(term_y - 1) < 1e-6 and tg_moved == 0 and gap == 0 and on_moved > 0
    say("PASS" if ok else "FAIL")
    finish("trace", {"max_target_diff": maxdiff, "terminal_target": term_y, "target_moved": tg_moved,
                     "sync_gap": gap, "pass": ok})


# ------------------------------------------------------------------------------------
def test_replay(dev):
    """Replay buffer: sampling uniformity, no stack crosses an episode boundary, true s' at truncation."""
    say("== replay: uniform sampling and frame-stack integrity (4 envs, stack 4, ring wraps many times) ==")
    n_env, stack, cap = 4, 4, 4 * 300
    buf = ReplayBuffer(n_env, cap, (2,), np.int32, stack, seed=1)
    st = FrameStacker(n_env, stack, (2,), np.int32)
    rng = np.random.default_rng(1)
    # Each frame encodes (global episode id, step in episode) so that any mixing is detectable.
    ep_id = np.arange(n_env); next_ep = n_env; t = np.zeros(n_env, dtype=int)
    length = rng.integers(1, 40, size=n_env)
    actor = {}                                  # (episode, t) -> state the actor saw
    for e in range(n_env):
        st.reset(e, np.array([ep_id[e], 0]))
    for _ in range(5000):
        for e in range(n_env):
            actor[(int(ep_id[e]), int(t[e]))] = st.s[e].copy()
            last = t[e] + 1 >= length[e]
            buf.add(e, st.s[e, -1], t[e] % 7, float(ep_id[e]), last and (ep_id[e] % 2 == 0), t[e])
            if last:                             # odd episodes are "truncated": cut = 0
                buf.add_final(e, np.array([ep_id[e], t[e] + 1]), t[e] + 1)
                ep_id[e] = next_ep; next_ep += 1; t[e] = 0; length[e] = rng.integers(1, 40)
                st.reset(e, np.array([ep_id[e], 0]))
            else:
                t[e] += 1
                st.push(e, np.array([ep_id[e], t[e]]))
    # (a) uniformity: chi-square of sample counts over the sampleable set
    E, I = np.meshgrid(np.arange(n_env), np.arange(buf.cap), indexing="ij")
    mask = buf.sampleable(E.ravel(), I.ravel()).reshape(n_env, buf.cap)
    n_ok = int(mask.sum()); draws = 400_000
    e, i = buf.sample_indices(draws)
    counts = np.zeros((n_env, buf.cap), dtype=int)
    np.add.at(counts, (e, i), 1)
    outside = int(counts[~mask].sum())
    from scipy import stats
    chi2, pval = stats.chisquare(counts[mask])
    say(f"(a) {n_ok} sampleable transitions of {n_env * buf.cap} slots; {draws} draws")
    say(f"    draws that hit a non-sampleable slot: {outside} (predicted 0)")
    say(f"    chi-square vs uniform: chi2 = {chi2:.1f}, dof = {n_ok - 1}, p = {pval:.3f} (predicted: not small)")
    # (b) stack integrity on 100k sampled transitions
    e, i = buf.sample_indices(100_000)
    s, s2 = buf.stacked(e, i), buf.stacked(e, (i + 1) % buf.cap)
    ep, tt = s[:, -1, 0], s[:, -1, 1]
    cross = int((s[:, :, 0] != ep[:, None]).sum() + (s2[:, :, 0] != ep[:, None]).sum())
    want = np.maximum(tt[:, None] - np.arange(stack - 1, -1, -1)[None, :], 0)
    wrong_t = int((s[:, :, 1] != want).sum())
    want2 = np.maximum(tt[:, None] + 1 - np.arange(stack - 1, -1, -1)[None, :], 0)
    wrong_t2 = int((s2[:, :, 1] != want2).sum())
    mism = sum(int(not np.array_equal(s[k], actor[(int(ep[k]), int(tt[k]))])) for k in range(0, 100_000, 10))
    say(f"(b) frames from another episode inside a sampled state or next state: {cross} of {2 * 100_000 * stack} (predicted 0)")
    say(f"    wrong time index in state / next state stacks: {wrong_t} / {wrong_t2} (predicted 0 / 0)")
    say(f"    sampled state != state the actor saw at that step (10,000 checked): {mism} (predicted 0)")
    # (c) cut flag only on real terminations; truncated transitions keep their true next frame
    last = s2[:, -1, 1] == tt + 1
    cut = buf.cut[e, i]
    bad_cut = int(((cut == 1) & (ep % 2 == 1)).sum())
    n_trunc_last = int(((buf.act[e, i] == tt % 7) & (cut == 0) & (ep % 2 == 1)).sum())
    say(f"(c) cut=1 on a truncated (odd) episode: {bad_cut} (predicted 0); next frame is always step t+1 of the same "
        f"episode: {bool(last.all())}")
    ok = outside == 0 and pval > 0.001 and cross == 0 and wrong_t == 0 and wrong_t2 == 0 and mism == 0 and bad_cut == 0 and last.all()
    say("PASS" if ok else "FAIL")
    finish("replay", {"sampleable": n_ok, "outside": outside, "chi2": chi2, "p": pval, "cross_episode_frames": cross,
                      "wrong_time": wrong_t + wrong_t2, "actor_mismatch": mism, "bad_cut": bad_cut, "pass": bool(ok)})


# ------------------------------------------------------------------------------------
def test_overfit(dev):
    """A single fixed batch with a frozen target must be fit almost exactly."""
    say("== overfit: one fixed batch of 64 CartPole transitions, target network frozen ==")
    say("Prediction: with fixed regression targets the loss goes to ~0 (the network can memorise 64 points).")
    torch.manual_seed(0)
    spec = ENV_SPECS["cartpole"]
    # double=False: with Double DQN the online net picks a', so the target would move while we fit.
    agent = DQNAgent(build_qnet("cartpole"), 2, 0.95, 1e-3, dev, loss="mse", double=False)
    env = make_env("cartpole"); rng = np.random.default_rng(0)
    buf = ReplayBuffer(1, 500, spec["obs_shape"], spec["obs_dtype"], 1, seed=0)
    for ep in range(6):
        s, _ = env.reset(seed=ep); t = 0
        while True:
            a = int(rng.integers(2)); s2, r, term, trunc, _ = env.step(a)
            buf.add(0, s, a, r, term, t)
            if term or trunc:
                buf.add_final(0, s2, t + 1); break
            s, t = s2, t + 1
    batch = buf.sample(64, dev)
    y0 = agent.td_target(batch[2], batch[3], batch[4])
    hist = []
    for k in range(3000):
        hist.append(agent.update(batch)[0])           # target net never synced -> targets stay y0
    q = agent.online(batch[0]).gather(1, batch[1].unsqueeze(1)).squeeze(1).detach()
    err = float((q - y0).abs().max())
    say(f"MSE loss: step 0 = {hist[0]:.4f}, step 100 = {hist[100]:.5f}, step 1000 = {hist[1000]:.2e}, step 2999 = {hist[-1]:.2e}")
    say(f"max |Q(s,a) - y| over the batch after training = {err:.2e}; number of terminal samples in batch = {int(batch[4].sum())}")
    ok = hist[-1] < 1e-4 * hist[0]
    say("PASS" if ok else "FAIL")
    finish("overfit", {"loss0": hist[0], "loss_end": hist[-1], "max_abs_err": err, "pass": ok})


# ------------------------------------------------------------------------------------
def _q_table(agent, S):
    with torch.no_grad():
        return agent.q_values(np.eye(S, dtype=np.float32)[:, None, :]).cpu().numpy().T     # (A, S)


def test_shuttle(dev, seeds=(0, 1, 2)):
    """DQN on a tabular MDP with exact Q* (value iteration), with and without the truncation bug."""
    from dqn.toy import ShuttleEnv, shuttle_spec, small_shuttle
    H, gamma, scale = 20, 0.95, 0.02
    mdp = small_shuttle(gamma)
    V, pi_star, _ = mdp.value_iteration()
    Q_star = mdp.q_values(V)
    g_eff = gamma * (1 - 1 / H)
    mdp_bug = small_shuttle(g_eff)
    Vb, _, _ = mdp_bug.value_iteration()
    Q_bug = mdp_bug.q_values(Vb)
    say("== shuttle: DQN vs exact Q* on the 21-state shuttle MDP from Assignment 1 (K=5, cf=20, ch=2) ==")
    say(f"No terminal states; every episode is truncated after {H} steps; rewards scaled by {scale} for training.")
    say("Prediction 1: with truncation handled correctly (bootstrap through the time limit), DQN -> Q* of gamma = 0.95.")
    say(f"Prediction 2: if truncation is (wrongly) treated as terminal, 1/{H} of all transitions lose their bootstrap,")
    say(f"              so DQN -> Q* of an MDP with gamma_eff = 0.95 * (1 - 1/{H}) = {g_eff:.4f}, i.e. much smaller |Q|.")
    say(f"Exact values: mean Q*(gamma=0.95) = {Q_star.mean():.2f}; mean Q*(gamma_eff) = {Q_bug.mean():.2f}")
    say("Prediction 3 (added after prediction 2 failed with the Huber loss, see report): the fixed point in prediction 2")
    say("              is a conditional MEAN, so it should hold for the MSE loss. With Huber (delta = 1) the 5% of")
    say("              targets without bootstrap are > 1 away from the rest and are down-weighted like outliers,")
    say("              so the Huber agent should land between Q*(gamma_eff) and Q*.")
    out = {"gamma_eff": g_eff, "mean_Qstar": Q_star.mean(), "mean_Qbug": Q_bug.mean(), "runs": []}
    for bug, loss in ((0, "huber"), (1, "mse"), (1, "huber")):
        for seed in seeds:
            tmp = tempfile.mkdtemp()
            a = make_args(tmp, seed=seed, total_steps=40_000, buffer=40_000, learning_starts=500, target_sync=250,
                          eps_mid=0.1, eps_mid_step=5_000, eps_end=0.1, eval_every=0, heldout_every=10**9, loss=loss,
                          reward_scale=scale, trunc_as_terminal=bug, run_name=f"shuttle_b{bug}_s{seed}", save_buffer=0)
            r = train(a, env_name=lambda: ShuttleEnv(mdp, H), spec=shuttle_spec(mdp), quiet=True)
            Q = _q_table(r["agent"], mdp.S) / scale
            shutil.rmtree(tmp, ignore_errors=True)
            row = {"bug": bug, "loss": loss, "seed": seed, "mean_Q": Q.mean(),
                   "mae_vs_Qstar": np.abs(Q - Q_star).mean(), "max_vs_Qstar": np.abs(Q - Q_star).max(),
                   "mae_vs_Qbug": np.abs(Q - Q_bug).mean(),
                   "rel_mae_vs_Qstar": np.abs(Q - Q_star).mean() / np.abs(Q_star).mean(),
                   "rel_mae_vs_Qbug": np.abs(Q - Q_bug).mean() / np.abs(Q_bug).mean(),
                   "policy_agree": float(((Q[1] > Q[0]).astype(int) == pi_star).mean())}
            out["runs"].append(row)
            say(f"  trunc_as_terminal={bug} loss={loss:5s} seed={seed}: mean Q = {row['mean_Q']:8.2f} | MAE vs Q* = "
                f"{row['mae_vs_Qstar']:6.2f} ({100 * row['rel_mae_vs_Qstar']:.1f}%), max err = {row['max_vs_Qstar']:6.2f} | "
                f"MAE vs Q*(gamma_eff) = {row['mae_vs_Qbug']:6.2f} ({100 * row['rel_mae_vs_Qbug']:.1f}%) | "
                f"greedy = pi* in {100 * row['policy_agree']:.0f}% of states")
    sel = lambda b, l: [r for r in out["runs"] if r["bug"] == b and r["loss"] == l]
    good, bug_mse, bug_hub = sel(0, "huber"), sel(1, "mse"), sel(1, "huber")
    m = lambda rows, k: float(np.mean([r[k] for r in rows]))
    say(f"Correct handling (Huber): mean Q {m(good, 'mean_Q'):.1f}, relative error vs Q* {100 * m(good, 'rel_mae_vs_Qstar'):.1f}%.")
    say(f"Bug + MSE:   mean Q {m(bug_mse, 'mean_Q'):.1f}; relative error vs predicted Q*(gamma_eff) {100 * m(bug_mse, 'rel_mae_vs_Qbug'):.1f}%, vs Q* {100 * m(bug_mse, 'rel_mae_vs_Qstar'):.1f}%.")
    say(f"Bug + Huber: mean Q {m(bug_hub, 'mean_Q'):.1f}; lies between Q*(gamma_eff) = {Q_bug.mean():.1f} and Q* = {Q_star.mean():.1f}.")
    ok = (all(r["rel_mae_vs_Qstar"] < 0.05 for r in good) and all(r["rel_mae_vs_Qbug"] < 0.10 for r in bug_mse)
          and all(Q_star.mean() < r["mean_Q"] < Q_bug.mean() for r in bug_hub))
    say("PASS" if ok else "FAIL")
    out["pass"] = ok
    finish("shuttle", out)


def test_controls(dev):
    """Constant-reward and zero-reward controls, and the predicted 1/(1-gamma) ceiling for several gammas."""
    from dqn.toy import ShuttleEnv, shuttle_spec, small_shuttle
    say("== controls: reward == 1 at every step, no termination (truncation every 20 steps) ==")
    say("Prediction: Q(s,a) = 1/(1-gamma) for every state and action; with reward == 0, Q = 0.")
    mdp = small_shuttle()
    out, ok = [], True
    for reward, gamma in (("const", 0.5), ("const", 0.9), ("const", 0.95), ("zero", 0.95)):
        tmp = tempfile.mkdtemp()
        a = make_args(tmp, seed=0, gamma=gamma, total_steps=30_000, buffer=30_000, learning_starts=500,
                      target_sync=100, eps_mid=0.1, eps_mid_step=3_000, eps_end=0.1, eval_every=0,
                      heldout_every=10**9, run_name=f"ctrl_{reward}_{gamma}", save_buffer=0)
        r = train(a, env_name=lambda: ShuttleEnv(mdp, 20, reward), spec=shuttle_spec(mdp), quiet=True)
        Q = _q_table(r["agent"], mdp.S)
        shutil.rmtree(tmp, ignore_errors=True)
        pred = 0.0 if reward == "zero" else 1 / (1 - gamma)
        row = {"reward": reward, "gamma": gamma, "predicted": pred, "mean_Q": Q.mean(), "min_Q": Q.min(),
               "max_Q": Q.max(), "max_abs_dev": np.abs(Q - pred).max()}
        out.append(row)
        ok &= row["max_abs_dev"] < max(0.03 * pred, 0.02)
        say(f"  reward={reward:5s} gamma={gamma}: predicted {pred:6.3f} | learned Q over all 42 (s,a): mean {row['mean_Q']:.4f}, "
            f"min {row['min_Q']:.4f}, max {row['max_Q']:.4f}, max |dev| = {row['max_abs_dev']:.4f}")
    say("PASS" if ok else "FAIL")
    finish("controls", {"rows": out, "pass": bool(ok)})


# ------------------------------------------------------------------------------------
def test_resume(dev):
    """Stop / restart: counters continue, logs have no duplicates, buffer is restored, the gap is not sampled."""
    import csv
    say("== resume: 6,000 CartPole steps in one go vs 3,000 + restart + 3,000 ==")
    tmp = tempfile.mkdtemp()
    kw = dict(seed=0, eval_every=2_000, eval_episodes=5, heldout_every=500, run_name="resume")
    train(make_args(tmp, total_steps=3_000, **kw), quiet=True)
    ck = torch.load(os.path.join(tmp, "ck", "resume", "ckpt.pt"), weights_only=False)
    r = train(make_args(tmp, total_steps=6_000, **kw), quiet=True)
    rows = list(csv.DictReader(open(os.path.join(tmp, "res", "resume.csv"))))
    eps = [int(x["episode"]) for x in rows]; steps = [int(x["step"]) for x in rows]
    held = [int(x["step"]) for x in csv.DictReader(open(os.path.join(tmp, "res", "resume_heldout.csv")))]
    buf = ReplayBuffer(1, 50_000, (4,), np.float32, 1); buf.load(os.path.join(tmp, "ck", "resume", "buffer.npz"))
    dup = len(eps) - len(set(eps)); contiguous = eps == list(range(len(eps)))
    say(f"checkpoint at step {ck['step']} / episode {ck['episode']}; after restart training ended at step {r['step']}")
    say(f"episode log: {len(eps)} rows, duplicates = {dup}, contiguous ids = {contiguous}, steps non-decreasing = {steps == sorted(steps)}")
    say(f"held-out log: steps strictly increasing = {all(b > a for a, b in zip(held, held[1:]))}")
    # Learning starts once the buffer holds 1,000 slots. Slots = env steps + one extra slot per finished episode
    # (the final frame), so the first update happens at the first step n with n + episodes_finished(n) >= 1000.
    n0 = next(n for n in range(1, 6001) if n + sum(st <= n for st in steps) >= 1000)
    pred = 6_000 - n0 + 1
    say(f"gradient updates = {r['agent'].updates}; predicted {pred} = one per step from step {n0}, the first step at which the "
        f"buffer holds 1,000 slots (my first prediction, 5,000, ignored the extra final-frame slot per episode)")
    say(f"replay buffer after restart holds {len(buf)} slots; non-sampleable slots (episode ends + restart gap) = {int((~buf.valid[0, :len(buf)]).sum())}")
    ok = dup == 0 and contiguous and r["step"] == 6_000 and r["agent"].updates == pred
    shutil.rmtree(tmp, ignore_errors=True)
    say("PASS" if ok else "FAIL")
    finish("resume", {"rows": len(eps), "duplicates": dup, "updates": r["agent"].updates, "pass": ok})


# ------------------------------------------------------------------------------------
def test_pacman_env(dev):
    """MsPacman environment facts the training code relies on, measured on random-policy episodes."""
    say("== pacman_env: ALE/MsPacman-v5 properties measured on 20 random-policy episodes ==")
    env = make_env("mspacman")
    say(f"action space: {env.action_space.n} actions: {env.unwrapped.get_action_meanings()}")
    spec = env.spec.kwargs
    say(f"env kwargs: frameskip={spec.get('frameskip')}, repeat_action_probability={spec.get('repeat_action_probability')}, "
        f"full_action_space={spec.get('full_action_space')}, max_num_frames_per_episode={spec.get('max_num_frames_per_episode')}")
    rng = np.random.default_rng(0)
    vals, pac_visible, n_frames, rewards, losses_per_ep, lens, scores, cut_at_end = set(), 0, 0, {}, [], [], [], 0
    for ep in range(20):
        obs, info = env.reset(seed=100 + ep); lives = info["lives"]; lost = 0; L = 0; R = 0
        while True:
            raw_has_pac = bool((obs[1:176:2, ::2].sum(axis=2) == 448).any())
            f = preprocess_observation(obs)
            assert f.shape == (88, 80, 1) and f.dtype == np.int8
            vals.update(np.unique(f).tolist()); n_frames += 1
            # Pac-Man's colour (sum 448) is mapped to 0 -> -128 after normalisation: the unique darkest value
            pac_visible += int((f == -128).any())
            assert raw_has_pac == bool((f == -128).any())
            obs, r, term, trunc, info = env.step(int(rng.integers(9)))
            L += 1; R += r
            if r != 0:
                rewards[r] = rewards.get(r, 0) + 1
            if info["lives"] < lives:
                lost += 1
                if term:
                    cut_at_end += 1
            lives = info["lives"]
            if term or trunc:
                break
        losses_per_ep.append(lost); lens.append(L); scores.append(R)
    total = sum(k * v for k, v in rewards.items())
    say(f"preprocessed frame: shape (88, 80, 1), dtype int8, {len(vals)} distinct values in [{min(vals)}, {max(vals)}]")
    say(f"value -128 (Pac-Man after the 'improve contrast' step) present in {pac_visible}/{n_frames} frames "
        f"({100 * pac_visible / n_frames:.1f}%); no other pixel maps to -128 (asserted for every frame)")
    say(f"episode length: mean {np.mean(lens):.0f} agent steps (min {min(lens)}, max {max(lens)}); random score mean {np.mean(scores):.0f}")
    say(f"life losses detected per episode: {sorted(set(losses_per_ep))} (predicted exactly 3); "
        f"the last one coincides with termination in {cut_at_end}/20 episodes")
    say(f"non-zero rewards seen (value: count): {dict(sorted(rewards.items()))}; all are positive: {all(k > 0 for k in rewards)}")
    clipped_total = sum(rewards.values())
    say(f"reward clipping: {clipped_total} clipped reward units carry a raw score of {total:.0f}; "
        f"{100 * rewards.get(10.0, 0) * 10 / total:.1f}% of the raw score comes from 10-point pellets")
    # determinism: same seed + same actions -> same episode, with and without sticky actions
    def play(seed, p, act_seed=0, n=400):
        e = make_env("mspacman", repeat_action_probability=p); e.reset(seed=seed); g = np.random.default_rng(act_seed); R = 0
        for _ in range(n):
            o, r, te, tr, _ = e.step(int(g.integers(9))); R += r
            if te or tr:
                break
        e.close(); return R, int(o.astype(np.int64).sum())
    same = play(1, 0.25) == play(1, 0.25); diff = play(1, 0.25) != play(2, 0.25); det0 = play(1, 0.0) == play(2, 0.0)
    say(f"sticky p=0.25: same seed & actions -> identical episode: {same}; different seed -> different episode: {diff}")
    say(f"sticky p=0: different seeds, same actions -> identical episode: {det0} (the env seed only drives sticky actions)")
    ok = set(losses_per_ep) == {3} and same and diff and det0 and all(k > 0 for k in rewards)
    say("PASS" if ok else "FAIL")
    finish("pacman_env", {"values": len(vals), "pac_visible_frac": pac_visible / n_frames, "rewards": {str(k): v for k, v in rewards.items()},
                          "life_losses": losses_per_ep, "mean_len": float(np.mean(lens)), "random_mean": float(np.mean(scores)),
                          "pellet_share": rewards.get(10.0, 0) * 10 / total, "pass": bool(ok)})


def test_sticky(dev, run, which="best", n=30):
    """Reduce stochasticity: greedy policy with sticky actions switched off must be deterministic."""
    from dqn.evaluate import load_agent
    say(f"== sticky: {run} ({which}.pt), greedy, {n} evaluation seeds, sticky-action probability 0.25 vs 0 ==")
    say("Prediction: with p=0 the emulator and the greedy policy are both deterministic, so every episode has the")
    say("            same score (std = 0). All spread in the 500-episode histogram therefore comes from sticky actions.")
    agent, cfg = load_agent("mspacman", run, which, "checkpoints", dev)
    out = {}
    for p in (0.25, 0.0):
        res = rollout(agent, "mspacman", [4_000_000_000 + k for k in range(n)], ENV_SPECS["mspacman"], n_envs=10,
                      workers=True, env_kwargs={"repeat_action_probability": p})
        R = res["returns"]
        out[str(p)] = {"mean": R.mean(), "std": R.std(ddof=1), "distinct": len(set(R.tolist())), "truncated": int(res["truncated"].sum())}
        say(f"  p={p}: mean {R.mean():.1f}, std {R.std(ddof=1):.1f}, distinct scores {len(set(R.tolist()))}/{n}, "
            f"mean length {res['lengths'].mean():.0f}, truncated {int(res['truncated'].sum())}")
    ok = out["0.0"]["distinct"] == 1 and out["0.25"]["distinct"] > 1
    say("PASS" if ok else "FAIL")
    out["pass"] = ok
    finish(f"sticky_{run}", out)


TESTS = {"trace": test_trace, "replay": test_replay, "overfit": test_overfit, "resume": test_resume,
         "shuttle": test_shuttle, "controls": test_controls, "pacman_env": test_pacman_env}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests", nargs="+", default=["trace", "replay", "overfit", "resume", "shuttle", "controls"])
    p.add_argument("--run", default="mspacman_s0")
    p.add_argument("--which", default="best")
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    dev = get_device(a.allow_cpu)
    for t in a.tests:
        if t == "sticky":
            test_sticky(dev, a.run, a.which)
        else:
            TESTS[t](dev)


if __name__ == "__main__":
    main()
