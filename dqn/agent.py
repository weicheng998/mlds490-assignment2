"""DQN agent: online network, target network, TD target, one gradient step.

Switches (all used by the ablation / validation experiments):
  double        Double DQN target (online net selects a', target net evaluates it)
  loss          'huber' | 'mse'
  use_target    False -> the online network is used for the target too ("no target network")
  mask_terminal False -> bootstrap even after a real termination (a deliberate bug, for ablation)
  target_max    clip the TD target at a known upper bound of the return (experiment, off by default)
"""
import copy

import torch
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, qnet, n_actions, gamma, lr, device, double=True, loss="huber",
                 use_target=True, mask_terminal=True, grad_clip=10.0, adam_eps=1e-8, target_max=None):
        self.online = qnet.to(device)
        self.target = copy.deepcopy(self.online)            # target network: a frozen copy
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr, eps=adam_eps)
        self.n_actions, self.gamma, self.device = n_actions, gamma, device
        self.double, self.loss_name = double, loss
        self.use_target, self.mask_terminal, self.grad_clip = use_target, mask_terminal, grad_clip
        self.target_max = target_max        # optional known upper bound on returns (experiment only)
        self.updates = 0

    @torch.no_grad()
    def q_values(self, states):
        """Q(s, .) of the online network for a batch of states (numpy or tensor)."""
        return self.online(torch.as_tensor(states, device=self.device))

    @torch.no_grad()
    def td_target(self, r, s2, cut):
        """y = r + gamma * (1 - cut) * Q_target(s', a*).

        cut = 1 only when the episode really terminated at s' (pole fell, life lost / game over).
        A time-limit truncation has cut = 0: the task would have continued, so we still bootstrap."""
        tnet = self.target if self.use_target else self.online
        q_next_all = tnet(s2)
        if self.double:
            a_star = self.online(s2).argmax(1, keepdim=True)     # online net picks the action
            q_next = q_next_all.gather(1, a_star).squeeze(1)     # target net evaluates it
        else:
            q_next = q_next_all.max(1).values                    # vanilla DQN: max over target net
        if not self.mask_terminal:
            cut = torch.zeros_like(cut)                          # ablation: ignore termination
        y = r + self.gamma * (1.0 - cut) * q_next                # terminal mask: no bootstrap if cut
        if self.target_max is not None:
            y = y.clamp(max=self.target_max)                     # experiment: clip at the known ceiling
        return y

    def update(self, batch):
        s, a, r, s2, cut = batch
        y = self.td_target(r, s2, cut)
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)  # Q(s, a) of the action taken
        loss = F.smooth_l1_loss(q, y) if self.loss_name == "huber" else F.mse_loss(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.opt.step()
        self.updates += 1
        return loss.item(), q.mean().item(), float(gnorm)

    def sync_target(self):
        """Hard target update: copy the online weights into the target network."""
        self.target.load_state_dict(self.online.state_dict())

    @torch.no_grad()
    def soft_update(self, tau):
        """Polyak target update: target <- (1 - tau) * target + tau * online."""
        for pt, po in zip(self.target.parameters(), self.online.parameters()):
            pt.mul_(1.0 - tau).add_(po, alpha=tau)

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def state(self):
        return {"online": self.online.state_dict(), "target": self.target.state_dict(),
                "opt": self.opt.state_dict(), "updates": self.updates}

    def load_state(self, st):
        self.online.load_state_dict(st["online"])
        self.target.load_state_dict(st["target"])
        self.opt.load_state_dict(st["opt"])
        self.updates = st["updates"]
