"""
EXPERIMENT 1: EBM Differentiability & Credit Assignment

Pure EBM with Langevin sampling. No Flow. No proposals.
Online EBM updates via InfoNCE.

The ONLY experimental variable: diff_mode (how many Langevin steps
retain the computation graph).

Three regimes x Three horizons = 9 runs.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os

from models import Actor, ValueNetwork, BilinearEBM, RewardModel
from regimes import DifferentiabilityRegime, REGIME_NAMES, REGIME_COLORS, REGIME_LIST
from gradient_diagnostics import GradientDiagnosticLogger


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "ENV_NAME": "Pendulum-v1",
    "ACTION_SCALE": 2.0,

    "TOTAL_STEPS": 50000,
    "BATCH_SIZE": 256,

    "DISCOUNT": 0.99,
    "LAMBDA": 0.95,
    "TAU": 0.005,

    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 3e-4,
    "LR_REWARD": 1e-3,
    "LR_EBM": 1e-4,

    "ENTROPY_COEFF": 0.01,

    # EBM online update
    "WM_UPDATE_EVERY": 50,
    "WM_UPDATE_BATCH": 128,

    # InfoNCE
    "NUM_NEGATIVES": 128,          # more negatives = sharper energy landscape
    "INFONCE_TEMPERATURE": 0.1,

    # Langevin
    "LANGEVIN_STEPS": 10,
    "LANGEVIN_STEP_SIZE": 0.01,
    "LANGEVIN_NOISE_MAX": 0.01,
    "LANGEVIN_NOISE_MIN": 0.0001,
    "LANGEVIN_INIT_NOISE": 0.05,
    "LANGEVIN_GRAD_CLIP": 1.0,

    # Trust
    "TRUST_THRESHOLD": 3.0,
    "TRUST_SHARPNESS": 2.0,

    # Logging
    "EVAL_INTERVAL": 1000,
    "EVAL_EPISODES": 10,
    "LOG_INTERVAL": 500,
    "DIAGNOSTIC_INTERVAL": 2000,
    "GRAD_NORM_INTERVAL": 50,
    "VARIANCE_SAMPLES": 8,

    # Reward model exploitation guard
    # Clamp predicted rewards to env's true range during imagined rollouts
    "REWARD_CLAMP_MIN": -16.28,  # Pendulum: -(pi^2 + 0.1*8^2 + 0.001*2^2)
    "REWARD_CLAMP_MAX": 0.0,     # Pendulum: best possible reward

    "HIDDEN_DIM": 128,             # wider for sharper energy landscape
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# REPLAY BUFFER (identical to original)
# =============================================================================
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=100000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._state_sum = np.zeros(state_dim, dtype=np.float64)
        self._state_sq_sum = np.zeros(state_dim, dtype=np.float64)
        self._state_count = 0

    def add(self, s, a, r, ns):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = ns
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._state_sum += ns.astype(np.float64)
        self._state_sq_sum += (ns.astype(np.float64)) ** 2
        self._state_count += 1

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        device = CONFIG["DEVICE"]
        return {
            "states": torch.tensor(self.states[idx], device=device),
            "actions": torch.tensor(self.actions[idx], device=device),
            "rewards": torch.tensor(self.rewards[idx], device=device),
            "next_states": torch.tensor(self.next_states[idx], device=device),
        }

    def sample_states(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return torch.tensor(self.states[idx], device=CONFIG["DEVICE"])

    def sample_negatives(self, batch_size, num_negatives):
        idx = np.random.randint(0, self.size, size=(batch_size, num_negatives))
        return torch.tensor(self.next_states[idx], device=CONFIG["DEVICE"])

    @property
    def state_mean(self):
        if self._state_count == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        return (self._state_sum / self._state_count).astype(np.float32)

    @property
    def state_std(self):
        if self._state_count < 2:
            return np.ones(self.state_dim, dtype=np.float32)
        mean = self._state_sum / self._state_count
        var = self._state_sq_sum / self._state_count - mean ** 2
        return np.sqrt(np.clip(var, 1e-6, None)).astype(np.float32)


# =============================================================================
# TRUST (identical to original)
# =============================================================================
class TrustComputer:
    def __init__(self, buffer, threshold=3.0, sharpness=2.0):
        self.buffer = buffer
        self.threshold = threshold
        self.sharpness = sharpness

    def compute_trust(self, predicted_states):
        device = predicted_states.device
        mean = torch.tensor(self.buffer.state_mean, device=device)
        std = torch.tensor(self.buffer.state_std, device=device)
        z = (predicted_states - mean) / std
        z_norm = torch.norm(z, dim=-1, keepdim=True)
        return torch.sigmoid(self.sharpness * (self.threshold - z_norm))


# =============================================================================
# TD(lambda) (identical to original)
# =============================================================================
def compute_lambda_returns(rewards, values, trust, discount, lam):
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)
    for t in reversed(range(H)):
        eff_gamma = discount * trust[:, t]
        if t == H - 1:
            returns[:, t] = rewards[:, t] + eff_gamma * values[:, t]
        else:
            td1 = rewards[:, t] + eff_gamma * values[:, t]
            cont = rewards[:, t] + eff_gamma * returns[:, t + 1]
            returns[:, t] = (1 - lam) * td1 + lam * cont
    return returns


# =============================================================================
# EBM ONLINE UPDATE (InfoNCE)
# =============================================================================
def update_ebm(ebm, ebm_optimizer, buffer, device):
    BS = CONFIG["WM_UPDATE_BATCH"]
    if buffer.size < BS:
        return {}

    batch = buffer.sample(BS)
    s, a, ns = batch["states"], batch["actions"], batch["next_states"]
    B = s.shape[0]

    ebm.train()
    ebm_optimizer.zero_grad()

    # --- InfoNCE (contrastive) — matching reference pipeline ---
    E_pos = ebm(s, a, ns)

    # Random negatives from buffer only — no hard negatives
    neg_ns = buffer.sample_negatives(B, CONFIG["NUM_NEGATIVES"])
    s_exp = s.unsqueeze(1).expand(B, CONFIG["NUM_NEGATIVES"], -1)
    a_exp = a.unsqueeze(1).expand(B, CONFIG["NUM_NEGATIVES"], -1)
    E_neg = ebm(s_exp, a_exp, neg_ns)

    logits = torch.cat([E_pos.unsqueeze(1), E_neg], dim=1) / CONFIG["INFONCE_TEMPERATURE"]
    labels = torch.zeros(B, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(ebm.parameters(), 1.0)
    ebm_optimizer.step()
    ebm.eval()

    with torch.no_grad():
        return {
            "ebm_loss": loss.item(),
            "ebm_acc": (logits.argmax(1) == 0).float().mean().item(),
            "E_gap": (E_pos.mean() - E_neg.mean()).item(),
        }


# =============================================================================
# EVAL (identical to original)
# =============================================================================
def evaluate_policy(actor, num_episodes=10):
    env = gym.make(CONFIG["ENV_NAME"])
    device = CONFIG["DEVICE"]
    results = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_r = 0
        done = False
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act = actor.sample(s_t).cpu().numpy()[0]
            act = np.clip(act, -CONFIG["ACTION_SCALE"], CONFIG["ACTION_SCALE"])
            state, r, term, trunc, _ = env.step(act)
            ep_r += r
            done = term or trunc
        results.append(ep_r)
    env.close()
    return np.mean(results), np.std(results)


# =============================================================================
# TRAIN ONE REGIME
# =============================================================================
def train_regime(regime_name, horizon, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = CONFIG["DEVICE"]
    env = gym.make(CONFIG["ENV_NAME"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"\n{'='*60}")
    print(f"Regime: {REGIME_NAMES[regime_name]} | Horizon: {horizon} | Seed: {seed}")
    print(f"{'='*60}")

    # --- Load pretrained EBM ---
    ebm = BilinearEBM(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    ebm.load_state_dict(torch.load("pretrained_ebm_pendulum.pth",
                                    map_location=device, weights_only=True))
    ebm.eval()

    # --- Create regime ---
    regime = DifferentiabilityRegime(
        regime_name, ebm, state_dim, device, CONFIG
    )

    # --- RL components ---
    actor = Actor(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"],
                  action_scale=CONFIG["ACTION_SCALE"]).to(device)
    critic = ValueNetwork(state_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target = deepcopy(critic)
    for p in critic_target.parameters():
        p.requires_grad = False

    reward_model = RewardModel(state_dim, action_dim,
                               hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=CONFIG["LR_ACTOR"])
    critic_opt = optim.Adam(critic.parameters(), lr=CONFIG["LR_CRITIC"])
    reward_opt = optim.Adam(reward_model.parameters(), lr=CONFIG["LR_REWARD"])
    ebm_optimizer = optim.Adam(ebm.parameters(), lr=CONFIG["LR_EBM"])

    buffer = ReplayBuffer(state_dim, action_dim)
    trust_comp = TrustComputer(buffer, CONFIG["TRUST_THRESHOLD"],
                               CONFIG["TRUST_SHARPNESS"])

    grad_logger = GradientDiagnosticLogger()

    # --- Seed buffer ---
    print("Collecting 2000 random transitions...")
    state, _ = env.reset()
    for _ in range(2000):
        a = env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a)
        buffer.add(state, a, r, ns)
        state = ns if not (term or trunc) else env.reset()[0]

    eval_steps, eval_rewards = [], []
    grad_norm_steps, grad_norms = [], []

    state, _ = env.reset()
    total_steps = 0
    H = horizon

    while total_steps < CONFIG["TOTAL_STEPS"]:
        # ==== 1. REAL ENV STEP ====
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = actor.sample(s_t).cpu().numpy()[0]
        action = np.clip(action, -CONFIG["ACTION_SCALE"], CONFIG["ACTION_SCALE"])
        next_state, reward, term, trunc, _ = env.step(action)
        buffer.add(state, action, reward, next_state)
        total_steps += 1
        state = next_state if not (term or trunc) else env.reset()[0]

        # ==== 2. EBM ONLINE UPDATE ====
        if total_steps % CONFIG["WM_UPDATE_EVERY"] == 0:
            wm_metrics = update_ebm(ebm, ebm_optimizer, buffer, device)
            if total_steps % CONFIG["LOG_INTERVAL"] == 0 and wm_metrics:
                print(f"  [EBM Update] loss={wm_metrics['ebm_loss']:.4f} | "
                      f"acc={wm_metrics['ebm_acc']:.3f} | "
                      f"E_gap={wm_metrics['E_gap']:.3f}")

        # ==== 3. REWARD MODEL UPDATE ====
        if buffer.size >= CONFIG["BATCH_SIZE"] and total_steps % 5 == 0:
            batch = buffer.sample(CONFIG["BATCH_SIZE"])
            pred_r = reward_model(batch["states"], batch["actions"],
                                  batch["next_states"])
            r_loss = F.mse_loss(pred_r, batch["rewards"])
            reward_opt.zero_grad()
            r_loss.backward()
            reward_opt.step()

        # ==== 4. IMAGINED ROLLOUT + CRITIC + ACTOR ====
        if buffer.size >= CONFIG["BATCH_SIZE"] and total_steps % 5 == 0:
            B = CONFIG["BATCH_SIZE"]

            regime.freeze()
            for p in reward_model.parameters():
                p.requires_grad = False

            # --- CRITIC (imagined rollout) ---
            curr = buffer.sample_states(B)
            i_states, i_rewards, i_next, i_trust = [], [], [], []

            for t in range(H):
                a = actor.sample(curr)
                ns = regime.predict_next_state(curr, a)
                r = reward_model(curr, a, ns).squeeze(-1)
                r = r.clamp(CONFIG["REWARD_CLAMP_MIN"], CONFIG["REWARD_CLAMP_MAX"])
                w = trust_comp.compute_trust(ns)

                i_states.append(curr)
                i_rewards.append(r)
                i_next.append(ns)
                i_trust.append(w)
                curr = ns

            rew_t = torch.stack(i_rewards, dim=1)
            ns_t = torch.stack(i_next, dim=1)
            st_t = torch.stack(i_states, dim=1)
            tr_t = torch.cat(i_trust, dim=1)

            with torch.no_grad():
                nv = critic_target(
                    ns_t.reshape(B * H, -1)
                ).squeeze(-1).reshape(B, H)
                targets = compute_lambda_returns(
                    rew_t.detach(), nv, tr_t.detach(),
                    CONFIG["DISCOUNT"], CONFIG["LAMBDA"]
                )

            vpred = critic(
                st_t.reshape(B * H, -1).detach()
            ).squeeze(-1).reshape(B, H)
            critic_loss = F.mse_loss(vpred, targets)

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

            # --- ACTOR (fresh rollout for gradient flow) ---
            for p in critic.parameters():
                p.requires_grad = False

            curr_a = buffer.sample_states(B)
            ret = torch.zeros(B, device=device)
            disc = torch.ones(B, device=device)

            for t in range(H):
                a = actor.sample(curr_a)
                ns = regime.predict_next_state(curr_a, a)
                r = reward_model(curr_a, a, ns).squeeze(-1)
                r = r.clamp(CONFIG["REWARD_CLAMP_MIN"], CONFIG["REWARD_CLAMP_MAX"])
                w = trust_comp.compute_trust(ns).squeeze(-1)

                mu, log_std = actor(curr_a)
                std = torch.exp(torch.clamp(log_std, -5, 0.5))
                ent = 0.5 * torch.log(2 * np.pi * np.e * std.pow(2)).sum(dim=-1)

                ret = ret + disc * (r + CONFIG["ENTROPY_COEFF"] * ent)
                disc = disc * CONFIG["DISCOUNT"] * w
                curr_a = ns

            ret = ret + disc * critic(curr_a).squeeze(-1)
            actor_loss = -ret.mean()

            actor_opt.zero_grad()
            actor_loss.backward()

            # Grad norms (pre-clipping)
            if total_steps % CONFIG["GRAD_NORM_INTERVAL"] == 0:
                grad_logger.log_after_backward(total_steps, actor)
                total_norm = grad_logger.history["grad_norm"][-1]
                grad_norm_steps.append(total_steps)
                grad_norms.append(total_norm)

            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()

            # --- FULL DIAGNOSTIC (periodic) ---
            if total_steps % CONFIG["DIAGNOSTIC_INTERVAL"] == 0:
                def make_rollout_loss():
                    c = buffer.sample_states(B)
                    r_ = torch.zeros(B, device=device)
                    d_ = torch.ones(B, device=device)
                    for tt in range(H):
                        aa = actor.sample(c)
                        nns = regime.predict_next_state(c, aa)
                        rr = reward_model(c, aa, nns).squeeze(-1)
                        rr = rr.clamp(CONFIG["REWARD_CLAMP_MIN"], CONFIG["REWARD_CLAMP_MAX"])
                        ww = trust_comp.compute_trust(nns).squeeze(-1)
                        r_ = r_ + d_ * rr
                        d_ = d_ * CONFIG["DISCOUNT"] * ww
                        c = nns
                    r_ = r_ + d_ * critic(c).squeeze(-1)
                    return -r_.mean()

                start_states = buffer.sample_states(min(B, 32))
                langevin_cfg = {
                    "LANGEVIN_STEPS": CONFIG["LANGEVIN_STEPS"],
                    "LANGEVIN_STEP_SIZE": CONFIG["LANGEVIN_STEP_SIZE"],
                    "LANGEVIN_NOISE_MAX": CONFIG["LANGEVIN_NOISE_MAX"],
                    "LANGEVIN_NOISE_MIN": CONFIG["LANGEVIN_NOISE_MIN"],
                    "LANGEVIN_INIT_NOISE": CONFIG["LANGEVIN_INIT_NOISE"],
                    "LANGEVIN_GRAD_CLIP": CONFIG["LANGEVIN_GRAD_CLIP"],
                }
                grad_logger.log_full_diagnostic(
                    step=total_steps,
                    actor=actor,
                    actor_opt=actor_opt,
                    make_rollout_loss_fn=make_rollout_loss,
                    ebm=ebm,
                    langevin_config=langevin_cfg,
                    start_states=start_states,
                    horizon=H,
                    num_variance_samples=CONFIG["VARIANCE_SAMPLES"],
                )

            # Unfreeze
            for p in critic.parameters():
                p.requires_grad = True
            for p in reward_model.parameters():
                p.requires_grad = True
            regime.unfreeze()
            ebm_optimizer.zero_grad()  # clear stray grads from actor rollout

            # Polyak
            with torch.no_grad():
                for p, tp in zip(critic.parameters(), critic_target.parameters()):
                    tp.data.copy_(CONFIG["TAU"] * p.data + (1 - CONFIG["TAU"]) * tp.data)

            if total_steps % CONFIG["LOG_INTERVAL"] == 0:
                gn = grad_norms[-1] if grad_norms else 0
                print(f"  [{regime.display_name}] Step {total_steps} | "
                      f"Critic: {critic_loss.item():.4f} | "
                      f"Actor: {actor_loss.item():.4f} | "
                      f"Grad norm: {gn:.4f}")

        # ==== 5. EVAL ====
        if total_steps % CONFIG["EVAL_INTERVAL"] == 0:
            em, es = evaluate_policy(actor, CONFIG["EVAL_EPISODES"])
            eval_steps.append(total_steps)
            eval_rewards.append(em)
            print(f"  Step {total_steps}/{CONFIG['TOTAL_STEPS']} | "
                  f"Eval: {em:.1f} +/- {es:.1f}")

    env.close()

    return {
        "regime": regime_name,
        "horizon": horizon,
        "display_name": regime.display_name,
        "color": regime.color,
        "eval_steps": eval_steps,
        "eval_rewards": eval_rewards,
        "grad_norm_steps": grad_norm_steps,
        "grad_norms": grad_norms,
        "diagnostics": grad_logger.summary(),
    }


# =============================================================================
# PLOTTING
# =============================================================================
def smooth(data, window=5):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_convergence(all_results, horizons, regimes):
    fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 5), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    for i, h in enumerate(horizons):
        ax = axes[i]
        ax.set_title(f"Horizon H={h}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Environment Steps", fontsize=11)
        if i == 0:
            ax.set_ylabel("Eval Return", fontsize=11)
        ax.grid(True, alpha=0.3)
        for regime in regimes:
            key = f"{regime}_H{h}"
            if key in all_results:
                d = all_results[key]
                ax.plot(d["eval_steps"], d["eval_rewards"],
                        label=d["display_name"],
                        color=d["color"], linewidth=2, alpha=0.8)
        ax.legend(fontsize=9)
    plt.suptitle(
        "Experiment 1: Policy Convergence by EBM Differentiability Regime\n"
        "(Pure EBM Langevin — only gradient flow differs)",
        fontsize=13, fontweight="bold", y=1.04
    )
    plt.tight_layout()
    plt.savefig("exp1_convergence.png", dpi=200, bbox_inches="tight")
    print("Saved: exp1_convergence.png")
    plt.close()


def plot_gradient_norms(all_results, horizons, regimes):
    fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 4), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    for i, h in enumerate(horizons):
        ax = axes[i]
        ax.set_title(f"H={h}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps")
        if i == 0:
            ax.set_ylabel("||dJ/dtheta|| (pre-clipping)")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        for regime in regimes:
            key = f"{regime}_H{h}"
            if key in all_results:
                d = all_results[key]
                norms = np.array(d["grad_norms"])
                steps = np.array(d["grad_norm_steps"])
                if len(norms) > 10:
                    norms_s = smooth(norms, 10)
                    steps_s = steps[:len(norms_s)]
                    ax.plot(steps_s, norms_s, label=d["display_name"],
                            color=d["color"], linewidth=2, alpha=0.8)
        ax.legend(fontsize=8)
    plt.suptitle("Actor Gradient Magnitude (Pre-Clipping)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("exp1_gradient_norms.png", dpi=200, bbox_inches="tight")
    print("Saved: exp1_gradient_norms.png")
    plt.close()


def plot_gradient_variance(all_results, horizons, regimes):
    fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 4), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    for i, h in enumerate(horizons):
        ax = axes[i]
        ax.set_title(f"H={h}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps")
        if i == 0:
            ax.set_ylabel("Var[||dJ/dtheta||]")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        for regime in regimes:
            key = f"{regime}_H{h}"
            if key in all_results:
                d = all_results[key]
                diag = d["diagnostics"]
                if diag["grad_variance"]:
                    steps = [gv["step"] for gv in diag["grad_variance"]]
                    variances = [gv["variance"] for gv in diag["grad_variance"]]
                    ax.plot(steps, variances, label=d["display_name"],
                            color=d["color"], linewidth=2, marker='o',
                            markersize=4, alpha=0.8)
        ax.legend(fontsize=8)
    plt.suptitle("Gradient Variance Across Rollout Samples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("exp1_gradient_variance.png", dpi=200, bbox_inches="tight")
    print("Saved: exp1_gradient_variance.png")
    plt.close()


def plot_spectral_analysis(all_results, horizons, regimes):
    fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 4), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    for i, h in enumerate(horizons):
        ax = axes[i]
        ax.set_title(f"H={h}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps")
        if i == 0:
            ax.set_ylabel("Cumulative Spectral Radius")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5,
                    label="Stability boundary")
        for regime in regimes:
            key = f"{regime}_H{h}"
            if key in all_results:
                d = all_results[key]
                diag = d["diagnostics"]
                if diag["cumulative_spectral_radius"]:
                    steps = [sr["step"] for sr in diag["cumulative_spectral_radius"]]
                    values = [sr["value"] for sr in diag["cumulative_spectral_radius"]]
                    ax.plot(steps, values, label=d["display_name"],
                            color=d["color"], linewidth=2, marker='s',
                            markersize=4, alpha=0.8)
        ax.legend(fontsize=8)
    plt.suptitle("Jacobian Spectral Radius (>1 = gradient explosion risk)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("exp1_spectral.png", dpi=200, bbox_inches="tight")
    print("Saved: exp1_spectral.png")
    plt.close()


def plot_effective_rank(all_results, horizons, regimes):
    fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 4), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    for i, h in enumerate(horizons):
        ax = axes[i]
        ax.set_title(f"H={h}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps")
        if i == 0:
            ax.set_ylabel("Effective Rank")
        ax.grid(True, alpha=0.3)
        for regime in regimes:
            key = f"{regime}_H{h}"
            if key in all_results:
                d = all_results[key]
                diag = d["diagnostics"]
                if diag["effective_rank"]:
                    steps = [er["step"] for er in diag["effective_rank"]]
                    ranks = [er["effective_rank"] for er in diag["effective_rank"]]
                    ax.plot(steps, ranks, label=d["display_name"],
                            color=d["color"], linewidth=2, marker='^',
                            markersize=4, alpha=0.8)
        ax.legend(fontsize=8)
    plt.suptitle("Effective Gradient Rank (higher = richer signal)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("exp1_effective_rank.png", dpi=200, bbox_inches="tight")
    print("Saved: exp1_effective_rank.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 1: EBM Differentiability & Credit Assignment")
    print(f"  Pure EBM — Langevin Sampling — Three Gradient-Flow Modes")
    print(f"  Only variable: EBM scoring gradient path (diff_mode)")
    print(f"{'#'*70}\n")

    for f in ["pretrained_ebm_pendulum.pth"]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found. Run pretrain_world_models.py first!")
            return

    regimes = REGIME_LIST
    horizons = [1, 3, 5]

    all_results = {}
    for horizon in horizons:
        for regime in regimes:
            key = f"{regime}_H{horizon}"
            try:
                t0 = time.time()
                all_results[key] = train_regime(regime, horizon, seed=42)
                elapsed = time.time() - t0
                print(f"\n  Completed {key} in {elapsed:.0f}s\n")
            except Exception as e:
                print(f"\n  !!! FAILED: {key} -- {e}")
                import traceback
                traceback.print_exc()

    print("\n--- Generating plots ---")
    plot_convergence(all_results, horizons, regimes)
    plot_gradient_norms(all_results, horizons, regimes)
    plot_gradient_variance(all_results, horizons, regimes)
    plot_spectral_analysis(all_results, horizons, regimes)
    plot_effective_rank(all_results, horizons, regimes)

    np.save("exp1_results.npy", all_results, allow_pickle=True)
    print("\nSaved: exp1_results.npy")
    print("\nExperiment 1 complete!")


if __name__ == "__main__":
    main()