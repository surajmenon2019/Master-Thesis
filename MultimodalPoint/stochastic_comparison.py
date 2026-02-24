"""
Stochastic Multimodality Experiment: Continuous Multimodal Point Environment

Compares 3 world model configurations:
  1. MDN (Explicit)       — mode-averages → unrealistic trajectories
  2. EBM E2E              — samples from true modes → faithful trajectories
  3. EBM Warm Start       — flow init + short Langevin → best of both worlds

Environment: MultimodalPoint-0.3 (slip_prob=0.3, deflection=90°, 3 obstacles)
Horizons:    H=1 (raw quality) and H=5 (compounding errors)

This experiment demonstrates EBM's representational advantage in a CONTINUOUS
environment with genuinely multimodal dynamics.

Usage:
    1. First run:  python pretrain_models.py
    2. Then run:   python stochastic_comparison.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# --- Path setup ---
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def _model_path(filename):
    return os.path.join(SCRIPT_DIR, filename)


# --- IMPORTS ---
try:
    from models import (
        Actor, BilinearEBM, RealNVP, MixtureDensityNetwork,
        ValueNetwork, RewardModel, Critic
    )
    from utils_sampling import predict_next_state_langevin_adaptive
    from multimodal_point_env import MultimodalPointEnv, MultimodalPointAdapter
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "ENV_NAME": "MultimodalPoint-0.3",
    "SLIP_PROB": 0.3,
    "DEFLECTION_ANGLE": 90.0,
    "N_OBSTACLES": 3,
    # Multiple seeds for statistical validity
    "SEEDS": [42, 123, 7],

    # Experiment matrix
    "CONFIGS": ["MDN", "EBM E2E", "EBM Warm Start"],
    "HORIZONS": [1, 5],

    # Training
    "TOTAL_STEPS": 15000,
    "BATCH_SIZE": 256,
    "DISCOUNT": 0.99,
    "LAMBDA": 0.95,
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 1e-3,
    "LR_REWARD": 1e-3,
    "ENTROPY_COEFF": 0.01,
    "UPDATE_FREQ": 10,

    # Gradient clipping — applied AFTER logging raw norm, preserving full
    # computation graph (E2E differentiability is unaffected; only caps magnitude).
    # Set to None to disable.
    "GRAD_CLIP_NORM": 1.0,

    # Sampling
    "LANGEVIN_STEPS_COLD": 30,
    "LANGEVIN_STEPS_WARM": 5,
    "LANGEVIN_STEP_SIZE": 0.05,
    "LANGEVIN_NOISE_SCALE": 0.01,

    # Evaluation
    "EVAL_INTERVAL": 500,
    "EVAL_EPISODES": 5,

    # System
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "HIDDEN_DIM": 128,

    # Warmup
    "WARMUP_STEPS": 3000,

    # Output
    "SAVE_DIR": "results_stochastic_comparison",
}


# =============================================================================
# HELPERS
# =============================================================================

def compute_lambda_values(next_values, rewards, dones, discount, lambda_):
    """GAE-style lambda returns."""
    batch_size, horizon = rewards.shape
    v_lambda = next_values[:, -1] * (1.0 - dones[:, -1])
    lambda_values = torch.zeros_like(rewards)
    for t in reversed(range(horizon)):
        v_lambda = rewards[:, t] + (1.0 - dones[:, t]) * discount * (
            (1.0 - lambda_) * next_values[:, t] + lambda_ * v_lambda
        )
        lambda_values[:, t] = v_lambda
    return lambda_values


def squashed_gaussian_entropy(actor, state, squashed_action, eps=1e-6):
        """
        Differential entropy proxy for tanh-squashed Gaussian policy.

        Uses change-of-variables:
            log pi(a|s) = log N(u; mu, sigma) - sum log(1 - tanh(u)^2)
            where a = tanh(u), u = atanh(a)
        and returns -log pi(a|s).
        """
        mu, log_std = actor.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        a = squashed_action.clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh = 0.5 * (torch.log1p(a) - torch.log1p(-a))

        log_prob_u = dist.log_prob(pre_tanh)
        log_det_jac = torch.log(1.0 - a.pow(2) + eps)
        log_prob_a = (log_prob_u - log_det_jac).sum(dim=-1, keepdim=True)
        return -log_prob_a


# =============================================================================
# TRAJECTORY BUFFER (continuous version)
# =============================================================================

class TrajectoryBuffer:
    """Buffer storing (s, a, r, s') transitions for continuous env."""

    def __init__(self, state_dim, action_dim, capacity=50000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.recent_trajectories = []
        self.max_recent_trajectories = 20
        self.current_trajectory = {'states': [], 'actions': [], 'rewards': []}

    def add_transition(self, state, action, reward, next_state):
        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def finish_trajectory(self):
        if len(self.current_trajectory['states']) == 0:
            return
        self.recent_trajectories.append({
            'states': np.array(self.current_trajectory['states']),
            'actions': np.array(self.current_trajectory['actions']),
            'rewards': np.array(self.current_trajectory['rewards'])
        })
        if len(self.recent_trajectories) > self.max_recent_trajectories:
            self.recent_trajectories.pop(0)
        self.current_trajectory = {'states': [], 'actions': [], 'rewards': []}

    def sample_imagination_states(self, batch_size, device):
        if self.size == 0:
            return torch.randn(batch_size, self.state_dim).to(device)
        # Mix recent and random
        states_list = []
        if len(self.recent_trajectories) > 0:
            n_recent = batch_size // 2
            recent_states = []
            for _ in range(n_recent):
                traj = self.recent_trajectories[
                    np.random.randint(0, len(self.recent_trajectories))
                ]
                s = traj['states'][np.random.randint(0, len(traj['states']))]
                recent_states.append(s)
            states_list.append(np.array(recent_states))
            n_remaining = batch_size - n_recent
        else:
            n_remaining = batch_size
        idx = np.random.randint(0, self.size, size=n_remaining)
        states_list.append(self.states[idx])
        batch_states = np.concatenate(states_list, axis=0)
        return torch.tensor(batch_states, dtype=torch.float32).to(device)

    def sample_for_reward_training(self, batch_size, device):
        if self.size == 0:
            return None
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'states': torch.tensor(self.states[idx], dtype=torch.float32).to(device),
            'actions': torch.tensor(self.actions[idx], dtype=torch.float32).to(device),
            'rewards': torch.tensor(self.rewards[idx], dtype=torch.float32).to(device)
        }

    def sample(self, batch_size):
        if self.size == 0:
            return None
        idx = np.random.randint(0, self.size, size=batch_size)
        device = CONFIG["DEVICE"]
        return (
            torch.tensor(self.states[idx], dtype=torch.float32).to(device),
            torch.tensor(self.actions[idx], dtype=torch.float32).to(device),
            torch.tensor(self.next_states[idx], dtype=torch.float32).to(device)
        )

    def clone(self):
        new_buf = TrajectoryBuffer(self.state_dim, self.action_dim, self.capacity)
        new_buf.states[:] = self.states[:]
        new_buf.actions[:] = self.actions[:]
        new_buf.rewards[:] = self.rewards[:]
        new_buf.next_states[:] = self.next_states[:]
        new_buf.ptr = self.ptr
        new_buf.size = self.size
        new_buf.recent_trajectories = [
            {k: v.copy() for k, v in traj.items()}
            for traj in self.recent_trajectories
        ]
        return new_buf


# =============================================================================
# INFONCE LOSS (for online EBM updates)
# =============================================================================

def infonce_loss(ebm, state, action, pos_next_state, buffer,
                 num_negatives=32, temperature=0.1):
    B = state.shape[0]
    device = state.device

    E_pos = ebm(state, action, pos_next_state)

    neg_indices = np.random.randint(0, buffer.size, size=(B, num_negatives))
    neg_next_states = torch.tensor(
        buffer.next_states[neg_indices],
        dtype=torch.float32, device=device
    )

    state_exp = state.unsqueeze(1).expand(B, num_negatives, -1)
    action_exp = action.unsqueeze(1).expand(B, num_negatives, -1)
    E_neg = ebm(state_exp, action_exp, neg_next_states)

    logits = torch.cat([E_pos.unsqueeze(1), E_neg], dim=1) / temperature
    labels = torch.zeros(B, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels)

    metrics = {
        'E_pos': E_pos.mean().item(),
        'E_neg': E_neg.mean().item(),
        'E_gap': (E_pos.mean() - E_neg.mean()).item()
    }
    return loss, metrics


# =============================================================================
# WORLD MODEL AGENT (Continuous)
# =============================================================================

class WorldModelAgent:
    """
    World model agent for continuous multimodal environment.
    Loads pretrained EBM / Flow / MDN.
    """
    def __init__(self, config_name, env_name, state_dim, action_dim, device):
        self.config_name = config_name
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ebm = None
        self.flow = None
        self.mdn = None

        if config_name == "MDN":
            self.mdn = MixtureDensityNetwork(
                state_dim, action_dim, num_gaussians=5,
                hidden_dim=CONFIG["HIDDEN_DIM"]
            ).to(device)
            self.mdn.load_state_dict(torch.load(
                _model_path(f"pretrained_mdn_{env_name}.pth"),
                map_location=device, weights_only=False
            ))
            self.mdn.eval()
            for p in self.mdn.parameters():
                p.requires_grad = False
        else:
            # EBM variants
            self.ebm = BilinearEBM(
                state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]
            ).to(device)
            self.ebm.load_state_dict(torch.load(
                _model_path(f"pretrained_ebm_{env_name}.pth"),
                map_location=device, weights_only=False
            ))
            self.ebm.train()
            for p in self.ebm.parameters():
                p.requires_grad = True

            if config_name == "EBM Warm Start":
                self.flow = RealNVP(
                    data_dim=state_dim,
                    context_dim=state_dim + action_dim,
                    hidden_dim=CONFIG["HIDDEN_DIM"]
                ).to(device)
                self.flow.load_state_dict(torch.load(
                    _model_path(f"pretrained_flow_{env_name}_ForwardKL.pth"),
                    map_location=device, weights_only=False
                ))
                self.flow.eval()
                for p in self.flow.parameters():
                    p.requires_grad = False

    def predict_next_state(self, state, action):
        """Predict next state using the configured world model."""
        if self.config_name == "MDN":
            return self.mdn.sample_differentiable(state, action)

        elif self.config_name == "EBM E2E":
            return predict_next_state_langevin_adaptive(
                self.ebm, state, action,
                use_ascent=True,
                config={
                    "LANGEVIN_STEPS": CONFIG["LANGEVIN_STEPS_COLD"],
                    "LANGEVIN_STEP_SIZE": CONFIG["LANGEVIN_STEP_SIZE"],
                    "LANGEVIN_NOISE_SCALE": CONFIG["LANGEVIN_NOISE_SCALE"]
                },
                differentiable=True
            )

        elif self.config_name == "EBM Warm Start":
            z = torch.randn_like(state).to(self.device)
            context = torch.cat([state, action], dim=1)
            init = self.flow.sample(z, context=context)
            return predict_next_state_langevin_adaptive(
                self.ebm, state, action,
                init_state=init,
                use_ascent=True,
                config={
                    "LANGEVIN_STEPS": CONFIG["LANGEVIN_STEPS_WARM"],
                    "LANGEVIN_STEP_SIZE": CONFIG["LANGEVIN_STEP_SIZE"],
                    "LANGEVIN_NOISE_SCALE": CONFIG["LANGEVIN_NOISE_SCALE"]
                },
                differentiable=True
            )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_policy(env_adapter, actor, num_episodes, device):
    """Run actor in real environment, return mean reward."""
    episode_rewards = []
    for _ in range(num_episodes):
        state = env_adapter.reset()
        done = False
        episode_reward = 0
        steps = 0
        while not done and steps < 200:
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = actor.forward(st)
                action = torch.tanh(mu)
                action_np = action.cpu().numpy()[0]
            next_state, reward, done, info = env_adapter.step(action_np)
            episode_reward += reward
            state = next_state
            steps += 1
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)


# =============================================================================
# ONLINE EBM UPDATE
# =============================================================================

def update_ebm_online(ebm, buffer, optimizer, num_steps=50, batch_size=32):
    """Fine-tune EBM on newly collected transitions."""
    ebm.train()
    if buffer.size < batch_size:
        return
    for _ in range(num_steps):
        data = buffer.sample(batch_size)
        if data is None:
            return
        s, a, real_ns = data
        loss, _ = infonce_loss(ebm, s, a, real_ns, buffer,
                               num_negatives=32, temperature=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# =============================================================================
# CORE: TRAIN SINGLE RUN
# =============================================================================

def train_single_run(config_name, horizon, env_adapter, warmup_buffer, run_seed):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    device = CONFIG["DEVICE"]
    state_dim = env_adapter.state_dim
    action_dim = env_adapter.action_dim

    state_buffer = warmup_buffer.clone()
    agent = WorldModelAgent(config_name, CONFIG["ENV_NAME"],
                            state_dim, action_dim, device)

    # Continuous actor (Gaussian policy, not discrete)
    actor = Actor(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic = ValueNetwork(state_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target = ValueNetwork(state_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False
    reward_model = RewardModel(state_dim, action_dim,
                               hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=CONFIG["LR_ACTOR"])
    critic_opt = optim.Adam(critic.parameters(), lr=CONFIG["LR_CRITIC"])
    reward_opt = optim.Adam(reward_model.parameters(), lr=CONFIG["LR_REWARD"])

    ebm_opt = None
    if agent.ebm is not None:
        ebm_opt = optim.Adam(agent.ebm.parameters(), lr=1e-4)

    eval_steps, eval_rewards = [], []
    grad_norms, grad_norm_steps = [], []
    recent_grad_norms = []

    total_steps = 0
    episode_count = 0
    state = env_adapter.reset()

    obs_low = torch.tensor(
        env_adapter.env.observation_space.low, dtype=torch.float32, device=device
    )
    obs_high = torch.tensor(
        env_adapter.env.observation_space.high, dtype=torch.float32, device=device
    )

    while total_steps < CONFIG["TOTAL_STEPS"]:
        # 1. Real environment interaction
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor.sample(st)
            action_np = action.cpu().numpy()[0]

        next_state, reward, done, info = env_adapter.step(action_np)
        state_buffer.add_transition(state, action_np, reward, next_state)
        total_steps += 1

        if done:
            state_buffer.finish_trajectory()
            episode_count += 1
            # Periodic online EBM update
            if ebm_opt and episode_count % 5 == 0 and state_buffer.size >= 500:
                update_ebm_online(agent.ebm, state_buffer, ebm_opt)
            state = env_adapter.reset()
        else:
            state = next_state

        # 2. Model-based imagined update (pathwise gradients)
        if (state_buffer.size >= CONFIG["BATCH_SIZE"]
                and total_steps % CONFIG["UPDATE_FREQ"] == 0):

            # Train reward model on real transitions
            batch = state_buffer.sample_for_reward_training(
                CONFIG["BATCH_SIZE"], device
            )
            if batch:
                pred_r = reward_model(batch['states'], batch['actions'])
                loss_r = F.smooth_l1_loss(pred_r, batch['rewards'])
                reward_opt.zero_grad()
                loss_r.backward()
                reward_opt.step()

            # Imagined rollout
            initial_states = state_buffer.sample_imagination_states(
                CONFIG["BATCH_SIZE"], device
            )
            curr = initial_states
            states_list, rewards_list, next_states_list = [], [], []

            for t in range(horizon):
                # Clip actor output to env action space [-1, 1]
                # (env clips internally, but world model / reward model must
                #  also see in-distribution actions, so clip before passing in)
                a_cont = actor.sample(curr).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

                # Entropy bonus computed in the same squashed-action space
                # as the policy output to avoid objective mismatch.
                entropy = squashed_gaussian_entropy(actor, curr, a_cont)

                # Keep reward model in realistic environment scale.
                r_pred = reward_model(curr, a_cont).clamp(-2.0, 12.0)

                ns_pred = agent.predict_next_state(curr, a_cont)
                ns_pred = torch.max(torch.min(ns_pred, obs_high), obs_low)

                states_list.append(curr)
                rewards_list.append(r_pred + CONFIG["ENTROPY_COEFF"] * entropy)
                next_states_list.append(ns_pred)
                curr = ns_pred

            states_seq = torch.stack(states_list, dim=1)
            rewards_seq = torch.stack(rewards_list, dim=1)
            next_states_seq = torch.stack(next_states_list, dim=1)
            dones = torch.zeros(CONFIG["BATCH_SIZE"], horizon, 1, device=device)

            # Critic update (TD-lambda)
            flat_next_states = next_states_seq.reshape(-1, state_dim)
            with torch.no_grad():
                flat_next_values = critic_target(flat_next_states)
                next_values = flat_next_values.reshape(
                    CONFIG["BATCH_SIZE"], horizon, 1
                )

            lambda_targets = compute_lambda_values(
                next_values.squeeze(-1),
                rewards_seq.squeeze(-1),
                dones.squeeze(-1),
                CONFIG["DISCOUNT"],
                CONFIG["LAMBDA"]
            ).unsqueeze(-1)

            flat_curr_states = states_seq.reshape(-1, state_dim).detach()
            v_values = critic(flat_curr_states).reshape(
                CONFIG["BATCH_SIZE"], horizon, 1
            )
            critic_loss = F.mse_loss(v_values, lambda_targets.detach())
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # Actor update (pathwise gradient through world model)
            flat_ns_grad = next_states_seq.reshape(-1, state_dim)
            v_next_pred = critic_target(flat_ns_grad).reshape(
                CONFIG["BATCH_SIZE"], horizon, 1
            )

            actor_objective = rewards_seq + CONFIG["DISCOUNT"] * v_next_pred
            actor_loss = -actor_objective.mean()
            actor_opt.zero_grad()
            actor_loss.backward()

            # Record raw gradient norm BEFORE clipping (diagnostic signal).
            # Clipping is applied after recording so the logged norm reflects
            # the true gradient magnitude the world model produced — not the
            # clipped version. The E2E computation graph is fully intact;
            # clipping only caps gradient magnitude to prevent explosion.
            total_norm = 0.0
            for p in actor.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            recent_grad_norms.append(total_norm)

            if CONFIG["GRAD_CLIP_NORM"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=CONFIG["GRAD_CLIP_NORM"]
                )
            actor_opt.step()

            # Soft target update
            for p, tp in zip(critic.parameters(), critic_target.parameters()):
                tp.data.copy_(0.005 * p.data + 0.995 * tp.data)

        # 3. Periodic evaluation
        if total_steps % CONFIG["EVAL_INTERVAL"] == 0:
            eval_mean = evaluate_policy(
                env_adapter, actor, CONFIG["EVAL_EPISODES"], device
            )
            eval_steps.append(total_steps)
            eval_rewards.append(eval_mean)

            if len(recent_grad_norms) > 0:
                avg_gnorm = np.mean(recent_grad_norms)
                grad_norms.append(avg_gnorm)
                grad_norm_steps.append(total_steps)
                recent_grad_norms = []

            gn_str = (f" | GradNorm: {grad_norms[-1]:.4f}"
                      if grad_norms else "")
            print(f"    Step {total_steps:>5d} | "
                  f"Eval: {eval_mean:.3f}{gn_str}")

    return {
        'eval_steps': np.array(eval_steps),
        'eval_rewards': np.array(eval_rewards),
        'grad_norms': np.array(grad_norms),
        'grad_norm_steps': np.array(grad_norm_steps),
    }


# =============================================================================
# PLOTTING
# =============================================================================

def smooth_curve(y, window=5):
    if len(y) < window:
        return y, 0
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode='valid')
    return smoothed, len(y) - len(smoothed)


def plot_results(all_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    horizons = CONFIG["HORIZONS"]
    configs = CONFIG["CONFIGS"]
    n_h = len(horizons)
    n_seeds = len(CONFIG["SEEDS"])

    colors = {
        "MDN": "#00CC96",            # Green (explicit)
        "EBM E2E": "#EF553B",        # Red
        "EBM Warm Start": "#FFA15A", # Orange
    }
    linestyles = {
        "MDN": "-",
        "EBM E2E": "-.",
        "EBM Warm Start": "--",
    }

    smooth_window = 5

    # FIGURE 1: Policy Convergence (mean ± std across seeds)
    fig1, axes1 = plt.subplots(1, n_h, figsize=(6 * n_h, 5.5), sharey=True)
    if n_h == 1:
        axes1 = [axes1]
    fig1.suptitle(
        f"Continuous Multimodal Env (slip={CONFIG['SLIP_PROB']}): "
        f"Explicit vs EBM Policy Convergence"
        f" (mean \u00b1 std, {n_seeds} seeds, clip={CONFIG['GRAD_CLIP_NORM']})",
        fontsize=13, fontweight='bold', y=1.02
    )

    for i, H in enumerate(horizons):
        ax = axes1[i]
        for config_name in configs:
            agg = aggregate_seeds(all_results, config_name, H)
            if agg is None:
                continue
            steps = agg['steps']
            mean_r = agg['reward_mean']
            std_r = agg['reward_std']

            smoothed, offset = smooth_curve(mean_r, smooth_window)
            smooth_steps = steps[offset:]
            smooth_std = std_r[offset:]

            c = colors[config_name]
            ax.plot(smooth_steps, smoothed, color=c,
                    linestyle=linestyles[config_name], linewidth=2.5,
                    label=config_name, zorder=2)
            ax.fill_between(smooth_steps,
                            smoothed - smooth_std,
                            smoothed + smooth_std,
                            color=c, alpha=0.15, zorder=1)

        ax.set_title(f"Horizon = {H}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Training Steps", fontsize=11)
        if i == 0:
            ax.set_ylabel("Eval Reward", fontsize=11)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
        ax.tick_params(labelsize=9)

    fig1.tight_layout()
    path1 = os.path.join(save_dir, "continuous_stochastic_convergence.png")
    fig1.savefig(path1, dpi=200, bbox_inches='tight')
    print(f"Saved: {path1}")
    plt.close(fig1)

    # FIGURE 2: Gradient Norms (mean ± std, pre-clip, log scale)
    fig2, axes2 = plt.subplots(1, n_h, figsize=(6 * n_h, 5.5), sharey=False)
    if n_h == 1:
        axes2 = [axes2]
    fig2.suptitle(
        f"Continuous Multimodal Env (slip={CONFIG['SLIP_PROB']}): "
        f"Actor Gradient Norms (raw, pre-clip)"
        f" (mean \u00b1 std, {n_seeds} seeds)",
        fontsize=13, fontweight='bold', y=1.02
    )

    for i, H in enumerate(horizons):
        ax = axes2[i]
        for config_name in configs:
            agg = aggregate_seeds(all_results, config_name, H)
            if agg is None or len(agg['gnorm_mean']) == 0:
                continue
            steps = agg['steps']
            mean_gn = agg['gnorm_mean']
            std_gn = agg['gnorm_std']

            smoothed, offset = smooth_curve(mean_gn, smooth_window)
            smooth_steps = steps[offset:]
            smooth_std = std_gn[offset:]

            c = colors[config_name]
            ax.plot(smooth_steps, smoothed, color=c,
                    linestyle=linestyles[config_name], linewidth=2.5,
                    label=config_name, zorder=2)
            ax.fill_between(smooth_steps,
                            np.maximum(smoothed - smooth_std, 1e-6),
                            smoothed + smooth_std,
                            color=c, alpha=0.15, zorder=1)

        ax.set_title(f"Horizon = {H}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Training Steps", fontsize=11)
        if i == 0:
            ax.set_ylabel("Gradient Norm (pre-clip)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
        ax.tick_params(labelsize=9)

    fig2.tight_layout()
    path2 = os.path.join(save_dir, "continuous_stochastic_gradient_norms.png")
    fig2.savefig(path2, dpi=200, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close(fig2)


# =============================================================================
# MAIN
# =============================================================================

def aggregate_seeds(all_results, config_name, horizon):
    """
    Collect per-seed results for a (config_name, horizon) pair and return
    mean ± std arrays interpolated onto a common step grid.
    """
    seed_rewards, seed_gnorms = [], []
    ref_steps = None

    for seed in CONFIG["SEEDS"]:
        key = f"{config_name}_H{horizon}_seed{seed}"
        if key not in all_results:
            continue
        r = all_results[key]
        if len(r['eval_rewards']) == 0:
            continue
        steps = r['eval_steps']
        if ref_steps is None:
            ref_steps = steps
        seed_rewards.append(np.interp(ref_steps, steps, r['eval_rewards']))
        if len(r['grad_norms']) > 0:
            gn_steps = r['grad_norm_steps']
            seed_gnorms.append(np.interp(ref_steps, gn_steps, r['grad_norms']))

    if ref_steps is None or len(seed_rewards) == 0:
        return None

    rw = np.array(seed_rewards)
    gn = np.array(seed_gnorms) if seed_gnorms else None
    return {
        'steps': ref_steps,
        'reward_mean': rw.mean(axis=0),
        'reward_std': rw.std(axis=0),
        'gnorm_mean': gn.mean(axis=0) if gn is not None else np.array([]),
        'gnorm_std': gn.std(axis=0) if gn is not None else np.array([]),
    }


def main():
    device = CONFIG["DEVICE"]
    save_dir = os.path.join(SCRIPT_DIR, CONFIG["SAVE_DIR"])
    os.makedirs(save_dir, exist_ok=True)

    n_seeds = len(CONFIG["SEEDS"])
    total_runs = len(CONFIG["CONFIGS"]) * len(CONFIG["HORIZONS"]) * n_seeds

    print(f"\n{'='*70}")
    print(f"  CONTINUOUS MULTIMODAL STOCHASTIC EXPERIMENT")
    print(f"  Environment: {CONFIG['ENV_NAME']}")
    print(f"    slip_prob={CONFIG['SLIP_PROB']}, "
          f"deflection={CONFIG['DEFLECTION_ANGLE']}\u00b0, "
          f"obstacles={CONFIG['N_OBSTACLES']}")
    print(f"  Configs:     {CONFIG['CONFIGS']}")
    print(f"  Horizons:    {CONFIG['HORIZONS']}")
    print(f"  Seeds:       {CONFIG['SEEDS']}")
    print(f"  Steps/run:   {CONFIG['TOTAL_STEPS']}")
    print(f"  Total runs:  {total_runs}")
    print(f"  Grad clip:   {CONFIG['GRAD_CLIP_NORM']}")
    print(f"  Device:      {device}")
    print(f"{'='*70}\n")

    # Create environment adapter
    env_adapter = MultimodalPointAdapter(
        slip_prob=CONFIG["SLIP_PROB"],
        deflection_angle=CONFIG["DEFLECTION_ANGLE"],
        n_obstacles=CONFIG["N_OBSTACLES"],
    )

    # Warmup: collect initial transitions with random policy (first seed)
    print("Collecting warmup data...")
    np.random.seed(CONFIG["SEEDS"][0])
    warmup_buffer = TrajectoryBuffer(env_adapter.state_dim, env_adapter.action_dim)
    state = env_adapter.reset()
    for i in range(CONFIG["WARMUP_STEPS"]):
        action = env_adapter.env.action_space.sample()
        ns, r, d, info = env_adapter.step(action)
        warmup_buffer.add_transition(state, action, r, ns)
        if d:
            warmup_buffer.finish_trajectory()
            state = env_adapter.reset()
        else:
            state = ns
    warmup_buffer.finish_trajectory()
    print(f"  Warmup done: {warmup_buffer.size} transitions collected\n")

    # Run all (config, horizon, seed) combinations
    all_results = {}
    run_idx = 0

    for config_name in CONFIG["CONFIGS"]:
        for horizon in CONFIG["HORIZONS"]:
            for seed in CONFIG["SEEDS"]:
                run_idx += 1
                print(f"\n{'='*55}")
                print(f"  Run {run_idx}/{total_runs}: "
                      f"{config_name}, H={horizon}, seed={seed}")
                print(f"{'='*55}")

                t0 = time.time()
                result = train_single_run(
                    config_name, horizon, env_adapter,
                    warmup_buffer, seed
                )
                elapsed = time.time() - t0

                # Key includes seed so runs don't overwrite each other
                key = f"{config_name}_H{horizon}_seed{seed}"
                all_results[key] = result

                final_reward = (result['eval_rewards'][-1]
                                if len(result['eval_rewards']) > 0 else 0)
                print(f"  Done in {elapsed:.1f}s. Final eval: {final_reward:.3f}")

                # Save incrementally after every run
                np.save(os.path.join(save_dir, "all_results.npy"),
                        all_results, allow_pickle=True)

    # Generate plots
    print("\n\nGenerating figures...")
    plot_results(all_results, save_dir)

    print(f"\n{'='*70}")
    print(f"  CONTINUOUS STOCHASTIC EXPERIMENT COMPLETE")
    print(f"  Results: {save_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
