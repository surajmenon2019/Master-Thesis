"""
Horizon Comparison: Explicit vs EBM World Models

Compares 5 world model configurations across 4 horizons (H=1,3,5,10):
  1. MDN (Explicit)       — clean gradient, single forward pass
  2. EBM Detached         — no gradient through model
  3. EBM E2E              — truly differentiable Langevin (fixed)
  4. EBM Warm Start       — flow init + 5 differentiable Langevin steps
  5. EBM SVGD             — differentiable SVGD (already correct)

Usage:
    python horizon_comparison.py
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
import json

# --- IMPORTS ---
# Ensure project root is on sys.path for local module imports.
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _model_path(filename):
    return os.path.join(SCRIPT_DIR, filename)

try:
    import minigrid
    import gymnasium as gym
    from models import BilinearEBM, RealNVP, MixtureDensityNetwork, ValueNetwork, RewardModel
    from models_minigrid import DiscreteActor
    from minigrid_adapter import MiniGridAdapter
    from utils_sampling import (
        predict_next_state_langevin_adaptive,
        predict_next_state_svgd_adaptive
    )
    from pretrain_ebm_minigrid import infonce_loss
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "ENV_NAME": "MiniGrid-Empty-8x8-v0",
    "SEED": 42,

    # Experiment matrix
    "CONFIGS": ["MDN", "EBM Detached", "EBM E2E", "EBM Warm Start", "EBM SVGD"],
    "HORIZONS": [1, 5, 10],

    # Training
    "TOTAL_STEPS": 30000,
    "BATCH_SIZE": 256,
    "DISCOUNT": 0.99,
    "LAMBDA": 0.95,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-3,
    "LR_REWARD": 1e-3,
    "ENTROPY_COEFF": 0.01,
    "UPDATE_FREQ": 10,
    "IMAGINATION_MIN_STATE": 0.0,
    "IMAGINATION_MAX_STATE": 1.0,

    # Sampling
    "LANGEVIN_STEPS_COLD": 30,
    "LANGEVIN_STEPS_WARM": 5,
    "LANGEVIN_STEP_SIZE": 0.05,
    "LANGEVIN_NOISE_SCALE": 0.01,
    "SVGD_STEPS": 10,
    "SVGD_PARTICLES": 10,

    # Evaluation
    "EVAL_INTERVAL": 500,
    "EVAL_EPISODES": 5,

    # System
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "HIDDEN_DIM": 128,

    # Warmup
    "WARMUP_TARGET_POSITIVES": 10,

    # Output
    "SAVE_DIR": "results_horizon_comparison",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_lambda_values(next_values, rewards, dones, discount, lambda_):
    batch_size, horizon = rewards.shape
    v_lambda = next_values[:, -1] * (1.0 - dones[:, -1])
    lambda_values = torch.zeros_like(rewards)
    for t in reversed(range(horizon)):
        v_lambda = rewards[:, t] + (1.0 - dones[:, t]) * discount * (
            (1.0 - lambda_) * next_values[:, t] + lambda_ * v_lambda
        )
        lambda_values[:, t] = v_lambda
    return lambda_values


def to_one_hot(action_idx, num_actions):
    vec = np.zeros(num_actions, dtype=np.float32)
    vec[action_idx] = 1.0
    return vec


# =============================================================================
# TRAJECTORY BUFFER
# =============================================================================

class TrajectoryBuffer:
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
        self.positive_indices = []

    def add_transition(self, state, action, reward, next_state):
        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        if reward > 0.0001:
            if self.ptr not in self.positive_indices:
                self.positive_indices.append(self.ptr)
        else:
            if self.ptr in self.positive_indices:
                self.positive_indices.remove(self.ptr)
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
        states_list = []
        num_pos = len(self.positive_indices)
        if num_pos > 0:
            n_pos = min(batch_size // 4, num_pos * 2)  # 25% cap, limit oversampling
            idx_pos = np.random.choice(self.positive_indices, size=n_pos, replace=True)
            states_list.append(self.states[idx_pos])
            n_remaining = batch_size - n_pos
        else:
            n_remaining = batch_size
        if len(self.recent_trajectories) > 0:
            recent_states = []
            for _ in range(n_remaining):
                traj = self.recent_trajectories[np.random.randint(0, len(self.recent_trajectories))]
                s = traj['states'][np.random.randint(0, len(traj['states']))]
                recent_states.append(s)
            states_list.append(np.array(recent_states))
        else:
            idx = np.random.randint(0, self.size, size=n_remaining)
            states_list.append(self.states[idx])
        batch_states = np.concatenate(states_list, axis=0)
        return torch.tensor(batch_states, dtype=torch.float32).to(device)

    def sample_for_reward_training(self, batch_size, device):
        if self.size == 0:
            return None
        num_pos = len(self.positive_indices)
        if num_pos > 0:
            pos_batch = batch_size // 4
            pos_idx = np.random.choice(self.positive_indices, size=pos_batch, replace=True)
            rand_idx = np.random.randint(0, self.size, size=batch_size - pos_batch)
            idx = np.concatenate([pos_idx, rand_idx])
            np.random.shuffle(idx)
        else:
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
        new_buf.positive_indices = list(self.positive_indices)
        new_buf.recent_trajectories = [
            {k: v.copy() for k, v in traj.items()}
            for traj in self.recent_trajectories
        ]
        return new_buf


# =============================================================================
# WORLD MODEL AGENT
# =============================================================================

class WorldModelAgent:
    """
    Unified world model agent supporting 5 configurations:
      - MDN: explicit model, fully differentiable forward pass
      - EBM Detached: Langevin sampling, gradients detached
      - EBM E2E: Langevin sampling, truly differentiable (fixed)
      - EBM Warm Start: Flow init + short differentiable Langevin
      - EBM SVGD: SVGD sampling, naturally differentiable
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
                state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]
            ).to(device)
            self.mdn.load_state_dict(torch.load(
                _model_path(f"pretrained_mdn_{env_name}.pth"),
                map_location=device, weights_only=False
            ))
            self.mdn.eval()
            for p in self.mdn.parameters():
                p.requires_grad = False
        else:
            # All EBM variants share the same pretrained EBM
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
        """
        Predict next state given current state and action.
        Returns a tensor with appropriate gradient connectivity.
        """
        if self.config_name == "MDN":
            # Explicit model: single forward pass, fully differentiable
            return self.mdn.sample_differentiable(state, action)

        elif self.config_name == "EBM Detached":
            # Langevin sampling, output detached (model as black-box)
            ns = predict_next_state_langevin_adaptive(
                self.ebm, state, action,
                use_ascent=True,
                config={
                    "LANGEVIN_STEPS": CONFIG["LANGEVIN_STEPS_COLD"],
                    "LANGEVIN_STEP_SIZE": CONFIG["LANGEVIN_STEP_SIZE"],
                    "LANGEVIN_NOISE_SCALE": CONFIG["LANGEVIN_NOISE_SCALE"]
                },
                differentiable=False  # detached
            )
            return ns.detach()  # explicitly detach the output too

        elif self.config_name == "EBM E2E":
            # Truly differentiable Langevin (FIXED)
            return predict_next_state_langevin_adaptive(
                self.ebm, state, action,
                use_ascent=True,
                config={
                    "LANGEVIN_STEPS": CONFIG["LANGEVIN_STEPS_COLD"],
                    "LANGEVIN_STEP_SIZE": CONFIG["LANGEVIN_STEP_SIZE"],
                    "LANGEVIN_NOISE_SCALE": CONFIG["LANGEVIN_NOISE_SCALE"]
                },
                differentiable=True  # FIXED: truly E2E
            )

        elif self.config_name == "EBM Warm Start":
            # Flow init + short differentiable Langevin refinement
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
                differentiable=True  # E2E through short chain
            )

        elif self.config_name == "EBM SVGD":
            # SVGD: naturally differentiable (no detach bug)
            return predict_next_state_svgd_adaptive(
                self.ebm, state, action,
                use_ascent=True,
                config={
                    "SVGD_STEPS": CONFIG["SVGD_STEPS"],
                    "SVGD_PARTICLES": CONFIG["SVGD_PARTICLES"],
                    "SVGD_STEP_SIZE": CONFIG["LANGEVIN_STEP_SIZE"]
                }
            )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_policy(env_adapter, actor, num_episodes, device):
    episode_rewards = []
    for _ in range(num_episodes):
        state = env_adapter.reset()
        done = False
        episode_reward = 0
        steps = 0
        while not done:
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_vec, _ = actor.sample(st, hard=False)
                action_vec = action_vec.cpu().numpy()[0]
            next_state, reward, done, info = env_adapter.step(action_vec)
            episode_reward += reward
            state = next_state
            steps += 1
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)


# =============================================================================
# ONLINE EBM UPDATE
# =============================================================================

def update_ebm_online(ebm, buffer, optimizer, num_steps=50, batch_size=32):
    ebm.train()
    if buffer.size < batch_size:
        return
    for _ in range(num_steps):
        data = buffer.sample(batch_size)
        if data is None:
            return
        s, a, real_ns = data
        loss, _ = infonce_loss(ebm, s, a, real_ns, buffer, num_negatives=32, temperature=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# =============================================================================
# CORE: TRAIN SINGLE RUN
# =============================================================================

def train_single_run(config_name, horizon, env_adapter, warmup_buffer, run_seed):
    """Train one (config, horizon) pair."""
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    device = CONFIG["DEVICE"]
    state_dim = env_adapter.state_dim
    action_dim = env_adapter.action_dim

    # Clone buffer so each run starts from same warmup data
    state_buffer = warmup_buffer.clone()

    # World model
    agent = WorldModelAgent(config_name, CONFIG["ENV_NAME"], state_dim, action_dim, device)

    # Policy networks (fresh for each run)
    actor = DiscreteActor(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic = ValueNetwork(state_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target = ValueNetwork(state_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False
    reward_model = RewardModel(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=CONFIG["LR_ACTOR"])
    critic_opt = optim.Adam(critic.parameters(), lr=CONFIG["LR_CRITIC"])
    reward_opt = optim.Adam(reward_model.parameters(), lr=CONFIG["LR_REWARD"])

    ebm_opt = None
    if agent.ebm is not None:
        ebm_opt = optim.Adam(agent.ebm.parameters(), lr=1e-4)

    # Tracking
    eval_steps, eval_rewards = [], []
    grad_norms, grad_norm_steps = [], []
    recent_grad_norms = []

    total_steps = 0
    episode_count = 0
    state = env_adapter.reset()


    while total_steps < CONFIG["TOTAL_STEPS"]:
        # --- 1. Real Environment Interaction ---
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_vec_tensor, _ = actor.sample(st, hard=True)
            action_vec = action_vec_tensor.cpu().numpy()[0]

        next_state, reward, done, info = env_adapter.step(action_vec)
        action_id = np.argmax(action_vec)
        action_onehot = to_one_hot(action_id, action_dim)
        state_buffer.add_transition(state, action_onehot, reward, next_state)
        total_steps += 1

        if done:
            state_buffer.finish_trajectory()
            episode_count += 1
            # Online EBM update
            if ebm_opt and episode_count % 5 == 0 and state_buffer.size >= 500:
                update_ebm_online(agent.ebm, state_buffer, ebm_opt)
            state = env_adapter.reset()
        else:
            state = next_state

        # --- 2. Imagined Rollout & Policy Update ---
        if state_buffer.size >= CONFIG["BATCH_SIZE"] and total_steps % CONFIG["UPDATE_FREQ"] == 0:
            # Reward model update
            batch = state_buffer.sample_for_reward_training(CONFIG["BATCH_SIZE"], device)
            if batch:
                pred_r = reward_model(batch['states'], batch['actions'])
                loss_r = F.smooth_l1_loss(pred_r, batch['rewards'])
                reward_opt.zero_grad()
                loss_r.backward()
                reward_opt.step()

            # Imagination rollout
            initial_states = state_buffer.sample_imagination_states(CONFIG["BATCH_SIZE"], device)
            curr = initial_states
            states_list, rewards_list, next_states_list = [], [], []

            for t in range(horizon):
                curr_input = curr
                a_soft, logits = actor.sample(curr, hard=True)
                dist = torch.distributions.Categorical(logits=logits)
                entropy = dist.entropy().unsqueeze(-1)

                r_pred = reward_model(curr_input, a_soft).clamp(0.0, 1.0)
                ns_pred = agent.predict_next_state(curr_input, a_soft).clamp(
                    CONFIG["IMAGINATION_MIN_STATE"],
                    CONFIG["IMAGINATION_MAX_STATE"]
                )

                rollout_reward = r_pred + CONFIG["ENTROPY_COEFF"] * entropy

                curr = ns_pred

                states_list.append(curr_input)
                rewards_list.append(rollout_reward)
                next_states_list.append(ns_pred)

            states_seq = torch.stack(states_list, dim=1)
            rewards_seq = torch.stack(rewards_list, dim=1)
            next_states_seq = torch.stack(next_states_list, dim=1)
            dones = torch.zeros(CONFIG["BATCH_SIZE"], horizon, 1, device=device)
            dones[:, -1, :] = 1.0

            # --- Critic Update ---
            flat_next_states = next_states_seq.reshape(-1, state_dim)
            with torch.no_grad():
                flat_next_values = critic_target(flat_next_states)
                next_values = flat_next_values.reshape(CONFIG["BATCH_SIZE"], horizon, 1)

            lambda_targets = compute_lambda_values(
                next_values.squeeze(-1),
                rewards_seq.squeeze(-1),
                dones.squeeze(-1),
                CONFIG["DISCOUNT"],
                CONFIG["LAMBDA"]
            ).unsqueeze(-1)

            flat_curr_states = states_seq.reshape(-1, state_dim).detach()
            v_values = critic(flat_curr_states).reshape(CONFIG["BATCH_SIZE"], horizon, 1)
            critic_loss = F.mse_loss(v_values, lambda_targets.detach())
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # --- Actor Update (Pathwise Policy Gradient) ---
            flat_ns_grad = next_states_seq.reshape(-1, state_dim)
            v_next_pred = critic_target(flat_ns_grad).reshape(CONFIG["BATCH_SIZE"], horizon, 1)

            actor_objective = rewards_seq + CONFIG["DISCOUNT"] * v_next_pred
            actor_loss = -actor_objective.mean()
            actor_opt.zero_grad()
            actor_loss.backward()

            # Record gradient norm
            total_norm = 0.0
            for p in actor.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            recent_grad_norms.append(total_norm)

            actor_opt.step()

            # Soft target update
            for p, tp in zip(critic.parameters(), critic_target.parameters()):
                tp.data.copy_(0.005 * p.data + 0.995 * tp.data)

        # --- 3. Evaluation ---
        if total_steps % CONFIG["EVAL_INTERVAL"] == 0:
            eval_mean = evaluate_policy(env_adapter, actor, CONFIG["EVAL_EPISODES"], device)
            eval_steps.append(total_steps)
            eval_rewards.append(eval_mean)

            if len(recent_grad_norms) > 0:
                avg_gnorm = np.mean(recent_grad_norms)
                grad_norms.append(avg_gnorm)
                grad_norm_steps.append(total_steps)
                recent_grad_norms = []

            gn_str = f" | GradNorm: {grad_norms[-1]:.4f}" if grad_norms else ""
            print(f"    Step {total_steps:>5d} | Eval: {eval_mean:.3f}{gn_str}")

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

    colors = {
        "MDN": "#00CC96",                # Green (explicit baseline)
        "EBM Detached": "#636EFA",       # Blue
        "EBM E2E": "#EF553B",           # Red
        "EBM Warm Start": "#FFA15A",     # Orange
        "EBM SVGD": "#AB63FA",          # Purple
    }
    linestyles = {
        "MDN": "-",
        "EBM Detached": "--",
        "EBM E2E": "-.",
        "EBM Warm Start": "-",
        "EBM SVGD": ":",
    }

    smooth_window = 5

    # =========================================================================
    # FIGURE 1: Policy Convergence
    # =========================================================================
    fig1, axes1 = plt.subplots(1, n_h, figsize=(5.5 * n_h, 5.5), sharey=True)
    fig1.suptitle(
        "Policy Convergence: Explicit vs EBM World Models Across Horizons",
        fontsize=15, fontweight='bold', y=1.02
    )

    for i, H in enumerate(horizons):
        ax = axes1[i]
        for config_name in configs:
            key = f"{config_name}_H{H}"
            if key not in all_results:
                continue
            r = all_results[key]
            steps = r['eval_steps']
            rewards = r['eval_rewards']

            ax.scatter(steps, rewards, color=colors[config_name], alpha=0.10, s=8, zorder=1)

            smoothed, offset = smooth_curve(rewards, smooth_window)
            smooth_steps = steps[offset:]
            ax.plot(smooth_steps, smoothed, color=colors[config_name],
                    linestyle=linestyles[config_name], linewidth=2.5,
                    label=config_name, zorder=2)

        ax.set_title(f"Horizon = {H}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Training Steps", fontsize=11)
        if i == 0:
            ax.set_ylabel("Eval Reward", fontsize=11)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax.set_ylim(-0.05, 1.0)
        ax.tick_params(labelsize=9)

    fig1.tight_layout()
    path1 = os.path.join(save_dir, "convergence_comparison.png")
    fig1.savefig(path1, dpi=200, bbox_inches='tight')
    print(f"Saved: {path1}")
    plt.close(fig1)

    # =========================================================================
    # FIGURE 2: Gradient Norms
    # =========================================================================
    fig2, axes2 = plt.subplots(1, n_h, figsize=(5.5 * n_h, 5.5), sharey=False)
    fig2.suptitle(
        "Actor Gradient Norms: Explicit vs EBM World Models",
        fontsize=15, fontweight='bold', y=1.02
    )

    for i, H in enumerate(horizons):
        ax = axes2[i]
        for config_name in configs:
            key = f"{config_name}_H{H}"
            if key not in all_results:
                continue
            r = all_results[key]
            gn_steps = r['grad_norm_steps']
            gn = r['grad_norms']
            if len(gn) == 0:
                continue

            ax.scatter(gn_steps, gn, color=colors[config_name], alpha=0.10, s=8, zorder=1)
            smoothed, offset = smooth_curve(gn, smooth_window)
            smooth_steps = gn_steps[offset:]
            ax.plot(smooth_steps, smoothed, color=colors[config_name],
                    linestyle=linestyles[config_name], linewidth=2.5,
                    label=config_name, zorder=2)

        ax.set_title(f"Horizon = {H}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Training Steps", fontsize=11)
        if i == 0:
            ax.set_ylabel("Gradient Norm", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax.tick_params(labelsize=9)

    fig2.tight_layout()
    path2 = os.path.join(save_dir, "gradient_norms_comparison.png")
    fig2.savefig(path2, dpi=200, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close(fig2)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    device = CONFIG["DEVICE"]
    save_dir = CONFIG["SAVE_DIR"]
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  HORIZON COMPARISON EXPERIMENT")
    print(f"  Environment: {CONFIG['ENV_NAME']}")
    print(f"  Configs: {CONFIG['CONFIGS']}")
    print(f"  Horizons: {CONFIG['HORIZONS']}")
    print(f"  Steps/run: {CONFIG['TOTAL_STEPS']}")
    print(f"  Total runs: {len(CONFIG['CONFIGS']) * len(CONFIG['HORIZONS'])}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    env_adapter = MiniGridAdapter(CONFIG['ENV_NAME'])

    # --- Warmup Buffer (shared across all runs) ---
    print("Collecting warmup data...")
    np.random.seed(CONFIG["SEED"])
    warmup_buffer = TrajectoryBuffer(env_adapter.state_dim, env_adapter.action_dim)
    s = env_adapter.reset()
    steps_collected = 0
    while len(warmup_buffer.positive_indices) < CONFIG["WARMUP_TARGET_POSITIVES"]:
        a_int = np.random.randint(0, env_adapter.action_dim)
        ns, r, d, _ = env_adapter.step(a_int)
        warmup_buffer.add_transition(s, to_one_hot(a_int, env_adapter.action_dim), r, ns)
        steps_collected += 1
        if d:
            warmup_buffer.finish_trajectory()
            s = env_adapter.reset()
        else:
            s = ns
    print(f"  Warmup done: {steps_collected} steps, {len(warmup_buffer.positive_indices)} positives\n")

    # --- Run All Configurations ---
    all_results = {}
    run_idx = 0
    total_runs = len(CONFIG["CONFIGS"]) * len(CONFIG["HORIZONS"])

    for config_name in CONFIG["CONFIGS"]:
        for horizon in CONFIG["HORIZONS"]:
            run_idx += 1
            print(f"\n{'='*50}")
            print(f"  Run {run_idx}/{total_runs}: {config_name}, H={horizon}")
            print(f"{'='*50}")

            t0 = time.time()
            result = train_single_run(config_name, horizon, env_adapter, warmup_buffer, CONFIG["SEED"])
            elapsed = time.time() - t0

            key = f"{config_name}_H{horizon}"
            all_results[key] = result

            final_reward = result['eval_rewards'][-1] if len(result['eval_rewards']) > 0 else 0
            print(f"  Done in {elapsed:.1f}s. Final eval: {final_reward:.3f}")

            # Save incrementally
            np.save(os.path.join(save_dir, "all_results.npy"), all_results, allow_pickle=True)

    # --- Plot ---
    print("\n\nGenerating figures...")
    plot_results(all_results, save_dir)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Results: {save_dir}/")
    print(f"  - convergence_comparison.png")
    print(f"  - gradient_norms_comparison.png")
    print(f"  - all_results.npy")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
