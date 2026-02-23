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
    "SEED": 42,

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
                action = actor.sample(st)
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
                # Sample continuous action from actor
                a_cont = actor.sample(curr)  # (B, action_dim)

                # Entropy for continuous policy
                mu, log_std = actor.forward(curr)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mu, std)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)

                r_pred = reward_model(curr, a_cont)
                ns_pred = agent.predict_next_state(curr, a_cont)

                states_list.append(curr)
                rewards_list.append(r_pred + CONFIG["ENTROPY_COEFF"] * entropy)
                next_states_list.append(ns_pred)
                curr = ns_pred

            states_seq = torch.stack(states_list, dim=1)
            rewards_seq = torch.stack(rewards_list, dim=1)
            next_states_seq = torch.stack(next_states_list, dim=1)
            dones = torch.zeros(CONFIG["BATCH_SIZE"], horizon, 1, device=device)
            dones[:, -1, :] = 1.0

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

            # Track gradient norms
            total_norm = 0.0
            for p in actor.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            recent_grad_norms.append(total_norm)

            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
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

    # FIGURE 1: Policy Convergence
    fig1, axes1 = plt.subplots(1, n_h, figsize=(6 * n_h, 5.5), sharey=True)
    if n_h == 1:
        axes1 = [axes1]
    fig1.suptitle(
        f"Continuous Multimodal Env (slip={CONFIG['SLIP_PROB']}): "
        f"Explicit vs EBM Policy Convergence",
        fontsize=14, fontweight='bold', y=1.02
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

            ax.scatter(steps, rewards, color=colors[config_name],
                       alpha=0.10, s=8, zorder=1)
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
        ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
        ax.tick_params(labelsize=9)

    fig1.tight_layout()
    path1 = os.path.join(save_dir, "continuous_stochastic_convergence.png")
    fig1.savefig(path1, dpi=200, bbox_inches='tight')
    print(f"Saved: {path1}")
    plt.close(fig1)

    # FIGURE 2: Gradient Norms
    fig2, axes2 = plt.subplots(1, n_h, figsize=(6 * n_h, 5.5), sharey=False)
    if n_h == 1:
        axes2 = [axes2]
    fig2.suptitle(
        f"Continuous Multimodal Env (slip={CONFIG['SLIP_PROB']}): "
        f"Actor Gradient Norms",
        fontsize=14, fontweight='bold', y=1.02
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

            ax.scatter(gn_steps, gn, color=colors[config_name],
                       alpha=0.10, s=8, zorder=1)
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

def main():
    device = CONFIG["DEVICE"]
    save_dir = os.path.join(SCRIPT_DIR, CONFIG["SAVE_DIR"])
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  CONTINUOUS MULTIMODAL STOCHASTIC EXPERIMENT")
    print(f"  Environment: {CONFIG['ENV_NAME']}")
    print(f"    slip_prob={CONFIG['SLIP_PROB']}, "
          f"deflection={CONFIG['DEFLECTION_ANGLE']}°, "
          f"obstacles={CONFIG['N_OBSTACLES']}")
    print(f"  Configs: {CONFIG['CONFIGS']}")
    print(f"  Horizons: {CONFIG['HORIZONS']}")
    print(f"  Steps/run: {CONFIG['TOTAL_STEPS']}")
    print(f"  Total runs: {len(CONFIG['CONFIGS']) * len(CONFIG['HORIZONS'])}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    # Create environment adapter
    env_adapter = MultimodalPointAdapter(
        slip_prob=CONFIG["SLIP_PROB"],
        deflection_angle=CONFIG["DEFLECTION_ANGLE"],
        n_obstacles=CONFIG["N_OBSTACLES"],
    )

    # Warmup: collect initial transitions with random policy
    print("Collecting warmup data...")
    np.random.seed(CONFIG["SEED"])
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
    print(f"  Warmup done: {warmup_buffer.size} transitions collected\n")

    # Run experiment matrix
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
            result = train_single_run(
                config_name, horizon, env_adapter,
                warmup_buffer, CONFIG["SEED"]
            )
            elapsed = time.time() - t0

            key = f"{config_name}_H{horizon}"
            all_results[key] = result

            final_reward = (result['eval_rewards'][-1]
                            if len(result['eval_rewards']) > 0 else 0)
            print(f"  Done in {elapsed:.1f}s. Final eval: {final_reward:.3f}")

            # Save intermediate
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
