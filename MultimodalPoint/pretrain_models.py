"""
Pretrain World Models (EBM, Flow, MDN) on the Multimodal Point Environment

Collects transition data from random exploration and trains all three
world model types. Models are saved to the MultimodalPoint/ directory.

Usage:
    python pretrain_models.py
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from models import BilinearEBM, RealNVP, MixtureDensityNetwork, ValueNetwork, RewardModel
from multimodal_point_env import MultimodalPointEnv

# --- CONFIGURATION ---
CONFIG = {
    "ENV_NAME": "MultimodalPoint-0.3",
    "PRETRAIN_STEPS": 15000,
    "COLLECT_STEPS": 20000,
    "BATCH_SIZE": 256,
    "LR_EBM": 1e-4,
    "LR_FLOW": 1e-4,
    "LR_MDN": 1e-3,
    "LR_REWARD": 1e-3,
    "LR_VALUE": 1e-3,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "HIDDEN_DIM": 128,
    "NUM_NEGATIVES": 128,
    "TEMPERATURE": 0.1,
    "SLIP_PROB": 0.3,
    "DEFLECTION_ANGLE": 90.0,
    "N_OBSTACLES": 3,
}


# --- REPLAY BUFFER ---
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=100000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, s, a, ns, r=0.0):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.next_states[self.ptr] = ns
        self.rewards[self.ptr] = r
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        device = CONFIG["DEVICE"]
        return (
            torch.tensor(self.states[idx], dtype=torch.float32).to(device),
            torch.tensor(self.actions[idx], dtype=torch.float32).to(device),
            torch.tensor(self.next_states[idx], dtype=torch.float32).to(device),
        )

    def sample_with_rewards(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        device = CONFIG["DEVICE"]
        return (
            torch.tensor(self.states[idx], dtype=torch.float32).to(device),
            torch.tensor(self.actions[idx], dtype=torch.float32).to(device),
            torch.tensor(self.next_states[idx], dtype=torch.float32).to(device),
            torch.tensor(self.rewards[idx], dtype=torch.float32).to(device),
        )


# --- INFONCE LOSS ---
def infonce_loss(ebm, state, action, pos_next_state, buffer,
                 num_negatives=128, temperature=0.1):
    B = state.shape[0]
    device = state.device

    # Positive energy
    E_pos = ebm(state, action, pos_next_state)

    # Negative samples from buffer
    neg_indices = np.random.randint(0, buffer.size, size=(B, num_negatives))
    neg_next_states = torch.tensor(
        buffer.next_states[neg_indices],
        dtype=torch.float32,
        device=device
    )

    # Expand state/action
    state_exp = state.unsqueeze(1).expand(B, num_negatives, -1)
    action_exp = action.unsqueeze(1).expand(B, num_negatives, -1)

    # Negative energies
    E_neg = ebm(state_exp, action_exp, neg_next_states)

    # InfoNCE loss
    logits = torch.cat([E_pos.unsqueeze(1), E_neg], dim=1) / temperature
    labels = torch.zeros(B, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels)

    metrics = {
        'E_pos': E_pos.mean().item(),
        'E_neg': E_neg.mean().item(),
        'E_gap': (E_pos.mean() - E_neg.mean()).item()
    }
    return loss, metrics


def train_all_models():
    print(f"\n>>> PRETRAINING START: {CONFIG['ENV_NAME']}")
    print(f"    Device: {CONFIG['DEVICE']}")
    device = CONFIG["DEVICE"]

    # --- Environment ---
    env = MultimodalPointEnv(
        slip_prob=CONFIG["SLIP_PROB"],
        deflection_angle=CONFIG["DEFLECTION_ANGLE"],
        n_obstacles=CONFIG["N_OBSTACLES"],
    )
    state_dim = env.state_dim
    action_dim = env.action_dim
    print(f"    State dim: {state_dim}, Action dim: {action_dim}")

    # --- Replay Buffer ---
    buffer = ReplayBuffer(state_dim, action_dim)

    # --- Collect Transition Data (Random Exploration) ---
    print(f"\n>>> Collecting {CONFIG['COLLECT_STEPS']} transitions with random policy...")
    obs, _ = env.reset()
    episodes = 0
    total_reward = 0.0
    for i in range(CONFIG["COLLECT_STEPS"]):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, next_obs, reward)
        total_reward += reward

        if terminated or truncated:
            obs, _ = env.reset()
            episodes += 1
        else:
            obs = next_obs

    avg_reward = total_reward / max(episodes, 1)
    print(f"    Collected {buffer.size} transitions across {episodes} episodes")
    print(f"    Avg episode reward (random): {avg_reward:.2f}")

    # --- Initialize Models ---
    ebm = BilinearEBM(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    flow = RealNVP(data_dim=state_dim, context_dim=state_dim + action_dim,
                   hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    mdn = MixtureDensityNetwork(state_dim, action_dim, num_gaussians=5,
                                hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    reward_model = RewardModel(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    value_net = ValueNetwork(state_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)

    ebm_opt = optim.Adam(ebm.parameters(), lr=CONFIG["LR_EBM"])
    flow_opt = optim.Adam(flow.parameters(), lr=CONFIG["LR_FLOW"])
    mdn_opt = optim.Adam(mdn.parameters(), lr=CONFIG["LR_MDN"])
    reward_opt = optim.Adam(reward_model.parameters(), lr=CONFIG["LR_REWARD"])
    value_opt = optim.Adam(value_net.parameters(), lr=CONFIG["LR_VALUE"])

    # --- Training Loop ---
    print(f"\n>>> Training for {CONFIG['PRETRAIN_STEPS']} steps...")
    for step in range(CONFIG["PRETRAIN_STEPS"]):
        s, a, real_ns = buffer.sample(CONFIG["BATCH_SIZE"])
        context = torch.cat([s, a], dim=1)

        # A. TRAIN EBM (InfoNCE)
        ebm_loss, ebm_metrics = infonce_loss(
            ebm, s, a, real_ns, buffer,
            num_negatives=CONFIG["NUM_NEGATIVES"],
            temperature=CONFIG["TEMPERATURE"]
        )
        ebm_opt.zero_grad()
        ebm_loss.backward()
        ebm_opt.step()

        # B. TRAIN FLOW (Forward KL)
        noise = torch.rand_like(real_ns) * 0.01
        real_ns_noisy = real_ns + noise
        log_prob = flow.log_prob(real_ns_noisy, context=context)
        loss_flow = -log_prob.mean() / state_dim
        flow_opt.zero_grad()
        loss_flow.backward()
        flow_opt.step()

        # C. TRAIN MDN (NLL)
        pi, mu, sigma = mdn(s, a)
        target = real_ns.unsqueeze(1).expand_as(mu)
        dist = torch.distributions.Normal(mu, sigma)
        log_prob_components = dist.log_prob(target).sum(dim=-1)
        log_pi = torch.log_softmax(pi, dim=1)
        log_likelihood = torch.logsumexp(log_pi + log_prob_components, dim=1)
        loss_mdn = -log_likelihood.mean() / state_dim
        mdn_opt.zero_grad()
        loss_mdn.backward()
        mdn_opt.step()

        # D. TRAIN REWARD MODEL
        s_r, a_r, _, r_r = buffer.sample_with_rewards(CONFIG["BATCH_SIZE"])
        r_pred = reward_model(s_r, a_r)
        loss_reward = F.mse_loss(r_pred, r_r)
        reward_opt.zero_grad()
        loss_reward.backward()
        reward_opt.step()

        # E. TRAIN VALUE NETWORK (on collected states with reward as target)
        with torch.no_grad():
            v_target = reward_model(s_r, a_r)
        v_pred = value_net(s_r)
        loss_value = F.mse_loss(v_pred, v_target)
        value_opt.zero_grad()
        loss_value.backward()
        value_opt.step()

        if step % 1000 == 0:
            print(f"  Step {step:5d} | "
                  f"EBM: {ebm_loss.item():.4f} (Gap: {ebm_metrics['E_gap']:.2f}) | "
                  f"Flow: {loss_flow.item():.4f} | "
                  f"MDN: {loss_mdn.item():.4f} | "
                  f"Reward: {loss_reward.item():.4f} | "
                  f"Value: {loss_value.item():.4f}")

    # --- Save Models ---
    save_dir = SCRIPT_DIR
    tag = CONFIG["ENV_NAME"]
    paths = {
        "ebm": os.path.join(save_dir, f"pretrained_ebm_{tag}.pth"),
        "flow": os.path.join(save_dir, f"pretrained_flow_{tag}_ForwardKL.pth"),
        "mdn": os.path.join(save_dir, f"pretrained_mdn_{tag}.pth"),
        "reward": os.path.join(save_dir, f"pretrained_reward_{tag}.pth"),
        "value": os.path.join(save_dir, f"pretrained_value_{tag}.pth"),
    }
    torch.save(ebm.state_dict(), paths["ebm"])
    torch.save(flow.state_dict(), paths["flow"])
    torch.save(mdn.state_dict(), paths["mdn"])
    torch.save(reward_model.state_dict(), paths["reward"])
    torch.save(value_net.state_dict(), paths["value"])

    print(f"\n>>> SUCCESS. Models saved:")
    for name, path in paths.items():
        print(f"    {name}: {os.path.basename(path)}")

    return paths


if __name__ == "__main__":
    train_all_models()
