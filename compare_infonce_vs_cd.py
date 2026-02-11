"""
InfoNCE vs Contrastive Divergence Comparison for EBM Training

This script compares two methods for training Energy-Based Models:
1. InfoNCE: Contrastive learning with random negatives from buffer
2. Contrastive Divergence (CD): MCMC-generated negatives via Langevin dynamics

Based on: "Building Minimal and Reusable Causal State Abstractions for RL" (arxiv 2401.12497)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import json
import os
import sys
from tqdm import tqdm
import argparse

# Import existing modules
try:
    import safety_gymnasium
    from models import EnergyBasedModel
    from utils_sampling import predict_next_state_langevin, get_energy_gradient
    # Try importing MiniGrid adapter
    try:
        from minigrid_adapter import MiniGridAdapter
    except ImportError:
        MiniGridAdapter = None
        print("Warning: MiniGridAdapter not found. MiniGrid environments will not work.")
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    print("Make sure you're running from the src1 directory")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "ENV_NAME": "MiniGrid-Empty-8x8-v0",  # Can be overwritten by args
    "COLLECT_STEPS": 5000,
    "TRAIN_STEPS": 5000,
    "BATCH_SIZE": 128,
    "LR": 1e-4,
    "HIDDEN_DIM": 128,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    # InfoNCE specific
    "NUM_NEGATIVES": 512,
    "TEMPERATURE": 0.1,
    
    # CD specific
    "LANGEVIN_STEPS": 30,
    "CD_REG_WEIGHT": 0.1,
    
    # Evaluation
    "EVAL_INTERVAL": 500,
    "NUM_TEST_SAMPLES": 100,
    "SAVE_DIR": "results_infonce_vs_cd"
}

# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=100000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

    def add(self, s, a, ns):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.next_states[self.ptr] = ns
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, indices=None):
        if indices is None:
            idx = np.random.randint(0, self.size, size=batch_size)
        else:
            # Sample from provided indices (for train/test split)
            idx = np.random.choice(indices, size=batch_size)
            
        return (
            torch.tensor(self.states[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.actions[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.next_states[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
        )
    
    def get_all_data(self):
        """Return all collected data"""
        return (
            self.states[:self.size],
            self.actions[:self.size],
            self.next_states[:self.size]
        )

    def get_subset(self, indices):
        """Return a subset of data based on indices"""
        return (
            torch.tensor(self.states[indices], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.actions[indices], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.next_states[indices], dtype=torch.float32).to(CONFIG["DEVICE"])
        )

# =============================================================================
# ENVIRONMENT ADAPTER
# =============================================================================

class SafetyGymAdapter:
    def __init__(self, env_name):
        self.env = safety_gymnasium.make(env_name, render_mode=None)
        # SafetyGym has Box observation and action spaces
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.is_discrete = False

    def reset(self):
        s, _ = self.env.reset()
        return s.astype(np.float32)

    def step(self, action):
        ns, reward, cost, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return ns.astype(np.float32), done

class UniversalAdapter:
    """Wrapper to handle both SafetyGym and MiniGrid"""
    def __init__(self, env_name):
        if "MiniGrid" in env_name:
            if MiniGridAdapter is None:
                raise ImportError("MiniGridAdapter not found but MiniGrid env requested.")
            self.adapter = MiniGridAdapter(env_name)
            self.is_discrete = True
            self.action_dim = self.adapter.action_dim # This is 'N' for discrete
        else:
            self.adapter = SafetyGymAdapter(env_name)
            self.is_discrete = False
            self.action_dim = self.adapter.action_dim
            
        self.state_dim = self.adapter.state_dim
        
    def reset(self):
        return self.adapter.reset()
        
    def step(self, action):
        # Handle MiniGrid implementation differences if any
        if self.is_discrete:
            # adapter.step returns (ns, reward, done, info)
            return self.adapter.step(action)[:2] # Return ns, done
        else:
            return self.adapter.step(action)

# =============================================================================
# BILINEAR EBM (for InfoNCE)
# =============================================================================

class BilinearEBM(nn.Module):
    """
    Energy model: E(s,a,s') = g(s,a)^T · h(s')
    
    This architecture is suitable for contrastive learning (InfoNCE).
    Higher energy = more compatible/likely transition.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(BilinearEBM, self).__init__()
        
        # Context encoder: g(s, a)
        # Context encoder: g(s, a)
        self.g = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Next state encoder: h(s')
        self.h = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, state, action, next_state):
        """
        Returns energy (higher = more compatible).
        
        Args:
            state: (B, state_dim) or (B, K, state_dim)
            action: (B, action_dim) or (B, K, action_dim)
            next_state: (B, state_dim) or (B, K, state_dim)
        
        Returns:
            energy: (B,) or (B, K)
        """
        # Handle both 2D and 3D inputs
        if state.dim() == 3:
            # Batch of negatives: (B, K, D)
            B, K, _ = state.shape
            context = torch.cat([state, action], dim=-1)  # (B, K, state_dim + action_dim)
            g_out = self.g(context)  # (B, K, hidden_dim)
            h_out = self.h(next_state)  # (B, K, hidden_dim)
            energy = (g_out * h_out).sum(dim=-1)  # (B, K)
        else:
            # Single samples: (B, D)
            context = torch.cat([state, action], dim=-1)
            g_out = self.g(context)  # (B, hidden_dim)
            h_out = self.h(next_state)  # (B, hidden_dim)
            energy = (g_out * h_out).sum(dim=-1)  # (B,)
        
        return energy

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def infonce_loss(ebm, state, action, pos_next_state, buffer, 
                 num_negatives=512, temperature=0.1):
    """
    InfoNCE loss as described in CBM paper (arxiv 2401.12497).
    
    ENERGY CONVENTION: Higher energy = more compatible/likely
    - We want E(s,a,s'_real) > E(s,a,s'_random)
    - This is a CONTRASTIVE learning objective
    
    Trains the model to assign high energy to real transitions and low energy
    to random pairings (negative samples from marginal distribution).
    
    Args:
        ebm: BilinearEBM model
        state: (B, state_dim)
        action: (B, action_dim)
        pos_next_state: (B, state_dim) - real next states
        buffer: ReplayBuffer to sample negatives from
        num_negatives: Number of negative samples per positive
        temperature: Temperature for softmax
    
    Returns:
        loss: InfoNCE loss (scalar)
        E_pos_mean: Mean positive energy (for logging)
        E_neg_mean: Mean negative energy (for logging)
    """
    B = state.shape[0]
    device = state.device
    
    # Optimized InfoNCE
    # 1. Compute context embedding g(s,a) -> (B, H)
    context = torch.cat([state, action], dim=-1)
    g_out = ebm.g(context) # (B, H)
    
    # 2. Compute positive next state embedding h(s') -> (B, H)
    h_pos = ebm.h(pos_next_state) # (B, H)
    
    # 3. Compute positive energy: dot(g, h_pos) -> (B,)
    E_pos = (g_out * h_pos).sum(dim=-1)
    
    # 4. Sample negatives from buffer (random states)
    neg_indices = np.random.randint(0, buffer.size, size=(B, num_negatives))
    neg_next_states = torch.tensor(
        buffer.next_states[neg_indices], 
        dtype=torch.float32, 
        device=device
    ) # (B, K, state_dim)
    
    # 5. Compute negative embeddings efficiently
    # Flatten: (B*K, state_dim)
    neg_states_flat = neg_next_states.view(-1, neg_next_states.shape[-1])
    h_neg_flat = ebm.h(neg_states_flat) # (B*K, H)
    h_neg = h_neg_flat.view(B, num_negatives, -1) # (B, K, H)
    
    # 6. Compute negative energy: sum(g_out[:, None, :] * h_neg, dim=-1)
    # Broadcasting g_out to (B, 1, H) to match (B, K, H)
    E_neg = (g_out.unsqueeze(1) * h_neg).sum(dim=-1) # (B, K)
    
    # InfoNCE Loss
    logits = torch.cat([E_pos.unsqueeze(1), E_neg], dim=1) / temperature  # (B, K+1)
    labels = torch.zeros(B, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels)
    
    return loss, E_pos.mean().item(), E_neg.mean().item()


def contrastive_divergence_loss(ebm, state, action, pos_next_state, 
                                 langevin_steps=30, reg_weight=0.1):
    """
    Contrastive Divergence loss (current implementation).
    
    ENERGY CONVENTION: Lower energy = more likely (standard EBM)
    - We want E(s,a,s'_real) < E(s,a,s'_langevin)
    - Langevin dynamics finds low-energy states
    
    Trains the model by contrasting real transitions with MCMC-generated samples.
    
    Args:
        ebm: EnergyBasedModel
        state: (B, state_dim)
        action: (B, action_dim)
        pos_next_state: (B, state_dim) - real next states
        langevin_steps: Number of Langevin dynamics steps
        reg_weight: Regularization weight
    
    Returns:
        loss: CD loss (scalar)
        E_pos_mean: Mean positive energy (for logging)
        E_neg_mean: Mean negative energy (for logging)
    """
    # Positive energy - should be LOW (negative values)
    E_pos = ebm(state, action, pos_next_state).mean()
    
    # Negative samples via Langevin dynamics (finds low-energy states)
    neg_next_state = predict_next_state_langevin(
        ebm, state, action, 
        init_state=None,
        config={"LANGEVIN_STEPS": langevin_steps}
    ).detach()
    
    # Negative energy - Langevin tries to find low energy, but we add noise
    E_neg = ebm(state, action, neg_next_state).mean()
    
    # CD loss: minimize E_pos while maximizing E_neg
    # E_pos - E_neg will be negative when E_pos < E_neg (desired)
    loss = E_pos - E_neg + (E_pos**2 + E_neg**2) * reg_weight
    
    return loss, E_pos.item(), E_neg.item()

# =============================================================================
# TRAINING
# =============================================================================

def train_both_models(buffer, config):
    """
    Train both EBM-InfoNCE and EBM-CD side by side.
    """
    device = config["DEVICE"]
    state_dim = buffer.states.shape[1]
    action_dim = buffer.actions.shape[1]
    
    # SPLIT DATA: Train/Test
    # This prevents data leakage in evaluation
    all_indices = np.arange(buffer.size)
    np.random.shuffle(all_indices)
    split_idx = int(buffer.size * 0.9)
    train_indices = all_indices[:split_idx]
    test_indices = all_indices[split_idx:]
    
    print(f"\n>>> Data Split: {len(train_indices)} Train / {len(test_indices)} Test")
    print("\n>>> Initializing models...")
    
    # Models
    ebm_infonce = BilinearEBM(state_dim, action_dim, config["HIDDEN_DIM"]).to(device)
    ebm_cd = EnergyBasedModel(state_dim, action_dim, config["HIDDEN_DIM"]).to(device)
    
    # Optimizers
    opt_infonce = optim.Adam(ebm_infonce.parameters(), lr=config["LR"])
    opt_cd = optim.Adam(ebm_cd.parameters(), lr=config["LR"])
    
    # Metrics tracking
    metrics = {
        "steps": [],
        "infonce_loss": [], "infonce_E_pos": [], "infonce_E_neg": [],
        "cd_loss": [], "cd_E_pos": [], "cd_E_neg": [],
        "infonce_mse": [], "cd_mse": []
    }
    
    # Get test set (static from test_indices)
    # We take a fixed subset of the test set for consistent evaluation
    test_subset_idx = np.random.choice(test_indices, size=min(config["NUM_TEST_SAMPLES"], len(test_indices)), replace=False)
    test_s, test_a, test_ns = buffer.get_subset(test_subset_idx)
    
    print(f"\n>>> Training for {config['TRAIN_STEPS']} steps...")
    print(f"    InfoNCE: {config['NUM_NEGATIVES']} negatives, T={config['TEMPERATURE']}")
    print(f"    CD: {config['LANGEVIN_STEPS']} Langevin steps, reg={config['CD_REG_WEIGHT']}")
    
    for step in tqdm(range(config["TRAIN_STEPS"]), desc="Training"):
        # Sample from TRAIN indices only
        s, a, real_ns = buffer.sample(config["BATCH_SIZE"], indices=train_indices)
        
        # ==================== Train InfoNCE ====================
        loss_infonce, E_pos_info, E_neg_info = infonce_loss(
            ebm_infonce, s, a, real_ns, buffer, 
            num_negatives=config["NUM_NEGATIVES"],
            temperature=config["TEMPERATURE"]
        )
        opt_infonce.zero_grad()
        loss_infonce.backward()
        opt_infonce.step()
        
        # ==================== Train CD ====================
        loss_cd, E_pos_cd, E_neg_cd = contrastive_divergence_loss(
            ebm_cd, s, a, real_ns,
            langevin_steps=config["LANGEVIN_STEPS"],
            reg_weight=config["CD_REG_WEIGHT"]
        )
        opt_cd.zero_grad()
        loss_cd.backward()
        opt_cd.step()
        
        # ==================== Log metrics ====================
        if step % config["EVAL_INTERVAL"] == 0:
            # Compute test MSE
            with torch.no_grad():
                # InfoNCE: predict by finding max energy next state
                mse_infonce = compute_prediction_mse_infonce(
                    ebm_infonce, test_s, test_a, test_ns, buffer
                )
                
                # CD: predict using Langevin dynamics
                mse_cd = compute_prediction_mse_cd(
                    ebm_cd, test_s, test_a, test_ns, config
                )
            
            metrics["steps"].append(step)
            metrics["infonce_loss"].append(loss_infonce.item())
            metrics["infonce_E_pos"].append(E_pos_info)
            metrics["infonce_E_neg"].append(E_neg_info)
            metrics["cd_loss"].append(loss_cd.item())
            metrics["cd_E_pos"].append(E_pos_cd)
            metrics["cd_E_neg"].append(E_neg_cd)
            metrics["infonce_mse"].append(mse_infonce)
            metrics["cd_mse"].append(mse_cd)
            
            if step % (config["EVAL_INTERVAL"] * 2) == 0:
                print(f"\nStep {step}:")
                print(f"  InfoNCE - Loss: {loss_infonce.item():.4f}, MSE: {mse_infonce:.4f}, "
                      f"E_pos: {E_pos_info:.2f}, E_neg: {E_neg_info:.2f}")
                print(f"  CD      - Loss: {loss_cd.item():.4f}, MSE: {mse_cd:.4f}, "
                      f"E_pos: {E_pos_cd:.2f}, E_neg: {E_neg_cd:.2f}")
    
    return ebm_infonce, ebm_cd, metrics


def get_test_set(buffer, num_samples):
    """Extract a fixed test set from buffer"""
    idx = np.random.choice(buffer.size, size=num_samples, replace=False)
    device = CONFIG["DEVICE"]
    return (
        torch.tensor(buffer.states[idx], dtype=torch.float32).to(device),
        torch.tensor(buffer.actions[idx], dtype=torch.float32).to(device),
        torch.tensor(buffer.next_states[idx], dtype=torch.float32).to(device)
    )


def langevin_ascent(model, state, action, steps=50, step_size=0.01, noise_scale=0.005):
    """
    Langevin gradient ASCENT to find HIGH energy states.
    
    For InfoNCE bilinear model: higher energy = more compatible
    So we maximize energy by doing gradient ascent.
    
    Args:
        model: Energy model
        state: (B, state_dim)
        action: (B, action_dim)
        steps: Number of Langevin steps
        step_size: Step size for gradient ascent
        noise_scale: Scale of Gaussian noise
    
    Returns:
        curr_state: (B, state_dim) - predicted next state with high energy
    """
    from utils_sampling import get_energy_gradient
    
    # Initialize from random noise
    curr_state = torch.randn_like(state)
    
    for i in range(steps):
        noise = torch.randn_like(curr_state) * noise_scale
        grad_energy = get_energy_gradient(model, state, action, curr_state)
        # ASCENT: move in direction of increasing energy
        curr_state = curr_state + step_size * grad_energy + noise
    
    return curr_state



def compute_prediction_mse_infonce(ebm, state, action, true_next_state, buffer, num_candidates=1000):
    """
    Predict next state by finding the candidate with highest energy.
    Sample candidates from the buffer.
    """
    B = state.shape[0]
    device = state.device
    
    # Sample candidate next states from buffer
    candidate_indices = np.random.randint(0, buffer.size, size=(B, num_candidates))
    candidates = torch.tensor(
        buffer.next_states[candidate_indices],
        dtype=torch.float32,
        device=device
    )  # (B, K, state_dim)
    
    # Expand state and action
    state_exp = state.unsqueeze(1).expand(B, num_candidates, -1)
    action_exp = action.unsqueeze(1).expand(B, num_candidates, -1)
    
    # Compute energies
    energies = ebm(state_exp, action_exp, candidates)  # (B, K)
    
    # Select candidate with highest energy
    best_idx = energies.argmax(dim=1)  # (B,)
    pred_next_state = candidates[torch.arange(B), best_idx]  # (B, state_dim)
    
    # Compute MSE
    mse = F.mse_loss(pred_next_state, true_next_state).item()
    return mse


def compute_prediction_mse_cd(ebm, state, action, true_next_state, config):
    """
    Predict next state using Langevin dynamics.
    """
    pred_next_state = predict_next_state_langevin(
        ebm, state, action,
        init_state=None,
        config={"LANGEVIN_STEPS": config["LANGEVIN_STEPS"]}
    )
    
    mse = F.mse_loss(pred_next_state, true_next_state).item()
    return mse

# =============================================================================
# ANALYSIS & VISUALIZATION
# =============================================================================




def analyze_energy_landscapes(ebm_infonce, ebm_cd, buffer, config, save_dir):
    """
    Visualize and compare energy landscapes.
    """
    print("\n>>> Analyzing energy landscapes...")
    
    device = config["DEVICE"]
    
    # Sample a few test transitions
    num_samples = 5
    test_s, test_a, test_ns = get_test_set(buffer, num_samples)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i in range(num_samples):
        s = test_s[i:i+1]
        a = test_a[i:i+1]
        true_ns = test_ns[i:i+1]
        
        # Sample many next states from buffer for visualization
        grid_size = 50
        candidate_indices = np.random.randint(0, buffer.size, size=grid_size**2)
        candidates = torch.tensor(
            buffer.next_states[candidate_indices],
            dtype=torch.float32,
            device=device
        )
        
        # Project to 2D using sklearn PCA
        pca = PCA(n_components=2)
        candidates_2d = pca.fit_transform(candidates.cpu().numpy())
        true_ns_2d = pca.transform(true_ns.cpu().numpy())

        
        # Compute energies
        s_exp = s.expand(len(candidates), -1)
        a_exp = a.expand(len(candidates), -1)
        
        with torch.no_grad():
            E_infonce = ebm_infonce(s_exp, a_exp, candidates).cpu().numpy()
            E_cd = ebm_cd(s_exp, a_exp, candidates).cpu().numpy()
        
        # Plot InfoNCE (High Energy = Good)
        ax = axes[0, i]
        scatter = ax.scatter(candidates_2d[:, 0], candidates_2d[:, 1], 
                            c=E_infonce, cmap='viridis', s=20, alpha=0.6)
        ax.scatter(true_ns_2d[:, 0], true_ns_2d[:, 1], 
                  c='red', marker='*', s=200, edgecolors='white', linewidths=1.5,
                  label='True next state')
        ax.set_title(f'InfoNCE - Sample {i+1}\n(Bright/High = Likely)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax, label='Energy (Higher is Better)')
        if i == 0:
            ax.legend()
        
        # Plot CD (Low Energy = Good)
        # We INVERT the color map for visualization so "Bright" still means "Likely"
        # This makes visual comparison easier
        ax = axes[1, i]
        scatter = ax.scatter(candidates_2d[:, 0], candidates_2d[:, 1], 
                            c=-E_cd, cmap='viridis', s=20, alpha=0.6) # Note the negative sign
        ax.scatter(true_ns_2d[:, 0], true_ns_2d[:, 1], 
                  c='red', marker='*', s=200, edgecolors='white', linewidths=1.5,
                  label='True next state')
        ax.set_title(f'CD - Sample {i+1}\n(Bright/Low = Likely)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax, label='Negative Energy (Higher is Better)')
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'energy_landscapes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: energy_landscapes.png")



def analyze_langevin_quality(ebm_infonce, ebm_cd, buffer, config, save_dir):
    """
    Analyze Langevin sampling quality for both models.
    
    InfoNCE: Gradient ASCENT to maximize energy (higher = better)
    CD: Gradient DESCENT to minimize energy (lower = better)
    """
    print("\n>>> Analyzing Langevin sampling quality...")
    
    device = config["DEVICE"]
    num_samples = 3
    test_s, test_a, test_ns = get_test_set(buffer, num_samples)
    
    max_steps = 50
    
    fig, axes = plt.subplots(2, num_samples, figsize=(5*num_samples, 10))
    
    for i in range(num_samples):
        s = test_s[i:i+1]
        a = test_a[i:i+1]
        true_ns = test_ns[i:i+1]
        
        # Track Langevin dynamics for InfoNCE
        energies_infonce = []
        curr_state = torch.randn_like(true_ns)
        
        for step in range(max_steps):
            with torch.no_grad():
                energy = ebm_infonce(s, a, curr_state).item()
                energies_infonce.append(energy)
            
            if step < max_steps - 1:
                grad = get_energy_gradient(ebm_infonce, s, a, curr_state)
                # ASCENT: move toward HIGHER energy (InfoNCE convention)
                curr_state = curr_state + 0.01 * grad + torch.randn_like(curr_state) * 0.005
        
        # Track Langevin dynamics for CD
        energies_cd = []
        curr_state = torch.randn_like(true_ns)
        
        for step in range(max_steps):
            with torch.no_grad():
                energy = ebm_cd(s, a, curr_state).item()
                energies_cd.append(energy)
            
            if step < max_steps - 1:
                grad = get_energy_gradient(ebm_cd, s, a, curr_state)
                # DESCENT: move toward LOWER energy (standard EBM convention)
                curr_state = curr_state - 0.01 * grad + torch.randn_like(curr_state) * 0.005
        
        # Plot InfoNCE (should go UP)
        ax = axes[0, i]
        ax.plot(energies_infonce, linewidth=2, label='Langevin ASCENT', color='blue')
        with torch.no_grad():
            true_energy = ebm_infonce(s, a, true_ns).item()
        ax.axhline(true_energy, color='red', linestyle='--', linewidth=2, label=f'True s\' E={true_energy:.2f}')
        ax.set_xlabel('Langevin Step')
        ax.set_ylabel('Energy (higher = better)')
        ax.set_title(f'InfoNCE Sample {i+1}: MAXIMIZE Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot CD (should go DOWN)
        ax = axes[1, i]
        ax.plot(energies_cd, linewidth=2, label='Langevin DESCENT', color='orange')
        with torch.no_grad():
            true_energy = ebm_cd(s, a, true_ns).item()
        ax.axhline(true_energy, color='red', linestyle='--', linewidth=2, label=f'True s\' E={true_energy:.2f}')
        ax.set_xlabel('Langevin Step')
        ax.set_ylabel('Energy (lower = better)')
        ax.set_title(f'CD Sample {i+1}: MINIMIZE Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'langevin_quality.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: langevin_quality.png")


def analyze_gradient_variance(ebm_infonce, ebm_cd, buffer, config, save_dir):
    """
    Analyze gradient variance and stability.
    """
    print("\n>>> Analyzing gradient variance...")
    
    device = config["DEVICE"]
    num_samples = 100
    test_s, test_a, test_ns = get_test_set(buffer, num_samples)
    
    # Compute gradients for both models
    grads_infonce = []
    grads_cd = []
    
    for i in range(num_samples):
        s = test_s[i:i+1]
        a = test_a[i:i+1]
        ns = test_ns[i:i+1]
        
        grad_info = get_energy_gradient(ebm_infonce, s, a, ns)
        grad_cd_val = get_energy_gradient(ebm_cd, s, a, ns)
        
        grads_infonce.append(grad_info.norm().item())
        grads_cd.append(grad_cd_val.norm().item())
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(grads_infonce, bins=30, alpha=0.6, label='InfoNCE', color='blue')
    ax.hist(grads_cd, bins=30, alpha=0.6, label='CD', color='orange')
    ax.set_xlabel('Gradient Norm')
    ax.set_ylabel('Frequency')
    ax.set_title('Gradient Magnitude Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1]
    ax.boxplot([grads_infonce, grads_cd], labels=['InfoNCE', 'CD'])
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Variance Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gradient_variance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: gradient_variance.png")
    
    # Print statistics
    print(f"\n    Gradient Statistics:")
    print(f"      InfoNCE - Mean: {np.mean(grads_infonce):.4f}, Std: {np.std(grads_infonce):.4f}")
    print(f"      CD      - Mean: {np.mean(grads_cd):.4f}, Std: {np.std(grads_cd):.4f}")


def plot_training_curves(metrics, save_dir):
    """
    Plot training curves.
    """
    print("\n>>> Plotting training curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = metrics["steps"]
    
    # Loss
    ax = axes[0, 0]
    ax.plot(steps, metrics["infonce_loss"], label='InfoNCE', linewidth=2)
    ax.plot(steps, metrics["cd_loss"], label='CD', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MSE
    ax = axes[0, 1]
    ax.plot(steps, metrics["infonce_mse"], label='InfoNCE', linewidth=2)
    ax.plot(steps, metrics["cd_mse"], label='CD', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('MSE')
    ax.set_title('Prediction MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy gap (E_pos - E_neg)
    ax = axes[1, 0]
    infonce_gap = np.array(metrics["infonce_E_pos"]) - np.array(metrics["infonce_E_neg"])
    cd_gap = np.array(metrics["cd_E_pos"]) - np.array(metrics["cd_E_neg"])
    ax.plot(steps, infonce_gap, label='InfoNCE', linewidth=2)
    ax.plot(steps, cd_gap, label='CD', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('E_pos - E_neg')
    ax.set_title('Energy Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Positive energy
    ax = axes[1, 1]
    ax.plot(steps, metrics["infonce_E_pos"], label='InfoNCE E_pos', linewidth=2)
    ax.plot(steps, metrics["infonce_E_neg"], label='InfoNCE E_neg', linewidth=2, linestyle='--')
    ax.plot(steps, metrics["cd_E_pos"], label='CD E_pos', linewidth=2)
    ax.plot(steps, metrics["cd_E_neg"], label='CD E_neg', linewidth=2, linestyle='--')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Energy')
    ax.set_title('Positive vs Negative Energies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: training_curves.png")


def create_summary_report(metrics, save_dir):
    """
    Create a text summary report.
    """
    print("\n>>> Creating summary report...")
    
    report_path = os.path.join(save_dir, 'comparison_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("InfoNCE vs Contrastive Divergence Comparison Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        for key, value in CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("FINAL RESULTS:\n")
        f.write("-" * 80 + "\n")
        
        # Get final values
        final_infonce_loss = metrics["infonce_loss"][-1]
        final_cd_loss = metrics["cd_loss"][-1]
        final_infonce_mse = metrics["infonce_mse"][-1]
        final_cd_mse = metrics["cd_mse"][-1]
        
        f.write(f"\nInfoNCE:\n")
        f.write(f"  Final Loss: {final_infonce_loss:.4f}\n")
        f.write(f"  Final MSE:  {final_infonce_mse:.4f}\n")
        f.write(f"  Final E_pos: {metrics['infonce_E_pos'][-1]:.4f}\n")
        f.write(f"  Final E_neg: {metrics['infonce_E_neg'][-1]:.4f}\n")
        
        f.write(f"\nContrastive Divergence:\n")
        f.write(f"  Final Loss: {final_cd_loss:.4f}\n")
        f.write(f"  Final MSE:  {final_cd_mse:.4f}\n")
        f.write(f"  Final E_pos: {metrics['cd_E_pos'][-1]:.4f}\n")
        f.write(f"  Final E_neg: {metrics['cd_E_neg'][-1]:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("=" * 80 + "\n\n")
        
        # Determine winner
        if final_infonce_mse < final_cd_mse:
            f.write(f"[+] InfoNCE achieves LOWER prediction MSE ({final_infonce_mse:.4f} vs {final_cd_mse:.4f})\n")
        else:
            f.write(f"[+] CD achieves LOWER prediction MSE ({final_cd_mse:.4f} vs {final_infonce_mse:.4f})\n")
        
        f.write(f"\n[+] InfoNCE training is FASTER (no MCMC during training)\n")
        f.write(f"[+] CD requires {CONFIG['LANGEVIN_STEPS']} Langevin steps per batch\n")
        
        f.write("\nSee visualizations for detailed analysis of:\n")
        f.write("  - Energy landscape smoothness\n")
        f.write("  - Langevin sampling quality\n")
        f.write("  - Gradient variance and stability\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"    Saved: comparison_report.txt")
    
    # Also save metrics as JSON
    metrics_path = os.path.join(save_dir, 'metrics.json')
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {k: [float(x) for x in v] if isinstance(v, list) else v 
                    for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"    Saved: metrics.json")

# =============================================================================
# MAIN
# =============================================================================

def main(args):
    """Main execution function"""
    
    # Update config from args
    if args.steps:
        CONFIG["TRAIN_STEPS"] = args.steps
    if args.env:
        CONFIG["ENV_NAME"] = args.env
    if args.quick_test:
        CONFIG["TRAIN_STEPS"] = 1000
        CONFIG["COLLECT_STEPS"] = 1000
        CONFIG["EVAL_INTERVAL"] = 100
    
    # Create save directory
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
    
    print("=" * 80)
    print("InfoNCE vs Contrastive Divergence Comparison")
    print("=" * 80)
    print(f"\nEnvironment: {CONFIG['ENV_NAME']}")
    print(f"Device: {CONFIG['DEVICE']}")
    
    # ==================== Data Collection ====================
    # ==================== Data Collection ====================
    print(f"\n>>> Collecting {CONFIG['COLLECT_STEPS']} transitions...")
    env_adapter = UniversalAdapter(CONFIG["ENV_NAME"]) # Use Universal Adapter
    
    # Handle One-Hot encoding for discrete actions
    action_size = env_adapter.action_dim
    
    buffer = ReplayBuffer(env_adapter.state_dim, action_size)
    
    s = env_adapter.reset()
    for i in tqdm(range(CONFIG['COLLECT_STEPS']), desc="Collecting data"):
        if env_adapter.is_discrete:
            # Sample integer action from the underlying adapter's env
            a_int = env_adapter.adapter.env.action_space.sample()
            
            # One-hot encode
            a_onehot = np.zeros(action_size, dtype=np.float32)
            a_onehot[a_int] = 1.0
            
            # Step env with integer
            ns, done = env_adapter.step(a_int)
            
            # Store one-hot
            buffer.add(s, a_onehot, ns)
        else:
            # Continuous action
            a = env_adapter.adapter.env.action_space.sample()
            ns, done = env_adapter.step(a)
            buffer.add(s, a, ns)
            
        if done:
            s = env_adapter.reset()
        else:
            s = ns
    
    print(f"    Collected {buffer.size} transitions")
    
    # ==================== Training ====================
    ebm_infonce, ebm_cd, metrics = train_both_models(buffer, CONFIG)
    
    # ==================== Analysis ====================
    print("\n" + "=" * 80)
    print("ANALYSIS & VISUALIZATION")
    print("=" * 80)
    
    plot_training_curves(metrics, CONFIG["SAVE_DIR"])
    analyze_energy_landscapes(ebm_infonce, ebm_cd, buffer, CONFIG, CONFIG["SAVE_DIR"])
    analyze_langevin_quality(ebm_infonce, ebm_cd, buffer, CONFIG, CONFIG["SAVE_DIR"])
    analyze_gradient_variance(ebm_infonce, ebm_cd, buffer, CONFIG, CONFIG["SAVE_DIR"])
    create_summary_report(metrics, CONFIG["SAVE_DIR"])
    
    # ==================== Save Models ====================
    print("\n>>> Saving models...")
    torch.save(ebm_infonce.state_dict(), 
               os.path.join(CONFIG["SAVE_DIR"], 'ebm_infonce.pth'))
    torch.save(ebm_cd.state_dict(), 
               os.path.join(CONFIG["SAVE_DIR"], 'ebm_cd.pth'))
    print(f"    Saved models to {CONFIG['SAVE_DIR']}/")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {CONFIG['SAVE_DIR']}/")
    print("\nGenerated files:")
    print("  - training_curves.png")
    print("  - energy_landscapes.png")
    print("  - langevin_quality.png")
    print("  - gradient_variance.png")
    print("  - comparison_report.txt")
    print("  - metrics.json")
    print("  - ebm_infonce.pth")
    print("  - ebm_cd.pth")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare InfoNCE vs CD for EBM training')
    parser.add_argument('--env', type=str, default=None, help='Environment name')
    parser.add_argument('--steps', type=int, default=None, help='Number of training steps')
    parser.add_argument('--quick_test', action='store_true', help='Run quick test (1000 steps)')
    
    args = parser.parse_args()
    main(args)
