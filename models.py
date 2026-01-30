import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

# =============================================================================
# 1. CORE RL COMPONENTS (Actor & Critic)
# =============================================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mu = self.net(state)
        # Numerical stability for log_std
        log_std = torch.clamp(self.log_std, min=-20, max=2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = D.Normal(mu, std)
        action = dist.rsample() # Reparameterization trick
        return torch.tanh(action)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# =============================================================================
# 2. WORLD MODELS (EBM & Flow & MDN)
# =============================================================================

class EnergyBasedModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(EnergyBasedModel, self).__init__()
        # E(s, a, s') -> Scalar Energy
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
            nn.SiLU(), # Swish is often better for EBMs than ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action, next_state):
        # Concatenate: (Batch, Dim_s + Dim_a + Dim_s)
        x = torch.cat([state, action, next_state], dim=1)
        energy = self.net(x)
        return energy

class RealNVP(nn.Module):
    def __init__(self, data_dim, hidden_dim=64, context_dim=0):
        super(RealNVP, self).__init__()
        self.split_dim = data_dim // 2
        in_dim = self.split_dim + context_dim
        out_dim = data_dim - self.split_dim
        
        # --- LAYER 1 (Transforms Second Half) ---
        self.s1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, out_dim))
        self.t1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
            
        # --- LAYER 2 (Transforms First Half - The Fix) ---
        self.s2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, out_dim))
        self.t2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x, context=None):
        """ Used internally for log_prob (x -> z) """
        x1 = x[:, :self.split_dim]
        x2 = x[:, self.split_dim:]
        
        # Step 1: Transform x2 based on x1
        inp1 = torch.cat([x1, context], dim=1) if context is not None else x1
        s1, t1 = self.s1(inp1), self.t1(inp1)
        z2 = (x2 - t1) * torch.exp(-s1)
        
        # Step 2: Transform x1 based on z2 (Swap dependency)
        inp2 = torch.cat([z2, context], dim=1) if context is not None else z2
        s2, t2 = self.s2(inp2), self.t2(inp2)
        z1 = (x1 - t2) * torch.exp(-s2)
        
        z = torch.cat([z1, z2], dim=1)
        log_det = -torch.sum(s1, dim=1) - torch.sum(s2, dim=1)
        return z, log_det

    def log_prob(self, x, context=None):
        """ REQUIRED for Training (Likelihood Loss) """
        z, log_det = self.forward(x, context)
        # Standard Normal Log Likelihood
        log_prob_z = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
        return log_prob_z + log_det

    def sample(self, z, context=None):
        """ Used for Benchmarking (z -> x) """
        z1 = z[:, :self.split_dim]
        z2 = z[:, self.split_dim:]
        
        # Inverse Step 2 (Must be first to un-swap)
        inp2 = torch.cat([z2, context], dim=1) if context is not None else z2
        s2, t2 = self.s2(inp2), self.t2(inp2)
        x1 = z1 * torch.exp(s2) + t2
        
        # Inverse Step 1
        inp1 = torch.cat([x1, context], dim=1) if context is not None else x1
        s1, t1 = self.s1(inp1), self.t1(inp1)
        x2 = z2 * torch.exp(s1) + t1
        
        x = torch.cat([x1, x2], dim=1)
        return x

class MixtureDensityNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_gaussians=5, hidden_dim=64):
        super(MixtureDensityNetwork, self).__init__()
        self.state_dim = state_dim
        self.num_gaussians = num_gaussians
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.pi_head = nn.Linear(hidden_dim, num_gaussians)
        self.mu_head = nn.Linear(hidden_dim, num_gaussians * state_dim)
        self.sigma_head = nn.Linear(hidden_dim, num_gaussians * state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        h = self.net(x)
        
        pi = F.log_softmax(self.pi_head(h), dim=1)
        mu = self.mu_head(h).view(-1, self.num_gaussians, self.state_dim)
        sigma = torch.exp(torch.clamp(self.sigma_head(h), -5, 2)).view(-1, self.num_gaussians, self.state_dim)
        return pi, mu, sigma