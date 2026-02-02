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
        # Numerical stability: clamp log_std to prevent exploding variance
        log_std = torch.clamp(self.log_std, min=-20, max=2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = D.Normal(mu, std)
        # Reparameterization trick (rsample) allows gradients to flow
        action = dist.rsample() 
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
# 2. WORLD MODELS (EBM, Flow, MDN)
# =============================================================================

class EnergyBasedModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(EnergyBasedModel, self).__init__()
        # E(s, a, s') -> Scalar Energy
        # Note: Input is (state + action + next_state)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
            nn.SiLU(), # Swish (SiLU) is standard for EBMs
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action, next_state):
        x = torch.cat([state, action, next_state], dim=1)
        return self.net(x)

class RealNVP(nn.Module):
    """
    Robust RealNVP that handles data dimensions safely.
    It splits input into (d // 2) and (d - d // 2).
    """
    def __init__(self, data_dim, hidden_dim=64, context_dim=0):
        super(RealNVP, self).__init__()
        
        self.split_dim = data_dim // 2
        dim1 = self.split_dim
        dim2 = data_dim - self.split_dim
        
        # Layer 1: Takes dim1, transforms dim2
        in_dim1 = dim1 + context_dim
        self.s1 = nn.Sequential(nn.Linear(in_dim1, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, dim2))
        self.t1 = nn.Sequential(nn.Linear(in_dim1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim2))
            
        # Layer 2: Takes dim2, transforms dim1
        in_dim2 = dim2 + context_dim
        self.s2 = nn.Sequential(nn.Linear(in_dim2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, dim1))
        self.t2 = nn.Sequential(nn.Linear(in_dim2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim1))

    def forward(self, x, context=None):
        """ Internal use for log_prob (x -> z) """
        x1 = x[:, :self.split_dim]
        x2 = x[:, self.split_dim:]
        
        # 1. Transform x2 using x1
        inp1 = torch.cat([x1, context], dim=1) if context is not None else x1
        s1, t1 = self.s1(inp1), self.t1(inp1)
        z2 = (x2 - t1) * torch.exp(-s1)
        
        # 2. Transform x1 using z2
        inp2 = torch.cat([z2, context], dim=1) if context is not None else z2
        s2, t2 = self.s2(inp2), self.t2(inp2)
        z1 = (x1 - t2) * torch.exp(-s2)
        
        z = torch.cat([z1, z2], dim=1)
        log_det = -torch.sum(s1, dim=1) - torch.sum(s2, dim=1)
        return z, log_det

    def log_prob(self, x, context=None):
        """ Calculates p(x|context) """
        z, log_det = self.forward(x, context)
        log_prob_z = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
        return log_prob_z + log_det

    def sample(self, z, context=None):
        """ Generates x from z (z -> x) """
        z1 = z[:, :self.split_dim]
        z2 = z[:, self.split_dim:]
        
        # Inverse 2
        inp2 = torch.cat([z2, context], dim=1) if context is not None else z2
        s2, t2 = self.s2(inp2), self.t2(inp2)
        x1 = z1 * torch.exp(s2) + t2
        
        # Inverse 1
        inp1 = torch.cat([x1, context], dim=1) if context is not None else x1
        s1, t1 = self.s1(inp1), self.t1(inp1)
        x2 = z2 * torch.exp(s1) + t1
        
        return torch.cat([x1, x2], dim=1)

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
        
        pi_logits = self.pi_head(h)
        mu = self.mu_head(h).view(-1, self.num_gaussians, self.state_dim)
        # Constrain sigma to reasonable values
        sigma = torch.exp(torch.clamp(self.sigma_head(h), -5, 2)).view(-1, self.num_gaussians, self.state_dim)
        return pi_logits, mu, sigma

    def sample_differentiable(self, state, action):
        """ Gumbel-Softmax Sampling for Differentiable Mode Selection """
        pi_logits, mu, sigma = self.forward(state, action)
        
        # 1. Soft Categorical (Gumbel-Softmax)
        weights = F.gumbel_softmax(pi_logits, tau=1.0, hard=False).unsqueeze(-1)
        
        # 2. Mixture
        mixed_mu = (weights * mu).sum(dim=1)
        mixed_sigma = (weights * sigma).sum(dim=1)
        
        # 3. Sample
        noise = torch.randn_like(mixed_mu)
        return mixed_mu + mixed_sigma * noise