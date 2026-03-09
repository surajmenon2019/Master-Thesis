import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

# =============================================================================
# 1. CORE RL COMPONENTS
# =============================================================================

class Actor(nn.Module):
    """
    Gaussian policy with state-dependent mean and global log_std.
    Outputs tanh-squashed actions for bounded action spaces.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_scale=1.0):
        super().__init__()
        self.action_scale = action_scale
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
        log_std = torch.clamp(self.log_std, min=-5, max=0.5)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = D.Normal(mu, std)
        action = dist.rsample()  # reparameterized
        return torch.tanh(action) * self.action_scale

    def log_prob(self, state, action):
        """Log probability of action under the policy (with tanh correction)."""
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = D.Normal(mu, std)
        # Inverse tanh to get pre-squash action
        raw_action = torch.atanh(torch.clamp(action / self.action_scale, -0.999, 0.999))
        log_p = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        # Tanh correction: log|det(d tanh/d raw)| = log(1 - tanh^2)
        log_p -= torch.log(1 - (action / self.action_scale).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return log_p


class ValueNetwork(nn.Module):
    """State value function V(s)."""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)


class TwinCritic(nn.Module):
    """
    TD3-style twin V(s) critics.

    Trains two independent value networks. Uses min(V1, V2) for
    actor bootstrapping and target computation. This prevents the
    actor from exploiting overconfident value estimates — the
    root cause of the late-training instability collapses.
    """
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.v1 = ValueNetwork(state_dim, hidden_dim)
        self.v2 = ValueNetwork(state_dim, hidden_dim)

    def forward(self, state):
        """Returns (v1, v2) tuple — use for critic loss on both."""
        return self.v1(state), self.v2(state)

    def min_value(self, state):
        """Returns min(V1, V2) — use for actor bootstrap and targets."""
        with torch.no_grad():
            v1 = self.v1(state)
            v2 = self.v2(state)
        return torch.min(v1, v2)


# =============================================================================
# 2. WORLD MODELS
# =============================================================================

class BilinearEBM(nn.Module):
    """
    Bilinear Energy-Based Model for InfoNCE training.

    ENERGY CONVENTION: Higher energy = more compatible (s,a) <-> s'.
    E(s,a,s') = g(s,a)^T W h(s')

    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Context encoder: g(s, a) -> hidden_dim
        # Deeper (5 layers) with residual connections for sharper energy landscape
        self.context_proj = nn.utils.spectral_norm(nn.Linear(state_dim + action_dim, hidden_dim))
        self.context_layers = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            ) for _ in range(3)
        ])
        self.context_act = nn.SiLU()

        # Next state encoder: h(s') -> hidden_dim
        self.ns_proj = nn.utils.spectral_norm(nn.Linear(state_dim, hidden_dim))
        self.ns_layers = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            ) for _ in range(3)
        ])
        self.ns_act = nn.SiLU()

        # Learned bilinear interaction
        self.W = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False))

    def _encode(self, state, action, next_state):
        """Shared encoding logic with residual connections."""
        context = torch.cat([state, action], dim=-1)
        g = self.context_act(self.context_proj(context))
        for layer in self.context_layers:
            g = g + layer(g)  # residual
        g = self.context_act(g)

        h = self.ns_act(self.ns_proj(next_state))
        for layer in self.ns_layers:
            h = h + layer(h)  # residual
        h = self.ns_act(h)
        return g, h

    def forward(self, state, action, next_state):
        """
        Compute energy E(s,a,s') = g(s,a)^T W h(s').

        Handles:
          (B, D) inputs -> (B,) output
          (B, K, D) inputs -> (B, K) output
        """
        is_batched_neg = (state.dim() == 3)

        if is_batched_neg:
            B, K, _ = state.shape
            s_flat = state.reshape(B * K, -1)
            a_flat = action.reshape(B * K, -1)
            ns_flat = next_state.reshape(B * K, -1)
            g, h = self._encode(s_flat, a_flat, ns_flat)
            # Bilinear: g^T W h
            energy = (g * self.W(h)).sum(dim=-1)  # (B*K,)
            return energy.reshape(B, K)
        else:
            g, h = self._encode(state, action, next_state)
            energy = (g * self.W(h)).sum(dim=-1)  # (B,)
            return energy


class RealNVP(nn.Module):
    """
    Conditional RealNVP normalizing flow.
    p(s' | s, a) via invertible transformations.

    No changes needed from original — this was architecturally sound.
    Added deeper coupling layers for better expressiveness.
    """
    def __init__(self, data_dim, hidden_dim=128, context_dim=0, n_layers=6):
        super().__init__()
        self.data_dim = data_dim
        self.n_layers = n_layers
        self.split_dim = data_dim // 2
        dim1 = self.split_dim
        dim2 = data_dim - self.split_dim

        # Build n_layers coupling pairs
        self.s_nets = nn.ModuleList()
        self.t_nets = nn.ModuleList()

        for i in range(n_layers):
            if i % 2 == 0:
                # Even layers: condition on dim1, transform dim2
                in_d = dim1 + context_dim
                out_d = dim2
            else:
                # Odd layers: condition on dim2, transform dim1
                in_d = dim2 + context_dim
                out_d = dim1

            self.s_nets.append(nn.Sequential(
                nn.Linear(in_d, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, out_d)
            ))
            self.t_nets.append(nn.Sequential(
                nn.Linear(in_d, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, out_d)
            ))

    def forward(self, x, context=None):
        """Forward pass x -> z, returns (z, log_det_jacobian)."""
        x1 = x[:, :self.split_dim]
        x2 = x[:, self.split_dim:]
        log_det = torch.zeros(x.shape[0], device=x.device)

        for i in range(self.n_layers):
            if i % 2 == 0:
                inp = torch.cat([x1, context], dim=1) if context is not None else x1
                s, t = self.s_nets[i](inp), self.t_nets[i](inp)
                s = torch.clamp(s, -5, 5)  # stability
                x2 = (x2 - t) * torch.exp(-s)
                log_det -= s.sum(dim=1)
            else:
                inp = torch.cat([x2, context], dim=1) if context is not None else x2
                s, t = self.s_nets[i](inp), self.t_nets[i](inp)
                s = torch.clamp(s, -5, 5)
                x1 = (x1 - t) * torch.exp(-s)
                log_det -= s.sum(dim=1)

        z = torch.cat([x1, x2], dim=1)
        return z, log_det

    def inverse(self, z, context=None):
        """Inverse pass z -> x."""
        z1 = z[:, :self.split_dim]
        z2 = z[:, self.split_dim:]

        x1, x2 = z1, z2

        # Reverse order of layers
        for i in reversed(range(self.n_layers)):
            if i % 2 == 0:
                inp = torch.cat([x1, context], dim=1) if context is not None else x1
                s, t = self.s_nets[i](inp), self.t_nets[i](inp)
                s = torch.clamp(s, -5, 5)
                x2 = x2 * torch.exp(s) + t
            else:
                inp = torch.cat([x2, context], dim=1) if context is not None else x2
                s, t = self.s_nets[i](inp), self.t_nets[i](inp)
                s = torch.clamp(s, -5, 5)
                x1 = x1 * torch.exp(s) + t

        return torch.cat([x1, x2], dim=1)

    def log_prob(self, x, context=None):
        """Log probability p(x | context)."""
        z, log_det = self.forward(x, context)
        log_prob_z = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
        return log_prob_z + log_det

    def sample(self, z, context=None):
        """Generate x from latent z."""
        return self.inverse(z, context)


class MixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network: p(s' | s, a) = sum_k pi_k N(mu_k, sigma_k).
    Uses Gumbel-Softmax for differentiable sampling.
    """
    def __init__(self, state_dim, action_dim, num_gaussians=10, hidden_dim=128):
        super().__init__()
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
        sigma = torch.exp(torch.clamp(self.sigma_head(h), -5, 2))
        sigma = sigma.view(-1, self.num_gaussians, self.state_dim)
        return pi_logits, mu, sigma

    def sample_differentiable(self, state, action):
        """Gumbel-Softmax weighted mixture sample (differentiable)."""
        pi_logits, mu, sigma = self.forward(state, action)

        weights = F.gumbel_softmax(pi_logits, tau=0.5, hard=False).unsqueeze(-1)
        z = torch.randn_like(mu)
        component_samples = mu + sigma * z
        return (weights * component_samples).sum(dim=1)

    def log_prob(self, state, action, target):
        """Log-likelihood of target under the mixture."""
        pi_logits, mu, sigma = self.forward(state, action)
        target_exp = target.unsqueeze(1).expand_as(mu)
        dist = D.Normal(mu, sigma)
        log_prob_components = dist.log_prob(target_exp).sum(dim=-1)  # (B, K)
        log_pi = torch.log_softmax(pi_logits, dim=1)
        return torch.logsumexp(log_pi + log_prob_components, dim=1)  # (B,)


class RewardModel(nn.Module):
    """
    Transition reward model r(s, a, s').
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action, next_state):
        x = torch.cat([state, action, next_state], dim=1)
        return self.net(x)