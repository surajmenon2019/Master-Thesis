import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

# Import existing models to reuse or re-implement if needed
# We can re-use Critic, EBM, Flow, MDN from models.py since they are generic MLPs
# We only need a new Actor for Discrete actions.

class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DiscreteActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Output: Logits for each action
        )
    
    def forward(self, state):
        return self.net(state)

    def sample(self, state, temperature=1.0, hard=False):
        """
        Gumbel-Softmax sampling for differentiable discrete actions.
        """
        logits = self.forward(state)
        action = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        return action, logits
