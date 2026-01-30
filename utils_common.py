import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import EnergyBasedModel, Actor, Critic
from utils_sampling import predict_next_state

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=1000000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, s, a, r, ns, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.next_states[self.ptr] = ns
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.float32),
            torch.tensor(self.next_states[idx], dtype=torch.float32),
            torch.tensor(self.rewards[idx], dtype=torch.float32),
            torch.tensor(self.dones[idx], dtype=torch.float32)
        )

