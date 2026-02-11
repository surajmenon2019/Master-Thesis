import gymnasium as gym
import minigrid
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper
from gymnasium.wrappers import FlattenObservation

class MiniGridAdapter:
    def __init__(self, env_name="MiniGrid-Empty-8x8-v0", render_mode=None):
        self.env = gym.make(env_name, render_mode=render_mode)
        
        # 1. Image only (remove mission text)
        self.env = ImgObsWrapper(self.env)
        
        # 2. Flatten (7x7x3 -> 147)
        self.env = FlattenObservation(self.env)
        
        # Check observation space
        self.state_dim = self.env.observation_space.shape[0]
        # (Likely 147 for 7x7 view, or larger if fully observable)
        
        # Action space is Discrete(N)
        self.num_actions = self.env.action_space.n
        self.action_dim = self.num_actions 
        
    def reset(self):
        s, _ = self.env.reset()
        return self._process_obs(s)

    def step(self, action):
        """
        Args:
            action: Scalar ID or Soft Vector
        """
        if hasattr(action, 'shape') and len(action.shape) > 0:
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            action_id = np.argmax(action)
        else:
            action_id = action

        ns, reward, terminated, truncated, info = self.env.step(action_id)
        done = terminated or truncated
        
        return self._process_obs(ns), reward, done, info

    def _process_obs(self, obs):
        """
        Normalize observations [0, 10] -> [0, 1]
        MiniGrid values are usually object IDs < 10.
        """
        return obs.astype(np.float32) / 10.0
