import gymnasium as gym
import minigrid
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper
from gymnasium.wrappers import FlattenObservation
from minigrid_adapter import MiniGridAdapter

class StochasticMiniGridAdapter(MiniGridAdapter):
    def __init__(self, env_name="MiniGrid-Empty-8x8-v0", render_mode=None, slip_prob=0.3):
        super().__init__(env_name, render_mode)
        self.slip_prob = slip_prob
        print(f"initialized StochasticMiniGridAdapter with slip_prob={slip_prob}")

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

        # --- STOCHASTICITY INJECTION ---
        # 0: left, 1: right, 2: forward
        # If action is Forward (2), chance to slip and Turn Left (0) or Turn Right (1)
        # If action is Turn (0/1), chance to stay put or move forward?
        # Let's keep it simple: "Slippery Floor" affects movement.
        
        final_action = action_id
        
        # Only inject stochasticity if action is Forward (2)
        if action_id == 2: 
            if np.random.rand() < self.slip_prob:
                # Slip!
                # 50% chance to turn left, 50% chance to turn right (Simulation of sliding)
                if np.random.rand() < 0.5:
                    final_action = 0 # Left
                else:
                    final_action = 1 # Right
                # print(f"  -> SLIP! Intended: {action_id}, Executed: {final_action}")
        
        # Execute
        ns, reward, terminated, truncated, info = self.env.step(final_action)
        done = terminated or truncated
        
        return self._process_obs(ns), reward, done, info
