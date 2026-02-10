import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from collections import deque

# --- IMPORTS ---
try:
    import safety_gymnasium
    from models import Actor, Critic, BilinearEBM, RealNVP, MixtureDensityNetwork
    from utils_sampling import (
        predict_next_state_langevin_adaptive, 
        predict_next_state_svgd_adaptive
    )
    # Import InfoNCE loss for online EBM updates
    from pretrain_ebm_safety_gym import infonce_loss
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
CONFIG = {
    "ENV_NAME": "SafetyPointGoal1-v0",
    
    # Agent types to test (matching benchmarks_safety.py)
    "AGENT_TYPES": [
        "Warm Start (ReverseKL)",
        "Warm Start (ForwardKL)",
        "Cold Start",
        "MDN",
        "SVGD",
        "Flow Only"
    ],
    
    # Training
    "TOTAL_STEPS": 30000,      # Total training steps per agent
    "BATCH_SIZE": 256,         # Batch size for imagined rollouts
    "HORIZON": 5,              # H for imagined rollouts
    
    # TD-Lambda (Algorithm 2, Lines 8-9)
    "DISCOUNT": 0.99,
    "LAMBDA": 0.95,
    
    # Optimizers
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 1e-3,
    "LR_REWARD": 1e-3,  # Reward model learning rate
    
    # Sampling (for EBM-based methods)
    "LANGEVIN_STEPS_COLD": 30,   # Cold start needs more steps
    "LANGEVIN_STEPS_WARM": 5,    # Warm start needs fewer steps
    "SVGD_STEPS": 10,
    
    # Tuning: 0.05 seems safer than 0.5 (exploded) but better than 0.01 (stagnant)
    "LANGEVIN_STEP_SIZE": 0.05,
    "LANGEVIN_NOISE_SCALE": 0.01,
    
    # Logging
    "LOG_INTERVAL": 100,
    "EVAL_INTERVAL": 1000,
    "EVAL_EPISODES": 5,
    
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "HIDDEN_DIM": 128,
}

# --- REWARD MODEL (LAMBDA Line 112-113) ---
class RewardModel(nn.Module):
    """
    Reward decoder: r̂(s,a) 
    Trained on real environment data (Line 113)
    Used in imagined rollouts (Line 149)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(RewardModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """Predict reward given state and action"""
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# --- LAMBDA-STYLE TRAJECTORY BUFFER ---
class TrajectoryBuffer:
    """
    LAMBDA-style buffer (Algorithm 2, Line 1: Initialize D)
    Stores full trajectories with states, actions, rewards
    """
    def __init__(self, state_dim, action_dim, capacity=50000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        
        # Full dataset D (all historical transitions)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        
        # Recent trajectories (for on-policy sampling)
        self.recent_trajectories = []
        self.max_recent_trajectories = 20  # Keep last 20 episodes
        self.current_trajectory = {'states': [], 'actions': [], 'rewards': []}
        
        # Track positive rewards for weighted sampling (Reward Model fix)
        self.positive_indices = []
    
    def add_transition(self, state, action, reward, next_state):
        """Add transition to current trajectory"""
        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        
        # Also add to dataset D for reward model training
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        
        # Track positive rewards (threshold > 0.0001 to capture small rewards)
        if reward > 0.0001:
            if self.ptr not in self.positive_indices:
                self.positive_indices.append(self.ptr)
        else:
            # If overwriting a positive index with a negative one, remove it
            if self.ptr in self.positive_indices:
                self.positive_indices.remove(self.ptr)
                
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def finish_trajectory(self):
        """
        Called when episode ends
        Line 17: Update dataset D ← D ∪ {o_{1:T}, a_{1:T}, r_{1:T}}
        """
        if len(self.current_trajectory['states']) == 0:
            return
        
        # Add to recent trajectories
        self.recent_trajectories.append({
            'states': np.array(self.current_trajectory['states']),
            'actions': np.array(self.current_trajectory['actions']),
            'rewards': np.array(self.current_trajectory['rewards'])
        })
        if len(self.recent_trajectories) > self.max_recent_trajectories:
            self.recent_trajectories.pop(0)
        
        # Clear current trajectory
        self.current_trajectory = {'states': [], 'actions': [], 'rewards': []}
    
    def sample_recent_states(self, batch_size):
        """Sample states from recent trajectories for imagined rollouts"""
        if len(self.recent_trajectories) == 0:
            return self.sample_diverse_states(batch_size)
        
        states = []
        for _ in range(batch_size):
            traj_idx = np.random.randint(0, len(self.recent_trajectories))
            trajectory = self.recent_trajectories[traj_idx]
            state_idx = np.random.randint(0, len(trajectory['states']))
            states.append(trajectory['states'][state_idx])
        
        return torch.tensor(np.array(states), dtype=torch.float32).to(CONFIG["DEVICE"])
    
    def sample_diverse_states(self, batch_size):
        """Sample uniformly from full dataset D"""
        if self.size == 0:
            return torch.randn(batch_size, self.state_dim).to(CONFIG["DEVICE"])
        
        idx = np.random.randint(0, self.size, size=batch_size)
        return torch.tensor(self.states[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
    
    def sample_for_reward_training(self, batch_size):
        """
        Sample (s, a, r) pairs for training reward model.
        Uses Weighted Sampling: 50% positive rewards, 50% random.
        """
        if self.size == 0:
            return None
        
        # Weighted Sampling
        num_pos = len(self.positive_indices)
        if num_pos > 0:
            half_batch = batch_size // 2
            # Sample 50% from positive indices
            pos_idx = np.random.choice(self.positive_indices, size=half_batch, replace=True)
            # Sample 50% from all indices (random)
            rand_idx = np.random.randint(0, self.size, size=batch_size - half_batch)
            idx = np.concatenate([pos_idx, rand_idx])
            np.random.shuffle(idx)
        else:
            # Fallback if no positive rewards yet
            idx = np.random.randint(0, self.size, size=batch_size)
            
        return {
            'states': torch.tensor(self.states[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            'actions': torch.tensor(self.actions[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            'rewards': torch.tensor(self.rewards[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
        }

# --- LAMBDA VALUE COMPUTATION (Algorithm 1) ---
def compute_lambda_values(next_values, rewards, dones, discount, lambda_):
    """
    Compute TD-lambda returns using PREDICTED rewards
    rewards: (B, H) - from reward model, not zeros!
    """
    batch_size, horizon = rewards.shape
    lambda_values = torch.zeros_like(rewards)
    
    # Backward computation
    v_lambda = next_values[:, -1] * (1.0 - dones[:, -1])
    
    for t in reversed(range(horizon)):
        td = rewards[:, t] + (1.0 - dones[:, t]) * (1.0 - lambda_) * discount * next_values[:, t]
        v_lambda = td + v_lambda * lambda_ * discount * (1.0 - dones[:, t])
        lambda_values[:, t] = v_lambda
    
    return lambda_values

# --- AGENT WRAPPER ---
class Agent:
    """Wrapper for different agent types using pretrained models"""
    def __init__(self, agent_type, env_name, state_dim, action_dim, device):
        self.agent_type = agent_type
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Load pretrained models (frozen - no training)
        self.ebm = None
        self.flow = None
        self.mdn = None
        
        # CRITICAL: BilinearEBM uses higher=better energy convention
        # Need gradient ASCENT for sampling
        self.use_ascent = False  # Will be set to True for BilinearEBM
        
        if "MDN" in agent_type:
            self.mdn = MixtureDensityNetwork(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
            self.mdn.load_state_dict(torch.load(f"pretrained_mdn_{env_name}.pth", map_location=device, weights_only=False))
            self.mdn.eval()
            for param in self.mdn.parameters():
                param.requires_grad = False
                
        elif "Flow Only" in agent_type:
            self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
            self.flow.load_state_dict(torch.load(f"pretrained_flow_{env_name}_ForwardKL.pth", map_location=device, weights_only=False))
            self.flow.eval()
            for param in self.flow.parameters():
                param.requires_grad = False
                
        else:  # EBM-based methods (Langevin or SVGD)
            # Load BilinearEBM (pretrained with InfoNCE)
            self.ebm = BilinearEBM(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
            self.ebm.load_state_dict(torch.load(f"pretrained_ebm_{env_name}.pth", map_location=device, weights_only=False))
            
            # Keep in training mode for online updates
            self.ebm.train()
            for param in self.ebm.parameters():
                param.requires_grad = True  # Unfreeze for online learning
            
            # BilinearEBM uses higher=better → need gradient ASCENT
            self.use_ascent = True
            
            if "Warm Start" in agent_type:
                suffix = "ForwardKL" if "ForwardKL" in agent_type else "ReverseKL"
                self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
                self.flow.load_state_dict(torch.load(f"pretrained_flow_{env_name}_{suffix}.pth", map_location=device, weights_only=False))
                self.flow.eval()
                for param in self.flow.parameters():
                    param.requires_grad = False
    
    def predict_next_state(self, state, action):
        """
        Predict next state using the agent's sampling method with correct energy convention.
        
        Note: NO torch.no_grad() here because Langevin/SVGD need gradients for sampling!
        - Gradients are for computing ∂E/∂s' (part of sampling algorithm)
        - Model parameters are already frozen (requires_grad=False in __init__)
        - So no model training happens, just gradient-based sampling
        """
        if "MDN" in self.agent_type:
            return self.mdn.sample_differentiable(state, action)
        
        elif "Flow Only" in self.agent_type:
            z = torch.randn_like(state).to(self.device)
            context = torch.cat([state, action], dim=1)
            return self.flow.sample(z, context=context)
        
        elif "SVGD" in self.agent_type:
            # Use adaptive SVGD with gradient ASCENT for BilinearEBM
            return predict_next_state_svgd_adaptive(
                self.ebm, state, action,
                use_ascent=self.use_ascent,  # True for BilinearEBM
                config={"SVGD_STEPS": CONFIG["SVGD_STEPS"]}
            )
        
        else:  # Langevin-based (Cold Start or Warm Start)
            init = None
            steps = CONFIG["LANGEVIN_STEPS_COLD"]
            
            if "Warm Start" in self.agent_type:
                z = torch.randn_like(state).to(self.device)
                context = torch.cat([state, action], dim=1)
                init = self.flow.sample(z, context=context)
                steps = CONFIG["LANGEVIN_STEPS_WARM"]
            
            # Use adaptive Langevin with gradient ASCENT for BilinearEBM
            return predict_next_state_langevin_adaptive(
                self.ebm, state, action,
                init_state=init,
                use_ascent=self.use_ascent,  # True for BilinearEBM
                config={
                    "LANGEVIN_STEPS": steps,
                    "LANGEVIN_STEP_SIZE": CONFIG["LANGEVIN_STEP_SIZE"],
                    "LANGEVIN_NOISE_SCALE": CONFIG["LANGEVIN_NOISE_SCALE"]
                }
            )

# --- IMAGINED ROLLOUT WITH REWARD MODEL (Line 149) ---
def generate_imagined_rollout(agent, actor, reward_model, initial_states, horizon, device):
    """
    Generate imagined trajectories with PREDICTED rewards
    Line 149: stacked['reward'] = self._reward_decoder(stacked['features']).mean()
    """
    batch_size = initial_states.shape[0]
    
    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = [] # New: Collect next states
    
    current_state = initial_states
    
    for t in range(horizon):
        # Sample action from policy
        action = actor.sample(current_state)
        
        # Predict reward using reward model (Line 149!)
        predicted_reward = reward_model(current_state, action)
        
        # Predict next state using world model
        next_state = agent.predict_next_state(current_state, action)
        
        # Clip next state to prevent explosions
        next_state = torch.clamp(next_state, -10.0, 10.0)
        
        states_list.append(current_state)
        actions_list.append(action)
        rewards_list.append(predicted_reward)
        next_states_list.append(next_state) # Collect next state
        
        current_state = next_state
    
    states = torch.stack(states_list, dim=1)
    actions = torch.stack(actions_list, dim=1)
    rewards = torch.stack(rewards_list, dim=1)
    next_states = torch.stack(next_states_list, dim=1) # (B, H, state_dim)
    
    # Debug: Check if reward model predicts anything useful
    if np.random.rand() < 0.05: # Log 5% of the time to avoid spam
        print(f"    [Rollout Rewards] Mean: {rewards.mean().item():.6f} | Max: {rewards.max().item():.6f} | Min: {rewards.min().item():.6f}")
        
    dones = torch.zeros(batch_size, horizon, 1).to(device)
    
    return states, actions, rewards, next_states, dones

# --- CRITIC UPDATE ---
def update_critic(critic, critic_target, states, actions, lambda_values, optimizer):
    """Update critic (state-action value function)"""
    batch_size, horizon, state_dim = states.shape
    states_flat = states.reshape(-1, state_dim)
    actions_flat = actions.reshape(-1, actions.shape[-1]).detach()  # Detach to prevent graph reuse!
    
    # Critic predicts Q(s,a)
    values = critic(states_flat, actions_flat)
    values = values.reshape(batch_size, horizon)
    
    # TD-lambda targets
    loss = F.mse_loss(values, lambda_values.detach())
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

# --- ACTOR UPDATE ---
def update_actor(actor, critic, states, actions, optimizer):
    """
    Update actor to maximize expected value.
    
    Re-samples actions (not from trajectory) because rollout actions 
    lack gradients due to frozen world model context.
    """
    batch_size, horizon, state_dim = states.shape
    states_flat = states.reshape(-1, state_dim).detach()  # Detach states, only train actor
    
    # Re-sample actions WITH gradients (rollout actions have no gradients)
    actions_new = actor.sample(states_flat)
    values = critic(states_flat, actions_new)
    
    loss = -values.mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

# --- REWARD MODEL UPDATE (Line 113) ---
def update_reward_model(reward_model, buffer, optimizer, batch_size=256):
    """
    Train reward decoder on real environment data
    Line 113: log_p_rewards = tf.reduce_mean(self._reward_decoder(features).log_prob(batch['reward']))
    """
    batch = buffer.sample_for_reward_training(batch_size)
    if batch is None:
        return 0.0
    
    # Predict rewards
    predicted_rewards = reward_model(batch['states'], batch['actions'])
    
    # MSE loss with real rewards
    loss = F.mse_loss(predicted_rewards, batch['rewards'])
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

# --- EBM ONLINE UPDATE ---
def update_ebm_online(ebm, buffer, optimizer, num_steps=500, batch_size=128):
    """
    Update EBM on real environment data AND hard negatives.
    Uses InfoNCE on REAL transitions + Langevin sampled negatives.
    """
    if buffer.size < batch_size:
        return {'E_pos': 0, 'E_neg': 0, 'E_gap': 0, 'loss': 0}
    
    ebm.train()
    metrics = {}
    
    for step in range(num_steps):
        # 1. Sample Real Positive Data
        idx = np.random.randint(0, buffer.size, size=batch_size)
        states = torch.tensor(buffer.states[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
        actions = torch.tensor(buffer.actions[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
        next_states = torch.tensor(buffer.next_states[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
        
        # 2. Generate Hard Negatives (Langevin Sampling)
        # Generate K=1 explicit negative per positive to keep cost manageable
        # We start from random noise or valid next states? 
        # Structure needs to be (B, K=1, D). 
        # Let's run Langevin on a copy of next_states (Contrastive Divergence style start)
        # or Random start (Cold start). Random is safer to find far-away modes.
        
        # Use simple random initialization for negatives
        # We can re-use the function but need to reshape inputs to be simple
        with torch.no_grad():
             # Using the updated 0.05 step size + clipping
             generated_negatives = predict_next_state_langevin_adaptive(
                ebm, states, actions,
                init_state=None, # Random init
                use_ascent=True, # Ascent because we want to find HIGH energy modes to penalize
                config={"LANGEVIN_STEPS": 20} # Shorter chain for speed
            )
             generated_negatives = generated_negatives.unsqueeze(1) # (B, 1, D)
        
        # 3. Compute Loss with Hard Negatives
        loss, metrics = infonce_loss(
            ebm, states, actions, next_states, buffer,
            num_negatives=256, # Random negatives
            temperature=0.1,
            explicit_negatives=generated_negatives # Helper negatives
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ebm.parameters(), 1.0)
        optimizer.step()
    
    metrics['loss'] = loss.item()
    return metrics

# --- EVALUATION ---
def evaluate_policy(env, actor, num_episodes=5):
    """Evaluate current policy in real environment"""
    episode_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(CONFIG["DEVICE"])
            with torch.no_grad():
                action = actor.sample(state_tensor).cpu().numpy()[0]
            action = np.clip(action, -1.0, 1.0)
            
            next_state, reward, cost, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
    
    env.reset()  # Reset after evaluation
    return np.mean(episode_rewards), np.std(episode_rewards)

# --- TRAIN SINGLE AGENT ---
def train_agent(agent_type, env, state_buffer):
    """
    Train policy for a single agent type
    LAMBDA Algorithm 2 with reward model
    """
    print(f"\n{'='*60}")
    print(f"Training Agent: {agent_type}")
    print(f"{'='*60}")
    
    device = CONFIG["DEVICE"]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent with frozen pretrained model
    agent = Agent(agent_type, CONFIG["ENV_NAME"], state_dim, action_dim, device)
    
    # Initialize trainable networks
    actor = Actor(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic = Critic(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target = Critic(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target.load_state_dict(critic.state_dict())
    
    # Initialize reward model 
    reward_model = RewardModel(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    # Optimizers
    actor_opt = optim.Adam(actor.parameters(), lr=CONFIG["LR_ACTOR"])
    critic_opt = optim.Adam(critic.parameters(), lr=CONFIG["LR_CRITIC"])
    reward_opt = optim.Adam(reward_model.parameters(), lr=CONFIG["LR_REWARD"])
    
    # EBM optimizer for online updates
    ebm_opt = None
    if agent.ebm is not None:
        ebm_opt = optim.Adam(agent.ebm.parameters(), lr=1e-4)
        print(f"  EBM optimizer created - Will update every 5 episodes")
    
    # Tracking
    eval_rewards = []
    eval_steps = []
    episode_count = 0
    
    # Loss tracking
    track_reward_loss = []
    track_critic_loss = []
    track_actor_loss = []
    
    # Training loop
    total_steps = 0
    state, _ = env.reset()
    
    while total_steps < CONFIG["TOTAL_STEPS"]:
        # Algorithm 2, Lines 12-17: Real environment interaction
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor.sample(state_tensor).cpu().numpy()[0]
        action = np.clip(action, -1.0, 1.0)
        
        next_state, reward, cost, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
       # Add to buffer (Line 17)
        state_buffer.add_transition(state, action, reward, next_state)
        
        total_steps += 1
        
        if done:
            state_buffer.finish_trajectory()
            episode_count += 1
            
            # Online EBM update every 5 episodes
            if ebm_opt is not None and episode_count % 5 == 0 and state_buffer.size >= 500:
                print(f"  [Episode {episode_count}] Updating EBM on {state_buffer.size} real transitions...")
                
                # Log weight norm BEFORE update
                norm_before = sum(p.norm().item() for p in agent.ebm.parameters())
                
                metrics = update_ebm_online(agent.ebm, state_buffer, ebm_opt, num_steps=500)
                
                # Log weight norm AFTER update
                norm_after = sum(p.norm().item() for p in agent.ebm.parameters())
                delta = norm_after - norm_before
                
                print(f"    Energy gap: {metrics['E_gap']:.4f} (Pos={metrics['E_pos']:.4f}, Neg={metrics['E_neg']:.4f})")
                print(f"    EBM Weight Delta: {delta:.6f} (Norm: {norm_after:.4f})")
                
                if abs(delta) < 1e-6:
                    print("    WARNING: EBM weights did not change! Check optimizer.")
            
            state, _ = env.reset()
        else:
            state = next_state
        
        # Algorithm 2, Lines 3-11: Update on imagined data
        if state_buffer.size >= CONFIG["BATCH_SIZE"] and total_steps % 10 == 0:
            # Train reward model on real data (Line 113)
            reward_loss = update_reward_model(reward_model, state_buffer, reward_opt)
            track_reward_loss.append(reward_loss)
            
            # Sample initial states (70% recent, 30% diverse)
            n_recent = int(CONFIG["BATCH_SIZE"] * 0.7)
            n_diverse = CONFIG["BATCH_SIZE"] - n_recent
            recent_states = state_buffer.sample_recent_states(n_recent)
            diverse_states = state_buffer.sample_diverse_states(n_diverse)
            initial_states = torch.cat([recent_states, diverse_states], dim=0)
            
            # Generate imagined rollouts WITH predicted rewards (Line 149)
            states, actions, rewards, next_states, dones = generate_imagined_rollout(
                agent, actor, reward_model, initial_states, CONFIG["HORIZON"], device
            )
            
            # Log Energy of Imagined States (Verify if sampling climbs energy surface)
            if agent.ebm is not None and total_steps % 100 == 0:
                with torch.no_grad():
                    # Check energy of the LAST state in rollout
                    last_states = states[:, -1, :] # (B, D)
                    # We need "previous" state/action for energy, but here we just check 
                    # compatibility of (s_{H-1}, a_{H-1}, s_H). 
                    # Let's just approximate by checking avg energy of transitions in rollout
                    # Flatten: (B*H, D)
                    s_flat = states[:, :-1, :].reshape(-1, state_dim)
                    a_flat = actions[:, :-1, :].reshape(-1, action_dim)
                    ns_flat = states[:, 1:, :].reshape(-1, state_dim)
                    
                    if s_flat.shape[0] > 0:
                        energies = agent.ebm(s_flat, a_flat, ns_flat)
                        print(f"    [Img Rollout] Avg Energy: {energies.mean().item():.4f} +/- {energies.std().item():.4f}")
            
            # Compute values with target critic
            # FIX: Use next_states for bootstrapping V(s'), NOT current states!
            batch_size, horizon, state_dim = next_states.shape
            
            # Use optimal actions for next state (target policy)
            # We don't have next_actions, so we sample them using the ACTOR (target policy approx)
            # or just use Q(s', a')? 
            # Standard Actor-Critic: V(s') = Q(s', \pi(s'))
            
            # Flatten next states
            next_states_flat = next_states.reshape(-1, state_dim)
            
            with torch.no_grad():
                # Sample next actions from current actor (or target actor if we had one)
                next_actions_flat = actor.sample(next_states_flat)
                
                # Compute Q(s', a')
                next_values = critic_target(next_states_flat, next_actions_flat)
                next_values = next_values.reshape(batch_size, horizon)
            
            # Compute lambda values from PREDICTED rewards (Line 102-106)
            lambda_values = compute_lambda_values(
                next_values, rewards.squeeze(-1), dones.squeeze(-1),
                CONFIG["DISCOUNT"], CONFIG["LAMBDA"]
            )
            
            # Update critic (Line 118-120)
            critic_loss = update_critic(critic, critic_target, states, actions, lambda_values, critic_opt)
            track_critic_loss.append(critic_loss)
            
            # Update actor (Line 106) - reuse actions from trajectory
            actor_loss = update_actor(actor, critic, states, actions, actor_opt)
            track_actor_loss.append(actor_loss)
            
            # Soft update of target critic
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
                
            # Log Policy/Reward stats periodically
            if total_steps % 1000 == 0:
                avg_r_loss = np.mean(track_reward_loss[-100:]) if track_reward_loss else 0
                avg_c_loss = np.mean(track_critic_loss[-100:]) if track_critic_loss else 0
                avg_a_loss = np.mean(track_actor_loss[-100:]) if track_actor_loss else 0
                
                # Check reward stats in buffer
                r_mean = state_buffer.rewards[:state_buffer.size].mean()
                r_max = state_buffer.rewards[:state_buffer.size].max()
                r_min = state_buffer.rewards[:state_buffer.size].min()
                
                print(f"    [Step {total_steps}] Losses -> Reward: {avg_r_loss:.6f} | Critic: {avg_c_loss:.6f} | Actor: {avg_a_loss:.6f}")
                print(f"    [Buffer Stats] Reward Mean: {r_mean:.6f} | Max: {r_max:.6f} | Min: {r_min:.6f}")
        
        # Evaluation
        if total_steps % CONFIG["EVAL_INTERVAL"] == 0:
            eval_mean, eval_std = evaluate_policy(env, actor, CONFIG["EVAL_EPISODES"])
            eval_rewards.append(eval_mean)
            eval_steps.append(total_steps)
            print(f"Step {total_steps}/{CONFIG['TOTAL_STEPS']} | Eval: {eval_mean:.2f} ± {eval_std:.2f}")
            state, _ = env.reset()
    
    # Finish any incomplete trajectory
    state_buffer.finish_trajectory()
    
    print(f"\nAgent training complete: {episode_count} episodes, {state_buffer.size} transitions in D")
    return eval_steps, eval_rewards

# --- MAIN TRAINING LOOP ---
def train():
    print(f"\n{'='*60}")
    print(f"POLICY CONVERGENCE TRAINING (WITH REWARD MODEL)")
    print(f"Environment: {CONFIG['ENV_NAME']}")
    print(f"Device: {CONFIG['DEVICE']}")
    print(f"{'='*60}")
    
    device = CONFIG["DEVICE"]
    
    # Initialize environment
    if CONFIG["ENV_NAME"] == "StochasticGridWorld-v0":
        from stochastic_grid_world import StochasticGridWorld
        env = StochasticGridWorld()
        print("Initialized Custom Stochastic Grid World!")
    else:
        env = safety_gymnasium.make(CONFIG["ENV_NAME"], render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize trajectory buffer
    state_buffer = TrajectoryBuffer(state_dim, action_dim, capacity=50000)
    
    # Collect initial trajectories with random policy
    print(f"\nCollecting initial trajectories (Dataset D)...")
    state, _ = env.reset()
    episodes_collected = 0
    
    while episodes_collected < 5:
        action = env.action_space.sample()
        next_state, reward, cost, terminated, truncated, _ = env.step(action)
        
        state_buffer.add_transition(state, action, reward, next_state)
        
        if terminated or truncated:
            state_buffer.finish_trajectory()
            episodes_collected += 1
            state, _ = env.reset()
        else:
            state = next_state
    
    print(f"Collected {episodes_collected} episodes ({state_buffer.size} transitions) in dataset D")
    
    # Train each agent type
    results = {}
    for agent_type in CONFIG["AGENT_TYPES"]:
        try:
            steps, rewards = train_agent(agent_type, env, state_buffer)
            results[agent_type] = {"steps": steps, "rewards": rewards}
        except FileNotFoundError as e:
            print(f"!!! Skipping {agent_type}: Pretrained model not found")
            print(f"    Error: {e}")
            continue
    
    # Plot convergence comparison
    plt.figure(figsize=(12, 6))
    
    colors = {
        "Cold Start": "tab:blue",
        "Warm Start (ForwardKL)": "tab:orange",
        "Warm Start (ReverseKL)": "tab:green",
        "MDN": "tab:brown",
        "SVGD": "tab:red",
        "Flow Only": "tab:purple"
    }
    
    for agent_type, data in results.items():
        plt.plot(data["steps"], data["rewards"], 
                marker='o', label=agent_type, 
                linewidth=2, color=colors.get(agent_type, "black"))
    
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title("Policy Convergence Comparison (with Reward Model)", fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("policy_convergence_comparison.png", dpi=300)
    print(f"\n{'='*60}")
    print(f"Training complete! Plot saved to policy_convergence_comparison.png")
    print(f"{'='*60}")
    
    # Save results
    np.save("policy_convergence_results.npy", results)

if __name__ == "__main__":
    train()
