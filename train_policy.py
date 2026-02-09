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
    from models import Actor, Critic, EnergyBasedModel, RealNVP, MixtureDensityNetwork
    from utils_sampling import predict_next_state_langevin, predict_next_state_svgd
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
CONFIG = {
    "ENV_NAME": "SafetyPointGoal1-v0",
    
    # Agent types to test (matching benchmarks_safety.py)
    "AGENT_TYPES": [
        "Cold Start",
        "Warm Start (ForwardKL)",
        "Warm Start (ReverseKL)",
        "MDN",
        "SVGD",
        "Flow Only"
    ],
    
    # Training
    "TOTAL_STEPS": 50000,      # Total training steps per agent
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
        """Sample (s, a, r) pairs for training reward model"""
        if self.size == 0:
            return None
        
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
                
        else:  # EBM-based methods
            self.ebm = EnergyBasedModel(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
            self.ebm.load_state_dict(torch.load(f"pretrained_ebm_{env_name}.pth", map_location=device, weights_only=False))
            self.ebm.eval()
            for param in self.ebm.parameters():
                param.requires_grad = False
            
            if "Warm Start" in agent_type:
                suffix = "ForwardKL" if "ForwardKL" in agent_type else "ReverseKL"
                self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
                self.flow.load_state_dict(torch.load(f"pretrained_flow_{env_name}_{suffix}.pth", map_location=device, weights_only=False))
                self.flow.eval()
                for param in self.flow.parameters():
                    param.requires_grad = False
    
    def predict_next_state(self, state, action):
        """Predict next state using the agent's sampling method"""
        with torch.no_grad():
            if "MDN" in self.agent_type:
                return self.mdn.sample_differentiable(state, action)
            
            elif "Flow Only" in self.agent_type:
                z = torch.randn_like(state).to(self.device)
                context = torch.cat([state, action], dim=1)
                return self.flow.sample(z, context=context)
            
            elif "SVGD" in self.agent_type:
                return predict_next_state_svgd(
                    self.ebm, state, action, 
                    config={"SVGD_STEPS": CONFIG["SVGD_STEPS"]}
                )
            
            else:  # Langevin-based
                init = None
                steps = CONFIG["LANGEVIN_STEPS_COLD"]
                
                if "Warm Start" in self.agent_type:
                    z = torch.randn_like(state).to(self.device)
                    context = torch.cat([state, action], dim=1)
                    init = self.flow.sample(z, context=context)
                    steps = CONFIG["LANGEVIN_STEPS_WARM"]
                
                return predict_next_state_langevin(
                    self.ebm, state, action, 
                    init_state=init,
                    config={"LANGEVIN_STEPS": steps}
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
        
        current_state = next_state
    
    states = torch.stack(states_list, dim=1)  # (B, H, state_dim)
    actions = torch.stack(actions_list, dim=1)  # (B, H, action_dim)
    rewards = torch.stack(rewards_list, dim=1)  # (B, H, 1) - PREDICTED!
    dones = torch.zeros(batch_size, horizon, 1).to(device)
    
    return states, actions, rewards, dones

# --- CRITIC UPDATE ---
def update_critic(critic, critic_target, states, actions, lambda_values, optimizer):
    """Update critic (state-action value function)"""
    batch_size, horizon, state_dim = states.shape
    states_flat = states.reshape(-1, state_dim)
    actions_flat = actions.reshape(-1, actions.shape[-1])
    
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
    Update actor to maximize expected value
    Uses actions from trajectory generation (supervisor feedback fix)
    """
    batch_size, horizon, state_dim = states.shape
    states_flat = states.reshape(-1, state_dim)
    actions_flat = actions.reshape(-1, actions.shape[-1])
    
    # Use actions from trajectory 
    values = critic(states_flat, actions_flat)
    
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
    
    # Tracking
    eval_rewards = []
    eval_steps = []
    episode_count = 0
    
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
            state, _ = env.reset()
        else:
            state = next_state
        
        # Algorithm 2, Lines 3-11: Update on imagined data
        if state_buffer.size >= CONFIG["BATCH_SIZE"] and total_steps % 10 == 0:
            # Train reward model on real data (Line 113)
            reward_loss = update_reward_model(reward_model, state_buffer, reward_opt)
            
            # Sample initial states (70% recent, 30% diverse)
            n_recent = int(CONFIG["BATCH_SIZE"] * 0.7)
            n_diverse = CONFIG["BATCH_SIZE"] - n_recent
            recent_states = state_buffer.sample_recent_states(n_recent)
            diverse_states = state_buffer.sample_diverse_states(n_diverse)
            initial_states = torch.cat([recent_states, diverse_states], dim=0)
            
            # Generate imagined rollouts WITH predicted rewards (Line 149)
            states, actions, rewards, dones = generate_imagined_rollout(
                agent, actor, reward_model, initial_states, CONFIG["HORIZON"], device
            )
            
            # Compute values with target critic
            batch_size, horizon, state_dim = states.shape
            states_flat = states.reshape(-1, state_dim)
            actions_flat = actions.reshape(-1, action_dim)
            with torch.no_grad():
                next_values = critic_target(states_flat, actions_flat)
                next_values = next_values.reshape(batch_size, horizon)
            
            # Compute lambda values from PREDICTED rewards (Line 102-106)
            lambda_values = compute_lambda_values(
                next_values, rewards.squeeze(-1), dones.squeeze(-1),
                CONFIG["DISCOUNT"], CONFIG["LAMBDA"]
            )
            
            # Update critic (Line 118-120)
            critic_loss = update_critic(critic, critic_target, states, actions, lambda_values, critic_opt)
            
            # Update actor (Line 106) - reuse actions from trajectory
            actor_loss = update_actor(actor, critic, states, actions, actor_opt)
            
            # Soft update of target critic
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
        
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
    env = safety_gymnasium.make(CONFIG["ENV_NAME"], render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize trajectory buffer
    state_buffer = TrajectoryBuffer(state_dim, action_dim, capacity=50000)
    
    # Collect initial trajectories with random policy
    print(f"\nCollecting initial trajectories (Dataset D)...")
    state, _ = env.reset()
    episodes_collected = 0
    
    while episodes_collected < 50:
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
