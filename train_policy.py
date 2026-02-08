import torch
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

# --- LAMBDA-STYLE TRAJECTORY BUFFER ---
class TrajectoryBuffer:
    """
    LAMBDA-style buffer (Algorithm 2, Line 1: Initialize D)
    - Stores full dataset D of all trajectories (for diversity)
    - Maintains recent trajectories separately (for on-policy initial states)
    - Samples from recent trajectories for imagined rollouts (Line 7)
    """
    def __init__(self, state_dim, capacity=50000):
        self.state_dim = state_dim
        self.capacity = capacity
        
        # Full dataset D (all historical states)
        self.all_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        
        # Recent trajectories (for on-policy sampling)
        self.recent_trajectories = []
        self.max_recent_trajectories = 20  # Keep last 20 episodes
        self.current_trajectory = []
    
    def add_state(self, state):
        """Add state to current trajectory"""
        self.current_trajectory.append(state)
    
    def finish_trajectory(self):
        """
        Called when episode ends - stores trajectory and adds states to dataset D
        This implements Line 17: Update dataset D ← D ∪ {o_{1:T}, ...}
        """
        if len(self.current_trajectory) == 0:
            return
        
        # Add to recent trajectories
        self.recent_trajectories.append(np.array(self.current_trajectory))
        if len(self.recent_trajectories) > self.max_recent_trajectories:
            self.recent_trajectories.pop(0)  # Remove oldest
        
        # Add all states to full dataset D
        for state in self.current_trajectory:
            self.all_states[self.ptr] = state
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        
        # Clear current trajectory
        self.current_trajectory = []
    
    def sample_recent(self, batch_size):
        """
        Sample states from recent trajectories (LAMBDA approach)
        Line 7: Use each state in s_{τ':τ'+L} as initial state
        """
        if len(self.recent_trajectories) == 0:
            # Fallback to random sampling if no recent trajectories
            return self.sample_diverse(batch_size)
        
        states = []
        for _ in range(batch_size):
            # Sample a random recent trajectory
            traj_idx = np.random.randint(0, len(self.recent_trajectories))
            trajectory = self.recent_trajectories[traj_idx]
            
            # Sample a random state from that trajectory
            state_idx = np.random.randint(0, len(trajectory))
            states.append(trajectory[state_idx])
        
        return torch.tensor(np.array(states), dtype=torch.float32).to(CONFIG["DEVICE"])
    
    def sample_diverse(self, batch_size):
        """
        Sample uniformly from full dataset D (for diversity)
        Used for model training to see diverse states
        """
        if self.size == 0:
            # Return random states if buffer is empty
            return torch.randn(batch_size, self.state_dim).to(CONFIG["DEVICE"])
        
        idx = np.random.randint(0, self.size, size=batch_size)
        return torch.tensor(self.all_states[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
    
    def sample_hybrid(self, batch_size, recent_ratio=0.7):
        """
        Hybrid sampling: Mix of recent (70%) and diverse (30%)
        Balances on-policy learning with exploration
        """
        n_recent = int(batch_size * recent_ratio)
        n_diverse = batch_size - n_recent
        
        recent_states = self.sample_recent(n_recent)
        diverse_states = self.sample_diverse(n_diverse)
        
        # Shuffle to mix them
        all_states = torch.cat([recent_states, diverse_states], dim=0)
        perm = torch.randperm(all_states.shape[0])
        return all_states[perm]

# --- LAMBDA VALUE COMPUTATION (Algorithm 2, Lines 8-9) ---
def compute_lambda_values(next_values, rewards, dones, discount, lambda_):
    """
    Compute TD-lambda returns (Algorithm 1 referenced in Line 7)
    Args:
        next_values: (B, H) - V(s_{t+1})
        rewards: (B, H) - r_t
        dones: (B, H) - terminal flags
        discount: scalar
        lambda_: scalar
    Returns:
        lambda_values: (B, H) - V^λ(s_t)
    """
    batch_size, horizon = rewards.shape
    lambda_values = torch.zeros_like(rewards)
    
    # Backward computation as in Algorithm 1
    v_lambda = next_values[:, -1] * (1.0 - dones[:, -1])
    
    for t in reversed(range(horizon)):
        td = rewards[:, t] + (1.0 - dones[:, t]) * (1.0 - lambda_) * discount * next_values[:, t]
        v_lambda = td + v_lambda * lambda_ * discount * (1.0 - dones[:, t])
        lambda_values[:, t] = v_lambda
    
    return lambda_values

# --- AGENT WRAPPER (matches benchmarks_safety.py) ---
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
            self.mdn.load_state_dict(torch.load(f"pretrained_mdn_{env_name}.pth", map_location=device))
            self.mdn.eval()  # Freeze
            for param in self.mdn.parameters():
                param.requires_grad = False
                
        elif "Flow Only" in agent_type:
            self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
            self.flow.load_state_dict(torch.load(f"pretrained_flow_{env_name}_ForwardKL.pth", map_location=device))
            self.flow.eval()  # Freeze
            for param in self.flow.parameters():
                param.requires_grad = False
                
        else:  # EBM-based methods (Cold Start, Warm Start, SVGD)
            self.ebm = EnergyBasedModel(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
            self.ebm.load_state_dict(torch.load(f"pretrained_ebm_{env_name}.pth", map_location=device))
            self.ebm.eval()  # Freeze
            for param in self.ebm.parameters():
                param.requires_grad = False
            
            # Load flow for warm start
            if "Warm Start" in agent_type:
                suffix = "ForwardKL" if "ForwardKL" in agent_type else "ReverseKL"
                self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
                self.flow.load_state_dict(torch.load(f"pretrained_flow_{env_name}_{suffix}.pth", map_location=device))
                self.flow.eval()  # Freeze
                for param in self.flow.parameters():
                    param.requires_grad = False
    
    def predict_next_state(self, state, action):
        """Predict next state using the agent's sampling method"""
        with torch.no_grad():  # Models are frozen
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
            
            else:  # Langevin-based (Cold Start or Warm Start)
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

# --- IMAGINED ROLLOUT (Algorithm 2, Line 7) ---
def generate_imagined_rollout(agent, actor, initial_states, horizon, device):
    """
    Generate imagined trajectories using the frozen world model
    This corresponds to Line 7: "Compute Σ V_λ(s_t), Σ V_λ^*(s_t) via Algorithm 1"
    
    Args:
        agent: Agent with frozen pretrained model
        actor: Actor network (being trained)
        initial_states: (B, state_dim) - starting states from real env
        horizon: int - rollout length H
        device: torch device
    Returns:
        states: (B, H, state_dim)
        actions: (B, H, action_dim)
        rewards: (B, H, 1) - zeros for now (no reward model)
        dones: (B, H, 1) - zeros (assuming no termination in short horizon)
    """
    batch_size = initial_states.shape[0]
    
    states_list = []
    actions_list = []
    
    current_state = initial_states
    
    for t in range(horizon):
        # Sample action from policy (Line 14: Sample a_t ~ π_ξ(·|s_t))
        action = actor.sample(current_state)
        
        # Predict next state using frozen world model (Line 13: Infer s_t)
        next_state = agent.predict_next_state(current_state, action)
        
        states_list.append(current_state)
        actions_list.append(action)
        
        current_state = next_state
    
    states = torch.stack(states_list, dim=1)  # (B, H, state_dim)
    actions = torch.stack(actions_list, dim=1)  # (B, H, action_dim)
    
    # Note: We don't have a reward model, so rewards are zeros
    # In full implementation, you'd train a reward predictor on real data
    rewards = torch.zeros(batch_size, horizon, 1).to(device)
    dones = torch.zeros(batch_size, horizon, 1).to(device)
    
    return states, actions, rewards, dones

# --- CRITIC UPDATE (Algorithm 2, Line 8) ---
def update_critic(critic, critic_target, states, lambda_values, optimizer):
    """
    Update ψ and ψ^λ via Equation (9) with Σ V_λ(s_t) and Σ V_λ^*(s_t)
    """
    batch_size, horizon, state_dim = states.shape
    states_flat = states.reshape(-1, state_dim)
    
    # Critic predicts V(s)
    values = critic(states_flat, torch.zeros(states_flat.shape[0], 2).to(states_flat.device))
    values = values.reshape(batch_size, horizon)
    
    # TD-lambda targets
    loss = F.mse_loss(values, lambda_values.detach())
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

# --- ACTOR UPDATE (Algorithm 2, Line 9) ---
def update_actor(actor, critic, states, optimizer):
    """
    Update ξ according to Equation (10) with Σ V_λ(s_t) and Σ V_λ^*(s_t)
    """
    batch_size, horizon, state_dim = states.shape
    states_flat = states.reshape(-1, state_dim)
    
    # Sample actions from current policy
    actions = actor.sample(states_flat)
    
    # Compute values
    values = critic(states_flat, actions)
    
    # Policy gradient: maximize expected value
    loss = -values.mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
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
            action = np.clip(action, -1.0, 1.0)  # Clip action
            
            next_state, reward, cost, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
    
    # CRITICAL: Reset env after evaluation to ensure clean state
    env.reset()
    
    return np.mean(episode_rewards), np.std(episode_rewards)

# --- TRAIN SINGLE AGENT ---
def train_agent(agent_type, env, state_buffer):
    """
    Train policy for a single agent type
    Algorithm 2, Lines 3-11 (model updates) + Lines 12-17 (real env interaction)
    """
    print(f"\n{'='*60}")
    print(f"Training Agent: {agent_type}")
    print(f"{'='*60}")
    
    device = CONFIG["DEVICE"]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent with frozen pretrained model
    agent = Agent(agent_type, CONFIG["ENV_NAME"], state_dim, action_dim, device)
    
    # Initialize trainable actor and critic
    actor = Actor(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic = Critic(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target = Critic(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target.load_state_dict(critic.state_dict())
    
    # Optimizers
    actor_opt = optim.Adam(actor.parameters(), lr=CONFIG["LR_ACTOR"])
    critic_opt = optim.Adam(critic.parameters(), lr=CONFIG["LR_CRITIC"])
    
    # Tracking
    eval_rewards = []
    eval_steps = []
    episode_count = 0
    
    # Training loop (Algorithm 2, Lines 2-18)
    total_steps = 0
    state, _ = env.reset()
    
    while total_steps < CONFIG["TOTAL_STEPS"]:
        # Add current state to trajectory (Line 17: building trajectory)
        state_buffer.add_state(state)
        
        # Algorithm 2, Lines 3-11: Update policy on imagined data
        if state_buffer.size >= CONFIG["BATCH_SIZE"]:
            # Sample initial states from recent trajectories (Line 7)
            # Use hybrid sampling: 70% recent, 30% diverse for balance
            initial_states = state_buffer.sample_hybrid(CONFIG["BATCH_SIZE"], recent_ratio=0.7)
            
            # Generate imagined rollouts (Line 7)
            states, actions, rewards, dones = generate_imagined_rollout(
                agent, actor, initial_states, CONFIG["HORIZON"], device
            )
            
            # Compute values with target critic
            batch_size, horizon, state_dim = states.shape
            states_flat = states.reshape(-1, state_dim)
            with torch.no_grad():
                next_values = critic_target(states_flat, torch.zeros(states_flat.shape[0], action_dim).to(device))
                next_values = next_values.reshape(batch_size, horizon)
            
            # Compute lambda values (Line 7: via Algorithm 1)
            lambda_values = compute_lambda_values(
                next_values, rewards.squeeze(-1), dones.squeeze(-1),
                CONFIG["DISCOUNT"], CONFIG["LAMBDA"]
            )
            
            # Update critic (Line 8)
            critic_loss = update_critic(critic, critic_target, states, lambda_values, critic_opt)
            
            # Update actor (Line 9)
            actor_loss = update_actor(actor, critic, states, actor_opt)
            
            # Soft update of target critic
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
        
        # Algorithm 2, Lines 12-17: Real environment interaction
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor.sample(state_tensor).cpu().numpy()[0]
        
        # Clip action to valid range for safety
        action = np.clip(action, -1.0, 1.0)
        
        next_state, reward, cost, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_steps += 1
        
        # Handle episode termination (Algorithm 2, Line 17)
        if done:
            # Finish trajectory and add to dataset D
            state_buffer.finish_trajectory()
            episode_count += 1
            state, _ = env.reset()
        else:
            state = next_state
        
        # Evaluation
        if total_steps % CONFIG["EVAL_INTERVAL"] == 0:
            eval_mean, eval_std = evaluate_policy(env, actor, CONFIG["EVAL_EPISODES"])
            eval_rewards.append(eval_mean)
            eval_steps.append(total_steps)
            print(f"Step {total_steps}/{CONFIG['TOTAL_STEPS']} | Eval: {eval_mean:.2f} ± {eval_std:.2f}")
            # Reset state after evaluation to continue training
            state, _ = env.reset()
    
    # Finish any incomplete trajectory at end of training
    state_buffer.finish_trajectory()
    
    print(f"\nAgent training complete: {episode_count} episodes, {state_buffer.size} states in D")
    return eval_steps, eval_rewards

# --- MAIN TRAINING LOOP ---
def train():
    print(f"\n{'='*60}")
    print(f"POLICY CONVERGENCE TRAINING")
    print(f"Environment: {CONFIG['ENV_NAME']}")
    print(f"Device: {CONFIG['DEVICE']}")
    print(f"{'='*60}")
    
    device = CONFIG["DEVICE"]
    
    # Initialize environment
    env = safety_gymnasium.make(CONFIG["ENV_NAME"], render_mode=None)
    state_dim = env.observation_space.shape[0]
    
    # Initialize trajectory buffer (Algorithm 2, Line 1: Initialize D)
    state_buffer = TrajectoryBuffer(state_dim, capacity=50000)
    
    # Collect initial trajectories with random policy
    print(f"\nCollecting initial trajectories (Dataset D)...")
    state, _ = env.reset()
    episodes_collected = 0
    steps_collected = 0
    
    while episodes_collected < 50:  # Collect 50 initial episodes
        state_buffer.add_state(state)
        action = env.action_space.sample()
        next_state, _, _, terminated, truncated, _ = env.step(action)
        steps_collected += 1
        
        if terminated or truncated:
            state_buffer.finish_trajectory()
            episodes_collected += 1
            state, _ = env.reset()
        else:
            state = next_state
    
    print(f"Collected {episodes_collected} episodes ({state_buffer.size} states) in dataset D")
    
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
    plt.title("Policy Convergence Comparison", fontsize=14, fontweight='bold')
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
