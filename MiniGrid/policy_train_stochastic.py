import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- IMPORTS ---
try:
    import minigrid
    import gymnasium as gym
    from models import Critic, BilinearEBM, RealNVP, MixtureDensityNetwork, ValueNetwork, RewardModel
    from models_minigrid import DiscreteActor
    from minigrid_stochastic import StochasticMiniGridAdapter # CHANGED
    from utils_sampling import (
        predict_next_state_langevin_adaptive, 
        predict_next_state_svgd_adaptive
    )
    # Import logic helper if possible, or redefine local
    from pretrain_ebm_minigrid import infonce_loss
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    sys.exit(1)

# --- HELPER FUNCTIONS ---

def compute_lambda_values(next_values, rewards, dones, discount, lambda_):
    """
    Compute TD-lambda returns.
    next_values: (B, H) - Value of state at t+1 (or smoothed value)
    rewards: (B, H)
    dones: (B, H)
    """
    batch_size, horizon = rewards.shape
    lambda_values = torch.zeros_like(rewards)
    
    last_next_val = next_values[:, -1]
    
    # Backward pass
    v_lambda = last_next_val * (1.0 - dones[:, -1])
    
    for t in reversed(range(horizon)):
        # TD error: r + gamma * V(s')
        # Here next_values[:, t] is V(s_{t+1})
        
        v_lambda = rewards[:, t] + (1.0 - dones[:, t]) * discount * (
            (1.0 - lambda_) * next_values[:, t] + lambda_ * v_lambda
        )
        lambda_values[:, t] = v_lambda
        
    return lambda_values

def update_ebm_online(ebm, buffer, optimizer, num_steps=50, batch_size=32):
    """
    Online EBM update using InfoNCE + Negative Sampling
    """
    ebm.train()
    if buffer.size < batch_size: return {}
    
    losses = []
    gap = 0
    
    for _ in range(num_steps):
        s, a, real_ns = buffer.sample(batch_size)
        
        loss, metrics = infonce_loss(
            ebm, s, a, real_ns, buffer,
            num_negatives=32, 
            temperature=0.1
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        gap = metrics['E_gap']
        
    return {'loss': np.mean(losses), 'E_gap': gap}


# --- CONFIGURATION ---
# --- CONFIGURATION ---
CONFIG = {
    "BASE_ENV": "MiniGrid-Empty-8x8-v0",   # The Gym ID for the underlying grid
    "MODEL_TAG": "MiniGrid-Stochastic-0.1", # UPDATED TAG: 0.1 Slip
    "AGENT_TYPES": [
        "Warm Start (ForwardKL)",
        # "Cold Start", 
    ],
    "TOTAL_STEPS": 50000, 
    "BATCH_SIZE": 256,
    "HORIZON": 10,          # Reverted to 10 per user request
    "DISCOUNT": 0.99,
    "LAMBDA": 0.95,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-3,
    "LR_REWARD": 1e-3,
    "ENTROPY_COEFF": 0.02,  # Compromise: 0.005 was too low, 0.05 too high
    "LANGEVIN_STEPS_COLD": 30,
    "LANGEVIN_STEPS_WARM": 5,
    "SVGD_STEPS": 10,
    "LANGEVIN_STEP_SIZE": 0.05,
    "LANGEVIN_NOISE_SCALE": 0.01,
    "EVAL_INTERVAL": 1000,
    "EVAL_EPISODES": 10,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "HIDDEN_DIM": 128,
}
# ... (rest of file)

def train():
    print("="*60)
    print(f"!!! STARTING STOCHASTIC TRAINING MODE !!!")
    print(f"Base Env: {CONFIG['BASE_ENV']}")
    print(f"Wrapper:  StochasticMiniGridAdapter (Slip Prob = 0.1)") # PRINT UPDATED
    print(f"Models:   {CONFIG['MODEL_TAG']}")
    print("="*60)
    
    device = CONFIG["DEVICE"]
    # Initialize Stochastic Adapter with BASE_ENV
    env_adapter = StochasticMiniGridAdapter(CONFIG['BASE_ENV'], slip_prob=0.1) # LOGIC UPDATED

# --- TRAJECTORY BUFFER ---
class TrajectoryBuffer:
    def __init__(self, state_dim, action_dim, capacity=50000):
        self.state_dim = state_dim
        self.action_dim = action_dim # Discrete One-Hot Size
        self.capacity = capacity
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        
        self.recent_trajectories = []
        self.max_recent_trajectories = 20
        self.current_trajectory = {'states': [], 'actions': [], 'rewards': []}
        self.positive_indices = []
    
    def add_transition(self, state, action, reward, next_state):
        # Action is vector (one-hot)
        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        
        if reward > 0.0001:
            if self.ptr not in self.positive_indices: self.positive_indices.append(self.ptr)
        else:
            if self.ptr in self.positive_indices: self.positive_indices.remove(self.ptr)
                
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def finish_trajectory(self):
        if len(self.current_trajectory['states']) == 0: return
        self.recent_trajectories.append({
            'states': np.array(self.current_trajectory['states']),
            'actions': np.array(self.current_trajectory['actions']),
            'rewards': np.array(self.current_trajectory['rewards'])
        })
        if len(self.recent_trajectories) > self.max_recent_trajectories:
            self.recent_trajectories.pop(0)
        self.current_trajectory = {'states': [], 'actions': [], 'rewards': []}
    
    def sample_imagination_states(self, batch_size):
        """
        Sample starting states for imagination.
        Crucial: Mix "Recent" states with "Positive" states to ensure the agent 
        imagines successful trajectories and propagates the reward signal.
        """
        if self.size == 0: return self.sample_diverse_states(batch_size)
        
        states_list = []
        num_pos = len(self.positive_indices)
        
        # 1. POSITIVES (50%)
        if num_pos > 0:
            n_pos = batch_size // 2
            idx_pos = np.random.choice(self.positive_indices, size=n_pos, replace=True)
            states_list.append(self.states[idx_pos])
            n_remaining = batch_size - n_pos
        else:
            n_remaining = batch_size
            
        # 2. RECENT OR DIVERSE (50%)
        if len(self.recent_trajectories) > 0:
            # Flatten recent trajectories (flatten lists)
            recent_states = []
            for _ in range(n_remaining):
                traj = self.recent_trajectories[np.random.randint(0, len(self.recent_trajectories))]
                s = traj['states'][np.random.randint(0, len(traj['states']))]
                recent_states.append(s)
            states_list.append(np.array(recent_states))
        else:
            # Fallback to random history
            idx = np.random.randint(0, self.size, size=n_remaining)
            states_list.append(self.states[idx])
            
        batch_states = np.concatenate(states_list, axis=0)
        return torch.tensor(batch_states, dtype=torch.float32).to(CONFIG["DEVICE"])
    
    def sample_diverse_states(self, batch_size):
        if self.size == 0: return torch.randn(batch_size, self.state_dim).to(CONFIG["DEVICE"])
        idx = np.random.randint(0, self.size, size=batch_size)
        return torch.tensor(self.states[idx], dtype=torch.float32).to(CONFIG["DEVICE"])

    def sample_for_reward_training(self, batch_size):
        if self.size == 0: return None
        num_pos = len(self.positive_indices)
        if num_pos > 0:
            half_batch = batch_size // 2
            pos_idx = np.random.choice(self.positive_indices, size=half_batch, replace=True)
            rand_idx = np.random.randint(0, self.size, size=batch_size - half_batch)
            idx = np.concatenate([pos_idx, rand_idx])
            np.random.shuffle(idx)
        else:
            idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'states': torch.tensor(self.states[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            'actions': torch.tensor(self.actions[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            'rewards': torch.tensor(self.rewards[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
        }

    def sample(self, batch_size):
        """
        Generic sampling for EBM training (s, a, ns)
        """
        if self.size == 0: return None
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.actions[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.next_states[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
        )

# --- AGENT WRAPPER ---
# --- AGENT WRAPPER ---
class Agent:
    def __init__(self, agent_type, model_tag, state_dim, action_dim, device):
        self.agent_type = agent_type
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.ebm = None
        self.flow = None
        self.mdn = None
        self.use_ascent = False
        
        if "MDN" in agent_type:
            self.mdn = MixtureDensityNetwork(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
            try:
                # Load STOCHASTIC pre-trained model dynamically
                fname = f"pretrained_mdn_{model_tag}.pth"
                self.mdn.load_state_dict(torch.load(fname, map_location=device, weights_only=False))
                print(f"Loaded pretrained MDN ({fname}).")
            except Exception as e:
                print(f"No pretrained MDN found ({e}). Initialized random.")
            self.mdn.train() 
            
        elif "Flow Only" in agent_type:
            self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
            try:
                fname = f"pretrained_flow_{model_tag}_ForwardKL.pth"
                self.flow.load_state_dict(torch.load(fname, map_location=device, weights_only=False))
            except: pass
            self.flow.train() 
        else:
            self.ebm = BilinearEBM(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
            try:
                fname = f"pretrained_ebm_{model_tag}.pth"
                self.ebm.load_state_dict(torch.load(fname, map_location=device, weights_only=False))
            except: pass
            self.ebm.train()
            for p in self.ebm.parameters(): p.requires_grad = True
            self.use_ascent = True
            
            if "Warm Start" in agent_type:
                suffix = "ForwardKL" if "ForwardKL" in agent_type else "ReverseKL"
                self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
                try:
                    fname = f"pretrained_flow_{model_tag}_{suffix}.pth"
                    self.flow.load_state_dict(torch.load(fname, map_location=device, weights_only=False))
                except: pass
                self.flow.train()

    def predict_next_state(self, state, action):
        # Action here is (B, action_dim) Soft One-Hot
        if "MDN" in self.agent_type:
            return self.mdn.sample_differentiable(state, action)
        elif "Flow Only" in self.agent_type:
            z = torch.randn_like(state).to(self.device)
            context = torch.cat([state, action], dim=1)
            return self.flow.sample(z, context=context)
        elif "SVGD" in self.agent_type:
            return predict_next_state_svgd_adaptive(
                self.ebm, state, action,
                use_ascent=self.use_ascent,
                config={"SVGD_STEPS": CONFIG["SVGD_STEPS"]}
            )
        else:
            init = None
            steps = CONFIG["LANGEVIN_STEPS_COLD"]
            if "Warm Start" in self.agent_type:
                z = torch.randn_like(state).to(self.device)
                context = torch.cat([state, action], dim=1)
                init = self.flow.sample(z, context=context)
                steps = CONFIG["LANGEVIN_STEPS_WARM"]
            
            return predict_next_state_langevin_adaptive(
                self.ebm, state, action,
                init_state=init,
                use_ascent=self.use_ascent,
                config={
                    "LANGEVIN_STEPS": steps,
                    "LANGEVIN_STEP_SIZE": CONFIG["LANGEVIN_STEP_SIZE"],
                    "LANGEVIN_NOISE_SCALE": CONFIG["LANGEVIN_NOISE_SCALE"]
                }
            )

# --- ONE-HOT HELPER ---
def to_one_hot(action_idx, num_actions):
    vec = np.zeros(num_actions, dtype=np.float32)
    vec[action_idx] = 1.0
    return vec

# --- EVALUATION ---
def evaluate_policy(env_adapter, actor, num_episodes=5):
    episode_rewards = []
    for _ in range(num_episodes):
        state = env_adapter.reset()
        done = False
        episode_reward = 0
        while not done:
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(CONFIG["DEVICE"])
            with torch.no_grad():
                # actor.sample returns (B, action_dim)
                action_vec, _ = actor.sample(st, hard=False)
                action_vec = action_vec.cpu().numpy()[0]
                
            # Adapter handles argmax internally for step
            next_state, reward, done, info = env_adapter.step(action_vec)
            episode_reward += reward
            state = next_state
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards), np.std(episode_rewards)

# --- TRAIN SINGLE AGENT ---
def train_agent(agent_type, env_adapter, state_buffer):
    print(f"\nTraining Agent: {agent_type} on STOCHASTIC ENV ({CONFIG['MODEL_TAG']})")
    device = CONFIG["DEVICE"]
    state_dim = env_adapter.state_dim
    action_dim = env_adapter.action_dim
    
    agent = Agent(agent_type, CONFIG["MODEL_TAG"], state_dim, action_dim, device)
    
    # ACTOR
    actor = DiscreteActor(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    # CRITIC (Value Function)
    critic = ValueNetwork(state_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target = ValueNetwork(state_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target.load_state_dict(critic.state_dict())
    
    reward_model = RewardModel(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    actor_opt = optim.Adam(actor.parameters(), lr=CONFIG["LR_ACTOR"])
    critic_opt = optim.Adam(critic.parameters(), lr=CONFIG["LR_CRITIC"])
    reward_opt = optim.Adam(reward_model.parameters(), lr=CONFIG["LR_REWARD"])
    
    ebm_opt = None
    if agent.ebm is not None:
        ebm_opt = optim.Adam(agent.ebm.parameters(), lr=1e-4)

    eval_rewards = []
    eval_steps = []
    episode_count = 0
    total_steps = 0
    
    state = env_adapter.reset()
    
    while total_steps < CONFIG["TOTAL_STEPS"]:
        # 1. Real interaction
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_vec_tensor, _ = actor.sample(st, hard=False)
            action_vec = action_vec_tensor.cpu().numpy()[0]
            
        next_state, reward, done, info = env_adapter.step(action_vec)
        
        action_id = np.argmax(action_vec)
        action_onehot = to_one_hot(action_id, action_dim)
        
        state_buffer.add_transition(state, action_onehot, reward, next_state)
        total_steps += 1
        
        if done:
            state_buffer.finish_trajectory()
            episode_count += 1
            if ebm_opt and episode_count % 5 == 0 and state_buffer.size >= 500:
                metrics = update_ebm_online(agent.ebm, state_buffer, ebm_opt)
                if episode_count % 20 == 0:
                   print(f"  [EBM Online] Gap: {metrics['E_gap']:.4f} Loss: {metrics['loss']:.4f}")
            state = env_adapter.reset()
        else:
            state = next_state
            
        # 2. Imagined Update
        if state_buffer.size >= CONFIG["BATCH_SIZE"] and total_steps % 10 == 0:
            # Reward Model Update
            batch = state_buffer.sample_for_reward_training(CONFIG["BATCH_SIZE"])
            if batch:
                pred_r = reward_model(batch['states'], batch['actions'])
                loss_r = F.mse_loss(pred_r, batch['rewards'])
                reward_opt.zero_grad(); loss_r.backward(); reward_opt.step()
            
            # Imagination (Prioritized Sampling)
            initial_states = state_buffer.sample_imagination_states(CONFIG["BATCH_SIZE"])
            
            # Unroll
            curr = initial_states
            states_list, actions_list, rewards_list, next_states_list = [], [], [], []
            dones = torch.zeros(CONFIG["BATCH_SIZE"], CONFIG["HORIZON"], 1).to(device)
            
            for t in range(CONFIG["HORIZON"]):
                # Differentiable Action Sampling
                a_soft, logits = actor.sample(curr, hard=False) 
                
                # Entropy
                dist = torch.distributions.Categorical(logits=logits)
                entropy = dist.entropy().unsqueeze(-1) # (B, 1)
                
                r_pred = reward_model(curr, a_soft)
                ns_pred = agent.predict_next_state(curr, a_soft)
                
                states_list.append(curr)
                actions_list.append(a_soft)
                rewards_list.append(r_pred + CONFIG["ENTROPY_COEFF"] * entropy)
                next_states_list.append(ns_pred)
                curr = ns_pred
            
            states_seq = torch.stack(states_list, dim=1)     # (B, H, D)
            actions_seq = torch.stack(actions_list, dim=1)   # (B, H, A)
            rewards_seq = torch.stack(rewards_list, dim=1)   # (B, H, 1)
            next_states_seq = torch.stack(next_states_list, dim=1) # (B, H, D)
            
            # CRITIC UPDATE (Target Computation)
            flat_next_states = next_states_seq.reshape(-1, state_dim)
            with torch.no_grad():
                # Target Value V(s')
                flat_next_values = critic_target(flat_next_states)
                next_values = flat_next_values.reshape(CONFIG["BATCH_SIZE"], CONFIG["HORIZON"], 1)
            
            # Compute Lambda Returns (Bootstrapped from V)
            lambda_targets = compute_lambda_values(
                next_values.squeeze(-1), 
                rewards_seq.squeeze(-1), 
                dones.squeeze(-1), 
                CONFIG["DISCOUNT"], 
                CONFIG["LAMBDA"]
            )
            lambda_targets = lambda_targets.unsqueeze(-1) # (B, H, 1)
            
            # CRITIC UPDATE (V-function)
            flat_curr_states = states_seq.reshape(-1, state_dim).detach() 
            v_values = critic(flat_curr_states).reshape(CONFIG["BATCH_SIZE"], CONFIG["HORIZON"], 1)
            
            critic_loss = F.mse_loss(v_values, lambda_targets.detach())
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            
            # ACTOR UPDATE: Model-Based Policy Gradient
            # Maximize: r + gamma * V(s')
            
            flat_next_states_grad = next_states_seq.reshape(-1, state_dim) # Has Grad
            # Freeze Critic Weights for Actor Update
            for p in critic.parameters(): p.requires_grad = False
            v_next_pred = critic(flat_next_states_grad).reshape(CONFIG["BATCH_SIZE"], CONFIG["HORIZON"], 1)
            for p in critic.parameters(): p.requires_grad = True
            
            # Actor Objective: Maximize expected return
            actor_objective = rewards_seq + CONFIG["DISCOUNT"] * v_next_pred
            actor_loss = -actor_objective.mean()
            
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            
            # Soft Update
            for p, tp in zip(critic.parameters(), critic_target.parameters()):
                tp.data.copy_(0.005 * p.data + 0.995 * tp.data)

            eval_mean, eval_std = evaluate_policy(env_adapter, actor, CONFIG["EVAL_EPISODES"])
            eval_rewards.append(eval_mean)
            eval_steps.append(total_steps)
            print(f"Step {total_steps} | Eval: {eval_mean:.2f}")

    return eval_steps, eval_rewards

def train():
    print("="*60)
    print(f"!!! STARTING STOCHASTIC TRAINING MODE !!!")
    print(f"Base Env: {CONFIG['BASE_ENV']}")
    print(f"Wrapper:  StochasticMiniGridAdapter (Slip Prob = 0.1)")
    print(f"Models:   {CONFIG['MODEL_TAG']}")
    print("="*60)
    
    device = CONFIG["DEVICE"]
    # Initialize Stochastic Adapter with BASE_ENV
    env_adapter = StochasticMiniGridAdapter(CONFIG['BASE_ENV'], slip_prob=0.3)
    state_buffer = TrajectoryBuffer(env_adapter.state_dim, env_adapter.action_dim)
    
    # Collect initial (Ensure we have some POSITIVE examples)
    print("Collecting initial data (Warmup)...")
    s = env_adapter.reset()
    steps_collected = 0
    # Collect until we have at least 10 positive transitions or 5000 steps
    # This is crucial for Sparse Reward environments
    while len(state_buffer.positive_indices) < 10 and steps_collected < 5000:
        a_int = np.random.randint(0, env_adapter.action_dim)
        ns, r, d, _ = env_adapter.step(a_int)
        
        state_buffer.add_transition(s, to_one_hot(a_int, env_adapter.action_dim), r, ns)
        steps_collected += 1
        
        if d: s = env_adapter.reset()
        else: s = ns
        
        if steps_collected % 1000 == 0:
            print(f"  Collected {steps_collected} steps. Positives: {len(state_buffer.positive_indices)}")

    print(f"Initial collection done. Steps: {steps_collected}, Positives: {len(state_buffer.positive_indices)}")
        
    for agent_type in CONFIG["AGENT_TYPES"]:
        train_agent(agent_type, env_adapter, state_buffer)

if __name__ == "__main__":
    train()
