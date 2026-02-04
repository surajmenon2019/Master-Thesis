"""
Script C: Policy Convergence (The Final Thesis Experiment)
Environment: SafetyPointGoal1-v0
Goal: Prove that 'Warm Start' agents learn faster/safer than 'Cold Start' agents
      because they can plan through the 'Energy Barriers' (Hazards).
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os

# --- IMPORTS ---
try:
    import safety_gymnasium
    from models import EnergyBasedModel, RealNVP, Actor, Critic, MixtureDensityNetwork
    # Ensure utils_sampling is the one with gradient clamping!
    from utils_sampling import predict_next_state_langevin, predict_next_state_svgd
except ImportError as e:
    print(f"CRITICAL: Missing modules. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
CONFIG = {
    "ENV_NAME": "SafetyPointGoal1-v0", 
    "METHODS": ["Cold Start", "Warm Start (ForwardKL)", "Warm Start (ReverseKL)", "Flow Only", "MDN", "SVGD"],
    
    # CHANGE THIS for each run to generate your comparison lines
    "CURRENT_METHOD": "Flow Only", 
    
    "EPISODES": 100,            
    "STEPS_PER_EPISODE": 1000,  # Long horizon needed for navigation
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "HIDDEN_DIM": 128,          # Matches Pretraining
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 1e-3,
    "GAMMA": 0.99,
    "BATCH_SIZE": 64
}

# --- REPLAY BUFFER ---
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=100000):
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
            torch.tensor(self.states[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.actions[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.next_states[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.rewards[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.dones[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
        )

# --- SAFETY GYM ADAPTER ---
class SafetyGymAdapter:
    def __init__(self, env_name):
        # We assume PointGoal1-v0. If v0 fails, try v1 or check installed version.
        self.env = safety_gymnasium.make(env_name, render_mode=None)
        # Flatten observations (Lidar + Vels)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        print(f"Initialized {env_name} | State: {self.state_dim} | Action: {self.action_dim}")

    def reset(self):
        s, _ = self.env.reset()
        return s.astype(np.float32)

    def step(self, action):
        # Unpack Safety Gym return values (6 values)
        ns, r, cost, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # CRITICAL: Augment Reward with Cost (Hazards)
        # This creates the "Energy Barrier" for the EBM
        total_reward = r - 5.0 * cost 
        
        return ns.astype(np.float32), total_reward, done

# --- WEIGHT LOADING HELPER ---
def load_pretrained_models(ebm, flow, mdn, method):
    print(f">>> Loading Weights for {method}...")
    env_name = CONFIG["ENV_NAME"]
    
    # 1. Load EBM (Shared Physics)
    if "Cold" in method or "Warm" in method or "SVGD" in method:
        try:
            ebm.load_state_dict(torch.load(f"pretrained_ebm_{env_name}.pth"))
            print("    [+] Loaded EBM Physics")
        except FileNotFoundError:
            print("    [!] Warning: EBM weights not found (Using Random)")

    # 2. Load Flow (The Warm Start)
    if "Warm" in method or "Flow" in method:
        # Determine suffix based on method string
        suffix = "ReverseKL" if "ReverseKL" in method else "ForwardKL"
        try:
            flow.load_state_dict(torch.load(f"pretrained_flow_{env_name}_{suffix}.pth"))
            print(f"    [+] Loaded Flow ({suffix})")
        except FileNotFoundError:
            print(f"    [!] Warning: Flow weights not found (Using Random)")

    # 3. Load MDN
    if "MDN" in method:
         try:
            mdn.load_state_dict(torch.load(f"pretrained_mdn_{env_name}.pth"))
            print("    [+] Loaded MDN Weights")
         except FileNotFoundError:
            print("    [!] Warning: MDN weights not found (Using Random)")

    # 4. Freeze Weights (CRITICAL: We test Policy Learning on FIXED Physics)
    for p in ebm.parameters(): p.requires_grad = False
    for p in flow.parameters(): p.requires_grad = False
    for p in mdn.parameters(): p.requires_grad = False

# --- MAIN TRAINING LOOP ---
def train_agent():
    print(f"\n>>> EXPERIMENT: Policy Convergence using [{CONFIG['CURRENT_METHOD']}]")
    device = CONFIG["DEVICE"]
    env_adapter = SafetyGymAdapter(CONFIG["ENV_NAME"])
    
    # 1. Initialize Models
    ebm = EnergyBasedModel(env_adapter.state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    flow = RealNVP(data_dim=env_adapter.state_dim, context_dim=env_adapter.state_dim+env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    mdn = MixtureDensityNetwork(env_adapter.state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    # Agent Models
    actor = Actor(env_adapter.state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic = Critic(env_adapter.state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    # Load Physics Knowledge (Frozen)
    load_pretrained_models(ebm, flow, mdn, CONFIG["CURRENT_METHOD"])
    
    # Optimizers
    actor_opt = optim.Adam(actor.parameters(), lr=CONFIG["LR_ACTOR"])
    critic_opt = optim.Adam(critic.parameters(), lr=CONFIG["LR_CRITIC"])
    
    buffer = ReplayBuffer(env_adapter.state_dim, env_adapter.action_dim)
    reward_history = []
    
    # 2. Pre-fill Buffer
    print(">>> Collecting Initial Exploration Data...")
    s = env_adapter.reset()
    for _ in range(1000):
        a = env_adapter.env.action_space.sample()
        ns, r, d = env_adapter.step(a)
        buffer.add(s, a, r, ns, d)
        s = ns if not d else env_adapter.reset()

    # 3. Training Loop
    print(">>> Starting RL Loop...")
    for episode in range(CONFIG["EPISODES"]):
        s = env_adapter.reset()
        ep_reward = 0
        
        for t in range(CONFIG["STEPS_PER_EPISODE"]):
            # A. Interaction (Real Environment)
            s_t = torch.FloatTensor(s).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor.sample(s_t).cpu().numpy()[0]
            
            ns, r, done = env_adapter.step(action)
            buffer.add(s, action, r, ns, done)
            s = ns if not done else env_adapter.reset()
            ep_reward += r
            
            if done: break

        reward_history.append(ep_reward)

        # B. Gradient Updates
        if buffer.size > CONFIG["BATCH_SIZE"]:
            for _ in range(50): 
                bs, ba, bns, br, bd = buffer.sample(CONFIG["BATCH_SIZE"])
                
                # ----------------------------
                # 1. Train Critic (Real Data)
                # ----------------------------
                with torch.no_grad():
                    next_actions = actor.sample(bns)
                    target_q = critic(bns, next_actions)
                    target_val = br + (1 - bd) * CONFIG["GAMMA"] * target_q
                
                current_q = critic(bs, ba)
                loss_critic = F.mse_loss(current_q, target_val)
                critic_opt.zero_grad(); loss_critic.backward(); critic_opt.step()

                # ----------------------------
                # 2. Train Actor (The "Deep Planning" Horizon)
                # ----------------------------
                # Start planning from real states
                curr_state = bs 
                
                # THIS IS THE STUDY: How deep can we go?
                # For Cold Start, gradients die if this > 1. 
                # For Warm Start, we should be able to push this to 5, 10, etc.
                PLANNING_HORIZON = 5  
                
                accumulated_loss = 0
                gamma_decay = 1.0
                
                for h in range(PLANNING_HORIZON):
                    # a. Policy Action at step h
                    a_plan = actor.sample(curr_state)
                    
                    # b. World Model Prediction (Differentiable Transition)
                    method = CONFIG["CURRENT_METHOD"]
                    next_state = None
                    
                    # NOTE: We use a SMALL number of Langevin steps here (e.g., 5)
                    # This relies on the Warm Start to be accurate.
                    # If Cold Start uses 5 steps, the state will be garbage.
                    LANGEVIN_REFINEMENT = 5 
                    
                    if method == "MDN":
                        next_state = mdn.sample_differentiable(curr_state, a_plan)
                    elif method == "Flow Only":
                        z = torch.randn_like(curr_state).to(device)
                        next_state = flow.sample(z, context=torch.cat([curr_state, a_plan], dim=1))
                    elif method == "SVGD":
                        next_state = predict_next_state_svgd(ebm, curr_state, a_plan)
                    else: 
                        # Langevin (Cold or Warm)
                        init = None
                        if "Warm" in method:
                            z = torch.randn_like(curr_state).to(device)
                            init = flow.sample(z, context=torch.cat([curr_state, a_plan], dim=1))
                        
                        next_state = predict_next_state_langevin(
                            ebm, curr_state, a_plan, init_state=init, config={"LANGEVIN_STEPS": LANGEVIN_REFINEMENT}
                        )
                    
                    # c. Calculate Loss for this step
                    # We want to maximize Critic Value (Q) and minimize Energy (Physical Validity)
                    q_val = critic(curr_state, a_plan)
                    energy_penalty = ebm(curr_state, a_plan, next_state).mean()
                    
                    # Loss = -Q + lambda * Energy
                    step_loss = -q_val.mean() + 0.5 * energy_penalty
                    
                    # Accumulate discounted loss over time
                    accumulated_loss += gamma_decay * step_loss
                    gamma_decay *= CONFIG["GAMMA"]
                    
                    # Pass the differentiable state to the next step
                    curr_state = next_state
                
                # d. Backpropagate through TIME (through the loop of 5 horizons)
                actor_opt.zero_grad()
                accumulated_loss.backward()
                
                # Optional: Clip gradients to prevent explosion in deep horizons
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                
                actor_opt.step()
            
        print(f"Episode {episode} | Reward: {ep_reward:.2f}")

    # Save
    np.save(f"rewards_{CONFIG['CURRENT_METHOD']}.npy", reward_history)
    print(f">>> SAVED rewards_{CONFIG['CURRENT_METHOD']}.npy")

if __name__ == "__main__":
    train_agent()