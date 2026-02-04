import torch
import torch.optim as optim
import numpy as np
import sys
import os

# --- IMPORTS ---
try:
    import safety_gymnasium
    from models import EnergyBasedModel, RealNVP 
    from utils_sampling import predict_next_state_langevin
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    print("Run: pip install safety-gymnasium")
    sys.exit(1)

# --- CONFIGURATION ---
FLOW_TYPE = "ReverseKL"  # Options: "ForwardKL" or "ReverseKL"
CONFIG = {
    "ENV_NAME": "SafetyPointGoal1-v0", 
    "PRETRAIN_STEPS": 10000,   # Total training steps
    "COLLECT_STEPS": 5000,     # Initial data collection
    "BATCH_SIZE": 128,
    "LR_EBM": 1e-4,
    "LR_FLOW": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "LANGEVIN_STEPS": 30,
    "HIDDEN_DIM": 128
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

    def add(self, s, a, ns):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.next_states[self.ptr] = ns
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.actions[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.next_states[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
        )

# --- ADAPTER ---
class SafetyGymAdapter:
    def __init__(self, env_name):
        self.env = safety_gymnasium.make(env_name, render_mode=None)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

    def reset(self):
        s, _ = self.env.reset()
        return s.astype(np.float32)

    def step(self, action):
        ns, reward, cost, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return ns.astype(np.float32), done

def train_unified_models():
    print(f"\n>>> PRETRAINING START: Safety Gym [{FLOW_TYPE}]")
    device = CONFIG["DEVICE"]
    env_adapter = SafetyGymAdapter(CONFIG["ENV_NAME"])
    state_dim = env_adapter.state_dim # Needed for Normalization
    
    # 1. Initialize Replay Buffer
    buffer = ReplayBuffer(env_adapter.state_dim, env_adapter.action_dim)

    # 2. Collect Initial Data (Random Policy)
    print(f">>> Collecting {CONFIG['COLLECT_STEPS']} transitions...")
    s = env_adapter.reset()
    for i in range(CONFIG['COLLECT_STEPS']):
        a = env_adapter.env.action_space.sample()
        ns, done = env_adapter.step(a)
        buffer.add(s, a, ns)
        if done: s = env_adapter.reset()
        else: s = ns
        if (i+1) % 1000 == 0: print(f"    Collected {i+1} transitions...")

    # 3. Models
    ebm = EnergyBasedModel(env_adapter.state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    flow = RealNVP(data_dim=env_adapter.state_dim, context_dim=env_adapter.state_dim + env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    ebm_opt = optim.Adam(ebm.parameters(), lr=CONFIG["LR_EBM"])
    flow_opt = optim.Adam(flow.parameters(), lr=CONFIG["LR_FLOW"])

    # 4. Training Loop
    print("\n>>> Starting Training Loop...")
    for step in range(CONFIG["PRETRAIN_STEPS"]):
        # Sample Batch
        s, a, real_ns = buffer.sample(CONFIG["BATCH_SIZE"])
        context = torch.cat([s, a], dim=1)

        # ==========================
        # A. TRAIN EBM (CD)
        # ==========================
        pos_energy = ebm(s, a, real_ns).mean()
        
        # Negative Sample (Langevin with Warm Start if ReverseKL)
        with torch.no_grad():
            if FLOW_TYPE == "ReverseKL":
                z = torch.randn(s.shape[0], state_dim).to(device)
                init_state = flow.sample(z, context=context)
            else:
                init_state = None # Cold start training for EBM if Flow is just MLE

        fake_ns = predict_next_state_langevin(
            ebm, s, a, init_state=init_state, 
            config={"LANGEVIN_STEPS": CONFIG["LANGEVIN_STEPS"]}
        ).detach()
        
        neg_energy = ebm(s, a, fake_ns).mean()
        
        # Loss: Minimize Positive (Real), Maximize Negative (Fake)
        ebm_loss = pos_energy - neg_energy + (pos_energy**2 + neg_energy**2) * 0.1
        
        ebm_opt.zero_grad(); ebm_loss.backward(); ebm_opt.step()

        # ==========================
        # B. TRAIN FLOW
        # ==========================
        if FLOW_TYPE == "ForwardKL":
            # MLE: Maximize log_prob of real data
            # FIX 1: Normalize by dimension to keep gradients stable
            log_prob = flow.log_prob(real_ns, context=context)
            loss_flow = -log_prob.mean() / state_dim 

        else: 
            # ReverseKL: Minimize KL(q || p) = E_q [ log(q) + E(x) ]
            z = torch.randn(s.shape[0], state_dim).to(device)
            fake = flow.sample(z, context=context)
            
            # Freeze EBM for this step
            for p in ebm.parameters(): p.requires_grad = False
            energy = ebm(s, a, fake).mean()
            for p in ebm.parameters(): p.requires_grad = True
            
            log_p = flow.log_prob(fake, context=context)
            
            # FIX 2: Normalize LogProb by Dimension
            avg_log_p = log_p.mean() / state_dim
            
            # FIX 3: SIGN FLIP (Minimize both Energy and Entropy-Penalty)
            # We want High Entropy (Spread) -> Minimize LogProb
            # We want Low Energy (Safe) -> Minimize Energy
            loss_flow = energy + avg_log_p 

        flow_opt.zero_grad(); loss_flow.backward(); flow_opt.step()
        
        if step % 1000 == 0:
            print(f"Step {step} | EBM Loss: {ebm_loss.item():.4f} | Flow Loss: {loss_flow.item():.4f}")

    # 5. Save
    torch.save(ebm.state_dict(), f"pretrained_ebm_{CONFIG['ENV_NAME']}.pth")
    torch.save(flow.state_dict(), f"pretrained_flow_{CONFIG['ENV_NAME']}_{FLOW_TYPE}.pth")
    print("\n>>> SUCCESS. Saved models.")

if __name__ == "__main__":
    train_unified_models()