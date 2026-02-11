import torch
import torch.optim as optim
import numpy as np
import sys
import os

# --- IMPORTS ---
try:
    import minigrid
    import gymnasium as gym
    from models import BilinearEBM, RealNVP
    from minigrid_adapter import MiniGridAdapter
    import torch.nn.functional as F
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
CONFIG = {
    "ENV_NAME": "MiniGrid-Stochastic-0.1", 
    "TRAIN_STEPS": 5000,   
    "COLLECT_STEPS": 5000, # Just need some context states (s, a)
    "BATCH_SIZE": 128,
    "LR_FLOW": 1e-4,
    "DEVICE": "cpu", # Use CPU for safety if CUDA was stuck, or change to "cuda"
    "HIDDEN_DIM": 128,
}

# --- REPLAY BUFFER (Simple) ---
class ContextBuffer:
    def __init__(self, state_dim, action_dim, capacity=10000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)

    def add(self, s, a):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx], dtype=torch.float32).to(CONFIG["DEVICE"]),
            torch.tensor(self.actions[idx], dtype=torch.float32).to(CONFIG["DEVICE"])
        )

# --- ONE-HOT HELPER ---
def to_one_hot(action_idx, num_actions):
    vec = np.zeros(num_actions, dtype=np.float32)
    vec[action_idx] = 1.0
    return vec

def train_reverse_kl():
    print(f"\n>>> TRAINING FLOW (Reverse KL): {CONFIG['ENV_NAME']}")
    device = CONFIG["DEVICE"]
    env_adapter = MiniGridAdapter(CONFIG["ENV_NAME"])
    state_dim = env_adapter.state_dim
    action_dim = env_adapter.action_dim
    
    # 1. Collect Context Data (s, a)
    print(f">>> Collecting {CONFIG['COLLECT_STEPS']} context transitions...")
    buffer = ContextBuffer(state_dim, action_dim)
    s = env_adapter.reset()
    for i in range(CONFIG['COLLECT_STEPS']):
        a_int = np.random.randint(0, action_dim) 
        ns, reward, done, info = env_adapter.step(a_int)
        a_vec = to_one_hot(a_int, action_dim)
        buffer.add(s, a_vec)
        if done: s = env_adapter.reset()
        else: s = ns

    # 2. Load Pretrained EBM (Frozen)
    print(">>> Loading Pretrained EBM...")
    ebm = BilinearEBM(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    try:
        ebm.load_state_dict(torch.load(f"pretrained_ebm_{CONFIG['ENV_NAME']}.pth", map_location=device))
        print("    Loaded EBM weights.")
    except FileNotFoundError:
        print("    CRITICAL: 'pretrained_ebm_MiniGrid-Empty-8x8-v0.pth' not found.")
        print("    Please run pretrain_ebm_minigrid.py first or ensure the file exists.")
        sys.exit(1)
    
    ebm.eval()
    for p in ebm.parameters(): p.requires_grad = False
    
    # 3. Initialize Flow
    flow = RealNVP(data_dim=state_dim, context_dim=state_dim + action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    flow_opt = optim.Adam(flow.parameters(), lr=CONFIG["LR_FLOW"])

    # 4. Training Loop (Reverse KL)
    print("\n>>> Starting Reverse KL Training...")
    print("    Objective: Min KL(p_flow || p_ebm) => Min E_flow [ log p_flow(x) - E(x) ]")
    print("    (Assuming EBM Convention: Higher Energy = Better/Higher Probability)")
    
    for step in range(CONFIG["TRAIN_STEPS"]):
        # Get context (s, a)
        s, a = buffer.sample(CONFIG["BATCH_SIZE"])
        context = torch.cat([s, a], dim=1)
        
        # A. Sample from Flow: x ~ p_flow(.|s,a)
        z = torch.randn(s.shape[0], state_dim).to(device)
        fake_ns, log_det = flow.forward(z, context=context) # forward maps z->x usually in RealNVP impl? 
        # Wait, check RealNVP implementation in models.py
        # forward(x, context) -> z, log_det (Inference)
        # sample(z, context) -> x (Generation)
        
        fake_ns = flow.sample(z, context=context)
        
        # B. Compute Log Prob of Generated Samples: log p_flow(fake_ns)
        # We need to run log_prob on the *generated* samples
        log_prob_flow = flow.log_prob(fake_ns, context=context)
        
        # C. Compute Energy of Generated Samples: E(s, a, fake_ns)
        # EBM is frozen
        with torch.no_grad(): # Wait, we need grads through fake_ns to flow?
            # Standard Reverse KL:
            # We differentiate L = log p(x) - E(x) w.r.t x (via reparam) ... 
            # Yes, we need gradients of E(x) w.r.t x propagated to Flow.
            pass
        
        # EBM forward pass creates graph connected to fake_ns?
        # Yes, if we just call ebm(...)
        energy = ebm(s, a, fake_ns) # (B,)
        
        # D. Loss Construction
        # Minimize: log_prob_flow - energy
        # Because Target Density ~ exp(Energy)
        # log(Target) ~ Energy
        # KL = log_prob - log_target = log_prob - Energy
        
        loss_reverse_kl = (log_prob_flow - energy).mean() / state_dim
        
        flow_opt.zero_grad()
        loss_reverse_kl.backward()
        flow_opt.step()

        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss_reverse_kl.item():.4f} | Avg Energy: {energy.mean().item():.4f} | Avg LogP: {log_prob_flow.mean().item():.4f}")

    # 5. Save
    save_path = f"pretrained_flow_{CONFIG['ENV_NAME']}_ReverseKL.pth"
    torch.save(flow.state_dict(), save_path)
    print(f"\n>>> SUCCESS. Saved ReverseKL Flow to: {save_path}")

if __name__ == "__main__":
    train_reverse_kl()
