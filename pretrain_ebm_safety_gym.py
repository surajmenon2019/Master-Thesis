import torch
import torch.optim as optim
import numpy as np
import sys
import os

# --- IMPORTS ---
try:
    import safety_gymnasium
    # Added MDN
    from models import EnergyBasedModel, RealNVP, MixtureDensityNetwork 
    from utils_sampling import predict_next_state_langevin
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    print("Run: pip install safety-gymnasium")
    sys.exit(1)

# --- CONFIGURATION ---
FLOW_TYPE = "ForwardKL"  # Options: "ForwardKL" or "ReverseKL"
CONFIG = {
    "ENV_NAME": "SafetyPointGoal1-v0", 
    "PRETRAIN_STEPS": 10000,   
    "COLLECT_STEPS": 5000,     
    "BATCH_SIZE": 128,
    "LR_EBM": 1e-4,
    "LR_FLOW": 1e-4,
    "LR_MDN": 1e-3, # Added specific LR for MDN
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
    print(f"\n>>> PRETRAINING START: Safety Gym [{FLOW_TYPE}] + MDN")
    device = CONFIG["DEVICE"]
    env_adapter = SafetyGymAdapter(CONFIG["ENV_NAME"])
    state_dim = env_adapter.state_dim
    
    # 1. Initialize Replay Buffer
    buffer = ReplayBuffer(env_adapter.state_dim, env_adapter.action_dim)

    # 2. Collect Initial Data
    print(f">>> Collecting {CONFIG['COLLECT_STEPS']} transitions...")
    s = env_adapter.reset()
    for i in range(CONFIG['COLLECT_STEPS']):
        a = env_adapter.env.action_space.sample()
        ns, done = env_adapter.step(a)
        buffer.add(s, a, ns)
        if done: s = env_adapter.reset()
        else: s = ns

    # 3. Models
    ebm = EnergyBasedModel(state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    flow = RealNVP(data_dim=state_dim, context_dim=state_dim + env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    mdn = MixtureDensityNetwork(state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    ebm_opt = optim.Adam(ebm.parameters(), lr=CONFIG["LR_EBM"])
    flow_opt = optim.Adam(flow.parameters(), lr=CONFIG["LR_FLOW"])
    mdn_opt = optim.Adam(mdn.parameters(), lr=CONFIG["LR_MDN"])

    # 4. Training Loop
    print("\n>>> Starting Training Loop...")
    for step in range(CONFIG["PRETRAIN_STEPS"]):
        s, a, real_ns = buffer.sample(CONFIG["BATCH_SIZE"])
        context = torch.cat([s, a], dim=1)

        # ==========================
        # A. TRAIN EBM (Contrastive Divergence)
        # ==========================
        pos_energy = ebm(s, a, real_ns).mean()
        
        with torch.no_grad():
            if FLOW_TYPE == "ReverseKL":
                z = torch.randn(s.shape[0], state_dim).to(device)
                init_state = flow.sample(z, context=context)
            else:
                init_state = None 

        fake_ns = predict_next_state_langevin(
            ebm, s, a, init_state=init_state, 
            config={"LANGEVIN_STEPS": CONFIG["LANGEVIN_STEPS"]}
        ).detach()
        
        neg_energy = ebm(s, a, fake_ns).mean()
        ebm_loss = pos_energy - neg_energy + (pos_energy**2 + neg_energy**2) * 0.1
        
        ebm_opt.zero_grad(); ebm_loss.backward(); ebm_opt.step()

        # ==========================
        # B. TRAIN FLOW
        # ==========================
        if FLOW_TYPE == "ForwardKL":
            # Normalized MLE
            log_prob = flow.log_prob(real_ns, context=context)
            loss_flow = -log_prob.mean() / state_dim 
        else: 
            # Normalized ReverseKL with fixed signs
            z = torch.randn(s.shape[0], state_dim).to(device)
            fake = flow.sample(z, context=context)
            
            for p in ebm.parameters(): p.requires_grad = False
            energy = ebm(s, a, fake).mean()
            for p in ebm.parameters(): p.requires_grad = True
            
            log_p = flow.log_prob(fake, context=context)
            avg_log_p = log_p.mean() / state_dim
            loss_flow = energy + avg_log_p 

        flow_opt.zero_grad(); loss_flow.backward(); flow_opt.step()

        # ==========================
        # C. TRAIN MDN (Strictly Normalized)
        # ==========================
        pi, mu, sigma = mdn(s, a)
        
        # Expand target for mixture broadcasting
        target = real_ns.unsqueeze(1).expand_as(mu)
        
        # 1. Log Probability per component
        dist = torch.distributions.Normal(mu, sigma)
        # Sum over dimensions to get prob of full state vector (Batch, K)
        log_prob_components = dist.log_prob(target).sum(dim=-1) 
        
        # 2. LogSumExp for Mixture
        log_pi = torch.log_softmax(pi, dim=1)
        log_likelihood = torch.logsumexp(log_pi + log_prob_components, dim=1)
        
        # 3. NORMALIZE by dimension so it matches Flow scale (Approx 1.0 instead of 60.0)
        loss_mdn = -log_likelihood.mean() / state_dim
        
        mdn_opt.zero_grad(); loss_mdn.backward(); mdn_opt.step()
        
        if step % 1000 == 0:
            print(f"Step {step} | EBM: {ebm_loss.item():.4f} | Flow: {loss_flow.item():.4f} | MDN: {loss_mdn.item():.4f}")

    # 5. Save
    torch.save(ebm.state_dict(), f"pretrained_ebm_{CONFIG['ENV_NAME']}.pth")
    torch.save(flow.state_dict(), f"pretrained_flow_{CONFIG['ENV_NAME']}_{FLOW_TYPE}.pth")
    torch.save(mdn.state_dict(), f"pretrained_mdn_{CONFIG['ENV_NAME']}.pth")
    print("\n>>> SUCCESS. Saved models (EBM, Flow, MDN).")

if __name__ == "__main__":
    train_unified_models()