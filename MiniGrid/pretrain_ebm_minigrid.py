import torch
import torch.optim as optim
import numpy as np
import sys
import os

# --- IMPORTS ---
try:
    import minigrid
    import gymnasium as gym
    # InfoNCE: BilinearEBM, RealNVP, MixtureDensityNetwork (from existing models)
    from models import BilinearEBM, RealNVP, MixtureDensityNetwork 
    from minigrid_adapter import MiniGridAdapter
    import torch.nn.functional as F
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    print("Run: pip install minigrid gymnasium")
    sys.exit(1)

# --- CONFIGURATION ---
FLOW_TYPE = "ForwardKL"  # Options: "ForwardKL" or "ReverseKL"
CONFIG = {
    "ENV_NAME": "MiniGrid-Empty-8x8-v0", 
    "PRETRAIN_STEPS": 10000,   
    "COLLECT_STEPS": 5000,     
    "BATCH_SIZE": 128,
    "LR_EBM": 1e-4,
    "LR_FLOW": 1e-4,
    "LR_MDN": 1e-3,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "HIDDEN_DIM": 128,
    "NUM_NEGATIVES": 128,
    "TEMPERATURE": 0.1,  
}

# --- REPLAY BUFFER ---
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=100000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        # Action is now a vector (one-hot or soft), same size as action_dim
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

# --- ONE-HOT HELPER ---
def to_one_hot(action_idx, num_actions):
    """Convert scalar action index to one-hot vector"""
    vec = np.zeros(num_actions, dtype=np.float32)
    vec[action_idx] = 1.0
    return vec

# --- INFONCE LOSS (SAME AS ORIGINAL) ---
def infonce_loss(ebm, state, action, pos_next_state, buffer, 
                 num_negatives=512, temperature=0.1, explicit_negatives=None):
    """
    InfoNCE loss for training BilinearEBM dynamics models.
    """
    B = state.shape[0]
    device = state.device
    
    # 1. Positive energy: E(s,a,s'_real) - should be HIGH
    E_pos = ebm(state, action, pos_next_state)  # (B,)
    
    # 2. Sample negatives from buffer
    neg_indices = np.random.randint(0, buffer.size, size=(B, num_negatives))
    neg_next_states = torch.tensor(
        buffer.next_states[neg_indices], 
        dtype=torch.float32, 
        device=device
    )  # (B, K, state_dim)
    
    # 3. Add explicit negatives if provided
    if explicit_negatives is not None:
        neg_next_states = torch.cat([neg_next_states, explicit_negatives], dim=1)
        num_negatives += explicit_negatives.shape[1]
    
    # 4. Expand state/action
    state_exp = state.unsqueeze(1).expand(B, num_negatives, -1)
    action_exp = action.unsqueeze(1).expand(B, num_negatives, -1)
    
    # 5. Negative energies
    E_neg = ebm(state_exp, action_exp, neg_next_states)
    
    # 6. Loss
    logits = torch.cat([E_pos.unsqueeze(1), E_neg], dim=1) / temperature
    labels = torch.zeros(B, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels)
    
    metrics = {
        'E_pos': E_pos.mean().item(),
        'E_neg': E_neg.mean().item(),
        'E_gap': (E_pos.mean() - E_neg.mean()).item()
    }
    
    return loss, metrics

def train_unified_models():
    print(f"\n>>> PRETRAINING START: MiniGrid [{FLOW_TYPE}] + MDN")
    device = CONFIG["DEVICE"]
    env_adapter = MiniGridAdapter(CONFIG["ENV_NAME"])
    state_dim = env_adapter.state_dim
    action_dim = env_adapter.action_dim # This is 'num_actions' size
    
    # 1. Initialize Replay Buffer
    buffer = ReplayBuffer(state_dim, action_dim)

    # 2. Collect Initial Data
    print(f">>> Collecting {CONFIG['COLLECT_STEPS']} transitions...")
    s = env_adapter.reset()
    for i in range(CONFIG['COLLECT_STEPS']):
        # Sample random discrete action from adapter's env
        # Note: env_adapter.env.action_space is Discrete/Box wrapped? 
        # minigrid vanilla is Discrete. adapter.action_dim is discrete size.
        # We need integer action for Env, but Vector for Buffer.
        
        # Sample integer
        a_int = np.random.randint(0, action_dim) 
        
        # Step env (adapter handles integer)
        ns, reward, done, info = env_adapter.step(a_int)
        
        # Convert to one-hot for buffer
        a_vec = to_one_hot(a_int, action_dim)
        
        buffer.add(s, a_vec, ns)
        if done: s = env_adapter.reset()
        else: s = ns

    # 3. Models
    # Inputs are (state, action_vector)
    ebm = BilinearEBM(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    flow = RealNVP(data_dim=state_dim, context_dim=state_dim + action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    mdn = MixtureDensityNetwork(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    ebm_opt = optim.Adam(ebm.parameters(), lr=CONFIG["LR_EBM"])
    flow_opt = optim.Adam(flow.parameters(), lr=CONFIG["LR_FLOW"])
    mdn_opt = optim.Adam(mdn.parameters(), lr=CONFIG["LR_MDN"])

    # 4. Training Loop
    print("\n>>> Starting Training Loop...")
    for step in range(CONFIG["PRETRAIN_STEPS"]):
        if step % 100 == 0: print(f"Beginning Step {step}...")
        
        s, a, real_ns = buffer.sample(CONFIG["BATCH_SIZE"])
        context = torch.cat([s, a], dim=1)

        # A. TRAIN EBM
        if step % 100 == 0: print(f"  Training EBM...")
        ebm_loss, ebm_metrics = infonce_loss(
            ebm, s, a, real_ns, buffer,
            num_negatives=CONFIG["NUM_NEGATIVES"],
            temperature=CONFIG["TEMPERATURE"]
        )
        
        ebm_opt.zero_grad()
        ebm_loss.backward()
        ebm_opt.step()

        # B. TRAIN FLOW
        if FLOW_TYPE == "ForwardKL":
            # Dequantization: Add noise to discrete data to prevent density collapse
            # Data is scaled by 0.1 (Adapter), so we add noise ~ U(0, 0.1)
            noise = torch.rand_like(real_ns) * 0.1
            real_ns_continuous = real_ns + noise
            
            log_prob = flow.log_prob(real_ns_continuous, context=context)
            loss_flow = -log_prob.mean() / state_dim 
        else: 
            z = torch.randn(s.shape[0], state_dim).to(device)
            fake = flow.sample(z, context=context)
            for p in ebm.parameters(): p.requires_grad = False
            energy = ebm(s, a, fake).mean()
            for p in ebm.parameters(): p.requires_grad = True
            log_p = flow.log_prob(fake, context=context)
            avg_log_p = log_p.mean() / state_dim
            loss_flow = energy + avg_log_p 

        flow_opt.zero_grad(); loss_flow.backward(); flow_opt.step()

        # C. TRAIN MDN
        pi, mu, sigma = mdn(s, a)
        target = real_ns.unsqueeze(1).expand_as(mu)
        dist = torch.distributions.Normal(mu, sigma)
        log_prob_components = dist.log_prob(target).sum(dim=-1) 
        log_pi = torch.log_softmax(pi, dim=1)
        log_likelihood = torch.logsumexp(log_pi + log_prob_components, dim=1)
        loss_mdn = -log_likelihood.mean() / state_dim
        
        mdn_opt.zero_grad(); loss_mdn.backward(); mdn_opt.step()
        
        if step % 100 == 0:
            print(f"Step {step} | EBM: {ebm_loss.item():.4f} (Gap: {ebm_metrics['E_gap']:.2f}) | Flow: {loss_flow.item():.4f} | MDN: {loss_mdn.item():.4f}")
        elif step < 10: # Print first 10 steps
            print(f"Step {step} | EBM: {ebm_loss.item():.4f} | Flow: {loss_flow.item():.4f} | MDN: {loss_mdn.item():.4f}")

    # 5. Save
    torch.save(ebm.state_dict(), f"pretrained_ebm_{CONFIG['ENV_NAME']}.pth")
    torch.save(flow.state_dict(), f"pretrained_flow_{CONFIG['ENV_NAME']}_{FLOW_TYPE}.pth")
    torch.save(mdn.state_dict(), f"pretrained_mdn_{CONFIG['ENV_NAME']}.pth")
    print(f"\n>>> SUCCESS. Saved models for {CONFIG['ENV_NAME']}")

if __name__ == "__main__":
    train_unified_models()
