import torch
import torch.optim as optim
import numpy as np
import sys
import os

# --- IMPORTS ---
try:
    import safety_gymnasium
    # InfoNCE: BilinearEBM, Flow, MDN
    from models import BilinearEBM, RealNVP, MixtureDensityNetwork 
    import torch.nn.functional as F
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
    "LR_MDN": 1e-3,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "HIDDEN_DIM": 128,
    # InfoNCE-specific parameters
    "NUM_NEGATIVES": 128,  # Number of random negatives for InfoNCE (was 512, reduced for speed)
    "TEMPERATURE": 0.1,    # InfoNCE temperature
    # Langevin only needed for Flow warm-start (not EBM training)
    "LANGEVIN_STEPS": 30,
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

# --- INFONCE LOSS FUNCTION ---
def infonce_loss(ebm, state, action, pos_next_state, buffer, 
                 num_negatives=512, temperature=0.1, explicit_negatives=None):
    """
    InfoNCE loss for training BilinearEBM dynamics models.
    Supports explicit negatives (e.g., from Langevin sampling) for Hard Negative Mining.
    
    Args:
        ebm: BilinearEBM model
        state: (B, state_dim)
        action: (B, action_dim)
        pos_next_state: (B, state_dim) - real next states
        buffer: ReplayBuffer with .next_states and .size
        num_negatives: Number of random negatives per positive
        temperature: Temperature for softmax (lower = harder)
        explicit_negatives: Optional (B, M, state_dim) tensor of hard negatives
    
    Returns:
        loss: Scalar InfoNCE loss
        metrics: Dict with E_pos and E_neg for logging
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
    
    # 3. Add explicit negatives if provided (Hard Negative Mining)
    if explicit_negatives is not None:
        neg_next_states = torch.cat([neg_next_states, explicit_negatives], dim=1)
        num_negatives += explicit_negatives.shape[1]
    
    # 4. Expand state/action for all negatives
    state_exp = state.unsqueeze(1).expand(B, num_negatives, -1)  # (B, K_total, state_dim)
    action_exp = action.unsqueeze(1).expand(B, num_negatives, -1)  # (B, K_total, action_dim)
    
    # 5. Negative energies: E(s,a,s'_neg) - should be LOW
    E_neg = ebm(state_exp, action_exp, neg_next_states)  # (B, K_total)
    
    # 6. InfoNCE loss: -log(exp(E_pos/T) / (exp(E_pos/T) + sum(exp(E_neg/T))))
    # Numerically stable version using cross-entropy
    logits = torch.cat([E_pos.unsqueeze(1), E_neg], dim=1) / temperature  # (B, K_total+1)
    labels = torch.zeros(B, dtype=torch.long, device=device)  # Positive at index 0
    loss = F.cross_entropy(logits, labels)
    
    # 7. Metrics for logging
    metrics = {
        'E_pos': E_pos.mean().item(),
        'E_neg': E_neg.mean().item(),
        'E_gap': (E_pos.mean() - E_neg.mean()).item()
    }
    
    return loss, metrics

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
    ebm = BilinearEBM(state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    flow = RealNVP(data_dim=state_dim, context_dim=state_dim + env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    mdn = MixtureDensityNetwork(state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    ebm_opt = optim.Adam(ebm.parameters(), lr=CONFIG["LR_EBM"])
    flow_opt = optim.Adam(flow.parameters(), lr=CONFIG["LR_FLOW"])
    mdn_opt = optim.Adam(mdn.parameters(), lr=CONFIG["LR_MDN"])

    # 4. Training Loop
    print("\n>>> Starting Training Loop...")
    print(f">>> EBM Training: InfoNCE (no MCMC!) with {CONFIG['NUM_NEGATIVES']} negatives")
    for step in range(CONFIG["PRETRAIN_STEPS"]):
        s, a, real_ns = buffer.sample(CONFIG["BATCH_SIZE"])
        context = torch.cat([s, a], dim=1)

        # ==========================
        # A. TRAIN EBM (InfoNCE - NO LANGEVIN!)
        # ==========================
        ebm_loss, ebm_metrics = infonce_loss(
            ebm, s, a, real_ns, buffer,
            num_negatives=CONFIG["NUM_NEGATIVES"],
            temperature=CONFIG["TEMPERATURE"]
        )
        
        ebm_opt.zero_grad()
        ebm_loss.backward()
        ebm_opt.step()

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
            print(f"Step {step} | EBM: {ebm_loss.item():.4f} (E_pos: {ebm_metrics['E_pos']:.2f}, E_neg: {ebm_metrics['E_neg']:.2f}, gap: {ebm_metrics['E_gap']:.2f}) | Flow: {loss_flow.item():.4f} | MDN: {loss_mdn.item():.4f}")

    # 5. Save
    torch.save(ebm.state_dict(), f"pretrained_ebm_{CONFIG['ENV_NAME']}.pth")
    torch.save(flow.state_dict(), f"pretrained_flow_{CONFIG['ENV_NAME']}_{FLOW_TYPE}.pth")
    torch.save(mdn.state_dict(), f"pretrained_mdn_{CONFIG['ENV_NAME']}.pth")
    print("\n>>> SUCCESS. Saved models (EBM, Flow, MDN).")

if __name__ == "__main__":
    train_unified_models()