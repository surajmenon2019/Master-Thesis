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
    from models import BilinearEBM, RealNVP, MixtureDensityNetwork, ValueNetwork, RewardModel 
    from minigrid_stochastic import StochasticMiniGridAdapter
    import torch.nn.functional as F
    from pretrain_ebm_minigrid import infonce_loss, ReplayBuffer, to_one_hot
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
FLOW_TYPE = "ForwardKL" 
CONFIG = {
    "ENV_NAME": "MiniGrid-Stochastic-0.1", # Match the new Policy Tag
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

def train_unified_models():
    print(f"\n>>> STOCHASTIC PRETRAINING START: {CONFIG['ENV_NAME']}")
    device = CONFIG["DEVICE"]
    # Initialize Stochastic Adapter
    env_adapter = StochasticMiniGridAdapter(render_mode=None, slip_prob=0.1)
    
    state_dim = env_adapter.state_dim
    action_dim = env_adapter.action_dim
    
    # 1. Initialize Replay Buffer (Reusable class)
    buffer = ReplayBuffer(state_dim, action_dim)

    # 2. Collect Initial Data
    print(f">>> Collecting {CONFIG['COLLECT_STEPS']} transitions...")
    s = env_adapter.reset()
    for i in range(CONFIG['COLLECT_STEPS']):
        # Sample integer
        a_int = np.random.randint(0, action_dim) 
        
        # Step env
        ns, reward, done, info = env_adapter.step(a_int)
        
        # Convert to one-hot for buffer
        a_vec = to_one_hot(a_int, action_dim)
        
        buffer.add(s, a_vec, ns)
        if done: s = env_adapter.reset()
        else: s = ns

    # 3. Models
    ebm = BilinearEBM(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    flow = RealNVP(data_dim=state_dim, context_dim=state_dim + action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    mdn = MixtureDensityNetwork(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    ebm_opt = optim.Adam(ebm.parameters(), lr=CONFIG["LR_EBM"])
    flow_opt = optim.Adam(flow.parameters(), lr=CONFIG["LR_FLOW"])
    mdn_opt = optim.Adam(mdn.parameters(), lr=CONFIG["LR_MDN"])

    # 4. Training Loop
    print("\n>>> Starting Training Loop...")
    for step in range(CONFIG["PRETRAIN_STEPS"]):
        
        s, a, real_ns = buffer.sample(CONFIG["BATCH_SIZE"])
        context = torch.cat([s, a], dim=1)

        # A. TRAIN EBM
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
        
        if step % 500 == 0:
            print(f"Step {step} | EBM: {ebm_loss.item():.4f} (Gap: {ebm_metrics['E_gap']:.2f}) | Flow: {loss_flow.item():.4f} | MDN: {loss_mdn.item():.4f}")

    # 5. Save with STOCHASTIC suffix
    torch.save(ebm.state_dict(), f"pretrained_ebm_{CONFIG['ENV_NAME']}.pth")
    torch.save(flow.state_dict(), f"pretrained_flow_{CONFIG['ENV_NAME']}_{FLOW_TYPE}.pth")
    torch.save(mdn.state_dict(), f"pretrained_mdn_{CONFIG['ENV_NAME']}.pth")
    print(f"\n>>> SUCCESS. Saved models for {CONFIG['ENV_NAME']}")

if __name__ == "__main__":
    train_unified_models()
