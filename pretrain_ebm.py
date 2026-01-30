import torch
import torch.optim as optim
import numpy as np
import os
import sys

# --- IMPORTS ---
# Ensure models.py is the updated version
try:
    from models import EnergyBasedModel, RealNVP 
    from env_stochastic_tree import StochasticTreeEnv
    from env_mobile_robot import PointGoalEnv 
    from utils_sampling import predict_next_state_langevin
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
# CHANGE THIS VARIABLE TO TRAIN DIFFERENT WARM STARTS
# Options: "ForwardKL", "ReverseKL"
FLOW_TYPE = "ForwardKL" 

CONFIG = {
    "ENV_NAME": "StochasticTree", 
    "STEPS": 10000,
    "BATCH_SIZE": 256,
    "LR_EBM": 1e-4,
    "LR_FLOW": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "LANGEVIN_STEPS": 20, # Steps used inside CD training
}

def train_unified_models():
    print(f"\n>>> EXPERIMENT START: Fixed EBM (CD) + Flow via [{FLOW_TYPE}]")
    device = CONFIG["DEVICE"]
    env_name = CONFIG["ENV_NAME"]
    
    # 1. Environment Setup
    if env_name == "StochasticTree":
        env = StochasticTreeEnv()
        state_dim = 2
        action_dim = 2
    elif env_name == "PointGoal":
        env = PointGoalEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError("Unknown Environment")

    # 2. Model Initialization
    # A. The World Model (EBM) - Always Fixed Architecture
    ebm = EnergyBasedModel(state_dim, action_dim).to(device)
    ebm_opt = optim.Adam(ebm.parameters(), lr=CONFIG["LR_EBM"])
    
    # B. The Warm Starter (Flow) - Conditioning on (s, a)
    # data_dim = state_dim (Predicting s')
    # context_dim = state_dim + action_dim (Conditioning on s, a)
    flow = RealNVP(data_dim=state_dim, context_dim=state_dim + action_dim, hidden_dim=64).to(device)
    flow_opt = optim.Adam(flow.parameters(), lr=CONFIG["LR_FLOW"])

    # 3. Data Collection (The "World Knowledge")
    print(f">>> Collecting {CONFIG['STEPS']} transitions...")
    buffer = []
    obs, _ = env.reset()
    for _ in range(CONFIG["STEPS"]):
        action = env.action_space.sample()
        next_obs, _, done, _, _ = env.step(action)
        buffer.append((obs, action, next_obs))
        obs = next_obs
        if done: obs, _ = env.reset()
    
    # 4. Main Training Loop
    print(">>> Starting Training Loop...")
    
    for step in range(CONFIG["STEPS"]):
        # A. Sample Batch
        idxs = np.random.randint(0, len(buffer), CONFIG["BATCH_SIZE"])
        batch = [buffer[k] for k in idxs]
        
        s = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
        a = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(device)
        real_ns = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).to(device)
        context = torch.cat([s, a], dim=1)

        # ==========================================
        # PART 1: TRAIN THE EBM (ALWAYS CD)
        # ==========================================
        # Positive Phase: Pull down energy of real data
        pos_energy = ebm(s, a, real_ns).mean()
        
        # Negative Phase: Push up energy of "Fantasy" particles
        # We use the Flow to initialize the Langevin chain.
        # This helps the EBM explore "Likely" regions even early in training.
        with torch.no_grad():
            z = torch.randn(s.shape[0], state_dim).to(device)
            init_state = flow.sample(z, context=context)
        
        # Langevin Dynamics (Gradient flows w.r.t s' but detached for EBM update)
        fake_ns = predict_next_state_langevin(
            ebm, s, a, init_state=init_state, 
            config={"LANGEVIN_STEPS": CONFIG["LANGEVIN_STEPS"]}
        ).detach()
        
        neg_energy = ebm(s, a, fake_ns).mean()
        
        # Contrastive Divergence Loss
        ebm_loss = pos_energy - neg_energy + (pos_energy**2 + neg_energy**2) * 0.1
        
        ebm_opt.zero_grad()
        ebm_loss.backward()
        ebm_opt.step()

        # ==========================================
        # PART 2: TRAIN THE FLOW (VARIABLE METHOD)
        # ==========================================
        
        if FLOW_TYPE == "ForwardKL":
            # --- STANDARD / MLE ---
            # Maximize Log Likelihood of Real Data.
            # "Learn the Replay Buffer distribution"
            log_prob = flow.log_prob(real_ns, context=context)
            flow_loss = -log_prob.mean()
            
        elif FLOW_TYPE == "ReverseKL":
            # --- AMORTIZED / ENERGY-SEEKING ---
            # Minimize KL(Flow || EBM)
            # "Learn to generate samples that the EBM thinks are low energy"
            
            # 1. Generate Samples
            z = torch.randn(s.shape[0], state_dim).to(device)
            fake_sample = flow.sample(z, context=context)
            
            # 2. Calculate Energy (Target)
            # CRITICAL: Freeze EBM. We want Flow to move to EBM, not EBM to move to Flow.
            for p in ebm.parameters(): p.requires_grad = False
            energy_score = ebm(s, a, fake_sample).mean()
            for p in ebm.parameters(): p.requires_grad = True
            
            # 3. Entropy Regularization (Prevent Mode Collapse)
            # Loss = Energy - Entropy
            log_prob_fake = flow.log_prob(fake_sample, context=context)
            entropy = -log_prob_fake.mean()
            
            flow_loss = energy_score - 0.1 * entropy

            
        else:
            raise ValueError(f"Unknown FLOW_TYPE: {FLOW_TYPE}")

        flow_opt.zero_grad()
        flow_loss.backward()
        flow_opt.step()
        
        if step % 1000 == 0:
            print(f"Step {step}: EBM Loss={ebm_loss.item():.4f} | Flow Loss={flow_loss.item():.4f}")

    # 5. Saving
    # EBM is generally the same across runs (physics is physics), but we overwrite to keep it synced.
    torch.save(ebm.state_dict(), f"pretrained_ebm_{env_name}.pth")
    
    # Save Flow with unique name for the benchmark script
    flow_filename = f"pretrained_flow_{env_name}_{FLOW_TYPE}.pth"
    torch.save(flow.state_dict(), flow_filename)
    
    print(f"\n>>> SUCCESS. Saved models:")
    print(f"    EBM:  pretrained_ebm_{env_name}.pth")
    print(f"    Flow: {flow_filename}")
    print(">>> You may now run the next Flow Type or proceed to benchmarking.")

if __name__ == "__main__":
    # Ensure no file permission errors by removing old temp files if needed
    # (Optional cleanup logic could go here)
    train_unified_models()