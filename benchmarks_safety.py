import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- IMPORTS ---
try:
    import safety_gymnasium
    from models import EnergyBasedModel, RealNVP, MixtureDensityNetwork
    # We import the exact same sampling logic used in training
    from utils_sampling import predict_next_state_langevin, predict_next_state_svgd
except ImportError as e:
    print(f"CRITICAL: Missing modules. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
CONFIG = {
    "ENV_NAME": "SafetyPointGoal1-v0",
    "METHODS": [
        "Cold Start", 
        "Flow Only", 
        "Warm Start (ForwardKL)", 
        "Warm Start (ReverseKL)", 
        "SVGD"
    ],
    # THE STUDY: How deep can we plan before gradients vanish?
    # We test trajectory lengths from 1 to 20 steps.
    "HORIZONS": [1, 3, 5, 10, 15, 20], 
    
    "BATCH_SIZE": 32,
    "HIDDEN_DIM": 128,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- 1. SAFETY GYM ADAPTER (Matches train_mbrl.py) ---
class SafetyGymAdapter:
    def __init__(self, env_name, device):
        self.device = device
        self.env = safety_gymnasium.make(env_name, render_mode=None)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

    def reset_batch(self, batch_size):
        states = []
        for _ in range(batch_size):
            s, _ = self.env.reset()
            states.append(s)
        return torch.tensor(np.array(states), dtype=torch.float32).to(self.device)

# --- 2. WEIGHT LOADING ---
def load_weights(model, method, env_name):
    try:
        if isinstance(model, EnergyBasedModel):
            model.load_state_dict(torch.load(f"pretrained_ebm_{env_name}.pth"))
        elif isinstance(model, RealNVP):
            suffix = "ForwardKL" if "ForwardKL" in method or "Flow" in method else "ReverseKL"
            model.load_state_dict(torch.load(f"pretrained_flow_{env_name}_{suffix}.pth"))
        print(f"-> Loaded weights for {method}")
    except FileNotFoundError:
        print(f"!! Warning: No weights found for {method}, using random.")

# --- 3. BENCHMARK LOOP ---
def run_benchmark():
    device = CONFIG["DEVICE"]
    env_adapter = SafetyGymAdapter(CONFIG["ENV_NAME"], device)
    
    # Initialize Models
    ebm = EnergyBasedModel(env_adapter.state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    flow = RealNVP(env_adapter.state_dim, context_dim=env_adapter.state_dim + env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    mdn = MixtureDensityNetwork(env_adapter.state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    # Initial State
    s_init = env_adapter.reset_batch(CONFIG["BATCH_SIZE"])
    
    # We want to measure d(Final_State) / d(Initial_Action)
    # This proves we can optimize the first action based on the future outcome.
    a_init = torch.randn(CONFIG["BATCH_SIZE"], env_adapter.action_dim).to(device).requires_grad_(True)
    
    results = {m: {"grad": [], "cost": []} for m in CONFIG["METHODS"]}

    print(f"\n>>> STARTING HORIZON BENCHMARK (Outer Loop Unroll)")
    print(f"    Testing Gradient Flow through Time: {CONFIG['HORIZONS']}")

    for method in CONFIG["METHODS"]:
        print(f"\n--- Testing: {method} ---")
        
        # Load Weights
        load_weights(ebm, method, CONFIG["ENV_NAME"])
        if "Warm" in method or "Flow" in method: 
            load_weights(flow, method, CONFIG["ENV_NAME"])

        # Determine Refinement Steps (Inner Loop)
        # Cold Start needs more steps to be physically valid.
        # Warm Start works with fewer steps (This is the efficiency gain).
        if "Cold" in method:
            INNER_STEPS = 30
        else:
            INNER_STEPS = 5 
        
        for H in CONFIG["HORIZONS"]:
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            
            # --- TRAJECTORY UNROLLING ---
            curr_state = s_init
            
            # We unroll the loop H times
            for t in range(H):
                # 1. Select Action
                # For t=0, we use a_init (which requires grad).
                # For t>0, we use a random action (representing a policy).
                # Note: We don't need a trained actor to check gradient flow through dynamics.
                if t == 0:
                    action = a_init
                else:
                    # Random policy for future steps
                    action = torch.randn(CONFIG["BATCH_SIZE"], env_adapter.action_dim).to(device)
                
                # 2. Predict Next State (The Transition)
                ns_pred = None
                
                if method == "MDN":
                    ns_pred = mdn.sample_differentiable(curr_state, action)
                elif method == "Flow Only":
                    z = torch.randn_like(curr_state).to(device)
                    # For Flow Only, prediction is 1-step generative
                    ns_pred = flow.sample(z, context=torch.cat([curr_state, action], dim=1))
                elif method == "SVGD":
                    ns_pred = predict_next_state_svgd(ebm, curr_state, action, config={"SVGD_STEPS": INNER_STEPS})
                else:
                    # Langevin (Cold or Warm)
                    init = None
                    if "Warm" in method:
                        z = torch.randn_like(curr_state).to(device)
                        init = flow.sample(z, context=torch.cat([curr_state, action], dim=1))
                    
                    ns_pred = predict_next_state_langevin(
                        ebm, curr_state, action, init_state=init, 
                        config={"LANGEVIN_STEPS": INNER_STEPS} # Fixed small budget for Warm
                    )
                
                # Move to next step
                curr_state = ns_pred
            
            # --- GRADIENT CHECK ---
            # We check the gradient of the FINAL state w.r.t INITIAL action
            loss = curr_state.sum()
            
            if a_init.grad is not None: a_init.grad.zero_()
            loss.backward()
            
            t1 = time.perf_counter()
            cost = (t1 - t0) * 1000
            
            # If gradients vanished, this will be 0.
            # If gradients exploded, this will be NaN or Huge.
            grad_mag = a_init.grad.norm().item()
            
            results[method]["grad"].append(grad_mag)
            results[method]["cost"].append(cost)
            print(f"Horizon={H:<2} | Inner={INNER_STEPS} | Mag: {grad_mag:.4f} | Cost: {cost:.2f}ms")

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    for m in CONFIG["METHODS"]:
        ax.plot(CONFIG["HORIZONS"], results[m]["grad"], marker='o', label=m, linewidth=2)
    ax.set_title("Gradient Signal vs Planning Horizon")
    ax.set_yscale("log")
    ax.set_xlabel("Planning Horizon (Steps)")
    ax.set_ylabel("Gradient Norm (Initial Action)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    ax = axes[1]
    for m in CONFIG["METHODS"]:
        ax.plot(CONFIG["HORIZONS"], results[m]["cost"], marker='o', label=m, linewidth=2)
    ax.set_title("Computational Cost vs Horizon")
    ax.set_yscale("log")
    ax.set_xlabel("Planning Horizon (Steps)")
    ax.set_ylabel("Time (ms)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("thesis_horizon_benchmark2.png", dpi=300)
    print("\n>>> SUCCESS. Plot saved to 'thesis_horizon_benchmark2.png'")

if __name__ == "__main__":
    run_benchmark()