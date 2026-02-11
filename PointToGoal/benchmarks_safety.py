import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- IMPORTS ---
try:
    import safety_gymnasium
    from models import BilinearEBM, RealNVP, MixtureDensityNetwork
    from utils_sampling import (
        predict_next_state_langevin_adaptive,
        predict_next_state_svgd_adaptive
    )
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
        "SVGD",
        "MDN"  # Added MDN
    ],
    "HORIZONS": [1, 3, 5, 10, 15, 20, 25, 30, 35, 40], 
    "BATCH_SIZE": 32,
    "HIDDEN_DIM": 128,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- 1. ADAPTER ---
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

# --- 2. WEIGHT LOADING (UPDATED) ---
def load_weights(model, method, env_name):
    try:
        if isinstance(model, BilinearEBM):
            model.load_state_dict(torch.load(f"pretrained_ebm_{env_name}.pth", map_location="cpu", weights_only=False))
        elif isinstance(model, RealNVP):
            suffix = "ForwardKL" if "ForwardKL" in method or "Flow" in method else "ReverseKL"
            model.load_state_dict(torch.load(f"pretrained_flow_{env_name}_{suffix}.pth", map_location="cpu", weights_only=False))
        elif isinstance(model, MixtureDensityNetwork):
            model.load_state_dict(torch.load(f"pretrained_mdn_{env_name}.pth", map_location="cpu", weights_only=False))
            
        print(f"-> Loaded weights for {method}")
    except FileNotFoundError:
        print(f"!! Warning: No weights found for {method}, using random.")

# --- 3. BENCHMARK LOOP ---
def run_benchmark():
    device = CONFIG["DEVICE"]
    env_adapter = SafetyGymAdapter(CONFIG["ENV_NAME"], device)
    
    # Initialize Models
    # BilinearEBM uses higher=better energy convention (trained with InfoNCE)
    ebm = BilinearEBM(env_adapter.state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    flow = RealNVP(env_adapter.state_dim, context_dim=env_adapter.state_dim + env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    mdn = MixtureDensityNetwork(env_adapter.state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    # Initial State for Benchmark
    s_init = env_adapter.reset_batch(CONFIG["BATCH_SIZE"])
    a_init = torch.randn(CONFIG["BATCH_SIZE"], env_adapter.action_dim).to(device).requires_grad_(True)
    
    results = {m: {"grad": [], "cost": []} for m in CONFIG["METHODS"]}

    print(f"\n>>> STARTING HORIZON BENCHMARK (Outer Loop Unroll)")

    for method in CONFIG["METHODS"]:
        print(f"\n--- Testing: {method} ---")
        
        # Load correct models
        load_weights(ebm, method, CONFIG["ENV_NAME"])
        if "Warm" in method or "Flow" in method: load_weights(flow, method, CONFIG["ENV_NAME"])
        if "MDN" in method: load_weights(mdn, method, CONFIG["ENV_NAME"])

        INNER_STEPS = 30 if "Cold" in method else 5 
        
        for H in CONFIG["HORIZONS"]:
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            
            curr_state = s_init
            
            for t in range(H):
                # 1. Select Action (Gradient required at t=0)
                action = a_init if t == 0 else torch.randn(CONFIG["BATCH_SIZE"], env_adapter.action_dim).to(device)
                
                # 2. Predict Next State
                ns_pred = None
                
                if method == "MDN":
                    ns_pred = mdn.sample_differentiable(curr_state, action)
                elif method == "Flow Only":
                    z = torch.randn_like(curr_state).to(device)
                    ns_pred = flow.sample(z, context=torch.cat([curr_state, action], dim=1))
                elif method == "SVGD":
                    # BilinearEBM: use gradient ASCENT (higher energy = better)
                    ns_pred = predict_next_state_svgd_adaptive(
                        ebm, curr_state, action,
                        use_ascent=True,  # BilinearEBM needs ascent
                        config={"SVGD_STEPS": INNER_STEPS}
                    )
                else: # Langevin
                    init = None
                    if "Warm" in method:
                        z = torch.randn_like(curr_state).to(device)
                        init = flow.sample(z, context=torch.cat([curr_state, action], dim=1))
                    
                    # BilinearEBM: use gradient ASCENT (higher energy = better)
                    ns_pred = predict_next_state_langevin_adaptive(
                        ebm, curr_state, action,
                        init_state=init,
                        use_ascent=True,
                        config={
                            "LANGEVIN_STEPS": INNER_STEPS,
                            "LANGEVIN_NOISE_SCALE": 0.0  # No noise for gradient flow
                        }
                    )
                
                curr_state = ns_pred
            
            # --- GRADIENT CHECK ---
            loss = curr_state.sum()
            if a_init.grad is not None: a_init.grad.zero_()
            loss.backward()
            
            t1 = time.perf_counter()
            grad_mag = a_init.grad.norm().item()
            cost = (t1 - t0) * 1000
            
            results[method]["grad"].append(grad_mag)
            results[method]["cost"].append(cost)
            print(f"Horizon={H:<2} | Inner={INNER_STEPS} | Mag: {grad_mag:.4f} | Cost: {cost:.2f}ms")

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define explicit colors to prevent collisions/invisibility
    COLOR_MAP = {
        "Cold Start": "tab:blue",
        "Flow Only": "tab:purple",             # Distinct Purple
        "Warm Start (ForwardKL)": "tab:orange",# Distinct Orange
        "Warm Start (ReverseKL)": "tab:green", # Distinct Green
        "SVGD": "tab:red",
        "MDN": "tab:brown"                     # Distinct Brown
    }

    # Plot 1: Gradient Magnitude
    ax = axes[0]
    for m in CONFIG["METHODS"]:
        # Safety check: if method failed, skip plotting to prevent crash
        if len(results[m]["grad"]) == 0: continue
        
        ax.plot(CONFIG["HORIZONS"], results[m]["grad"], 
                marker='o', label=m, linewidth=2, color=COLOR_MAP.get(m, "black"))
                
    ax.set_title("Gradient Signal vs Planning Horizon")
    ax.set_yscale("log")
    ax.set_xlabel("Planning Horizon (Steps)")
    ax.set_ylabel("Gradient Norm (Initial Action)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Plot 2: Computational Cost
    ax = axes[1]
    for m in CONFIG["METHODS"]:
        if len(results[m]["cost"]) == 0: continue

        ax.plot(CONFIG["HORIZONS"], results[m]["cost"], 
                marker='o', label=m, linewidth=2, color=COLOR_MAP.get(m, "black"))
                
    ax.set_title("Computational Cost vs Horizon")
    ax.set_yscale("log")
    ax.set_xlabel("Planning Horizon (Steps)")
    ax.set_ylabel("Time (ms)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("thesis_horizon_benchmark2.png", dpi=300)
    print("\n>>> SUCCESS. Plot saved.")

if __name__ == "__main__":
    run_benchmark()