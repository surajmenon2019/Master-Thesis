import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- IMPORTS ---
try:
    import minigrid
    import gymnasium as gym
    from models import BilinearEBM, RealNVP, MixtureDensityNetwork
    # Models Minigrid not needed here unless we use DiscreteActor for init?
    # Actually we just benchmark Gumbel-Softmax gradients through model.
    # So we need DiscreteActor to generate the initial action distribution.
    from models_minigrid import DiscreteActor 
    from minigrid_adapter import MiniGridAdapter
    from utils_sampling import (
        predict_next_state_langevin_adaptive, 
        predict_next_state_svgd_adaptive
    )
except ImportError as e:
    print(f"CRITICAL: Missing modules. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
CONFIG = {
    "ENV_NAME": "MiniGrid-Empty-8x8-v0",
    "METHODS": [
        "Cold Start", 
        "Flow Only", 
        "Warm Start (ForwardKL)", 
        "Warm Start (ReverseKL)", 
        "SVGD",
        "MDN"
    ],
    "HORIZONS": [1, 3, 5, 10, 15, 20], 
    "BATCH_SIZE": 128,
    "HIDDEN_DIM": 128,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- WEIGHT LOADING ---
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

def run_benchmark():
    device = CONFIG["DEVICE"]
    env_adapter = MiniGridAdapter(CONFIG["ENV_NAME"])
    
    # Models
    ebm = BilinearEBM(env_adapter.state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    flow = RealNVP(env_adapter.state_dim, context_dim=env_adapter.state_dim + env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    mdn = MixtureDensityNetwork(env_adapter.state_dim, env_adapter.action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    
    # Random initial state
    s_init = torch.tensor(env_adapter.reset(), dtype=torch.float32).unsqueeze(0).repeat(CONFIG["BATCH_SIZE"], 1).to(device)
    
    # Initial Action: We need a differentiable object.
    # In Continuous: Action Vector.
    # In Discrete: Logits or Softmax Vector.
    # We will optimize the LOGITS of the first action.
    a_logits_init = torch.randn(CONFIG["BATCH_SIZE"], env_adapter.action_dim).to(device).requires_grad_(True)
    
    results = {m: {"grad": [], "cost": []} for m in CONFIG["METHODS"]}
    
    print(f"\n>>> STARTING BENCHMARK: {CONFIG['ENV_NAME']}")
    
    for method in CONFIG["METHODS"]:
        print(f"\n--- Testing: {method} ---")
        load_weights(ebm, method, CONFIG["ENV_NAME"])
        if "Warm" in method or "Flow" in method: load_weights(flow, method, CONFIG["ENV_NAME"])
        if "MDN" in method: load_weights(mdn, method, CONFIG["ENV_NAME"])
        
        INNER_STEPS = 30 if "Cold" in method else 5 
        
        for H in CONFIG["HORIZONS"]:
            t0 = time.perf_counter()
            
            curr_state = s_init
            
            for t in range(H):
                # 1. Action
                if t == 0:
                    # Gumbel-Softmax for first action (differentiable)
                    action = torch.nn.functional.gumbel_softmax(a_logits_init, tau=1.0, hard=False)
                else:
                    # Random (but must be soft vector)
                    rand_logits = torch.randn(CONFIG["BATCH_SIZE"], env_adapter.action_dim).to(device)
                    action = torch.nn.functional.gumbel_softmax(rand_logits, tau=1.0, hard=False)
                
                # 2. Predict
                ns_pred = None
                if method == "MDN":
                    ns_pred = mdn.sample_differentiable(curr_state, action)
                elif method == "Flow Only":
                    z = torch.randn_like(curr_state).to(device)
                    ns_pred = flow.sample(z, context=torch.cat([curr_state, action], dim=1))
                elif method == "SVGD":
                    ns_pred = predict_next_state_svgd_adaptive(
                        ebm, curr_state, action,
                        use_ascent=True,
                        config={"SVGD_STEPS": INNER_STEPS}
                    )
                else: 
                    init = None
                    if "Warm" in method:
                        z = torch.randn_like(curr_state).to(device)
                        init = flow.sample(z, context=torch.cat([curr_state, action], dim=1))
                    
                    ns_pred = predict_next_state_langevin_adaptive(
                        ebm, curr_state, action,
                        init_state=init,
                        use_ascent=True,
                        config={
                            "LANGEVIN_STEPS": INNER_STEPS,
                            "LANGEVIN_NOISE_SCALE": 0.0
                        }
                    )
                
                curr_state = ns_pred
            
            # --- GRADIENT CHECK ---
            loss = curr_state.sum()
            if a_logits_init.grad is not None: a_logits_init.grad.zero_()
            loss.backward()
            
            t1 = time.perf_counter()
            grad_mag = a_logits_init.grad.norm().item()
            cost = (t1 - t0) * 1000
            
            results[method]["grad"].append(grad_mag)
            results[method]["cost"].append(cost)
            print(f"Horizon={H:<2} | Mag: {grad_mag:.4f} | Cost: {cost:.2f}ms")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for m in CONFIG["METHODS"]:
        if len(results[m]["grad"]) == 0: continue
        axes[0].plot(CONFIG["HORIZONS"], results[m]["grad"], marker='o', label=m)
    axes[0].set_title(f"Gradient Magnitude ({CONFIG['ENV_NAME']})")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True)
    
    for m in CONFIG["METHODS"]:
        if len(results[m]["cost"]) == 0: continue
        axes[1].plot(CONFIG["HORIZONS"], results[m]["cost"], marker='o', label=m)
    axes[1].set_title("Computational Cost")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.savefig(f"benchmark_{CONFIG['ENV_NAME']}.png")
    print(f"\n>>> SUCCESS. Plot saved to benchmark_{CONFIG['ENV_NAME']}.png")

if __name__ == "__main__":
    run_benchmark()
