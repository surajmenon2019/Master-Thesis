import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
import os

# --- IMPORTS ---
try:
    from models import EnergyBasedModel, RealNVP, MixtureDensityNetwork
    from utils_sampling import predict_next_state_langevin, predict_next_state_svgd
    from env_stochastic_tree import StochasticTreeEnv
    from env_mobile_robot import PointGoalEnv
except ImportError as e:
    print(f"CRITICAL ERROR: Missing Modules. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
CONFIG = {
    "ENV": "StochasticTree", # Change to "StochasticTree" or "PointGoal"
    "BATCH_SIZE": 64,
    "HORIZONS": [5, 10, 20, 30, 50, 75, 100], 
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "METHODS": [
        "Cold Start", 
        "Flow Only", 
        "Warm Start (ForwardKL)", 
        "Warm Start (ReverseKL)", 
        #"Warm Start (Hybrid)",
        "SVGD",
        "MDN"
    ]
}

def get_analytical_reward_gradient_batch(states, actions, env_name):
    """
    Returns the Ground Truth REWARD Gradient w.r.t ACTION.
    Shape must match ACTION space.
    """
    # CRITICAL FIX: Gradient w.r.t Action, so use action shape!
    a_np = actions.detach().cpu().numpy()
    grad = np.zeros_like(a_np) # (Batch, Action_Dim)
    
    s = states.detach().cpu().numpy()

    if env_name == "StochasticTree":
        # Goal: Move UP (+y), Avoid Tree (0, 1)
        # 1. Attraction (Global Goal at 0, 2)
        goal_pos = np.array([0.0, 2.0])
        to_goal = goal_pos - s
        dist_goal = np.linalg.norm(to_goal, axis=1, keepdims=True)
        force_att = to_goal / (dist_goal + 1e-6)

        # 2. Repulsion (Tree at 0, 1)
        tree_pos = np.array([0.0, 1.0])
        to_tree = tree_pos - s
        dist_tree = np.linalg.norm(to_tree, axis=1, keepdims=True)
        
        repulsion_strength = 1.0 / (dist_tree**3 + 0.1)
        repulsion_strength = np.clip(repulsion_strength, 0, 10.0)
        
        force_rep = -to_tree * repulsion_strength
        
        # Total Force in State Space
        total_force_state = force_att + 2.0 * force_rep
        
        # For simple kinematics (s' = s + a), Gradient_Action ~= Gradient_State
        grad = total_force_state

    elif env_name == "PointGoal":
        # Heuristic for PointGoal:
        # Action[0] is usually Forward Velocity.
        # Action[1] is usually Angular Velocity.
        # Optimal simple policy: Just go forward.
        grad[:, 0] = 1.0 
        
    # Normalize
    norm = np.linalg.norm(grad, axis=1, keepdims=True)
    grad = grad / (norm + 1e-8)
    
    return torch.tensor(grad).float().to(states.device)

def load_weights(model, method_name, env_name, model_type="flow"):
    """Helper to load specific weights"""
    filename = None
    if model_type == "flow":
        if "ForwardKL" in method_name or "Standard" in method_name or "Flow Only" in method_name:
            filename = f"pretrained_flow_{env_name}_ForwardKL.pth"
        elif "ReverseKL" in method_name:
            filename = f"pretrained_flow_{env_name}_ReverseKL.pth"
        #elif "Hybrid" in method_name:
        #    filename = f"pretrained_flow_{env_name}_Hybrid.pth"
    
    if filename and os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
        return True
    elif filename:
        # Suppress warning for MDN/SVGD/Cold which don't use flow weights
        if "Warm" in method_name or "Flow" in method_name:
            print(f"!! WARNING: Weights file '{filename}' not found. Skipping {method_name}.")
            return False
    return True 

def run_thesis_benchmark():
    print(f">>> STARTING FINAL THESIS BENCHMARK FOR [{CONFIG['ENV']}]")
    device = CONFIG["DEVICE"]
    
    # 1. SETUP ENVIRONMENT & DIMENSIONS
    if CONFIG["ENV"] == "StochasticTree":
        env = StochasticTreeEnv()
        state_dim = 2
        action_dim = 2
    elif CONFIG["ENV"] == "PointGoal":
        env = PointGoalEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print(f">>> Detected Dimensions: State={state_dim}, Action={action_dim}")
    
    # 2. Initialize Models
    try:
        # Load EBM (Shared Physics)
        ebm = EnergyBasedModel(state_dim, action_dim).to(device)
        try:
            ebm.load_state_dict(torch.load(f"pretrained_ebm_{CONFIG['ENV']}.pth"))
        except:
            print(f"!! Warning: No EBM weights found for {CONFIG['ENV']}. Using random weights.")
        
        # Initialize Flow 
        flow = RealNVP(data_dim=state_dim, context_dim=state_dim + action_dim, hidden_dim=64).to(device)
        
        # Initialize MDN
        mdn = MixtureDensityNetwork(state_dim, action_dim).to(device)
    except Exception as e:
        print(f"CRITICAL: Model Init Error. {e}")
        return

    # 3. Create Evaluation Batch
    states_np = np.random.uniform(-1, 1, size=(CONFIG["BATCH_SIZE"], state_dim))
    
    # Jitter for StochasticTree to avoid symmetry collapse
    if CONFIG["ENV"] == "StochasticTree":
        jitter = np.random.uniform(-0.1, 0.1, size=(CONFIG["BATCH_SIZE"],))
        states_np[:, 0] += jitter
        n_crit = CONFIG["BATCH_SIZE"] // 4
        states_np[:n_crit] = np.random.normal(loc=[0, 0.8], scale=0.1, size=(n_crit, 2))
        
    states = torch.tensor(states_np).float().to(device)
    actions = torch.zeros(CONFIG["BATCH_SIZE"], action_dim).to(device).requires_grad_(True)
    
    # Get Ground Truth (Reward Direction)
    true_reward_grads = get_analytical_reward_gradient_batch(states, actions, CONFIG["ENV"])
    
    results = defaultdict(lambda: defaultdict(dict))

    # 4. Main Loop
    for H in CONFIG["HORIZONS"]:
        print(f"\n=== Horizon (Steps): {H} ===")
        
        for method in CONFIG["METHODS"]:
            # Load Weights
            if "Flow" in method or "Warm" in method:
                if not load_weights(flow, method, CONFIG['ENV'], "flow"): continue
            
            # --- EXECUTION ---
            t0 = time.perf_counter()
            
            # A. Sampling (Forward Pass)
            init_state = None
            pred_ns = None
            
            if method == "MDN":
                # MDN Direct Sampling
                pi_logits, mu, sigma = mdn(states, actions)
                weights = torch.softmax(pi_logits, dim=1).unsqueeze(-1)
                pred_ns = (weights * mu).sum(dim=1)
            
            else:
                # Flow / EBM / SVGD Logic
                if "Flow" in method or "Warm" in method:
                    z = torch.randn(CONFIG["BATCH_SIZE"], state_dim).to(device)
                    context = torch.cat([states, actions], dim=1)
                    init_state = flow.sample(z, context=context)
                
                if method == "Flow Only":
                    pred_ns = init_state
                elif method == "SVGD":
                    pred_ns = predict_next_state_svgd(ebm, states, actions, num_particles=10, config={"SVGD_STEPS": H})
                else: # Langevin
                    pred_ns = predict_next_state_langevin(
                        ebm, states, actions, 
                        init_state=init_state, 
                        config={"LANGEVIN_STEPS": H}
                    )
            
            # B. Gradient Calculation (Backward Pass)
            # Objective: Maximize value of first dimension (Forward Velocity heuristic)
            goal_pos = torch.tensor([0.0, 2.0]).to(device)
            dist = torch.norm(pred_ns - goal_pos, dim=1)
            loss = dist.sum()
            
            if actions.grad is not None: actions.grad.zero_()
            loss.backward()
            grad_est = actions.grad.clone()
            
            t1 = time.perf_counter()
            
            # --- METRICS ---
            
            # 1. Distance from Ground Truth (Cosine Similarity)
            grad_est_norm = grad_est / (grad_est.norm(dim=1, keepdim=True) + 1e-8)
            # Compare Estimated Action Grad vs True Action Grad
            # Loss Gradient should be opposite to Reward Gradient
            acc = torch.sum(grad_est_norm * (-true_reward_grads), dim=1).mean().item()
            
            # 2. Gradient Stability (Magnitude)
            mag = grad_est.norm(dim=1).mean().item()
            
            # 3. Cost
            cost = (t1 - t0) * 1000.0 

            results[method][H] = {"acc": acc, "mag": mag, "cost": cost}
            print(f"[{method:25}] Acc: {acc:6.3f} | Mag: {mag:6.4f} | Cost: {cost:6.2f}ms")

    # 5. Plotting
    plot_final_graphs(results, CONFIG["HORIZONS"])

def plot_final_graphs(results, horizons):
    methods = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.set_style("whitegrid")
    
    metrics = ["acc", "mag", "cost"]
    titles = ["Metric 1: Gradient Accuracy", "Metric 2: Stability (Mag)", "Metric 3: Cost (Log)"]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for m in methods:
            y = [results[m][h][metric] for h in horizons]
            ax.plot(horizons, y, marker='o', label=m)
        ax.set_title(titles[i])
        ax.set_xlabel("Horizon")
        if metric == "cost": ax.set_yscale("log")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"thesis_benchmark_{CONFIG['ENV']}.png")
    print(f"\n>>> SAVED: thesis_benchmark_{CONFIG['ENV']}.png")

if __name__ == "__main__":
    run_thesis_benchmark()