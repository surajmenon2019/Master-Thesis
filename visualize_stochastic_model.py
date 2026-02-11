
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from minigrid_stochastic import StochasticMiniGridAdapter
from models import BilinearEBM, RealNVP
from utils_sampling import predict_next_state_langevin_adaptive
import torch.nn.functional as F
from sklearn.decomposition import PCA

# --- CONFIG ---
ENV_NAME = "MiniGrid-Empty-8x8-v0"
SLIP_PROB = 0.1
MODEL_TAG = "MiniGrid-Stochastic-0.1"
EBM_PATH = f"pretrained_ebm_{MODEL_TAG}.pth"
FLOW_PATH = f"pretrained_flow_{MODEL_TAG}_ForwardKL.pth" # Flow for generic proposal
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM = 128

def to_one_hot(action_idx, num_actions):
    vec = np.zeros(num_actions, dtype=np.float32)
    vec[action_idx] = 1.0
    return vec

def main():
    print(f"--- Visualizing EBM Stochasticity (Slip={SLIP_PROB}) ---")
    
    # 1. Initialize Environment
    env = StochasticMiniGridAdapter(ENV_NAME, slip_prob=SLIP_PROB, render_mode=None)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")
    
    # 2. Load Models
    # EBM
    ebm = BilinearEBM(state_dim, action_dim, hidden_dim=HIDDEN_DIM).to(DEVICE)
    try:
        ebm.load_state_dict(torch.load(EBM_PATH, map_location=DEVICE))
        print(f"Loaded EBM from {EBM_PATH}")
    except FileNotFoundError:
        print(f"ERROR: EBM file {EBM_PATH} not found!")
        return
    ebm.eval()
    
    # Flow (for Proposal / Warm Start)
    flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=HIDDEN_DIM).to(DEVICE)
    try:
        flow.load_state_dict(torch.load(FLOW_PATH, map_location=DEVICE))
        print(f"Loaded Flow from {FLOW_PATH}")
    except FileNotFoundError:
        print(f"WARNING: Flow file {FLOW_PATH} not found! Using Random Initialization.")
        flow = None
    if flow: flow.eval()
    
    # 3. Setup Experiment
    obs = env.reset()
    
    # Action: Move Forward (2)
    action_idx = 2 
    action_onehot = to_one_hot(action_idx, action_dim)
    
    # Batch Predict (Parallel Sampling)
    num_samples = 100
    
    state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).repeat(num_samples, 1).to(DEVICE)
    action_tensor = torch.tensor(action_onehot, dtype=torch.float32).unsqueeze(0).repeat(num_samples, 1).to(DEVICE)

    print(f"Generating {num_samples} samples using Langevin Dynamics...")
    
    # A. Initialization (Flow or Random)
    if flow:
        with torch.no_grad():
            z = torch.randn_like(state_tensor)
            context = torch.cat([state_tensor, action_tensor], dim=1)
            init_state = flow.sample(z, context=context)
        print("  initialized with Flow")
    else:
        init_state = torch.randn_like(state_tensor)
        print("  initialized with Random Noise")

    # B. Refinement (Langevin)
    # We use adaptive langevin with use_ascent=True for BilinearEBM
    config = {
        "LANGEVIN_STEPS": 50,
        "LANGEVIN_STEP_SIZE": 0.05,
        "LANGEVIN_NOISE_SCALE": 0.01
    }
    
    samples = predict_next_state_langevin_adaptive(
        ebm, state_tensor, action_tensor, 
        init_state=init_state, 
        use_ascent=True, # Bilinear EBM needs Ascent
        config=config
    )
    
    predictions = samples.detach().cpu().numpy()

    # 4. Analyze Predictions (PCA)
    print("Computing PCA projection...")
    pca = PCA(n_components=2)
    proj = pca.fit_transform(predictions)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.6, c='blue', label='EBM Predictions')
    
    # Optionally, assume Clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2) # Assume 2 modes: Forward vs Slip (Left/Right might be merged or separate)
    labels = kmeans.fit_predict(predictions)
    centers = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
    
    plt.title(f"EBM Stochastic Predictions (Slip={SLIP_PROB})\nInit: {'Flow' if flow else 'Random'} + Langevin(50)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.legend()
    plt.savefig("ebm_predictions_pca.png")
    print("Saved 'ebm_predictions_pca.png'")
    
    print("Cluster Sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} samples ({c/num_samples:.1%})")

if __name__ == "__main__":
    main()
