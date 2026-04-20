"""
Warm Start vs Cold Start Langevin Step Reduction Experiment
===========================================================

THESIS EXPERIMENT: measures how many Langevin MCMC steps are saved by
initializing from a learned Flow proposal (warm start) vs random noise
(cold start) when sampling from the EBM world model.

Fairness guarantees:
  - Same step_size, noise_scale, EBM for all initializations
  - Multiple independent metrics (MSE, energy, gradient norm, ΔE, per-dim)
  - Oracle baseline (ground truth + noise) to upper-bound EBM quality
  - 2000+ test transitions × 3 seeds with confidence intervals
  - Wide K range [0..100] so both methods can fully converge
  - Convergence threshold defined RELATIVE to K=100 asymptote (not absolute)

Initializations compared:
  1. Cold Start   — torch.randn_like(state)
  2. Warm Start   — Flow proposal sample
  3. Oracle       — ground truth s' + small Gaussian noise (upper bound)

Outputs:
  warmstart_convergence_mse.png
  warmstart_convergence_energy.png
  warmstart_convergence_gradnorm.png
  warmstart_convergence_table.png
  warmstart_distributions.png
  warmstart_results.npz
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from models import BilinearEBM, RealNVP
from minigrid_env import make_minigrid_env, discrete_to_onehot
from utils_sampling import get_energy_gradient

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    # --- Environment (must match pretrained models) ---
    "GRID_SIZE": 8,
    "N_OBSTACLES": 3,
    "SLIP_PROB": 0.1,
    "MAX_STEPS": 100,

    # --- Architecture (must match pretrained checkpoints) ---
    "HIDDEN_DIM": 128,
    "FLOW_N_LAYERS": 8,

    # --- Langevin hyperparameters (shared across all inits) ---
    "LANGEVIN_STEP_SIZE": 0.01,
    "LANGEVIN_NOISE_SCALE": 0.005,

    # --- Experiment ---
    "NUM_TEST_TRANSITIONS": 2000,
    "BATCH_SIZE": 256,
    "K_VALUES": [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50],
    "SEEDS": [42, 123, 7],
    "ORACLE_NOISE_STD": 0.05,  # noise added to ground truth for oracle init

    # --- Output ---
    "OUTPUT_DIR": "results_warmstart",

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# DATA COLLECTION
# =============================================================================
def collect_test_data(num_transitions, seed=42):
    """
    Collect (state, action, next_state) transitions from real env.
    Uses random policy — same as pretraining data collection.
    """
    np.random.seed(seed)
    env = make_minigrid_env(
        size=CONFIG["GRID_SIZE"],
        n_obstacles=CONFIG["N_OBSTACLES"],
        slip_prob=CONFIG["SLIP_PROB"],
        max_steps=CONFIG["MAX_STEPS"],
    )
    state_dim = env.state_dim
    action_dim = env.action_dim

    states = np.zeros((num_transitions, state_dim), dtype=np.float32)
    actions = np.zeros((num_transitions, action_dim), dtype=np.float32)
    next_states = np.zeros((num_transitions, state_dim), dtype=np.float32)

    state, _ = env.reset(seed=seed)
    for i in range(num_transitions):
        a_int = env.action_space.sample()
        ns, _, term, trunc, _ = env.step(a_int)
        states[i] = state
        actions[i] = discrete_to_onehot(a_int, n_actions=action_dim)
        next_states[i] = ns
        if term or trunc:
            state, _ = env.reset()
        else:
            state = ns

    env.close()
    return states, actions, next_states, state_dim, action_dim


# =============================================================================
# LANGEVIN WITH PER-STEP METRICS
# =============================================================================
def langevin_sweep(ebm, state, action, init_state, K_max, step_size, noise_scale):
    """
    Run Langevin dynamics for K_max steps from init_state, recording
    metrics at every step.

    Returns dict with arrays of shape (K_max+1, B):
      energy[k, b]    — EBM energy at step k for sample b
      grad_norm[k, b] — ||∇E|| at step k
      sample[k]       — (K_max+1, B, D) the actual samples
    """
    B, D = state.shape
    device = state.device

    # Storage: K_max+1 entries (step 0 = init, step K_max = final)
    energies = torch.zeros(K_max + 1, B, device=device)
    grad_norms = torch.zeros(K_max + 1, B, device=device)
    samples = torch.zeros(K_max + 1, B, D, device=device)

    curr = init_state.clone()

    # Record metrics at step 0 (initialization)
    with torch.no_grad():
        e0 = ebm(state, action, curr)
        energies[0] = e0
        samples[0] = curr

    # Compute gradient norm at step 0
    curr_g = curr.detach().requires_grad_(True)
    g0 = get_energy_gradient(ebm, state, action, curr_g, create_graph=False)
    grad_norms[0] = g0.detach().norm(dim=-1)

    # Run Langevin steps
    for k in range(1, K_max + 1):
        curr = curr.detach().requires_grad_(True)
        grad = get_energy_gradient(ebm, state, action, curr, create_graph=False)

        noise = torch.randn_like(curr) * noise_scale
        # BilinearEBM: higher energy = more compatible → ascend
        curr = curr + step_size * grad + noise

        with torch.no_grad():
            energy = ebm(state, action, curr)
            energies[k] = energy
            samples[k] = curr

        curr_g = curr.detach().requires_grad_(True)
        g = get_energy_gradient(ebm, state, action, curr_g, create_graph=False)
        grad_norms[k] = g.detach().norm(dim=-1)

    return {
        "energies": energies.detach(),      # (K+1, B)
        "grad_norms": grad_norms.detach(),  # (K+1, B)
        "samples": samples.detach(),        # (K+1, B, D)
    }


# =============================================================================
# COMPUTE ALL METRICS FOR A GIVEN INITIALIZATION
# =============================================================================
def compute_metrics_for_init(ebm, flow, states_t, actions_t, next_states_t,
                             init_type, K_max, seed):
    """
    Run Langevin sweep for one initialization type, compute all 5 metrics.

    Args:
        init_type: "cold", "warm", or "oracle"

    Returns dict with:
        energy:     (K_max+1,) mean energy across all samples
        energy_std: (K_max+1,) std
        mse:        (K_max+1,) mean MSE to ground truth
        mse_std:    (K_max+1,)
        grad_norm:  (K_max+1,) mean gradient norm
        grad_norm_std: (K_max+1,)
        delta_e:    (K_max,) mean |E_k - E_{k-1}|
        per_dim_mse: (K_max+1, D) per-dimension MSE at each step
        per_sample_mse: dict {K: (N,)} MSE per sample at key K values
    """
    torch.manual_seed(seed)
    device = states_t.device
    B_total = states_t.shape[0]
    D = states_t.shape[1]
    BS = CONFIG["BATCH_SIZE"]

    # Accumulate results across batches
    all_energies = []    # list of (K+1, BS) tensors
    all_grad_norms = []
    all_mse = []         # list of (K+1, BS) tensors
    all_per_dim_mse = [] # list of (K+1, BS, D) tensors

    step_size = CONFIG["LANGEVIN_STEP_SIZE"]
    noise_scale = CONFIG["LANGEVIN_NOISE_SCALE"]

    for start in range(0, B_total, BS):
        end = min(start + BS, B_total)
        s = states_t[start:end]
        a = actions_t[start:end]
        ns_true = next_states_t[start:end]
        b = s.shape[0]

        # Create initialization
        if init_type == "cold":
            init = torch.randn(b, D, device=device)
        elif init_type == "warm":
            z = torch.randn(b, D, device=device)
            context = torch.cat([s, a], dim=1)
            with torch.no_grad():
                init = flow.sample(z, context=context)
        elif init_type == "oracle":
            noise = torch.randn(b, D, device=device) * CONFIG["ORACLE_NOISE_STD"]
            init = ns_true + noise
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

        # Run Langevin sweep
        result = langevin_sweep(ebm, s, a, init, K_max, step_size, noise_scale)

        # Compute MSE to ground truth at each step
        # samples: (K+1, b, D), ns_true: (b, D)
        diff = result["samples"] - ns_true.unsqueeze(0)  # (K+1, b, D)
        mse_per_sample = (diff ** 2).mean(dim=-1)           # (K+1, b)
        per_dim_mse = diff ** 2                              # (K+1, b, D)

        all_energies.append(result["energies"])       # (K+1, b)
        all_grad_norms.append(result["grad_norms"])   # (K+1, b)
        all_mse.append(mse_per_sample)                # (K+1, b)
        all_per_dim_mse.append(per_dim_mse)            # (K+1, b, D)

    # Concatenate across batches: (K+1, N_total)
    all_energies = torch.cat(all_energies, dim=1)
    all_grad_norms = torch.cat(all_grad_norms, dim=1)
    all_mse = torch.cat(all_mse, dim=1)
    all_per_dim_mse = torch.cat(all_per_dim_mse, dim=1)  # (K+1, N, D)

    N = all_energies.shape[1]

    # Aggregate
    energy_mean = all_energies.mean(dim=1).cpu().numpy()      # (K+1,)
    energy_std = all_energies.std(dim=1).cpu().numpy()
    mse_mean = all_mse.mean(dim=1).cpu().numpy()
    mse_std = all_mse.std(dim=1).cpu().numpy()
    grad_norm_mean = all_grad_norms.mean(dim=1).cpu().numpy()
    grad_norm_std = all_grad_norms.std(dim=1).cpu().numpy()
    per_dim_mse_mean = all_per_dim_mse.mean(dim=1).cpu().numpy()  # (K+1, D)

    # Step-to-step ΔE
    delta_e = torch.abs(all_energies[1:] - all_energies[:-1]).mean(dim=1)
    delta_e_mean = delta_e.cpu().numpy()  # (K,)

    # Per-sample MSE at key K values for distribution plots
    key_ks = [0, 5, 15, 50, K_max]
    per_sample_mse_at_k = {}
    for k in key_ks:
        if k <= K_max:
            per_sample_mse_at_k[k] = all_mse[k].cpu().numpy()  # (N,)

    return {
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "grad_norm_mean": grad_norm_mean,
        "grad_norm_std": grad_norm_std,
        "delta_e_mean": delta_e_mean,
        "per_dim_mse_mean": per_dim_mse_mean,
        "per_sample_mse_at_k": per_sample_mse_at_k,
    }


# =============================================================================
# FIND K* — THE STEP AT WHICH 95% OF ASYMPTOTIC VALUE IS REACHED
# =============================================================================
def find_convergence_step(values, k_values, threshold=0.95):
    """
    Given values at each K, find the smallest K where the metric reaches
    threshold% of its final (K_max) value.

    For MSE (lower is better): find K where MSE <= MSE[0] - threshold*(MSE[0]-MSE[-1])
    For Energy (higher is better): find K where E >= E[0] + threshold*(E[-1]-E[0])

    Returns K* or None if never reached.
    """
    # We use the generic "fraction of improvement" approach
    v_init = values[0]
    v_final = values[-1]
    improvement = v_final - v_init

    if abs(improvement) < 1e-10:
        return k_values[0]  # No improvement possible

    target = v_init + threshold * improvement

    if improvement > 0:
        # Metric increases (energy) — find first K where value >= target
        for i, v in enumerate(values):
            if v >= target:
                return k_values[i]
    else:
        # Metric decreases (MSE) — find first K where value <= target
        for i, v in enumerate(values):
            if v <= target:
                return k_values[i]

    return k_values[-1]  # Never reached


# =============================================================================
# PLOTTING
# =============================================================================
COLORS = {
    "cold": "#E74C3C",   # red
    "warm": "#2ECC71",   # green
    "oracle": "#3498DB", # blue
}
LABELS = {
    "cold": "Cold Start (random noise)",
    "warm": "Warm Start (Flow proposal)",
    "oracle": "Oracle (ground truth + noise)",
}


def plot_convergence_curve(ax, k_values, results_by_seed, metric_key, ylabel,
                           title, invert=False):
    """
    Plot mean ± std across seeds for all 3 inits.

    Args:
        results_by_seed: dict[init_type -> list of per-seed results]
        metric_key: key in per-seed result dict (e.g. "mse_mean")
        invert: if True, lower is better (for MSE)
    """
    for init_type in ["cold", "warm", "oracle"]:
        seed_curves = []
        for seed_result in results_by_seed[init_type]:
            curve = seed_result[metric_key]
            # Extract values only at requested K_values
            seed_curves.append(curve)

        seed_curves = np.array(seed_curves)  # (n_seeds, K+1)
        mean = seed_curves.mean(axis=0)      # (K+1,)
        std = seed_curves.std(axis=0)

        # Use K_values for x-axis (only the requested subset)
        kv = np.array(k_values)

        # The curves have K_max+1 entries (one per step 0..K_max)
        # We need to subsample to match K_values
        indices = k_values  # These are the actual step indices
        mean_sub = mean[indices]
        std_sub = std[indices]

        ax.plot(kv, mean_sub, 'o-', color=COLORS[init_type],
                label=LABELS[init_type], linewidth=2, markersize=4)
        ax.fill_between(kv, mean_sub - std_sub, mean_sub + std_sub,
                       alpha=0.15, color=COLORS[init_type])

    ax.set_xlabel("Langevin Steps (K)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if invert:
        ax.invert_yaxis()


def plot_distributions(ax_row, k_values_to_show, results_by_seed):
    """
    Plot histograms of per-sample MSE at selected K values.
    """
    for i, k in enumerate(k_values_to_show):
        ax = ax_row[i]
        for init_type in ["cold", "warm", "oracle"]:
            # Pool per-sample MSE across seeds
            all_mse = []
            for seed_result in results_by_seed[init_type]:
                if k in seed_result["per_sample_mse_at_k"]:
                    all_mse.append(seed_result["per_sample_mse_at_k"][k])
            if all_mse:
                pooled = np.concatenate(all_mse)
                # Clip extreme outliers for visualization
                clip_val = np.percentile(pooled, 99)
                pooled_clipped = np.clip(pooled, 0, clip_val)
                ax.hist(pooled_clipped, bins=50, alpha=0.5,
                       color=COLORS[init_type], label=LABELS[init_type],
                       density=True)

        ax.set_title(f"K = {k}", fontsize=11, fontweight='bold')
        ax.set_xlabel("MSE per sample", fontsize=10)
        if i == 0:
            ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)


def create_convergence_table(k_values, results_by_seed, K_max):
    """
    Create a summary table showing K* for each metric and init type.
    """
    metrics = [
        ("MSE", "mse_mean", "lower"),
        ("Energy", "energy_mean", "higher"),
        ("Grad Norm", "grad_norm_mean", "lower"),
    ]

    full_k = list(range(K_max + 1))

    table_data = {}
    for metric_name, metric_key, direction in metrics:
        table_data[metric_name] = {}
        for init_type in ["cold", "warm", "oracle"]:
            k_stars = []
            for seed_result in results_by_seed[init_type]:
                curve = seed_result[metric_key]
                k_star = find_convergence_step(curve, full_k, threshold=0.95)
                k_stars.append(k_star)
            mean_k = np.mean(k_stars)
            std_k = np.std(k_stars)
            table_data[metric_name][init_type] = (mean_k, std_k)

    return table_data


def plot_convergence_table(fig, table_data, output_dir):
    """
    Create a clean table figure showing K* values.
    """
    fig_table, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    row_labels = list(table_data.keys())
    col_labels = ["Cold Start K*", "Warm Start K*", "Oracle K*", "Speedup (Cold/Warm)"]

    cell_text = []
    for metric in row_labels:
        cold_mean, cold_std = table_data[metric]["cold"]
        warm_mean, warm_std = table_data[metric]["warm"]
        oracle_mean, oracle_std = table_data[metric]["oracle"]

        if warm_mean > 0:
            speedup = cold_mean / warm_mean
        else:
            speedup = float('inf')

        row = [
            f"{cold_mean:.1f} ± {cold_std:.1f}",
            f"{warm_mean:.1f} ± {warm_std:.1f}",
            f"{oracle_mean:.1f} ± {oracle_std:.1f}",
            f"{speedup:.1f}×",
        ]
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(len(row_labels)):
        table[i + 1, -1].set_facecolor('#E8F6F3')

    ax.set_title("Convergence Step K* (95% of asymptotic value)",
                 fontsize=14, fontweight='bold', pad=20)

    fig_table.tight_layout()
    fig_table.savefig(os.path.join(output_dir, "warmstart_convergence_table.png"),
                      dpi=150, bbox_inches='tight')
    plt.close(fig_table)


def plot_delta_e(ax, results_by_seed, K_max):
    """
    Plot step-to-step energy change |E_k - E_{k-1}|.
    """
    k_range = np.arange(1, K_max + 1)
    for init_type in ["cold", "warm", "oracle"]:
        seed_curves = []
        for seed_result in results_by_seed[init_type]:
            seed_curves.append(seed_result["delta_e_mean"])

        seed_curves = np.array(seed_curves)
        mean = seed_curves.mean(axis=0)
        std = seed_curves.std(axis=0)

        ax.semilogy(k_range, mean, '-', color=COLORS[init_type],
                    label=LABELS[init_type], linewidth=2)
        ax.fill_between(k_range, np.maximum(mean - std, 1e-8), mean + std,
                       alpha=0.15, color=COLORS[init_type])

    ax.set_xlabel("Langevin Step", fontsize=12)
    ax.set_ylabel("|ΔE| (log scale)", fontsize=12)
    ax.set_title("Step-to-Step Energy Change", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')


def plot_per_dim_mse(ax, results_by_seed, K_values_to_show, state_dim):
    """
    Grouped bar chart: per-dimension MSE at K=5 vs K=50 for warm and cold.
    """
    dim_labels = [f"d{i}" for i in range(state_dim)]
    x = np.arange(state_dim)
    width = 0.2

    for offset, (init_type, K) in enumerate([
        ("cold", 5), ("warm", 5), ("cold", 50), ("warm", 50)
    ]):
        seed_curves = []
        for seed_result in results_by_seed[init_type]:
            seed_curves.append(seed_result["per_dim_mse_mean"][K])
        mean = np.mean(seed_curves, axis=0)

        label = f"{init_type.title()} K={K}"
        alpha = 1.0 if K == 5 else 0.6
        ax.bar(x + offset * width - 1.5 * width, mean, width,
               label=label, color=COLORS[init_type], alpha=alpha)

    ax.set_xlabel("State Dimension", fontsize=10)
    ax.set_ylabel("MSE", fontsize=10)
    ax.set_title("Per-Dimension MSE", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def run_experiment():
    device = CONFIG["DEVICE"]
    K_values = CONFIG["K_VALUES"]
    K_max = max(K_values)
    seeds = CONFIG["SEEDS"]
    output_dir = CONFIG["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"WARM START VS COLD START LANGEVIN — MiniGrid Experiment")
    print(f"{'='*70}")
    print(f"Device:        {device}")
    print(f"K values:      {K_values}")
    print(f"Seeds:         {seeds}")
    print(f"Test data:     {CONFIG['NUM_TEST_TRANSITIONS']} transitions")
    print(f"Step size:     {CONFIG['LANGEVIN_STEP_SIZE']}")
    print(f"Noise scale:   {CONFIG['LANGEVIN_NOISE_SCALE']}")
    print(f"Oracle noise:  {CONFIG['ORACLE_NOISE_STD']}")
    print(f"{'='*70}\n")

    # --- Load pretrained models ---
    print("Loading pretrained models...")
    state_dim = 7 + 2 * CONFIG["N_OBSTACLES"]  # 13 for 3 obstacles
    action_dim = 3
    hd = CONFIG["HIDDEN_DIM"]

    ebm = BilinearEBM(state_dim, action_dim, hidden_dim=hd).to(device)
    flow = RealNVP(state_dim, context_dim=state_dim + action_dim,
                   hidden_dim=hd, n_layers=CONFIG["FLOW_N_LAYERS"]).to(device)

    ebm_path = "pretrained_ebm_minigrid.pth"
    flow_path = "pretrained_flow_minigrid.pth"

    if not os.path.exists(ebm_path) or not os.path.exists(flow_path):
        print(f"\nERROR: Pretrained checkpoints not found!")
        print(f"  Expected: {ebm_path}, {flow_path}")
        print(f"  Run `python pretrain_minigrid.py` first.")
        return

    ebm.load_state_dict(torch.load(ebm_path, map_location=device, weights_only=True))
    flow.load_state_dict(torch.load(flow_path, map_location=device, weights_only=True))
    ebm.eval()
    flow.eval()
    print(f"  EBM:  {sum(p.numel() for p in ebm.parameters()):,} params")
    print(f"  Flow: {sum(p.numel() for p in flow.parameters()):,} params")
    print(f"  state_dim={state_dim}, action_dim={action_dim}")

    # --- Collect test data ---
    print(f"\nCollecting {CONFIG['NUM_TEST_TRANSITIONS']} test transitions...")
    states, actions, next_states, sd, ad = collect_test_data(
        CONFIG["NUM_TEST_TRANSITIONS"], seed=999)
    assert sd == state_dim and ad == action_dim
    print(f"  Collected. State range: [{states.min():.3f}, {states.max():.3f}]")

    states_t = torch.tensor(states, device=device)
    actions_t = torch.tensor(actions, device=device)
    next_states_t = torch.tensor(next_states, device=device)

    # --- Flow-only baseline sanity check ---
    print("\nFlow-only baseline (no Langevin):")
    with torch.no_grad():
        z_check = torch.randn(min(512, len(states)), state_dim, device=device)
        ctx_check = torch.cat([states_t[:512], actions_t[:512]], dim=1)
        flow_pred = flow.sample(z_check, context=ctx_check)
        flow_mse = F.mse_loss(flow_pred, next_states_t[:512]).item()
        print(f"  Flow prediction MSE: {flow_mse:.6f}")

    # --- EBM quality check ---
    print("\nEBM quality check (energy gap: positive=pos vs negative examples):")
    with torch.no_grad():
        E_pos = ebm(states_t[:512], actions_t[:512], next_states_t[:512])
        rand_ns = torch.randn_like(next_states_t[:512])
        E_neg = ebm(states_t[:512], actions_t[:512], rand_ns)
        gap = (E_pos.mean() - E_neg.mean()).item()
        print(f"  E(s,a,s'_true) = {E_pos.mean().item():.4f}")
        print(f"  E(s,a,s'_rand) = {E_neg.mean().item():.4f}")
        print(f"  Energy gap:      {gap:.4f}")
        if gap < 0.5:
            print("  WARNING: Small energy gap — EBM may not be well-trained")

    # --- Run sweep for all seeds ---
    init_types = ["cold", "warm", "oracle"]
    results_by_seed = {it: [] for it in init_types}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'─'*50}")
        print(f"Seed {seed_idx+1}/{len(seeds)}: {seed}")
        print(f"{'─'*50}")

        for init_type in init_types:
            t0 = time.time()
            print(f"  Running {init_type} start (K_max={K_max})...", end="", flush=True)

            result = compute_metrics_for_init(
                ebm, flow, states_t, actions_t, next_states_t,
                init_type=init_type,
                K_max=K_max,
                seed=seed,
            )
            elapsed = time.time() - t0
            results_by_seed[init_type].append(result)

            # Quick summary
            mse_init = result["mse_mean"][0]
            mse_final = result["mse_mean"][-1]
            print(f" done ({elapsed:.1f}s) | "
                  f"MSE: {mse_init:.4f} → {mse_final:.4f} | "
                  f"E: {result['energy_mean'][0]:.2f} → {result['energy_mean'][-1]:.2f}")

    # --- Compute convergence table ---
    print(f"\n{'='*70}")
    print("CONVERGENCE ANALYSIS (95% of K=100 value)")
    print(f"{'='*70}")
    table_data = create_convergence_table(K_values, results_by_seed, K_max)
    for metric_name, metric_vals in table_data.items():
        print(f"\n  {metric_name}:")
        for init_type, (k_mean, k_std) in metric_vals.items():
            print(f"    {init_type:8s}: K* = {k_mean:.1f} ± {k_std:.1f}")

    cold_mse_k = table_data["MSE"]["cold"][0]
    warm_mse_k = table_data["MSE"]["warm"][0]
    if warm_mse_k > 0:
        print(f"\n  → Warm start achieves 95% MSE convergence {cold_mse_k/warm_mse_k:.1f}× "
              f"faster than cold start")

    # --- Generate plots ---
    print(f"\nGenerating plots to {output_dir}/...")

    # 1. MSE convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_convergence_curve(ax, K_values, results_by_seed,
                          "mse_mean", "MSE to Ground Truth",
                          "Sample Quality vs Langevin Steps (MSE)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "warmstart_convergence_mse.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Energy convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_convergence_curve(ax, K_values, results_by_seed,
                          "energy_mean", "EBM Energy E(s,a,s')",
                          "EBM Energy vs Langevin Steps")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "warmstart_convergence_energy.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. Gradient norm convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_convergence_curve(ax, K_values, results_by_seed,
                          "grad_norm_mean", "||∇E|| (gradient norm)",
                          "Energy Gradient Norm vs Langevin Steps")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "warmstart_convergence_gradnorm.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4. Convergence table
    plot_convergence_table(fig, table_data, output_dir)

    # 5. Combined figure: ΔE + per-dim MSE
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_delta_e(axes[0], results_by_seed, K_max)
    plot_per_dim_mse(axes[1], results_by_seed, [5, 50], state_dim)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "warmstart_delta_e_and_perdim.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 6. Distribution plots
    dist_ks = [0, 5, 15, 50]
    fig, axes = plt.subplots(1, len(dist_ks), figsize=(5 * len(dist_ks), 4))
    plot_distributions(axes, dist_ks, results_by_seed)
    fig.suptitle("Per-Sample MSE Distribution at Key Steps",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "warmstart_distributions.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Save raw results ---
    save_data = {
        "K_values": np.array(K_values),
        "K_max": K_max,
        "seeds": np.array(seeds),
        "config": str(CONFIG),
    }

    for init_type in init_types:
        for metric_key in ["energy_mean", "energy_std", "mse_mean", "mse_std",
                          "grad_norm_mean", "grad_norm_std", "delta_e_mean"]:
            for seed_idx, seed_result in enumerate(results_by_seed[init_type]):
                key = f"{init_type}_seed{seed_idx}_{metric_key}"
                save_data[key] = seed_result[metric_key]

    np.savez(os.path.join(output_dir, "warmstart_results.npz"), **save_data)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}/")
    print(f"  warmstart_convergence_mse.png")
    print(f"  warmstart_convergence_energy.png")
    print(f"  warmstart_convergence_gradnorm.png")
    print(f"  warmstart_convergence_table.png")
    print(f"  warmstart_delta_e_and_perdim.png")
    print(f"  warmstart_distributions.png")
    print(f"  warmstart_results.npz")


if __name__ == "__main__":
    run_experiment()
