import torch
import torch.nn.functional as F
import numpy as np

"""
Sampling utilities for EBM-based world models.

Four sampling strategies:
1. importance_weighted_sample — PRIMARY METHOD for BilinearEBM.
   Draw N candidates from a proposal (Flow/MDN/noise), score with EBM,
   resample via differentiable softmax weighting. No Langevin needed.

2. langevin_sample — Langevin dynamics with proper truncated backprop.
   Fixed: the `step_diff` variable was computed but never used in original.
   Now only the final step creates graph, all others detach properly.

3. langevin_refine — Flow init + short Langevin refinement (warm start).

4. svgd_sample — Stein Variational Gradient Descent. Evolves a set of
   interacting particles with gradient attraction + kernel repulsion.
   Mode-covering: repulsive force prevents particle collapse into a
   single mode, giving better coverage of multimodal energy landscapes.
"""

# =============================================================================
# DEFAULT CONFIG
# =============================================================================
DEFAULT_CONFIG = {
    "LANGEVIN_STEPS": 30,          # was 20 — more steps for higher dim energy landscape
    "LANGEVIN_STEP_SIZE": 0.005,   # was 0.01 — smaller steps for stability in 17D
    "LANGEVIN_NOISE_SCALE": 0.002, # was 0.005 — less noise in higher dim
    "IW_NUM_SAMPLES": 64,
    "IW_TEMPERATURE": 1.0,      # softmax temperature for resampling

    # SVGD config
    "SVGD_NUM_PARTICLES": 32,     # number of interacting particles
    "SVGD_STEPS": 20,             # gradient steps
    "SVGD_STEP_SIZE": 0.01,       # step size for particle update
    "SVGD_BANDWIDTH": None,       # None = median heuristic (auto-adapts)
}


# =============================================================================
# ENERGY GRADIENT (simplified, single code path)
# =============================================================================
def get_energy_gradient(model, state, action, next_state, create_graph=False):
    """
    Compute dE/d(next_state).

    Args:
        create_graph: If True, the gradient itself is differentiable
                      (needed for actor backprop through final Langevin step).
    Returns:
        grad: same shape as next_state
    """
    ns = next_state
    if not ns.requires_grad:
        ns = ns.detach().requires_grad_(True)

    energy = model(state, action, ns)
    grad = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=ns,
        create_graph=create_graph,
        retain_graph=create_graph
    )[0]
    return grad


# =============================================================================
# 1. IMPORTANCE-WEIGHTED RESAMPLING (Primary method for EBM)
# =============================================================================
def importance_weighted_sample(ebm, state, action, proposal_fn, config=None):
    """
    Differentiable importance-weighted sampling from EBM.

    Instead of running Langevin on a flat energy surface, we:
    1. Draw N candidate next-states from a proposal distribution
    2. Score each candidate with the EBM
    3. Softmax-weight and combine (differentiable via Gumbel-Softmax)

    This directly tests the thesis question: "does the EBM improve
    over the proposal (Flow/MDN) alone?"

    Args:
        ebm: BilinearEBM model (higher energy = more compatible)
        state: (B, state_dim)
        action: (B, action_dim)
        proposal_fn: callable(B, N) -> (B, N, state_dim) candidate samples.
                     Must be differentiable (reparameterized).
        config: optional config overrides

    Returns:
        predicted_ns: (B, state_dim) — differentiable w.r.t. actor params
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    B = state.shape[0]
    N = cfg["IW_NUM_SAMPLES"]
    temperature = cfg["IW_TEMPERATURE"]

    # 1. Generate N candidate next-states from proposal
    candidates = proposal_fn(B, N)  # (B, N, state_dim)

    # 2. Score each candidate with EBM
    state_exp = state.unsqueeze(1).expand(B, N, -1)    # (B, N, state_dim)
    action_exp = action.unsqueeze(1).expand(B, N, -1)   # (B, N, action_dim)
    energies = ebm(state_exp, action_exp, candidates)    # (B, N)

    # 3. Differentiable soft resampling
    # Gumbel-Softmax gives differentiable categorical-like weights
    weights = F.gumbel_softmax(energies / temperature, tau=1.0, hard=False)  # (B, N)

    # 4. Weighted combination of candidates
    predicted_ns = (weights.unsqueeze(-1) * candidates).sum(dim=1)  # (B, state_dim)

    return predicted_ns


# =============================================================================
# 2. LANGEVIN DYNAMICS (fixed truncated backprop)
# =============================================================================
def langevin_sample(model, state, action, init_state=None,
                    use_ascent=True, config=None, differentiable=True):
    """
    Langevin dynamics sampling from EBM.

    FIXES from original:
    - `step_diff` was computed but NEVER USED — now it controls create_graph.
    - Proper truncated backprop: steps 0..K-2 detach, only step K-1 keeps graph.
    - This bounds gradient magnitude while still allowing actor signal to flow.

    Args:
        model: EBM (BilinearEBM or standard)
        state, action: (B, D), (B, A)
        init_state: optional warm-start initialization
        use_ascent: True for BilinearEBM (higher energy = better)
        config: optional overrides
        differentiable: if True, final step retains computation graph
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    if init_state is not None:
        curr = init_state
    else:
        curr = torch.randn_like(state)

    n_steps = cfg["LANGEVIN_STEPS"]
    step_size = cfg["LANGEVIN_STEP_SIZE"]
    noise_scale = cfg["LANGEVIN_NOISE_SCALE"]

    for i in range(n_steps):
        is_last = (i == n_steps - 1)

        # TRUNCATED BACKPROP: only the last step creates graph
        step_create_graph = differentiable and is_last

        # Ensure curr has requires_grad on ALL steps
        # Detach to prevent exploding gradient chains through all steps
        # Only the final step's create_graph=True allows actor gradient flow
        curr = curr.detach().requires_grad_(True)

        grad = get_energy_gradient(model, state, action, curr,
                                   create_graph=step_create_graph)

        noise = torch.randn_like(curr) * noise_scale

        if use_ascent:
            curr = curr + step_size * grad + noise
        else:
            curr = curr - step_size * grad + noise

    return curr


def langevin_refine(ebm, state, action, flow, config=None, differentiable=True):
    """
    Warm-start Langevin: initialize from Flow, refine with EBM.

    This is the "Warm Start" agent variant. The flow provides a good
    initial guess, and Langevin nudges it toward the EBM's energy peak.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    # Initialize from flow (differentiable)
    B = state.shape[0]
    z = torch.randn(B, state.shape[1], device=state.device)
    context = torch.cat([state, action], dim=1)
    init = flow.sample(z, context=context)

    # Short Langevin refinement (fewer steps since we start close)
    refine_cfg = cfg.copy()
    refine_cfg["LANGEVIN_STEPS"] = max(cfg["LANGEVIN_STEPS"] // 3, 5)

    return langevin_sample(
        ebm, state, action,
        init_state=init,
        use_ascent=True,  # BilinearEBM convention
        config=refine_cfg,
        differentiable=differentiable
    )


# =============================================================================
# 4. STEIN VARIATIONAL GRADIENT DESCENT (SVGD)
# =============================================================================
def rbf_kernel_batched(x, bandwidth=None):
    """
    Batched RBF kernel with median heuristic bandwidth.

    Computes N×N kernel matrices for all B batch elements simultaneously
    using pure tensor ops — no Python loops.

    Args:
        x: (B, N, D) particles
        bandwidth: float or None (auto via median heuristic)

    Returns:
        K:      (B, N, N) kernel matrices
        grad_K: (B, N, N, D) gradient of K w.r.t. first index: ∇_{x_i} K(x_i, x_j)
    """
    # Pairwise differences: (B, N, 1, D) - (B, 1, N, D) → (B, N, N, D)
    diff = x.unsqueeze(2) - x.unsqueeze(1)
    dist_sq = (diff ** 2).sum(dim=-1)  # (B, N, N)

    if bandwidth is None:
        with torch.no_grad():
            # Per-batch median heuristic
            # Mask out zero (self) distances, compute median per batch
            mask = dist_sq > 0                          # (B, N, N)
            # Flatten N×N per batch, take median of positive entries
            flat = dist_sq.reshape(x.shape[0], -1)      # (B, N*N)
            flat_mask = mask.reshape(x.shape[0], -1)
            # Replace zeros with inf so median ignores them
            flat_masked = flat.clone()
            flat_masked[~flat_mask] = float('inf')
            median_sq = flat_masked.median(dim=1).values  # (B,)
            bandwidth = median_sq / max(np.log(x.shape[1] + 1), 1.0)
            bandwidth = bandwidth.clamp(min=1e-5)         # (B,)
            bandwidth = bandwidth.reshape(-1, 1, 1)        # (B, 1, 1)

    K = torch.exp(-dist_sq / (2 * bandwidth))              # (B, N, N)

    # grad_K[b,i,j,:] = ∇_{x_i} K(x_i, x_j) = K(x_i,x_j) * (x_j - x_i) / h
    # Note: diff = x_i - x_j, so we negate to get (x_j - x_i).
    # FIX: original had positive diff, giving the WRONG sign for the
    # kernel gradient. The derivative of exp(-||x_i-x_j||²/2h) w.r.t. x_i
    # is K * (-(x_i-x_j)/h) = K * (x_j-x_i)/h.
    grad_K = -K.unsqueeze(-1) * diff / bandwidth.unsqueeze(-1)  # (B, N, N, D)

    return K, grad_K


def svgd_sample(model, state, action, init_state=None,
                use_ascent=True, config=None, differentiable=True):
    """
    Stein Variational Gradient Descent sampling from EBM.
    Fully batched — no Python loops over the batch dimension.

    The SVGD update for particle x_i is:
      x_i += eps * (1/N) * sum_j [ K(x_j, x_i) * grad_E(x_j) + grad_K(x_j, x_i) ]

    Args:
        model: EBM (BilinearEBM) — higher energy = more compatible
        state: (B, state_dim) — batch of current states
        action: (B, action_dim) — batch of actions
        init_state: optional (B, N, state_dim) or None for random init
        use_ascent: True for BilinearEBM (ascend energy)
        config: optional overrides for SVGD_* keys
        differentiable: if True, final step retains graph for actor backprop

    Returns:
        predicted_ns: (B, state_dim) — mean of converged particles
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    B = state.shape[0]
    D = state.shape[1]
    N = cfg.get("SVGD_NUM_PARTICLES", 10)
    n_steps = cfg.get("SVGD_STEPS", 20)
    step_size = cfg.get("SVGD_STEP_SIZE", 0.01)
    bandwidth = cfg.get("SVGD_BANDWIDTH", None)  # None = median heuristic

    # Initialize particles
    if init_state is not None:
        particles = init_state  # (B, N, D)
    else:
        particles = torch.randn(B, N, D, device=state.device) * 0.5

    # Expand state/action for batch-particle scoring
    state_exp = state.unsqueeze(1).expand(B, N, -1)    # (B, N, state_dim)
    action_exp = action.unsqueeze(1).expand(B, N, -1)   # (B, N, action_dim)

    for step_i in range(n_steps):
        is_last = (step_i == n_steps - 1)
        step_create_graph = differentiable and is_last

        # Detach particles from previous steps (truncated backprop)
        particles = particles.detach().requires_grad_(True)

        # --- Compute energy gradient for each particle ---
        # Flatten to (B*N, D) for the EBM forward pass
        s_flat = state_exp.reshape(B * N, -1)
        a_flat = action_exp.reshape(B * N, -1)
        p_flat = particles.reshape(B * N, -1)

        # Ensure p_flat tracks gradients
        if not p_flat.requires_grad:
            p_flat = p_flat.detach().requires_grad_(True)

        energy = model(s_flat, a_flat, p_flat)  # (B*N,)
        grad_energy = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=p_flat,
            create_graph=step_create_graph,
            retain_graph=step_create_graph
        )[0]  # (B*N, D)

        grad_energy = grad_energy.reshape(B, N, D)

        if not use_ascent:
            grad_energy = -grad_energy

        # --- Batched SVGD update (all B elements in parallel) ---
        K, grad_K = rbf_kernel_batched(particles, bandwidth=bandwidth)
        # K: (B, N, N), grad_K: (B, N, N, D)

        # Term 1: kernel-weighted gradient — attraction
        # K @ grad_energy: (B, N, N) @ (B, N, D) → (B, N, D)
        attract = torch.bmm(K, grad_energy)  # (B, N, D)

        # Term 2: kernel gradient — repulsion (sum over j)
        repulse = grad_K.sum(dim=1)  # (B, N, D)

        phi = (attract + repulse) / N
        particles = particles + step_size * phi

    # Return mean particle as prediction (differentiable through final step)
    predicted_ns = particles.mean(dim=1)  # (B, D)
    return predicted_ns