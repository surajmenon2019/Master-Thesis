import torch
import torch.nn.functional as F

"""
Sampling utilities for EBM-based world models.

Three sampling strategies:
1. importance_weighted_sample — PRIMARY METHOD for BilinearEBM.
   Draw N candidates from a proposal (Flow/MDN/noise), score with EBM,
   resample via differentiable softmax weighting. No Langevin needed.

2. langevin_sample — Langevin dynamics with proper truncated backprop.
   Fixed: the `step_diff` variable was computed but never used in original.
   Now only the final step creates graph, all others detach properly.

3. langevin_refine — Flow init + short Langevin refinement (warm start).
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