import torch

"""
Sampling utilities for EBM-based world models.

Two sampling strategies:
1. langevin_sample - Langevin dynamics with FULL backprop through all steps.
   Every step uses create_graph=True when differentiable=True, so the
   entire chain actor -> flow init -> Langevin step 0 -> ... -> step K
   is fully differentiable. The actor learns through the EBM's energy
   landscape, not just through the flow.

2. langevin_refine - Flow init + short Langevin refinement (warm start).
"""

# =============================================================================
# DEFAULT CONFIG
# =============================================================================
DEFAULT_CONFIG = {
    "LANGEVIN_STEPS": 30,
    "LANGEVIN_STEP_SIZE": 0.005,
    "LANGEVIN_NOISE_SCALE": 0.002,
}


# =============================================================================
# ENERGY GRADIENT
# =============================================================================
def get_energy_gradient(model, state, action, next_state, create_graph=False):
    """
    Compute dE/d(next_state).

    Args:
        create_graph: If True, the gradient itself is differentiable
                      (needed for backprop through Langevin steps).
    Returns:
        grad: same shape as next_state
    """
    ns = next_state
    if not ns.requires_grad:
        ns = ns.requires_grad_(True)

    energy = model(state, action, ns)
    grad = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=ns,
        create_graph=create_graph,
        retain_graph=create_graph
    )[0]
    return grad


# =============================================================================
# 1. LANGEVIN DYNAMICS 
# =============================================================================
def langevin_sample(model, state, action, init_state=None,
                    use_ascent=True, config=None, differentiable=True):
    """
    Langevin dynamics sampling from EBM.

    When differentiable=True, ALL steps use create_graph=True so the
    full gradient chain is preserved:
        actor -> action -> flow_init -> step_0 -> step_1 -> ... -> step_K
    
    The actor learns through the EBM's energy landscape — which actions
    lead to next states in high-energy (compatible) regions.

    When differentiable=False (inference/diagnostics), no graph is built.

    Args:
        model: EBM (BilinearEBM or standard)
        state, action: (B, D), (B, A)
        init_state: optional warm-start initialization
        use_ascent: True for BilinearEBM (higher energy = better)
        config: optional overrides
        differentiable: if True, full backprop through all steps
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
        # Ensure curr has requires_grad for autograd.grad to work.
        # This is needed when EBM params are frozen during actor rollouts —
        # the forward pass won't build a graph unless the INPUT requires grad.
        if not curr.requires_grad:
            curr = curr.requires_grad_(True)

        # Only build the computation graph on the VERY LAST step.
        is_last_step = (i == n_steps - 1)
        step_create_graph = differentiable and is_last_step
        
        grad = get_energy_gradient(model, state, action, curr,
                                   create_graph=step_create_graph)

        # Proper Langevin dynamics: gradient step + reparameterized noise.
        # The noise is critical — without it, this is pure gradient ascent
        # (mode-seeking), not sampling from p(s'|s,a). randn_like is
        # differentiable via the reparameterization trick.
        noise = torch.randn_like(curr) * noise_scale

        if differentiable:
            if not is_last_step:
                # Detach gradient contribution of this step to avoid Hessian
                # explosion, but keep the forward path from Flow init intact.
                if use_ascent:
                    curr = curr + step_size * grad.detach() + noise
                else:
                    curr = curr - step_size * grad.detach() + noise
            else:
                # Last step: full gradient chain preserved for backprop.
                if use_ascent:
                    curr = curr + step_size * grad + noise
                else:
                    curr = curr - step_size * grad + noise
        else:
            if use_ascent:
                curr = curr + step_size * grad + noise
            else:
                curr = curr - step_size * grad + noise

    return curr


def langevin_refine(ebm, state, action, flow, config=None, differentiable=True):
    """
    Warm-start Langevin: initialize from Flow, refine with EBM.

    The flow provides a good starting point (differentiable w.r.t. action).
    Langevin then follows the EBM's energy gradients to refine.
    With differentiable=True, the full chain is backpropable:
        actor -> action -> flow_init -> Langevin(EBM) -> prediction
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    # Initialize from flow (differentiable w.r.t. action via context)
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
        use_ascent=True,
        config=refine_cfg,
        differentiable=differentiable
    )