import torch

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    "LANGEVIN_STEPS": 30,
    "LANGEVIN_STEP_SIZE": 0.05,  # Moderate step size (was 0.5, exploded)
    "LANGEVIN_NOISE_SCALE": 0.01,
    "SVGD_STEPS": 10,
    "SVGD_STEP_SIZE": 0.05,
    "SVGD_PARTICLES": 10,
}

# =============================================================================
# ENERGY GRADIENT COMPUTATION
# =============================================================================
def get_energy_gradient(model, state, action, next_state):
    """
    Compute ∂E(s,a,s')/∂s' for any dimensionality.
    
    Handles both:
    - next_state: (B, D) → returns (B, D)
    - next_state: (B, P, D) → returns (B, P, D) for SVGD particles
    """
    original_shape = next_state.shape
    is_particles = len(original_shape) == 3
    
    if is_particles:
        # SVGD case: (B, P, D)
        B, P, D = original_shape
        # Flatten to (B*P, D)
        next_state_flat = next_state.reshape(B * P, D).requires_grad_(True)
        # Repeat state/action P times
        state_exp = state.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)
        action_exp = action.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)
        
        # Compute energies: (B*P,)
        energies = model(state_exp, action_exp, next_state_flat)
        
        # Gradient with create_graph=True for planning gradients
        grad = torch.autograd.grad(
            outputs=energies, inputs=next_state_flat,
            grad_outputs=torch.ones_like(energies),
            create_graph=True  # Enable for planning gradients
        )[0]
        
        # Reshape back to (B, P, D)
        return grad.reshape(B, P, D)
    else:
        # Langevin case: (B, D)
        # Enable grad to ensure we can take gradients w.r.t input even in no_grad mode
        with torch.enable_grad():
            next_state = next_state.detach().requires_grad_(True)
            energies = model(state, action, next_state)
            
            # Gradient with create_graph=True for planning gradients
            grad = torch.autograd.grad(
                outputs=energies, inputs=next_state,
                grad_outputs=torch.ones_like(energies),
                create_graph=True  # Enable for planning gradients
            )[0]
        
        return grad

# =============================================================================
# 1. LANGEVIN DYNAMICS
# =============================================================================
def predict_next_state_langevin(model, state, action, init_state=None, config=None):
    cfg = DEFAULT_CONFIG.copy()
    if config: cfg.update(config)
    
    if init_state is None:
        curr_state = torch.randn_like(state) 
    else:
        curr_state = init_state 

    for i in range(cfg["LANGEVIN_STEPS"]):
        noise = torch.randn_like(curr_state) * cfg["LANGEVIN_NOISE_SCALE"]
        grad_energy = get_energy_gradient(model, state, action, curr_state)
        curr_state = curr_state - cfg["LANGEVIN_STEP_SIZE"] * grad_energy + noise
    
    return curr_state

def predict_next_state_langevin_adaptive(model, state, action, init_state=None, use_ascent=False, config=None):
    """
    Adaptive Langevin dynamics for BilinearEBM and standard EBM.
    
    Args:
        model: Energy model (BilinearEBM or EnergyBasedModel)
        state: (B, state_dim)
        action: (B, action_dim)
        init_state: Optional initialization
        use_ascent: If True, use gradient ASCENT (for BilinearEBM: higher energy = better)
                    If False, use gradient DESCENT (for standard EBM: lower energy = better)
        config: Optional config dict
    
    Returns:
        predicted_state: (B, state_dim)
    """
    cfg = DEFAULT_CONFIG.copy()
    if config: cfg.update(config)
    
    if init_state is None:
        curr_state = torch.randn_like(state)
    else:
        curr_state = init_state
    
    for i in range(cfg["LANGEVIN_STEPS"]):
        noise = torch.randn_like(curr_state) * cfg["LANGEVIN_NOISE_SCALE"]
        grad_energy = get_energy_gradient(model, state, action, curr_state)
        
        if use_ascent:
            # BilinearEBM: higher energy = more compatible → gradient ASCENT
            curr_state = curr_state + cfg["LANGEVIN_STEP_SIZE"] * grad_energy + noise
        else:
            # Standard EBM: lower energy = higher probability → gradient DESCENT
            curr_state = curr_state - cfg["LANGEVIN_STEP_SIZE"] * grad_energy + noise
            
        # CLIP state to reasonable bounds to prevent divergence
        curr_state = torch.clamp(curr_state, -10.0, 10.0)
    
    return curr_state

# =============================================================================
# 2. DIFFERENTIABLE SVGD (Batch-Corrected)
# =============================================================================
def rbf_kernel_matrix_batched(x, h_min=1e-3):
    """
    Computes RBF kernel for a Batch of Particle sets.
    Input: (Batch, Particles, Dim)
    """
    B, N, D = x.shape
    
    # Pairwise differences: (B, N, 1, D) - (B, 1, N, D)
    diff = x.unsqueeze(2) - x.unsqueeze(1) 
    dist_sq = diff.pow(2).sum(dim=-1) # (B, N, N)
    
    # Median Heuristic
    h = torch.median(dist_sq.view(B, -1), dim=1, keepdim=True)[0]
    h = torch.clamp(h, min=h_min).unsqueeze(1)
    
    # Kernel
    k_xx = torch.exp(-dist_sq / h) # (B, N, N)
    
    # Gradient of kernel
    grad_k = -2.0 * k_xx.unsqueeze(-1) * diff / h.unsqueeze(-1)
    grad_k = grad_k.sum(dim=2)
    
    return k_xx, grad_k

def predict_next_state_svgd(model, state, action, config=None):
    cfg = DEFAULT_CONFIG.copy()
    if config: cfg.update(config)
    
    num_particles = 10
    B, D = state.shape
    
    # 1. Initialize Particles (B, P, D)
    # We broadcast state/action internally in get_energy_gradient
    particles = torch.randn(B, num_particles, D).to(state.device)
    
    # 2. SVGD Loop
    for i in range(cfg["SVGD_STEPS"]):
        # A. Score Function: (B, P, D)
        # get_energy_gradient handles the flattening and returns (B, P, D)
        grad_energy = get_energy_gradient(model, state, action, particles)
        score_func = -grad_energy 
        
        # B. Kernel: (B, P, P) and (B, P, D)
        k_xx, grad_k = rbf_kernel_matrix_batched(particles) 
        
        # C. Update Direction
        # Matmul: (B, P, P) @ (B, P, D) -> (B, P, D)
        term1 = torch.matmul(k_xx, score_func)
        phi = (term1 + grad_k) / num_particles
        
        # D. Step
        particles = particles + cfg["SVGD_STEP_SIZE"] * phi
    
    random_idx = torch.randint(0, num_particles, (B,), device=state.device)
    selected_particle = particles[torch.arange(B), random_idx]
    
    return selected_particle

def predict_next_state_svgd_adaptive(model, state, action, use_ascent=False, config=None):
    """
    Adaptive SVGD for BilinearEBM and standard EBM.
    
    Args:
        model: Energy model
        state: (B, state_dim)
        action: (B, action_dim)
        use_ascent: If True, maximize energy (BilinearEBM: higher = better)
                    If False, minimize energy (standard EBM: lower = better)
        config: Optional config dict
    
    Returns:
        predicted_state: (B, state_dim)
    """
    cfg = DEFAULT_CONFIG.copy()
    if config: cfg.update(config)
    
    B, D = state.shape
    num_particles = cfg.get("SVGD_PARTICLES", 10)
    
    # Initialize particles
    particles = torch.randn(B, num_particles, D).to(state.device)
    
    for step in range(cfg.get("SVGD_STEPS", 10)):
        # Compute energy gradients using existing function
        grad_energy = get_energy_gradient(model, state, action, particles)  # (B, P, D)
        
        # Kernel and gradients
        k_xx, grad_k = rbf_kernel_matrix_batched(particles)  # (B, P, P), (B, P, D)
        
        if use_ascent:
            # BilinearEBM: maximize energy (gradient ASCENT)
            # Move particles toward regions of HIGHER energy
            score_func = grad_energy  # Positive gradient = increase energy
        else:
            # Standard EBM: minimize energy (gradient DESCENT)
            # Move particles toward regions of LOWER energy
            score_func = -grad_energy  # Negative gradient = decrease energy
        
        # SVGD update
        term1 = torch.matmul(k_xx, score_func)  # (B, P, D)
        phi = (term1 + grad_k) / num_particles
        
        # Step
        particles = particles + cfg.get("SVGD_STEP_SIZE", 0.01) * phi
    
    # Return random particle from ensemble
    random_idx = torch.randint(0, num_particles, (B,), device=state.device)
    selected_particle = particles[torch.arange(B), random_idx]
    
    return selected_particle