import torch
import torch.autograd as autograd
import numpy as np

# --- CONFIGURATION ---
DEFAULT_CONFIG = {
    # Langevin Settings
    "LANGEVIN_STEPS": 30,
    "LANGEVIN_STEP_SIZE": 0.01,
    "LANGEVIN_NOISE_SCALE": 0.005, 
    "GRAD_CLIP": 1.0,
    
    # SVGD Settings
    "SVGD_STEPS": 10,
    "SVGD_STEP_SIZE": 0.05,
}

def get_energy_gradient(model, state, action, next_state_candidate):
    """
    Calculates gradient of Energy w.r.t next_state_candidate.
    Handles reshaping 3D inputs (Batch, Particles, Dim) -> 2D (Batch*Particles, Dim)
    for the model, and then reshapes gradients back to 3D.
    """
    s_prime = next_state_candidate
    
    # Detect shapes
    is_3d = (s_prime.dim() == 3)
    original_shape = s_prime.shape # (B, P, D) or (B, D)
    
    # 1. Flatten Everything to 2D for the Model
    if is_3d:
        B, P, _ = s_prime.shape # We don't care about D here, we let reshape handle it
        
        # Expand State: (B, D_s) -> (B, P, D_s) -> (B*P, D_s)
        # FIX: Explicitly use state.shape[-1] to avoid dimension mismatch with particles
        state_dim = state.shape[-1]
        state_in = state.unsqueeze(1).expand(B, P, state_dim).reshape(B * P, state_dim)
        
        # Expand Action: (B, D_a) -> (B, P, D_a) -> (B*P, D_a)
        # FIX: Explicitly use action.shape[-1] (e.g., 2) instead of D (e.g., 60)
        action_dim = action.shape[-1]
        action_in = action.unsqueeze(1).expand(B, P, action_dim).reshape(B * P, action_dim)
        
        # Flatten Candidate: (B, P, D_s) -> (B*P, D_s)
        # We simply collapse the first two dimensions
        s_prime_in = s_prime.reshape(B * P, -1)
    else:
        state_in = state
        action_in = action
        s_prime_in = s_prime

    # 2. Calculate Gradient
    with torch.set_grad_enabled(True):
        if not s_prime_in.requires_grad:
            s_prime_in.requires_grad_(True)
            
        # Model always receives 2D tensors: (N, D)
        energy = model(state_in, action_in, s_prime_in).sum()
        
        grads = autograd.grad(
            outputs=energy, 
            inputs=s_prime_in, 
            create_graph=True, 
            only_inputs=True
        )[0]
    
    # 3. Gradient Clipping
    grad_norm = grads.norm(dim=-1, keepdim=True)
    clipped_grads = grads / (grad_norm + 1e-8) * torch.clamp(grad_norm, max=DEFAULT_CONFIG["GRAD_CLIP"])
    
    # 4. Reshape back to Original (if input was 3D)
    if is_3d:
        clipped_grads = clipped_grads.view(original_shape) # (B, P, D)
        
    return clipped_grads

# =============================================================================
# 1. DIFFERENTIABLE LANGEVIN
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
    median_dist = dist_sq.view(B, -1).median(dim=1)[0].view(B, 1, 1)
    h = median_dist / np.log(N + 1)
    h = torch.maximum(h, torch.tensor(h_min).to(x.device))
    
    # Kernel Matrix
    k_xx = torch.exp(-dist_sq / h) # (B, N, N)
    
    # Gradient of Kernel: (B, N, N, D) -> Sum over dim 2 -> (B, N, D)
    grad_k = -k_xx.unsqueeze(-1) * diff * (2 / h.unsqueeze(-1))
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
    
    # 3. Return Mean Particle
    return particles.mean(dim=1)