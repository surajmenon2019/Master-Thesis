import torch
import torch.autograd as autograd
import numpy as np

# --- CONFIGURATION ---
# "Professor's Note": These hyperparameters are critical for the stability of 
# unrolled differentiation. If step_size is too high, gradients explode.
DEFAULT_CONFIG = {
    # Langevin Settings
    "LANGEVIN_STEPS": 30,
    "LANGEVIN_STEP_SIZE": 0.01,   # Decreased for stability
    "LANGEVIN_NOISE_SCALE": 0.005, 
    "GRAD_CLIP": 1.0,             # Critical for backprop through time
    
    # SVGD Settings
    "SVGD_STEPS": 20,             # SVGD converges faster but is more expensive
    "SVGD_STEP_SIZE": 0.01,
}

def get_energy_gradient(model, state, action, next_state_candidate):
    """
    Calculates \nabla_{s'} E(s, a, s').
    
    CRITICAL IMPLEMENTATION DETAILS:
    1. enable_grad(): Ensures we can calculate the gradient of Energy w.r.t Candidate.
    2. create_graph=True: The 'magic' switch. It stores the derivative of the gradient,
       allowing backprop to flow from 'next_state_candidate' back to 'action' 
       through the optimization path.
    """
    # We create a reference that requires grad for the inner optimization loop
    # We do NOT detach. If next_state_candidate comes from a Flow, it has history.
    s_prime = next_state_candidate

    with torch.set_grad_enabled(True):
        # We must ensure s_prime tracks gradients for the purpose of THIS calculation
        # even if the global context is no_grad (though it shouldn't be).
        if not s_prime.requires_grad:
            s_prime.requires_grad_(True)
            
        energy = model(state, action, s_prime).sum()
        
        # dE/ds'
        grads = autograd.grad(
            outputs=energy, 
            inputs=s_prime, 
            create_graph=True,  # REQUIRED for higher-order derivatives (differentiation through sampling)
            only_inputs=True
        )[0]
    
    # Differentiable Gradient Clipping
    # We normalize using standard torch ops to maintain the graph
    grad_norm = grads.norm(dim=-1, keepdim=True)
    
    # "Professor's Note": We add 1e-8 to avoid division by zero. 
    # We maintain the direction but cap the magnitude.
    clipped_grads = grads / (grad_norm + 1e-8) * torch.clamp(grad_norm, max=DEFAULT_CONFIG["GRAD_CLIP"])
    
    return clipped_grads

# =============================================================================
# 1. DIFFERENTIABLE LANGEVIN DYNAMICS (MCMC)
# =============================================================================
def predict_next_state_langevin(model, state, action, init_state=None, config=None):
    """
    Samples s' ~ exp(-E(s,a,s')) using Unrolled Langevin Dynamics.
    
    Args:
        state: Current state s_t
        action: Action a_t (Gradients must flow back to here!)
        init_state: If provided (Warm Start), creates the initial link in the graph.
                    If None (Cold Start), initializes with noise.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config: cfg.update(config)
    
    # 1. Initialization
    if init_state is None:
        # Cold Start: Random noise. 
        # Note: Gradients stop here for Cold Start (which is correct).
        curr_state = torch.randn_like(state) 
    else:
        # Warm Start: Comes from Flow or Approximate Model.
        # CRITICAL: Preserve the graph from the Flow.
        curr_state = init_state 

    # 2. Unrolled Loop (The "ResNet" over time)
    for i in range(cfg["LANGEVIN_STEPS"]):
        # Noise injection
        noise = torch.randn_like(curr_state) * cfg["LANGEVIN_NOISE_SCALE"]
        
        # Gradient of Energy
        # This connects curr_state to action via the Energy function
        grad_energy = get_energy_gradient(model, state, action, curr_state)
        
        # Dynamics Step: s_{k+1} = s_k - \eta \nabla E + \epsilon
        # We subtract gradient to minimize Energy
        curr_state = curr_state - cfg["LANGEVIN_STEP_SIZE"] * grad_energy + noise
        
        # "Professor's Note": In very deep unrolls, you might need a Tanh here 
        # if your state space is bounded [-1, 1], to prevent divergence.
        # For now, we assume the Energy function penalizes out-of-bounds.
    
    return curr_state

# =============================================================================
# 2. DIFFERENTIABLE SVGD (Stein Variational Gradient Descent)
# =============================================================================
def rbf_kernel_matrix(x, h_min=1e-3):
    """
    Computes RBF kernel and its gradient w.r.t x.
    Fully differentiable.
    """
    n = x.shape[0]
    dim = x.shape[1]
    
    # Pairwise differences: (n, n, dim)
    diff = x.unsqueeze(1) - x.unsqueeze(0) 
    
    # Squared L2 distance: (n, n)
    dist_sq = diff.pow(2).sum(dim=-1)
    
    # Median Heuristic for Bandwidth (Differentiable version)
    # We detach the median calculation to treat 'h' as a constant for stability, 
    # though technically it depends on x.
    median_dist = dist_sq.detach().median()
    h = median_dist / np.log(n + 1)
    h = torch.maximum(h, torch.tensor(h_min).to(x.device))
    
    # Kernel Matrix: (n, n)
    k_xx = torch.exp(-dist_sq / h)
    
    # Gradient of Kernel w.r.t x: \nabla_x k(x, x')
    # Derived: -k(x,x') * 2(x - x') / h
    grad_k = -k_xx.unsqueeze(-1) * diff * (2 / h) # (n, n, dim)
    
    # Sum over the second dimension (interactions with all other particles)
    grad_k_sum = grad_k.sum(dim=1) 
    
    return k_xx, grad_k_sum

def predict_next_state_svgd(model, state, action, num_particles=10, config=None):
    """
    Samples using SVGD.
    
    WARNING: This scales quadratically with num_particles.
    This creates a massive computation graph if LANGEVIN_STEPS is high.
    Use fewer steps for SVGD than Langevin.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config: cfg.update(config)

    # 1. Expand Inputs for Particles
    # We treat the batch as one large population or handle particles per batch item.
    # For simplicity in this thesis context, we assume Batch Size 1 or 
    # we flatten the batch to treat them as a pool of particles.
    
    # Strategy: Independent particles per batch item is too expensive for loop.
    # We assume 'state' is (Batch, Dim). We create (Batch * Particles, Dim).
    batch_size = state.shape[0]
    
    state_expanded = state.repeat_interleave(num_particles, dim=0)   # (B*P, D)
    action_expanded = action.repeat_interleave(num_particles, dim=0) # (B*P, D)
    
    # Initialize particles (Warm or Cold)
    # Here we default to Cold Start with spread to ensure interaction
    particles = torch.randn_like(state_expanded)
    
    # 2. Unrolled SVGD Loop
    for i in range(cfg["SVGD_STEPS"]):
        # A. Score Function: \nabla \log p(x) = -\nabla E(x)
        grad_energy = get_energy_gradient(model, state_expanded, action_expanded, particles)
        score_func = -grad_energy
        
        # B. Kernel Term
        # We compute kernel only within the batch for efficiency (Batch * Particles)
        k_xx, grad_k = rbf_kernel_matrix(particles)
        
        # C. Stein Variational Gradient
        # Phi(x) = (1/N) * sum( k(x,x')*score(x') + \nabla k(x,x') )
        
        # Term 1: k(x,x') * score(x') -> (N, N) @ (N, D) = (N, D)
        term1 = torch.matmul(k_xx, score_func)
        
        # Term 2: \nabla k (already summed)
        phi = (term1 + grad_k) / (batch_size * num_particles)
        
        # D. Update
        particles = particles + cfg["SVGD_STEP_SIZE"] * phi
    
    # 3. Aggregation
    # Return the mean particle for the next state prediction, 
    # OR return all particles if your Critic supports distributional input.
    # For standard Actor-Critic, we take the mean or one sample.
    
    # Reshape back to (Batch, Particles, Dim)
    particles_reshaped = particles.view(batch_size, num_particles, -1)
    
    # Return mean prediction (Differentiable averaging)
    next_state_pred = particles_reshaped.mean(dim=1)
    
    return next_state_pred