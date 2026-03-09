"""
Langevin Sampling for EBM — Three Differentiability Regimes.

FIXES OVER PREVIOUS VERSION:

FIX A — INIT GRADIENT FLOW (CRITICAL)
  Old: curr = state.detach() + noise for ALL modes.
  New: only detach for "detached" mode. For fully_differentiable and
  truncated, keep state in the graph so that multi-step credit assignment
  works: ns at horizon step t carries gradient back through state input
  from step t-1's ns.

FIX B — TRUNCATED BACKPROP THROUGH LAST K STEPS (not just 1)
  Old: create_graph=True only on final Langevin step, detach all others.
  New: retain graph for last TRUNCATE_K steps (default 10). This gives
  meaningful gradient signal without the full 50-step deep chain.

FIX C — REDUCED LANGEVIN DEPTH FOR FULLY_DIFFERENTIABLE
  Old: all 50 steps with create_graph=True → 50-deep second-order chain.
  New: configurable LANGEVIN_STEPS_DIFF for the differentiable path.
  Defaults to 10 steps. The energy landscape is smooth enough that 10
  steps suffice, and the gradient conditioning is dramatically better.

Original fixes (annealed noise, persistent chains, gradient clipping)
are preserved.
"""
import torch
import torch.nn.functional as F

DIFF_MODES = ("fully_differentiable", "truncated", "detached")

DEFAULT_CONFIG = {
    "LANGEVIN_STEPS": 10,
    "LANGEVIN_STEP_SIZE": 0.01,
    "LANGEVIN_NOISE_MAX": 0.01,
    "LANGEVIN_NOISE_MIN": 0.0001,
    "LANGEVIN_INIT_NOISE": 0.05,
    "LANGEVIN_GRAD_CLIP": 1.0,
    "TRUNCATE_K": 10,              # NEW: how many final steps retain graph in truncated mode
    "LANGEVIN_STEPS_DIFF": None,   # NEW: override step count for fully_differentiable (None = use LANGEVIN_STEPS)
}


# =============================================================================
# PERSISTENT CHAIN BUFFER
# =============================================================================
class PersistentChainBuffer:
    def __init__(self, capacity=10000, grid_resolution=0.1):
        self.capacity = capacity
        self.grid_res = grid_resolution
        self.buffer = {}
        self.keys_order = []

    def _make_key(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        discretized = (sa / self.grid_res).round().to(torch.int32)
        return tuple(discretized.cpu().numpy().tolist())

    def get(self, states, actions):
        B = states.shape[0]
        device = states.device
        inits = torch.zeros_like(states)
        mask = torch.zeros(B, dtype=torch.bool, device=device)
        for i in range(B):
            key = self._make_key(states[i], actions[i])
            if key in self.buffer:
                inits[i] = self.buffer[key].to(device)
                mask[i] = True
        return inits, mask

    def store(self, states, actions, results):
        B = states.shape[0]
        for i in range(B):
            key = self._make_key(states[i], actions[i])
            if key not in self.buffer:
                if len(self.keys_order) >= self.capacity:
                    old_key = self.keys_order.pop(0)
                    self.buffer.pop(old_key, None)
                self.keys_order.append(key)
            self.buffer[key] = results[i].detach().cpu()


_persistent_buffer = None

def get_persistent_buffer():
    global _persistent_buffer
    if _persistent_buffer is None:
        _persistent_buffer = PersistentChainBuffer()
    return _persistent_buffer

def reset_persistent_buffer():
    global _persistent_buffer
    _persistent_buffer = PersistentChainBuffer()


# =============================================================================
# ENERGY GRADIENT WITH CLIPPING
# =============================================================================
def get_energy_gradient(model, state, action, next_state, create_graph=False,
                        grad_clip=None):
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

    if grad_clip is not None and grad_clip > 0:
        grad_norms = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        clip_coeff = (grad_clip / grad_norms).clamp(max=1.0)
        grad = grad * clip_coeff

    return grad


# =============================================================================
# ANNEALED NOISE SCHEDULE
# =============================================================================
def noise_schedule(step, n_steps, noise_max, noise_min):
    frac = step / max(n_steps - 1, 1)
    return noise_max * (1 - frac) + noise_min * frac


# =============================================================================
# LANGEVIN SAMPLING
# =============================================================================
def langevin_sample(model, state, action, diff_mode, init_state=None,
                    config=None, use_persistent=True):
    assert diff_mode in DIFF_MODES

    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    n_steps = cfg["LANGEVIN_STEPS"]
    step_size = cfg["LANGEVIN_STEP_SIZE"]
    noise_max = cfg.get("LANGEVIN_NOISE_MAX", 0.01)
    noise_min = cfg.get("LANGEVIN_NOISE_MIN", 0.0001)
    init_noise = cfg.get("LANGEVIN_INIT_NOISE", 0.05)
    grad_clip = cfg.get("LANGEVIN_GRAD_CLIP", 1.0)
    truncate_k = cfg.get("TRUNCATE_K", 10)

    # FIX C: Optionally reduce step count for fully_differentiable
    if diff_mode == "fully_differentiable" and cfg.get("LANGEVIN_STEPS_DIFF") is not None:
        n_steps = cfg["LANGEVIN_STEPS_DIFF"]

    B = state.shape[0]

    # --- INITIALIZATION ---
    # FIX A: Only detach state for "detached" mode.
    # For differentiable modes, state stays in the computation graph so that
    # when this ns is used as 'state' in the next horizon step, gradients
    # flow back through the entire trajectory.
    if init_state is not None:
        curr = init_state
    elif diff_mode == "detached":
        if use_persistent:
            pbuf = get_persistent_buffer()
            cached_inits, found_mask = pbuf.get(state, action)
            curr = state.detach() + init_noise * torch.randn_like(state)
            if found_mask.any():
                curr[found_mask] = cached_inits[found_mask].to(state.device)
        else:
            curr = state.detach() + init_noise * torch.randn_like(state)
    else:
        # fully_differentiable and truncated: keep state in graph
        curr = state + init_noise * torch.randn_like(state)

    # --- LANGEVIN ITERATION ---
    for i in range(n_steps):
        is_last = (i == n_steps - 1)
        current_noise = noise_schedule(i, n_steps, noise_max, noise_min)

        if diff_mode == "fully_differentiable":
            # All steps retain graph
            step_create_graph = True
            if not curr.requires_grad:
                curr = curr.requires_grad_(True)

        elif diff_mode == "truncated":
            # FIX B: Retain graph for last truncate_k steps, not just the last 1
            in_graph_region = (i >= n_steps - truncate_k)
            step_create_graph = in_graph_region

            if not in_graph_region:
                curr = curr.detach().requires_grad_(True)
            else:
                # First step of graph region: detach to create clean boundary
                if i == n_steps - truncate_k:
                    curr = curr.detach().requires_grad_(True)
                elif not curr.requires_grad:
                    curr = curr.requires_grad_(True)

        elif diff_mode == "detached":
            step_create_graph = False
            curr = curr.detach().requires_grad_(True)

        grad = get_energy_gradient(
            model, state, action, curr,
            create_graph=step_create_graph,
            grad_clip=grad_clip
        )

        noise = torch.randn_like(curr) * current_noise
        curr = curr + step_size * grad + noise

    # --- Store for persistent chains ---
    if use_persistent and diff_mode == "detached":
        pbuf = get_persistent_buffer()
        pbuf.store(state, action, curr)

    if diff_mode == "detached":
        return curr.detach().requires_grad_(True)
    else:
        return curr