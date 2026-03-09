"""
Gradient Diagnostics — Pure EBM Langevin pipeline.
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def compute_gradient_norms(actor):
    layer_norms = {}
    total_sq = 0.0
    for name, p in actor.named_parameters():
        if p.grad is not None:
            norm = p.grad.data.norm(2).item()
            layer_norms[name] = norm
            total_sq += norm ** 2
    return total_sq ** 0.5, layer_norms


def compute_gradient_variance(actor, actor_opt, make_rollout_loss_fn, num_samples=8):
    grad_samples = defaultdict(list)
    norm_samples = []
    for _ in range(num_samples):
        actor_opt.zero_grad()
        loss = make_rollout_loss_fn()
        loss.backward()
        total_norm, _ = compute_gradient_norms(actor)
        norm_samples.append(total_norm)
        for name, p in actor.named_parameters():
            if p.grad is not None:
                grad_samples[name].append(p.grad.data.clone().flatten())
    norms = np.array(norm_samples)
    per_param_variance = {}
    for name, grads in grad_samples.items():
        stacked = torch.stack(grads, dim=0)
        per_param_variance[name] = stacked.var(dim=0).mean().item()
    return norms.mean(), norms.var(), per_param_variance


def compute_jacobian_singular_values(ebm, state, action, config, num_singular=5):
    """
    Compute ds'/ds through the full Langevin chain.
    State flows into both init (curr = s + noise) and energy E(s, a, curr).
    """
    B, D = state.shape
    b = min(B, 8)
    s = state[:b].detach().requires_grad_(True)
    a = action[:b].detach()

    step_size = config.get("LANGEVIN_STEP_SIZE", 0.01)
    n_steps = config.get("LANGEVIN_STEPS", 50)
    noise_max = config.get("LANGEVIN_NOISE_MAX", 0.01)
    noise_min = config.get("LANGEVIN_NOISE_MIN", 0.0001)
    init_noise = config.get("LANGEVIN_INIT_NOISE", 0.05)
    grad_clip = config.get("LANGEVIN_GRAD_CLIP", 1.0)

    # Init from s (NO detach) so gradient flows
    curr = s + init_noise * torch.randn_like(s)

    for i in range(n_steps):
        if not curr.requires_grad:
            curr = curr.requires_grad_(True)
        energy = ebm(s, a, curr)
        grad = torch.autograd.grad(
            energy.sum(), curr,
            create_graph=True, retain_graph=True
        )[0]

        if grad_clip > 0:
            grad_norms = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            clip_coeff = (grad_clip / grad_norms).clamp(max=1.0)
            grad = grad * clip_coeff

        frac = i / max(n_steps - 1, 1)
        current_noise = noise_max * (1 - frac) + noise_min * frac
        curr = curr + step_size * grad + current_noise * torch.randn_like(curr)

    ns = curr
    jac = torch.zeros(b, D, D, device=state.device)
    for d in range(D):
        g = torch.autograd.grad(
            ns[:, d].sum(), s,
            create_graph=False, retain_graph=True,
            allow_unused=True
        )[0]
        if g is not None:
            jac[:, d, :] = g

    U, S, Vh = torch.linalg.svd(jac)
    mean_S = S.mean(dim=0).detach().cpu().numpy()
    k = min(num_singular, D)
    return mean_S[:k], float(mean_S[0])


def compute_multistep_spectral(ebm, actor, start_states, horizon, config, num_singular=5):
    from utils_sampling2 import get_energy_gradient
    step_spectra = []
    cumulative_radius = 1.0
    curr = start_states.detach()
    for t in range(horizon):
        with torch.no_grad():
            a = actor.sample(curr)
        sv, sr = compute_jacobian_singular_values(ebm, curr, a, config, num_singular)
        step_spectra.append({"step": t, "singular_values": sv, "spectral_radius": sr})
        cumulative_radius *= sr
        # Step forward (no grad)
        with torch.no_grad():
            step_size = config.get("LANGEVIN_STEP_SIZE", 0.01)
            noise_max = config.get("LANGEVIN_NOISE_MAX", 0.01)
            noise_min = config.get("LANGEVIN_NOISE_MIN", 0.0001)
            n_steps = config.get("LANGEVIN_STEPS", 50)
            ns = curr + 0.05 * torch.randn_like(curr)
            for step_i in range(n_steps):
                ns = ns.requires_grad_(True)
                grad = get_energy_gradient(ebm, curr, a, ns, create_graph=False)
                frac = step_i / max(n_steps - 1, 1)
                current_noise = noise_max * (1 - frac) + noise_min * frac
                ns = ns + step_size * grad + current_noise * torch.randn_like(ns)
                ns = ns.detach()
            curr = ns
    return step_spectra, cumulative_radius


def compute_effective_rank(actor, threshold=0.99):
    grad_rows = []
    max_len = 0
    for name, p in actor.named_parameters():
        if p.grad is not None:
            g = p.grad.data.flatten()
            grad_rows.append(g)
            max_len = max(max_len, len(g))
    if len(grad_rows) == 0:
        return 0, 0, np.array([])
    padded = [F.pad(g, (0, max_len - len(g))) if len(g) < max_len else g for g in grad_rows]
    mat = torch.stack(padded, dim=0)
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    S_np = S.detach().cpu().numpy()
    total_var = (S_np ** 2).sum()
    if total_var < 1e-12:
        return 0, len(grad_rows), np.array([0.0])
    cumvar = np.cumsum(S_np ** 2) / total_var
    return int(np.searchsorted(cumvar, threshold)) + 1, len(grad_rows), cumvar


class GradientDiagnosticLogger:
    def __init__(self):
        self.history = {
            "steps": [], "grad_norm": [], "grad_variance": [],
            "mean_grad_norm_over_samples": [],
            "spectral_radius": [], "cumulative_spectral_radius": [],
            "effective_rank": [], "layer_norms": [],
            "step_spectra": [], "per_param_variance": [],
        }

    def log_after_backward(self, step, actor):
        total_norm, layer_norms = compute_gradient_norms(actor)
        self.history["steps"].append(step)
        self.history["grad_norm"].append(total_norm)
        self.history["layer_norms"].append(layer_norms)

    def log_full_diagnostic(self, step, actor, actor_opt, make_rollout_loss_fn,
                            ebm, langevin_config, start_states, horizon,
                            num_variance_samples=8):
        mean_gn, gv, ppv = compute_gradient_variance(
            actor, actor_opt, make_rollout_loss_fn, num_samples=num_variance_samples
        )
        self.history["grad_variance"].append({"step": step, "variance": gv, "mean_norm": mean_gn})
        self.history["per_param_variance"].append({"step": step, "ppv": ppv})

        try:
            step_spec, cum_sr = compute_multistep_spectral(
                ebm, actor, start_states, horizon, langevin_config
            )
            self.history["spectral_radius"].append({
                "step": step, "per_step": [s["spectral_radius"] for s in step_spec]
            })
            self.history["cumulative_spectral_radius"].append({"step": step, "value": cum_sr})
            self.history["step_spectra"].append({"step": step, "spectra": step_spec})
        except Exception as e:
            print(f"  [Diag] Spectral analysis failed at step {step}: {e}")

        actor_opt.zero_grad()
        loss = make_rollout_loss_fn()
        loss.backward()
        eff_rank, total_rank, cumvar = compute_effective_rank(actor)
        self.history["effective_rank"].append({
            "step": step, "effective_rank": eff_rank, "total_rank": total_rank
        })

    def summary(self):
        return dict(self.history)