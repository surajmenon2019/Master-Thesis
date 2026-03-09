"""
Three Differentiability Regimes — Pure EBM Langevin.
Uses improved sampling: annealed noise, persistent chains, gradient clipping.
"""
from utils_sampling2 import langevin_sample, DIFF_MODES, reset_persistent_buffer

REGIME_NAMES = {
    "fully_differentiable": "Fully Differentiable",
    "truncated": "Truncated Backprop",
    "detached": "Fully Detached",
}

REGIME_COLORS = {
    "fully_differentiable": "#e63946",
    "truncated": "#f4a261",
    "detached": "#457b9d",
}

REGIME_LIST = list(DIFF_MODES)


class DifferentiabilityRegime:
    def __init__(self, regime_name, ebm, state_dim, device, config):
        assert regime_name in DIFF_MODES
        self.name = regime_name
        self.display_name = REGIME_NAMES[regime_name]
        self.color = REGIME_COLORS[regime_name]
        self.diff_mode = regime_name
        self.ebm = ebm
        self.state_dim = state_dim
        self.device = device
        self.config = config
        reset_persistent_buffer()

    def predict_next_state(self, state, action):
        return langevin_sample(
            model=self.ebm,
            state=state,
            action=action,
            diff_mode=self.diff_mode,
            config={
                "LANGEVIN_STEPS": self.config.get("LANGEVIN_STEPS", 50),
                "LANGEVIN_STEP_SIZE": self.config.get("LANGEVIN_STEP_SIZE", 0.01),
                "LANGEVIN_NOISE_MAX": self.config.get("LANGEVIN_NOISE_MAX", 0.01),
                "LANGEVIN_NOISE_MIN": self.config.get("LANGEVIN_NOISE_MIN", 0.0001),
                "LANGEVIN_INIT_NOISE": self.config.get("LANGEVIN_INIT_NOISE", 0.05),
                "LANGEVIN_GRAD_CLIP": self.config.get("LANGEVIN_GRAD_CLIP", 1.0),
            },
            use_persistent=True,
        )

    def freeze(self):
        for p in self.ebm.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.ebm.parameters():
            p.requires_grad = True