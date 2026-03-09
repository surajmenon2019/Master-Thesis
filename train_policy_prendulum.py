"""
Policy Convergence Training — Pendulum-v1

THESIS EXPERIMENT:
  All world models are pretrained on the same random data, then ALL receive
  fair online updates during policy training:
    - EBM:  InfoNCE on real buffer data
    - Flow: Forward KL (MLE) on real buffer data
    - MDN:  Negative log-likelihood on real buffer data

  Same data, same frequency, same batch size. The ONLY variable is:
    - Model architecture (BilinearEBM vs RealNVP vs MDN)
    - Sampling method (importance-weighted vs Langevin vs direct)

  This tests: "As the policy explores new regions, which model architecture
  adapts fastest and produces the most useful imagined rollouts?"

  Critic trains on imagined rollouts ONLY — no real-data bailout.

Agents:
  1. EBM (IW)        — Flow proposal + EBM importance-weighted resampling
  2. EBM (Langevin)  — Flow init + EBM Langevin refinement
  3. EBM (SVGD)      — Flow init + EBM SVGD refinement (mode-covering particles)
  4. Flow            — RealNVP direct sampling
  5. MDN             — Mixture Density Network direct sampling

Outputs:
  1. pendulum_convergence.png  — eval reward vs steps, per horizon
  2. pendulum_trust.png        — rollout trust %, per horizon
  3. pendulum_mse.png          — 1-step prediction MSE over training
  4. pendulum_compute.png      — computational cost breakdown per agent
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy

from models import (
    Actor, ValueNetwork, BilinearEBM, RealNVP,
    MixtureDensityNetwork, RewardModel
)
from utils_sampling import importance_weighted_sample, langevin_refine, svgd_sample

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "ENV_NAME": "Pendulum-v1",
    "ACTION_SCALE": 2.0,

    "TOTAL_STEPS": 50000,
    "BATCH_SIZE": 256,

    "DISCOUNT": 0.99,
    "LAMBDA": 0.95,
    "TAU": 0.005,

    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 3e-4,
    "LR_REWARD": 1e-3,
    "LR_WORLD_MODEL": 1e-4,    # Same LR for ALL world model online updates

    "ENTROPY_COEFF": 0.01,

    # World model online update schedule
    "WM_UPDATE_EVERY": 50,      # Same frequency for all models
    "WM_UPDATE_BATCH": 128,     # Same batch size for all models

    # Importance-weighted sampling
    "IW_NUM_SAMPLES": 32,
    "IW_TEMPERATURE": 1.0,

    # Langevin
    "LANGEVIN_STEPS": 15,
    "LANGEVIN_STEP_SIZE": 0.01,
    "LANGEVIN_NOISE_SCALE": 0.005,

    # SVGD
    "SVGD_NUM_PARTICLES": 32,
    "SVGD_STEPS": 15,
    "SVGD_STEP_SIZE": 0.01,
    "SVGD_BANDWIDTH": None,       # None = median heuristic

    # InfoNCE (EBM online update)
    "NUM_NEGATIVES": 256,
    "INFONCE_TEMPERATURE": 0.1,

    # Trust weighting
    "TRUST_THRESHOLD": 3.0,
    "TRUST_SHARPNESS": 2.0,

    # Reward model exploitation guard
    # Clamp predicted rewards to env's true range during imagined rollouts
    # Prevents actor from chasing reward model extrapolation artifacts
    "REWARD_CLAMP_MIN": -16.28,  # Pendulum: -(pi^2 + 0.1*8^2 + 0.001*2^2)
    "REWARD_CLAMP_MAX": 0.0,     # Pendulum: best possible reward

    # Logging
    "EVAL_INTERVAL": 1000,
    "EVAL_EPISODES": 10,
    "LOG_INTERVAL": 500,
    "DIAGNOSTIC_INTERVAL": 2000,

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "HIDDEN_DIM": 128,

    # MDN config — must match pretrained checkpoint
    "MDN_NUM_GAUSSIANS": 5,
}


# =============================================================================
# REPLAY BUFFER
# =============================================================================
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=100000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

        self._state_sum = np.zeros(state_dim, dtype=np.float64)
        self._state_sq_sum = np.zeros(state_dim, dtype=np.float64)
        self._state_count = 0

    def add(self, s, a, r, ns):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = ns
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._state_sum += ns.astype(np.float64)
        self._state_sq_sum += (ns.astype(np.float64)) ** 2
        self._state_count += 1

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        device = CONFIG["DEVICE"]
        return {
            "states": torch.tensor(self.states[idx], device=device),
            "actions": torch.tensor(self.actions[idx], device=device),
            "rewards": torch.tensor(self.rewards[idx], device=device),
            "next_states": torch.tensor(self.next_states[idx], device=device),
        }

    def sample_states(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return torch.tensor(self.states[idx], device=CONFIG["DEVICE"])

    def sample_negatives(self, batch_size, num_negatives):
        idx = np.random.randint(0, self.size, size=(batch_size, num_negatives))
        return torch.tensor(self.next_states[idx], device=CONFIG["DEVICE"])

    @property
    def state_mean(self):
        if self._state_count == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        return (self._state_sum / self._state_count).astype(np.float32)

    @property
    def state_std(self):
        if self._state_count < 2:
            return np.ones(self.state_dim, dtype=np.float32)
        mean = self._state_sum / self._state_count
        var = self._state_sq_sum / self._state_count - mean ** 2
        return np.sqrt(np.clip(var, 1e-6, None)).astype(np.float32)


# =============================================================================
# TRUST WEIGHTING
# =============================================================================
class TrustComputer:
    def __init__(self, buffer, threshold=3.0, sharpness=2.0):
        self.buffer = buffer
        self.threshold = threshold
        self.sharpness = sharpness

    def compute_trust(self, predicted_states):
        device = predicted_states.device
        mean = torch.tensor(self.buffer.state_mean, device=device)
        std = torch.tensor(self.buffer.state_std, device=device)
        z = (predicted_states - mean) / std
        z_norm = torch.norm(z, dim=-1, keepdim=True)
        return torch.sigmoid(self.sharpness * (self.threshold - z_norm))

    def compute_rollout_stats(self, trust_list):
        with torch.no_grad():
            all_t = torch.cat(trust_list, dim=1)
            return {
                "pct_trusted": (all_t > 0.5).float().mean().item() * 100,
                "mean_trust": all_t.mean().item(),
                "trust_by_step": all_t.mean(dim=0).cpu().numpy(),
            }


# =============================================================================
# TD(lambda) WITH TRUST-WEIGHTED DISCOUNTING
# =============================================================================
def compute_lambda_returns(rewards, values, trust, discount, lam):
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)
    for t in reversed(range(H)):
        eff_gamma = discount * trust[:, t]
        if t == H - 1:
            returns[:, t] = rewards[:, t] + eff_gamma * values[:, t]
        else:
            td1 = rewards[:, t] + eff_gamma * values[:, t]
            cont = rewards[:, t] + eff_gamma * returns[:, t + 1]
            returns[:, t] = (1 - lam) * td1 + lam * cont
    return returns


# =============================================================================
# PREDICTION ACCURACY MONITOR
# =============================================================================
def evaluate_prediction_accuracy(agent, buffer, num_samples=512):
    if buffer.size < num_samples:
        return None
    batch = buffer.sample(num_samples)
    # Langevin agents need autograd.grad inside predict_next_state,
    # so we can't use @torch.no_grad(). Use torch.enable_grad() for
    # prediction, then detach for MSE computation.
    with torch.enable_grad():
        pred = agent.predict_next_state(batch["states"], batch["actions"])
    pred = pred.detach()
    target = batch["next_states"].detach()
    mse = F.mse_loss(pred, target).item()
    per_dim = ((pred - target) ** 2).mean(dim=0).cpu().numpy()
    return {"mse": mse, "per_dim_mse": per_dim}


# =============================================================================
# WORLD MODEL AGENT — with fair online updates
# =============================================================================
class WorldModelAgent:
    """
    Wraps pretrained world model + optimizer for online updates.

    ALL model types get online updates at the same frequency on the same
    data. The update method differs by architecture:
      - EBM:  InfoNCE (contrastive)
      - Flow: Forward KL (maximum likelihood)
      - MDN:  Negative log-likelihood

    During imagined rollouts, world model parameters are temporarily frozen
    so actor gradients don't corrupt them.
    """
    def __init__(self, agent_type, state_dim, action_dim, device):
        self.agent_type = agent_type
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ebm = None
        self.flow = None
        self.mdn = None
        self.wm_optimizer = None  # Single optimizer for whichever model(s) this agent uses

        hd = CONFIG["HIDDEN_DIM"]
        lr = CONFIG["LR_WORLD_MODEL"]

        if agent_type == "EBM (SVGD)":
            # SVGD: EBM only, no Flow — pure particle-based sampling
            self.ebm = BilinearEBM(state_dim, action_dim, hidden_dim=hd).to(device)
            self.ebm.load_state_dict(torch.load(
                "pretrained_ebm_pendulum.pth", map_location=device, weights_only=True))

            self.wm_optimizer = optim.Adam(self.ebm.parameters(), lr=lr)
            self._needs_ebm_grad = True

        elif "EBM" in agent_type:
            # EBM (IW) and EBM (Langevin): need both EBM and Flow
            self.ebm = BilinearEBM(state_dim, action_dim, hidden_dim=hd).to(device)
            self.ebm.load_state_dict(torch.load(
                "pretrained_ebm_pendulum.pth", map_location=device, weights_only=True))

            self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=hd).to(device)
            self.flow.load_state_dict(torch.load(
                "pretrained_flow_pendulum.pth", map_location=device, weights_only=True))

            # Both EBM and Flow get updated for these agents
            self.wm_optimizer = optim.Adam(
                list(self.ebm.parameters()) + list(self.flow.parameters()), lr=lr
            )

            # Langevin needs autograd.grad through EBM
            self._needs_ebm_grad = (agent_type == "EBM (Langevin)")

        elif agent_type == "Flow":
            self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=hd).to(device)
            self.flow.load_state_dict(torch.load(
                "pretrained_flow_pendulum.pth", map_location=device, weights_only=True))
            self.wm_optimizer = optim.Adam(self.flow.parameters(), lr=lr)

        elif agent_type == "MDN":
            self.mdn = MixtureDensityNetwork(
                state_dim, action_dim,
                num_gaussians=CONFIG["MDN_NUM_GAUSSIANS"],
                hidden_dim=hd
            ).to(device)
            self.mdn.load_state_dict(torch.load(
                "pretrained_mdn_pendulum.pth", map_location=device, weights_only=True))
            self.wm_optimizer = optim.Adam(self.mdn.parameters(), lr=lr)

    def update_world_model(self, buffer):
        """
        One gradient step of online adaptation. Same data, same frequency
        for all model types. Returns metrics dict.
        """
        BS = CONFIG["WM_UPDATE_BATCH"]
        if buffer.size < BS:
            return {}

        batch = buffer.sample(BS)
        s, a, ns = batch["states"], batch["actions"], batch["next_states"]
        B = s.shape[0]
        metrics = {}

        self.wm_optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device)

        if self.ebm is not None:
            # --- EBM: InfoNCE ---
            self.ebm.train()
            E_pos = self.ebm(s, a, ns)
            neg_ns = buffer.sample_negatives(B, CONFIG["NUM_NEGATIVES"])
            s_exp = s.unsqueeze(1).expand(B, CONFIG["NUM_NEGATIVES"], -1)
            a_exp = a.unsqueeze(1).expand(B, CONFIG["NUM_NEGATIVES"], -1)
            E_neg = self.ebm(s_exp, a_exp, neg_ns)

            logits = torch.cat([E_pos.unsqueeze(1), E_neg], dim=1) / CONFIG["INFONCE_TEMPERATURE"]
            labels = torch.zeros(B, dtype=torch.long, device=self.device)
            ebm_loss = F.cross_entropy(logits, labels)
            total_loss = total_loss + ebm_loss

            with torch.no_grad():
                metrics["ebm_loss"] = ebm_loss.item()
                metrics["E_gap"] = (E_pos.mean() - E_neg.mean()).item()
                metrics["ebm_acc"] = (logits.argmax(dim=1) == 0).float().mean().item()

        if self.flow is not None:
            # --- Flow: Forward KL (MLE) ---
            self.flow.train()
            context = torch.cat([s, a], dim=1)
            log_prob = self.flow.log_prob(ns, context=context)
            flow_loss = -log_prob.mean() / self.state_dim
            total_loss = total_loss + flow_loss

            with torch.no_grad():
                metrics["flow_loss"] = flow_loss.item()

        if self.mdn is not None:
            # --- MDN: NLL ---
            self.mdn.train()
            mdn_ll = self.mdn.log_prob(s, a, ns)
            mdn_loss = -mdn_ll.mean() / self.state_dim
            total_loss = total_loss + mdn_loss

            with torch.no_grad():
                metrics["mdn_loss"] = mdn_loss.item()

        total_loss.backward()
        # Clip all world model params uniformly
        if self.ebm is not None:
            torch.nn.utils.clip_grad_norm_(self.ebm.parameters(), 1.0)
        if self.flow is not None:
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
        if self.mdn is not None:
            torch.nn.utils.clip_grad_norm_(self.mdn.parameters(), 1.0)

        self.wm_optimizer.step()

        # Back to eval mode for inference
        if self.ebm is not None:
            self.ebm.eval()
        if self.flow is not None:
            self.flow.eval()
        if self.mdn is not None:
            self.mdn.eval()

        return metrics

    def freeze_for_rollout(self):
        """
        Temporarily freeze world model params so actor grads don't corrupt them.
        FIX: EBM (Langevin) and EBM (SVGD) keep EBM unfrozen — both need
        autograd.grad through the energy function to compute dE/ds'.
        """
        if self.ebm is not None:
            if not getattr(self, '_needs_ebm_grad', False):
                for p in self.ebm.parameters(): p.requires_grad = False
            # For Langevin/SVGD: EBM stays unfrozen (autograd needs live graph)
        if self.flow is not None:
            for p in self.flow.parameters(): p.requires_grad = False
        if self.mdn is not None:
            for p in self.mdn.parameters(): p.requires_grad = False

    def unfreeze_after_rollout(self):
        """Restore requires_grad for next online update and clear stray grads."""
        if self.ebm is not None:
            for p in self.ebm.parameters(): p.requires_grad = True
        if self.flow is not None:
            for p in self.flow.parameters(): p.requires_grad = True
        if self.mdn is not None:
            for p in self.mdn.parameters(): p.requires_grad = True
        # Clear any stray gradients accumulated during actor rollout
        # (especially important for Langevin where EBM was unfrozen)
        self.wm_optimizer.zero_grad()

    def predict_next_state(self, state, action):
        """Differentiable w.r.t. action for actor gradient flow."""
        if self.agent_type == "EBM (IW)":
            context = torch.cat([state, action], dim=1)

            def proposal_fn(B, N):
                z = torch.randn(B, N, self.state_dim, device=self.device)
                ctx_exp = context.unsqueeze(1).expand(B, N, -1)
                samples = self.flow.sample(
                    z.reshape(B * N, -1),
                    context=ctx_exp.reshape(B * N, -1)
                )
                return samples.reshape(B, N, -1)

            return importance_weighted_sample(
                self.ebm, state, action, proposal_fn,
                config={
                    "IW_NUM_SAMPLES": CONFIG["IW_NUM_SAMPLES"],
                    "IW_TEMPERATURE": CONFIG["IW_TEMPERATURE"],
                }
            )

        elif self.agent_type == "EBM (Langevin)":
            return langevin_refine(
                self.ebm, state, action, self.flow,
                config={
                    "LANGEVIN_STEPS": CONFIG["LANGEVIN_STEPS"],
                    "LANGEVIN_STEP_SIZE": CONFIG["LANGEVIN_STEP_SIZE"],
                    "LANGEVIN_NOISE_SCALE": CONFIG["LANGEVIN_NOISE_SCALE"],
                },
                differentiable=True,
            )

        elif self.agent_type == "EBM (SVGD)":
            return svgd_sample(
                self.ebm, state, action,
                config={
                    "SVGD_NUM_PARTICLES": CONFIG["SVGD_NUM_PARTICLES"],
                    "SVGD_STEPS": CONFIG["SVGD_STEPS"],
                    "SVGD_STEP_SIZE": CONFIG["SVGD_STEP_SIZE"],
                    "SVGD_BANDWIDTH": CONFIG["SVGD_BANDWIDTH"],
                },
                differentiable=True,
            )

        elif self.agent_type == "Flow":
            B = state.shape[0]
            z = torch.randn(B, self.state_dim, device=self.device)
            context = torch.cat([state, action], dim=1)
            return self.flow.sample(z, context=context)

        elif self.agent_type == "MDN":
            return self.mdn.sample_differentiable(state, action)

        else:
            raise ValueError(f"Unknown: {self.agent_type}")


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_policy(actor, num_episodes=10):
    env = gym.make("Pendulum-v1")
    device = CONFIG["DEVICE"]
    results = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_r = 0
        done = False
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act = actor.sample(s_t).cpu().numpy()[0]
            act = np.clip(act, -CONFIG["ACTION_SCALE"], CONFIG["ACTION_SCALE"])
            state, r, term, trunc, _ = env.step(act)
            ep_r += r
            done = term or trunc
        results.append(ep_r)
    env.close()
    return np.mean(results), np.std(results)


# =============================================================================
# TRAIN ONE AGENT
# =============================================================================
def train_agent(agent_type, horizon):
    device = CONFIG["DEVICE"]
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"\n{'='*60}")
    print(f"Agent: {agent_type} | Horizon: {horizon}")
    print(f"{'='*60}")

    # --- Components ---
    agent = WorldModelAgent(agent_type, state_dim, action_dim, device)

    actor = Actor(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"],
                  action_scale=CONFIG["ACTION_SCALE"]).to(device)
    critic = ValueNetwork(state_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target = deepcopy(critic)
    for p in critic_target.parameters():
        p.requires_grad = False

    reward_model = RewardModel(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=CONFIG["LR_ACTOR"])
    critic_opt = optim.Adam(critic.parameters(), lr=CONFIG["LR_CRITIC"])
    reward_opt = optim.Adam(reward_model.parameters(), lr=CONFIG["LR_REWARD"])

    buffer = ReplayBuffer(state_dim, action_dim)
    trust_comp = TrustComputer(buffer, CONFIG["TRUST_THRESHOLD"], CONFIG["TRUST_SHARPNESS"])

    # --- Seed buffer with random data ---
    print("Collecting 2000 random transitions...")
    state, _ = env.reset()
    for _ in range(2000):
        a = env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a)
        buffer.add(state, a, r, ns)
        state = ns if not (term or trunc) else env.reset()[0]

    # --- Histories ---
    eval_steps, eval_rewards = [], []
    diag_log, trust_log = [], []

    # --- Timing ---
    timing = {
        "wm_update": [],         # world model online update
        "reward_update": [],     # reward model update
        "rollout_predict": [],   # predict_next_state calls during imagined rollout
        "critic_update": [],     # critic loss + backward + step
        "actor_update": [],      # actor rollout + loss + backward + step
        "eval": [],              # evaluation episodes
        "wall_clock": [],        # cumulative wall time at each LOG_INTERVAL
    }
    train_start_time = time.time()

    state, _ = env.reset()
    total_steps = 0

    while total_steps < CONFIG["TOTAL_STEPS"]:
        # ==== 1. REAL ENVIRONMENT STEP ====
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = actor.sample(s_t).cpu().numpy()[0]
        action = np.clip(action, -CONFIG["ACTION_SCALE"], CONFIG["ACTION_SCALE"])
        next_state, reward, term, trunc, _ = env.step(action)
        buffer.add(state, action, reward, next_state)
        total_steps += 1
        state = next_state if not (term or trunc) else env.reset()[0]

        # ==== 2. WORLD MODEL ONLINE UPDATE (fair: same schedule for all) ====
        if total_steps % CONFIG["WM_UPDATE_EVERY"] == 0:
            _t0 = time.time()
            wm_metrics = agent.update_world_model(buffer)
            timing["wm_update"].append(time.time() - _t0)
            if total_steps % CONFIG["LOG_INTERVAL"] == 0 and wm_metrics:
                parts = []
                for k, v in wm_metrics.items():
                    parts.append(f"{k}={v:.4f}")
                print(f"  [WM Update] {' | '.join(parts)}")

        # ==== 3. REWARD MODEL UPDATE (real data, every 5 steps) ====
        if buffer.size >= CONFIG["BATCH_SIZE"] and total_steps % 5 == 0:
            _t0 = time.time()
            batch = buffer.sample(CONFIG["BATCH_SIZE"])
            pred_r = reward_model(batch["states"], batch["actions"], batch["next_states"])
            r_loss = F.mse_loss(pred_r, batch["rewards"])
            reward_opt.zero_grad()
            r_loss.backward()
            reward_opt.step()
            timing["reward_update"].append(time.time() - _t0)

        # ==== 4. IMAGINED ROLLOUT + CRITIC + ACTOR (every 5 steps) ====
        if buffer.size >= CONFIG["BATCH_SIZE"] and total_steps % 5 == 0:
            B = CONFIG["BATCH_SIZE"]
            H = horizon

            # Freeze world model so actor/critic gradients don't corrupt it
            agent.freeze_for_rollout()

            # Also freeze reward model during rollout
            for p in reward_model.parameters():
                p.requires_grad = False

            # --- Imagined rollout for CRITIC ---
            curr = buffer.sample_states(B)
            i_states, i_rewards, i_next, i_trust = [], [], [], []

            _t_predict = 0.0
            for t in range(H):
                a = actor.sample(curr)
                _tp0 = time.time()
                ns = agent.predict_next_state(curr, a)
                _t_predict += time.time() - _tp0
                r = reward_model(curr, a, ns).squeeze(-1)   # (B,)
                r = r.clamp(CONFIG["REWARD_CLAMP_MIN"], CONFIG["REWARD_CLAMP_MAX"])
                w = trust_comp.compute_trust(ns)             # (B, 1)

                i_states.append(curr)
                i_rewards.append(r)
                i_next.append(ns)
                i_trust.append(w)
                curr = ns

            rew_t = torch.stack(i_rewards, dim=1)           # (B, H)
            ns_t = torch.stack(i_next, dim=1)               # (B, H, D)
            st_t = torch.stack(i_states, dim=1)             # (B, H, D)
            tr_t = torch.cat(i_trust, dim=1)                # (B, H)

            # Critic targets (imagined data ONLY)
            _t0 = time.time()
            with torch.no_grad():
                nv = critic_target(
                    ns_t.reshape(B * H, -1)
                ).squeeze(-1).reshape(B, H)                 # (B, H)

                targets = compute_lambda_returns(
                    rew_t.detach(), nv, tr_t.detach(),
                    CONFIG["DISCOUNT"], CONFIG["LAMBDA"]
                )                                            # (B, H)

            vpred = critic(
                st_t.reshape(B * H, -1).detach()
            ).squeeze(-1).reshape(B, H)                     # (B, H)

            critic_loss = F.mse_loss(vpred, targets)
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()
            timing["critic_update"].append(time.time() - _t0)

            # --- ACTOR UPDATE (fresh rollout for proper gradient flow) ---
            _t0 = time.time()
            for p in critic.parameters():
                p.requires_grad = False

            curr_a = buffer.sample_states(B)
            ret = torch.zeros(B, device=device)             # (B,)
            disc = torch.ones(B, device=device)              # (B,)

            for t in range(H):
                a = actor.sample(curr_a)
                _tp0 = time.time()
                ns = agent.predict_next_state(curr_a, a)
                _t_predict += time.time() - _tp0
                r = reward_model(curr_a, a, ns).squeeze(-1)  # (B,)
                r = r.clamp(CONFIG["REWARD_CLAMP_MIN"], CONFIG["REWARD_CLAMP_MAX"])
                w = trust_comp.compute_trust(ns).squeeze(-1)  # (B,)

                # Entropy bonus
                mu, log_std = actor(curr_a)
                std = torch.exp(torch.clamp(log_std, -5, 0.5))
                ent = 0.5 * torch.log(2 * np.pi * np.e * std.pow(2)).sum(dim=-1)

                ret = ret + disc * (r + CONFIG["ENTROPY_COEFF"] * ent)
                disc = disc * CONFIG["DISCOUNT"] * w
                curr_a = ns

            # Bootstrap final value
            ret = ret + disc * critic(curr_a).squeeze(-1)
            actor_loss = -ret.mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()
            timing["actor_update"].append(time.time() - _t0)

            # Record total predict_next_state time (critic rollout + actor rollout)
            timing["rollout_predict"].append(_t_predict)

            # Unfreeze everything
            for p in critic.parameters():
                p.requires_grad = True
            for p in reward_model.parameters():
                p.requires_grad = True
            agent.unfreeze_after_rollout()

            # Polyak
            with torch.no_grad():
                for p, tp in zip(critic.parameters(), critic_target.parameters()):
                    tp.data.copy_(CONFIG["TAU"] * p.data + (1 - CONFIG["TAU"]) * tp.data)

            # --- LOGGING ---
            if total_steps % CONFIG["LOG_INTERVAL"] == 0:
                ts = trust_comp.compute_rollout_stats(i_trust)
                trust_log.append({"step": total_steps, **ts})
                timing["wall_clock"].append({
                    "step": total_steps,
                    "elapsed": time.time() - train_start_time,
                })
                bystep = " ".join(f"{v:.2f}" for v in ts["trust_by_step"])
                print(f"  [Step {total_steps}] Critic: {critic_loss.item():.4f} | "
                      f"Actor: {actor_loss.item():.4f} | "
                      f"Trust: {ts['pct_trusted']:.1f}% (mean={ts['mean_trust']:.3f}) | "
                      f"By step: [{bystep}]")

        # ==== 5. PREDICTION DIAGNOSTIC ====
        if total_steps % CONFIG["DIAGNOSTIC_INTERVAL"] == 0:
            diag = evaluate_prediction_accuracy(agent, buffer)
            if diag:
                diag_log.append({"step": total_steps, **diag})
                print(f"  [Diag] 1-step MSE: {diag['mse']:.6f}")

        # ==== 6. EVAL ====
        if total_steps % CONFIG["EVAL_INTERVAL"] == 0:
            _t0 = time.time()
            em, es = evaluate_policy(actor, CONFIG["EVAL_EPISODES"])
            timing["eval"].append(time.time() - _t0)
            eval_steps.append(total_steps)
            eval_rewards.append(em)
            print(f"Step {total_steps}/{CONFIG['TOTAL_STEPS']} | "
                  f"Eval: {em:.1f} +/- {es:.1f}")

    env.close()
    total_wall = time.time() - train_start_time
    print(f"\n  [{agent_type} H={horizon}] Total wall time: {total_wall:.1f}s")

    # Summarize timing
    def _mean(lst): return np.mean(lst) if lst else 0.0
    def _sum(lst): return np.sum(lst) if lst else 0.0
    print(f"    predict_next_state: {_sum(timing['rollout_predict']):.1f}s total "
          f"({_mean(timing['rollout_predict'])*1000:.2f}ms avg/call)")
    print(f"    wm_update:         {_sum(timing['wm_update']):.1f}s total "
          f"({_mean(timing['wm_update'])*1000:.2f}ms avg/call)")
    print(f"    critic_update:     {_sum(timing['critic_update']):.1f}s total")
    print(f"    actor_update:      {_sum(timing['actor_update']):.1f}s total")

    return {
        "eval_steps": eval_steps,
        "eval_rewards": eval_rewards,
        "diagnostic_log": diag_log,
        "rollout_quality_log": trust_log,
        "timing": timing,
        "total_wall_time": total_wall,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"\n{'#'*60}")
    print(f"PENDULUM-v1 MBRL POLICY CONVERGENCE")
    print(f"Fair online updates: all models adapt on same data/schedule")
    print(f"Critic on imagined data ONLY")
    print(f"{'#'*60}\n")

    agent_types = ["EBM (IW)", "EBM (Langevin)", "EBM (SVGD)", "Flow", "MDN"]
    horizons = [1, 3, 5]

    all_results = {}
    for horizon in horizons:
        for agent_type in agent_types:
            key = f"{agent_type} H={horizon}"
            try:
                all_results[key] = train_agent(agent_type, horizon)
            except Exception as e:
                print(f"!!! FAILED: {key} -- {e}")
                import traceback; traceback.print_exc()

    # ===== PLOT 1: Policy convergence =====
    colors = {"EBM (IW)": "tab:red", "EBM (Langevin)": "tab:orange",
              "EBM (SVGD)": "tab:purple", "Flow": "tab:blue", "MDN": "tab:green"}

    fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 5), sharey=True)
    if len(horizons) == 1: axes = [axes]
    for i, h in enumerate(horizons):
        ax = axes[i]
        ax.set_title(f"Horizon = {h}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps"); ax.grid(True, alpha=0.3)
        if i == 0: ax.set_ylabel("Eval Reward")
        for at in agent_types:
            k = f"{at} H={h}"
            if k in all_results:
                d = all_results[k]
                ax.plot(d["eval_steps"], d["eval_rewards"],
                        label=at, color=colors[at], linewidth=2)
        ax.legend(fontsize=9)
    plt.suptitle("Policy Convergence (Fair Online Updates, Imagined Rollouts Only)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("pendulum_convergence.png", dpi=200, bbox_inches="tight")
    print("\nSaved: pendulum_convergence.png")

    # ===== PLOT 2: Rollout quality =====
    fig2, axes2 = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 4), sharey=True)
    if len(horizons) == 1: axes2 = [axes2]
    for i, h in enumerate(horizons):
        ax = axes2[i]
        ax.set_title(f"Rollout Trust -- H={h}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps"); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)
        if i == 0: ax.set_ylabel("% Trusted Steps")
        for at in agent_types:
            k = f"{at} H={h}"
            if k in all_results and all_results[k]["rollout_quality_log"]:
                rq = all_results[k]["rollout_quality_log"]
                ax.plot([r["step"] for r in rq], [r["pct_trusted"] for r in rq],
                        label=at, color=colors[at], linewidth=2)
        ax.legend(fontsize=9)
    plt.suptitle("Rollout Quality Over Training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("pendulum_trust.png", dpi=200, bbox_inches="tight")
    print("Saved: pendulum_trust.png")

    # ===== PLOT 3: Prediction MSE =====
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.set_title("1-Step Prediction MSE", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Steps"); ax3.set_ylabel("MSE"); ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)
    for at in agent_types:
        k = f"{at} H=1"
        if k in all_results and all_results[k]["diagnostic_log"]:
            dl = all_results[k]["diagnostic_log"]
            ax3.plot([d["step"] for d in dl], [d["mse"] for d in dl],
                     label=at, color=colors[at], linewidth=2)
    ax3.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("pendulum_mse.png", dpi=200, bbox_inches="tight")
    print("Saved: pendulum_mse.png")

    # ===== PLOT 4: Computational cost =====
    # Two subplots:
    #   Left:  Total wall-clock time per agent (stacked bar by component)
    #   Right: Per-step predict_next_state cost (the key differentiator)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

    # Collect timing data at H=3 (middle horizon) for clearest comparison
    # Fall back to whatever horizon has data
    timing_horizon = 3 if 3 in horizons else horizons[0]

    bar_agents = []
    bar_wall = []
    bar_predict = []
    bar_critic = []
    bar_actor = []
    bar_wm = []
    bar_other = []
    per_step_predict_mean = []
    per_step_predict_std = []

    for at in agent_types:
        k = f"{at} H={timing_horizon}"
        if k not in all_results or "timing" not in all_results[k]:
            continue
        t = all_results[k]["timing"]
        total_w = all_results[k].get("total_wall_time", 0)

        t_predict = np.sum(t["rollout_predict"]) if t["rollout_predict"] else 0
        t_critic = np.sum(t["critic_update"]) if t["critic_update"] else 0
        t_actor = np.sum(t["actor_update"]) if t["actor_update"] else 0
        t_wm = np.sum(t["wm_update"]) if t["wm_update"] else 0
        t_other = max(total_w - t_predict - t_critic - t_actor - t_wm, 0)

        bar_agents.append(at)
        bar_wall.append(total_w)
        bar_predict.append(t_predict)
        bar_critic.append(t_critic)
        bar_actor.append(t_actor)
        bar_wm.append(t_wm)
        bar_other.append(t_other)

        if t["rollout_predict"]:
            per_step_predict_mean.append(np.mean(t["rollout_predict"]) * 1000)
            per_step_predict_std.append(np.std(t["rollout_predict"]) * 1000)
        else:
            per_step_predict_mean.append(0)
            per_step_predict_std.append(0)

    if bar_agents:
        x = np.arange(len(bar_agents))
        width = 0.6

        # Left: stacked bar chart of time breakdown
        b1 = ax4a.bar(x, bar_predict, width, label="predict_next_state",
                       color="tab:red", alpha=0.85)
        b2 = ax4a.bar(x, bar_actor, width, bottom=np.array(bar_predict),
                       label="actor update", color="tab:orange", alpha=0.85)
        b3 = ax4a.bar(x, bar_critic, width,
                       bottom=np.array(bar_predict) + np.array(bar_actor),
                       label="critic update", color="tab:blue", alpha=0.85)
        b4 = ax4a.bar(x, bar_wm, width,
                       bottom=np.array(bar_predict) + np.array(bar_actor) + np.array(bar_critic),
                       label="WM update", color="tab:green", alpha=0.85)
        b5 = ax4a.bar(x, bar_other, width,
                       bottom=(np.array(bar_predict) + np.array(bar_actor) +
                               np.array(bar_critic) + np.array(bar_wm)),
                       label="other (env, eval, etc)", color="tab:gray", alpha=0.5)

        ax4a.set_xticks(x)
        ax4a.set_xticklabels(bar_agents, rotation=20, ha="right", fontsize=9)
        ax4a.set_ylabel("Time (seconds)")
        ax4a.set_title(f"Total Wall-Clock Breakdown (H={timing_horizon})",
                        fontsize=12, fontweight="bold")
        ax4a.legend(fontsize=8, loc="upper left")
        ax4a.grid(True, alpha=0.3, axis="y")

        # Add total time labels on top
        for i, tw in enumerate(bar_wall):
            ax4a.text(i, tw + 0.5, f"{tw:.0f}s", ha="center", fontsize=9, fontweight="bold")

        # Right: per-call predict_next_state cost (bar + error bar)
        bar_colors = [colors.get(at, "tab:gray") for at in bar_agents]
        ax4b.bar(x, per_step_predict_mean, width, yerr=per_step_predict_std,
                  color=bar_colors, alpha=0.85, capsize=4)
        ax4b.set_xticks(x)
        ax4b.set_xticklabels(bar_agents, rotation=20, ha="right", fontsize=9)
        ax4b.set_ylabel("Time (ms)")
        ax4b.set_title(f"Per-Call predict_next_state Cost (H={timing_horizon})",
                        fontsize=12, fontweight="bold")
        ax4b.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for i, (m, s) in enumerate(zip(per_step_predict_mean, per_step_predict_std)):
            ax4b.text(i, m + s + 0.2, f"{m:.1f}ms", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Computational Cost Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("pendulum_compute.png", dpi=200, bbox_inches="tight")
    print("Saved: pendulum_compute.png")

    # ===== PLOT 5: Wall-clock convergence (reward vs time, not steps) =====
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    ax5.set_title(f"Reward vs Wall-Clock Time (H={timing_horizon})",
                   fontsize=12, fontweight="bold")
    ax5.set_xlabel("Wall Time (seconds)")
    ax5.set_ylabel("Eval Reward")
    ax5.grid(True, alpha=0.3)

    for at in agent_types:
        k = f"{at} H={timing_horizon}"
        if k not in all_results or "timing" not in all_results[k]:
            continue
        d = all_results[k]
        wc = d["timing"]["wall_clock"]
        if not wc or not d["eval_steps"]:
            continue

        # Interpolate wall time at eval steps
        wc_steps = [w["step"] for w in wc]
        wc_times = [w["elapsed"] for w in wc]
        if len(wc_steps) >= 2:
            eval_times = np.interp(d["eval_steps"], wc_steps, wc_times)
            ax5.plot(eval_times, d["eval_rewards"],
                     label=at, color=colors[at], linewidth=2)

    ax5.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("pendulum_wallclock.png", dpi=200, bbox_inches="tight")
    print("Saved: pendulum_wallclock.png")

    np.save("pendulum_results.npy", all_results, allow_pickle=True)
    print("Saved: pendulum_results.npy\nDone!")


if __name__ == "__main__":
    main()