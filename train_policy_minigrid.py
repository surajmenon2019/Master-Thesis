"""
Policy Convergence Training — MiniGrid-Dynamic-Obstacles

THESIS EXPERIMENT: identical structure to Pendulum version.
  All world models pretrained on same random data, all receive fair
  online updates. Critic trains on imagined rollouts ONLY.

Agents:
  1. EBM (IW)        — Flow proposal + EBM importance-weighted resampling
  2. EBM (Langevin)  — Flow init + EBM Langevin refinement
  3. EBM (SVGD)      — EBM SVGD refinement (mode-covering particles)
  4. Flow            — RealNVP direct sampling
  5. MDN             — Mixture Density Network direct sampling
  6. Direct          — Simple NN regression baseline (no flow/EBM/mixture)

Environment: MiniGrid-Dynamic-Obstacles-8x8 with 3 moving obstacles.
  Genuine environment-side stochasticity from obstacle movement.
  state_dim=15, action_dim=3.

Changes from original:
  - Added Direct NN baseline (simple MLP world model)
  - Multi-seed runs (NUM_SEEDS=3) with mean +/- std plots
  - Reward model accuracy validation (analytical vs learned)
  - Detailed computational cost breakdown per component

Outputs:
  minigrid_convergence.png, minigrid_trust.png, minigrid_mse.png,
  minigrid_compute.png, minigrid_wallclock.png, minigrid_results.npy
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy

from models import (
    Actor, ValueNetwork, BilinearEBM, RealNVP,
    MixtureDensityNetwork, RewardModel, DirectNN
)
from utils_sampling import importance_weighted_sample, langevin_refine, svgd_sample
from minigrid_env import (
    make_minigrid_env, continuous_to_discrete, discrete_to_onehot,
    minigrid_analytical_reward,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # --- Environment ---
    "GRID_SIZE": 8,
    "N_OBSTACLES": 3,
    "SLIP_PROB": 0.1,
    "MAX_STEPS": 100,
    "ACTION_SCALE": 1.0,

    # --- Training ---
    "TOTAL_STEPS": 50000,
    "BATCH_SIZE": 256,

    # --- Multi-seed ---
    "NUM_SEEDS": 3,
    "SEEDS": [42, 123, 7],

    # --- RL ---
    "DISCOUNT": 0.99,
    "LAMBDA": 0.95,
    "TAU": 0.005,
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 3e-4,
    "LR_WORLD_MODEL": 1e-4,
    "ENTROPY_COEFF": 0.01,

    # --- World model online updates ---
    "WM_UPDATE_EVERY": 50,
    "WM_UPDATE_BATCH": 128,

    # --- Sampling configs ---
    "IW_NUM_SAMPLES": 32,
    "IW_TEMPERATURE": 1.0,
    "LANGEVIN_STEPS": 15,
    "LANGEVIN_STEP_SIZE": 0.01,
    "LANGEVIN_NOISE_SCALE": 0.005,
    "SVGD_NUM_PARTICLES": 32,
    "SVGD_STEPS": 15,
    "SVGD_STEP_SIZE": 0.01,
    "SVGD_BANDWIDTH": None,

    # --- InfoNCE ---
    "NUM_NEGATIVES": 256,
    "INFONCE_TEMPERATURE": 0.1,

    # --- Trust ---
    "TRUST_THRESHOLD": 3.0,
    "TRUST_SHARPNESS": 2.0,

    # --- Logging ---
    "EVAL_INTERVAL": 1000,
    "EVAL_EPISODES": 10,
    "LOG_INTERVAL": 500,
    "DIAGNOSTIC_INTERVAL": 2000,

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "HIDDEN_DIM": 128,

    # --- Architecture (must match pretrained checkpoints) ---
    "MDN_NUM_GAUSSIANS": 10,
    "FLOW_N_LAYERS": 8,
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
# TRUST / TD(lambda) / DIAGNOSTICS  (unchanged — work on any dim)
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


def evaluate_prediction_accuracy(agent, buffer, num_samples=512):
    if buffer.size < num_samples:
        return None
    batch = buffer.sample(num_samples)
    with torch.enable_grad():
        pred = agent.predict_next_state(batch["states"], batch["actions"])
    pred = pred.detach()
    target = batch["next_states"].detach()
    mse = F.mse_loss(pred, target).item()
    per_dim = ((pred - target) ** 2).mean(dim=0).cpu().numpy()
    return {"mse": mse, "per_dim_mse": per_dim}


# =============================================================================
# REWARD MODEL ACCURACY — ONE-TIME SANITY CHECK
# =============================================================================
def reward_sanity_check(num_samples=2000):
    """
    One-time check that minigrid_analytical_reward matches real env rewards.
    Run once before training. If this fails, imagined rollouts optimize
    the wrong objective.

    Returns dict with mae, corr, sign_agreement, or None on failure.
    """
    env = make_minigrid_env(
        size=CONFIG["GRID_SIZE"], n_obstacles=CONFIG["N_OBSTACLES"],
        slip_prob=CONFIG["SLIP_PROB"], max_steps=CONFIG["MAX_STEPS"])
    device = CONFIG["DEVICE"]

    states, actions, next_states, rewards = [], [], [], []
    state, _ = env.reset()
    for _ in range(num_samples):
        a_int = env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a_int)
        states.append(state)
        actions.append(discrete_to_onehot(a_int, env.action_dim))
        next_states.append(ns)
        rewards.append(r)
        if term or trunc:
            state, _ = env.reset()
        else:
            state = ns
    env.close()

    s = torch.tensor(np.array(states), device=device)
    a = torch.tensor(np.array(actions), device=device)
    ns = torch.tensor(np.array(next_states), device=device)
    actual = torch.tensor(np.array(rewards), device=device)

    with torch.no_grad():
        analytical = minigrid_analytical_reward(s, ns, a)

    diff = analytical - actual
    mae = diff.abs().mean().item()
    mse = (diff ** 2).mean().item()

    a_mean, p_mean = actual.mean(), analytical.mean()
    cov = ((actual - a_mean) * (analytical - p_mean)).mean()
    corr = (cov / (actual.std() * analytical.std() + 1e-8)).item()
    sign_agree = ((actual > 0) == (analytical > 0)).float().mean().item()

    return {
        "mae": mae, "rmse": mse ** 0.5, "corr": corr,
        "sign_agreement": sign_agree,
        "actual_mean": actual.mean().item(),
        "analytical_mean": analytical.mean().item(),
    }


# =============================================================================
# WORLD MODEL AGENT
# =============================================================================
class WorldModelAgent:
    def __init__(self, agent_type, state_dim, action_dim, device):
        self.agent_type = agent_type
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ebm = None
        self.flow = None
        self.mdn = None
        self.direct = None
        self.wm_optimizer = None

        hd = CONFIG["HIDDEN_DIM"]
        lr = CONFIG["LR_WORLD_MODEL"]
        flow_nl = CONFIG["FLOW_N_LAYERS"]

        if agent_type == "EBM (SVGD)":
            self.ebm = BilinearEBM(state_dim, action_dim, hidden_dim=hd).to(device)
            self.ebm.load_state_dict(torch.load(
                "pretrained_ebm_minigrid.pth", map_location=device, weights_only=True))
            self.wm_optimizer = optim.Adam(self.ebm.parameters(), lr=lr)
            self._needs_ebm_grad = True

        elif "EBM" in agent_type:
            self.ebm = BilinearEBM(state_dim, action_dim, hidden_dim=hd).to(device)
            self.ebm.load_state_dict(torch.load(
                "pretrained_ebm_minigrid.pth", map_location=device, weights_only=True))
            self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim,
                                hidden_dim=hd, n_layers=flow_nl).to(device)
            self.flow.load_state_dict(torch.load(
                "pretrained_flow_minigrid.pth", map_location=device, weights_only=True))
            self.wm_optimizer = optim.Adam(
                list(self.ebm.parameters()) + list(self.flow.parameters()), lr=lr)
            self._needs_ebm_grad = (agent_type in ("EBM (Langevin)", "EBM (IW)"))

        elif agent_type == "Flow":
            self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim,
                                hidden_dim=hd, n_layers=flow_nl).to(device)
            self.flow.load_state_dict(torch.load(
                "pretrained_flow_minigrid.pth", map_location=device, weights_only=True))
            self.wm_optimizer = optim.Adam(self.flow.parameters(), lr=lr)

        elif agent_type == "MDN":
            self.mdn = MixtureDensityNetwork(
                state_dim, action_dim,
                num_gaussians=CONFIG["MDN_NUM_GAUSSIANS"], hidden_dim=hd).to(device)
            self.mdn.load_state_dict(torch.load(
                "pretrained_mdn_minigrid.pth", map_location=device, weights_only=True))
            self.wm_optimizer = optim.Adam(self.mdn.parameters(), lr=lr)

        elif agent_type == "Direct":
            self.direct = DirectNN(state_dim, action_dim, hidden_dim=hd).to(device)
            self.direct.load_state_dict(torch.load(
                "pretrained_direct_minigrid.pth", map_location=device, weights_only=True))
            self.wm_optimizer = optim.Adam(self.direct.parameters(), lr=lr)

    def update_world_model(self, buffer):
        BS = CONFIG["WM_UPDATE_BATCH"]
        if buffer.size < BS: return {}
        batch = buffer.sample(BS)
        s, a, ns = batch["states"], batch["actions"], batch["next_states"]
        B = s.shape[0]
        metrics = {}
        self.wm_optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device)

        if self.ebm is not None:
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
            self.flow.train()
            context = torch.cat([s, a], dim=1)
            log_prob = self.flow.log_prob(ns, context=context)
            flow_loss = -log_prob.mean() / self.state_dim
            total_loss = total_loss + flow_loss
            with torch.no_grad(): metrics["flow_loss"] = flow_loss.item()

        if self.mdn is not None:
            self.mdn.train()
            mdn_ll = self.mdn.log_prob(s, a, ns)
            mdn_loss = -mdn_ll.mean() / self.state_dim
            total_loss = total_loss + mdn_loss
            with torch.no_grad(): metrics["mdn_loss"] = mdn_loss.item()

        if self.direct is not None:
            self.direct.train()
            pred = self.direct(s, a)
            direct_loss = F.mse_loss(pred, ns)
            total_loss = total_loss + direct_loss
            with torch.no_grad(): metrics["direct_loss"] = direct_loss.item()

        total_loss.backward()
        if self.ebm: torch.nn.utils.clip_grad_norm_(self.ebm.parameters(), 1.0)
        if self.flow: torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
        if self.mdn: torch.nn.utils.clip_grad_norm_(self.mdn.parameters(), 1.0)
        if self.direct: torch.nn.utils.clip_grad_norm_(self.direct.parameters(), 1.0)
        self.wm_optimizer.step()
        if self.ebm: self.ebm.eval()
        if self.flow: self.flow.eval()
        if self.mdn: self.mdn.eval()
        if self.direct: self.direct.eval()
        return metrics

    def freeze_for_rollout(self):
        if self.ebm and not getattr(self, '_needs_ebm_grad', False):
            for p in self.ebm.parameters(): p.requires_grad = False
        if self.flow:
            for p in self.flow.parameters(): p.requires_grad = False
        if self.mdn:
            for p in self.mdn.parameters(): p.requires_grad = False
        if self.direct:
            for p in self.direct.parameters(): p.requires_grad = False

    def unfreeze_after_rollout(self):
        for m in [self.ebm, self.flow, self.mdn, self.direct]:
            if m:
                for p in m.parameters(): p.requires_grad = True
        self.wm_optimizer.zero_grad()

    def predict_next_state(self, state, action):
        if self.agent_type == "EBM (IW)":
            context = torch.cat([state, action], dim=1)
            def proposal_fn(B, N):
                z = torch.randn(B, N, self.state_dim, device=self.device)
                ctx_exp = context.unsqueeze(1).expand(B, N, -1)
                samples = self.flow.sample(
                    z.reshape(B * N, -1), context=ctx_exp.reshape(B * N, -1))
                return samples.reshape(B, N, -1)
            return importance_weighted_sample(
                self.ebm, state, action, proposal_fn,
                config={"IW_NUM_SAMPLES": CONFIG["IW_NUM_SAMPLES"],
                        "IW_TEMPERATURE": CONFIG["IW_TEMPERATURE"]})

        elif self.agent_type == "EBM (Langevin)":
            return langevin_refine(self.ebm, state, action, self.flow,
                config={"LANGEVIN_STEPS": CONFIG["LANGEVIN_STEPS"],
                        "LANGEVIN_STEP_SIZE": CONFIG["LANGEVIN_STEP_SIZE"],
                        "LANGEVIN_NOISE_SCALE": CONFIG["LANGEVIN_NOISE_SCALE"]},
                differentiable=True)

        elif self.agent_type == "EBM (SVGD)":
            return svgd_sample(self.ebm, state, action,
                config={"SVGD_NUM_PARTICLES": CONFIG["SVGD_NUM_PARTICLES"],
                        "SVGD_STEPS": CONFIG["SVGD_STEPS"],
                        "SVGD_STEP_SIZE": CONFIG["SVGD_STEP_SIZE"],
                        "SVGD_BANDWIDTH": CONFIG["SVGD_BANDWIDTH"]},
                differentiable=True)

        elif self.agent_type == "Flow":
            B = state.shape[0]
            z = torch.randn(B, self.state_dim, device=self.device)
            context = torch.cat([state, action], dim=1)
            return self.flow.sample(z, context=context)

        elif self.agent_type == "MDN":
            return self.mdn.sample_differentiable(state, action)

        elif self.agent_type == "Direct":
            return self.direct(state, action)

        else:
            raise ValueError(f"Unknown: {self.agent_type}")

    def count_parameters(self):
        """Count total trainable parameters in world model components."""
        total = 0
        for m in [self.ebm, self.flow, self.mdn, self.direct]:
            if m is not None:
                total += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_policy(actor, num_episodes=10):
    """Returns: (mean_reward, std_reward, goals_reached, collisions)"""
    env = make_minigrid_env(
        size=CONFIG["GRID_SIZE"], n_obstacles=CONFIG["N_OBSTACLES"],
        slip_prob=CONFIG["SLIP_PROB"], max_steps=CONFIG["MAX_STEPS"])
    device = CONFIG["DEVICE"]
    results = []
    goals, collisions = 0, 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_r, done = 0.0, False
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_logits = actor.sample(s_t).cpu().numpy()[0]
            action_int = continuous_to_discrete(action_logits)
            state, r, term, trunc, info = env.step(action_int)
            ep_r += r
            done = term or trunc
        results.append(ep_r)
        if info.get("goal_reached"): goals += 1
        elif info.get("collision"): collisions += 1
    env.close()
    return np.mean(results), np.std(results), goals, collisions


# =============================================================================
# TRAIN ONE AGENT (single seed)
# =============================================================================
def train_agent(agent_type, horizon, seed=42):
    device = CONFIG["DEVICE"]

    # Set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    env = make_minigrid_env(
        size=CONFIG["GRID_SIZE"], n_obstacles=CONFIG["N_OBSTACLES"],
        slip_prob=CONFIG["SLIP_PROB"], max_steps=CONFIG["MAX_STEPS"])
    state_dim = env.state_dim
    action_dim = env.action_dim

    print(f"\n{'='*60}")
    print(f"Agent: {agent_type} | Horizon: {horizon} | Seed: {seed}")
    print(f"state_dim={state_dim}, action_dim={action_dim}")
    print(f"{'='*60}")

    agent = WorldModelAgent(agent_type, state_dim, action_dim, device)
    actor = Actor(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"],
                  action_scale=CONFIG["ACTION_SCALE"]).to(device)
    critic = ValueNetwork(state_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target = deepcopy(critic)
    for p in critic_target.parameters(): p.requires_grad = False
    actor_opt = optim.Adam(actor.parameters(), lr=CONFIG["LR_ACTOR"])
    critic_opt = optim.Adam(critic.parameters(), lr=CONFIG["LR_CRITIC"])

    buffer = ReplayBuffer(state_dim, action_dim)
    trust_comp = TrustComputer(buffer, CONFIG["TRUST_THRESHOLD"], CONFIG["TRUST_SHARPNESS"])

    # Log parameter count
    wm_params = agent.count_parameters()
    print(f"World model parameters: {wm_params:,}")

    print("Collecting 2000 random transitions...")
    state, _ = env.reset()
    for _ in range(2000):
        a_int = env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a_int)
        buffer.add(state, discrete_to_onehot(a_int, action_dim), r, ns)
        state = ns if not (term or trunc) else env.reset()[0]

    eval_steps, eval_rewards = [], []
    diag_log, trust_log = [], []
    timing = {
        "wm_update": [], "reward_update": [], "rollout_predict": [],
        "critic_update": [], "actor_update": [], "eval": [], "wall_clock": [],
        "total_predict_calls": 0,
    }
    train_start_time = time.time()

    state, _ = env.reset()
    total_steps = 0

    while total_steps < CONFIG["TOTAL_STEPS"]:
        # 1. REAL STEP
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_logits = actor.sample(s_t).cpu().numpy()[0]
        action_int = continuous_to_discrete(action_logits)
        next_state, reward, term, trunc, _ = env.step(action_int)
        buffer.add(state, discrete_to_onehot(action_int, action_dim), reward, next_state)
        total_steps += 1
        state = next_state if not (term or trunc) else env.reset()[0]

        # 2. WM UPDATE
        if total_steps % CONFIG["WM_UPDATE_EVERY"] == 0:
            _t0 = time.time()
            wm_metrics = agent.update_world_model(buffer)
            timing["wm_update"].append(time.time() - _t0)
            if total_steps % CONFIG["LOG_INTERVAL"] == 0 and wm_metrics:
                parts = [f"{k}={v:.4f}" for k, v in wm_metrics.items()]
                print(f"  [WM Update] {' | '.join(parts)}")

        # 3. IMAGINED ROLLOUT + CRITIC + ACTOR
        if buffer.size >= CONFIG["BATCH_SIZE"] and total_steps % 5 == 0:
            B = CONFIG["BATCH_SIZE"]; H = horizon
            agent.freeze_for_rollout()

            curr = buffer.sample_states(B)
            i_states, i_rewards, i_next, i_trust = [], [], [], []
            _t_predict = 0.0
            for t in range(H):
                a = actor.sample(curr)
                _tp0 = time.time()
                ns = agent.predict_next_state(curr, a)
                _t_predict += time.time() - _tp0
                r = minigrid_analytical_reward(curr, ns, a)
                w = trust_comp.compute_trust(ns)
                i_states.append(curr); i_rewards.append(r)
                i_next.append(ns); i_trust.append(w)
                curr = ns

            timing["total_predict_calls"] += H

            rew_t = torch.stack(i_rewards, dim=1)
            ns_t = torch.stack(i_next, dim=1)
            st_t = torch.stack(i_states, dim=1)
            tr_t = torch.cat(i_trust, dim=1)

            _t0 = time.time()
            with torch.no_grad():
                nv = critic_target(ns_t.reshape(B*H, -1)).squeeze(-1).reshape(B, H)
                targets = compute_lambda_returns(rew_t.detach(), nv, tr_t.detach(),
                                                 CONFIG["DISCOUNT"], CONFIG["LAMBDA"])
            vpred = critic(st_t.reshape(B*H, -1).detach()).squeeze(-1).reshape(B, H)
            critic_loss = F.mse_loss(vpred, targets)
            critic_opt.zero_grad(); critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); critic_opt.step()
            timing["critic_update"].append(time.time() - _t0)

            _t0 = time.time()
            for p in critic.parameters(): p.requires_grad = False
            curr_a = buffer.sample_states(B)
            ret = torch.zeros(B, device=device)
            disc = torch.ones(B, device=device)
            for t in range(H):
                a = actor.sample(curr_a)
                _tp0 = time.time()
                ns = agent.predict_next_state(curr_a, a)
                _t_predict += time.time() - _tp0
                r = minigrid_analytical_reward(curr_a, ns, a)
                w = trust_comp.compute_trust(ns).squeeze(-1)
                mu, log_std = actor(curr_a)
                std = torch.exp(torch.clamp(log_std, -5, 0.5))
                ent = 0.5 * torch.log(2 * np.pi * np.e * std.pow(2)).sum(dim=-1)
                ret = ret + disc * (r + CONFIG["ENTROPY_COEFF"] * ent)
                disc = disc * CONFIG["DISCOUNT"] * w
                curr_a = ns

            timing["total_predict_calls"] += H

            ret = ret + disc * critic(curr_a).squeeze(-1)
            actor_loss = -ret.mean()
            actor_opt.zero_grad(); actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0); actor_opt.step()
            timing["actor_update"].append(time.time() - _t0)
            timing["rollout_predict"].append(_t_predict)

            for p in critic.parameters(): p.requires_grad = True
            agent.unfreeze_after_rollout()
            with torch.no_grad():
                for p, tp in zip(critic.parameters(), critic_target.parameters()):
                    tp.data.copy_(CONFIG["TAU"] * p.data + (1 - CONFIG["TAU"]) * tp.data)

            if total_steps % CONFIG["LOG_INTERVAL"] == 0:
                ts = trust_comp.compute_rollout_stats(i_trust)
                trust_log.append({"step": total_steps, **ts})
                timing["wall_clock"].append({"step": total_steps,
                                             "elapsed": time.time() - train_start_time})
                bystep = " ".join(f"{v:.2f}" for v in ts["trust_by_step"])
                print(f"  [Step {total_steps}] Critic: {critic_loss.item():.8f} | "
                      f"Actor: {actor_loss.item():.4f} | "
                      f"Trust: {ts['pct_trusted']:.1f}% | By step: [{bystep}]")

        # 4. DIAGNOSTICS
        if total_steps % CONFIG["DIAGNOSTIC_INTERVAL"] == 0:
            diag = evaluate_prediction_accuracy(agent, buffer)
            if diag:
                diag_log.append({"step": total_steps, **diag})
                print(f"  [Diag] 1-step MSE: {diag['mse']:.6f}")

        # 5. EVAL
        if total_steps % CONFIG["EVAL_INTERVAL"] == 0:
            _t0 = time.time()
            em, es, goals, cols = evaluate_policy(actor, CONFIG["EVAL_EPISODES"])
            timing["eval"].append(time.time() - _t0)
            eval_steps.append(total_steps)
            eval_rewards.append(em)
            n_ep = CONFIG["EVAL_EPISODES"]
            print(f"Step {total_steps}/{CONFIG['TOTAL_STEPS']} | "
                  f"Eval: {em:.1f} +/- {es:.1f} | "
                  f"Goals: {goals}/{n_ep} | Collisions: {cols}/{n_ep}")

    env.close()
    total_wall = time.time() - train_start_time
    print(f"\n  [{agent_type} H={horizon} seed={seed}] Wall time: {total_wall:.1f}s")

    def _mean(lst): return np.mean(lst) if lst else 0.0
    def _sum(lst): return np.sum(lst) if lst else 0.0
    print(f"    predict: {_sum(timing['rollout_predict']):.1f}s "
          f"({_mean(timing['rollout_predict'])*1000:.2f}ms/call)")
    print(f"    wm:      {_sum(timing['wm_update']):.1f}s")
    print(f"    critic:  {_sum(timing['critic_update']):.1f}s")
    print(f"    actor:   {_sum(timing['actor_update']):.1f}s")
    print(f"    total predict calls: {timing['total_predict_calls']}")
    print(f"    WM parameters: {wm_params:,}")

    return {"eval_steps": eval_steps, "eval_rewards": eval_rewards,
            "diagnostic_log": diag_log, "rollout_quality_log": trust_log,
            "timing": timing, "total_wall_time": total_wall,
            "wm_params": wm_params}


# =============================================================================
# MULTI-SEED AGGREGATION
# =============================================================================
def aggregate_seeds(seed_results):
    """
    Aggregate results across seeds into mean +/- std.

    Args:
        seed_results: list of dicts from train_agent (one per seed)

    Returns:
        dict with eval_steps, eval_rewards_mean, eval_rewards_std,
        and aggregated timing/diagnostic info
    """
    if not seed_results:
        return None

    # Use the eval_steps from the first seed as reference
    ref_steps = seed_results[0]["eval_steps"]

    # Interpolate all seeds to same steps and compute mean/std
    all_rewards = []
    for r in seed_results:
        if len(r["eval_steps"]) == len(ref_steps):
            all_rewards.append(r["eval_rewards"])
        else:
            interp = np.interp(ref_steps, r["eval_steps"], r["eval_rewards"])
            all_rewards.append(interp.tolist())

    all_rewards = np.array(all_rewards)  # (num_seeds, num_eval_points)
    mean_rewards = all_rewards.mean(axis=0)
    std_rewards = all_rewards.std(axis=0)

    # Aggregate timing across seeds (take mean)
    avg_wall = np.mean([r["total_wall_time"] for r in seed_results])

    # Detailed timing aggregation
    timing_agg = {}
    for key in ["wm_update", "critic_update", "actor_update", "rollout_predict"]:
        per_seed_totals = [np.sum(r["timing"][key]) if r["timing"][key] else 0
                           for r in seed_results]
        timing_agg[f"{key}_mean"] = np.mean(per_seed_totals)
        timing_agg[f"{key}_std"] = np.std(per_seed_totals)

    # Per-call predict cost
    avg_predict_ms = np.mean([
        np.mean(r["timing"]["rollout_predict"]) * 1000
        if r["timing"]["rollout_predict"] else 0
        for r in seed_results
    ])
    avg_predict_std_ms = np.std([
        np.mean(r["timing"]["rollout_predict"]) * 1000
        if r["timing"]["rollout_predict"] else 0
        for r in seed_results
    ])

    return {
        "eval_steps": ref_steps,
        "eval_rewards_mean": mean_rewards.tolist(),
        "eval_rewards_std": std_rewards.tolist(),
        "total_wall_time_mean": avg_wall,
        "avg_predict_ms": avg_predict_ms,
        "avg_predict_std_ms": avg_predict_std_ms,
        "timing_agg": timing_agg,
        "wm_params": seed_results[0].get("wm_params", 0),
        "seed_results": seed_results,
    }


# =============================================================================
# MAIN + PLOTTING
# =============================================================================
def main():
    print(f"\n{'#'*60}")
    print(f"MINIGRID DYNAMIC-OBSTACLES MBRL POLICY CONVERGENCE")
    print(f"Grid: {CONFIG['GRID_SIZE']}x{CONFIG['GRID_SIZE']}, "
          f"obstacles={CONFIG['N_OBSTACLES']}, slip={CONFIG['SLIP_PROB']}")
    print(f"Critic on imagined data ONLY")
    print(f"Seeds: {CONFIG['SEEDS']}")
    print(f"{'#'*60}\n")

    agent_types = ["Direct", "EBM (IW)", "EBM (Langevin)", "MDN", "EBM (SVGD)"]
    horizons = [1, 3, 5]
    seeds = CONFIG["SEEDS"]

    # --- One-time reward sanity check ---
    print("Running reward sanity check (analytical vs env)...")
    rsc = reward_sanity_check(num_samples=2000)
    print(f"  MAE={rsc['mae']:.4f} | RMSE={rsc['rmse']:.4f} | "
          f"Corr={rsc['corr']:.4f} | Sign agree={rsc['sign_agreement']:.2%}")
    print(f"  Env mean={rsc['actual_mean']:.4f} | "
          f"Analytical mean={rsc['analytical_mean']:.4f}")
    if rsc["corr"] < 0.9:
        print("  WARNING: Low correlation — analytical reward may not match env!")
    else:
        print("  OK: Analytical reward matches env rewards.")
    print()

    all_results = {}
    for horizon in horizons:
        for agent_type in agent_types:
            key = f"{agent_type} H={horizon}"
            seed_runs = []
            for seed in seeds:
                try:
                    result = train_agent(agent_type, horizon, seed=seed)
                    seed_runs.append(result)
                except Exception as e:
                    print(f"!!! FAILED: {key} seed={seed} -- {e}")
                    import traceback; traceback.print_exc()

            if seed_runs:
                all_results[key] = aggregate_seeds(seed_runs)

    # ===== PLOTTING =====
    colors = {
        "EBM (IW)": "tab:red", "EBM (Langevin)": "tab:orange",
        "EBM (SVGD)": "tab:purple", "Flow": "tab:blue",
        "MDN": "tab:green", "Direct": "tab:brown",
    }

    # -------------------------------------------------------------------------
    # Plot 1: Convergence (with shaded std bands)
    # -------------------------------------------------------------------------
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
                steps = d["eval_steps"]
                mean = np.array(d["eval_rewards_mean"])
                std = np.array(d["eval_rewards_std"])
                ax.plot(steps, mean, label=at, color=colors[at], linewidth=2)
                ax.fill_between(steps, mean - std, mean + std,
                                color=colors[at], alpha=0.15)
        ax.legend(fontsize=9)
    plt.suptitle(f"Policy Convergence — Dynamic-Obstacles-{CONFIG['GRID_SIZE']}x{CONFIG['GRID_SIZE']} "
                 f"({CONFIG['N_OBSTACLES']} obs, slip={CONFIG['SLIP_PROB']}) "
                 f"[{len(seeds)} seeds]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("minigrid_convergence.png", dpi=200, bbox_inches="tight")
    print("\nSaved: minigrid_convergence.png")

    # -------------------------------------------------------------------------
    # Plot 2: Trust (use first seed for trust curves)
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 4), sharey=True)
    if len(horizons) == 1: axes2 = [axes2]
    for i, h in enumerate(horizons):
        ax = axes2[i]
        ax.set_title(f"Rollout Trust -- H={h}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps"); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)
        if i == 0: ax.set_ylabel("% Trusted Steps")
        for at in agent_types:
            k = f"{at} H={h}"
            if k in all_results:
                sr = all_results[k]["seed_results"][0]
                if sr["rollout_quality_log"]:
                    rq = sr["rollout_quality_log"]
                    ax.plot([r["step"] for r in rq], [r["pct_trusted"] for r in rq],
                            label=at, color=colors[at], linewidth=2)
        ax.legend(fontsize=9)
    plt.suptitle("Rollout Quality Over Training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("minigrid_trust.png", dpi=200, bbox_inches="tight")
    print("Saved: minigrid_trust.png")

    # -------------------------------------------------------------------------
    # Plot 3: MSE
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.set_title("1-Step Prediction MSE", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Steps"); ax3.set_ylabel("MSE"); ax3.set_yscale("log"); ax3.grid(True, alpha=0.3)
    for at in agent_types:
        k = f"{at} H=1"
        if k in all_results:
            sr = all_results[k]["seed_results"][0]
            if sr["diagnostic_log"]:
                dl = sr["diagnostic_log"]
                ax3.plot([d["step"] for d in dl], [d["mse"] for d in dl],
                         label=at, color=colors[at], linewidth=2)
    ax3.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("minigrid_mse.png", dpi=200, bbox_inches="tight")
    print("Saved: minigrid_mse.png")

    # -------------------------------------------------------------------------
    # Plot 4: Compute cost (ENHANCED — 2x2 grid)
    # -------------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))
    timing_horizon = 3 if 3 in horizons else horizons[0]

    # --- 4a: Wall-clock breakdown (stacked bar) ---
    ax4a = axes4[0, 0]
    bar_agents, bar_wall = [], []
    bar_predict, bar_critic, bar_actor, bar_wm, bar_other = [], [], [], [], []
    per_step_predict_mean, per_step_predict_std = [], []
    bar_params = []
    for at in agent_types:
        k = f"{at} H={timing_horizon}"
        if k not in all_results: continue
        d = all_results[k]
        ta = d.get("timing_agg", {})
        total_w = d.get("total_wall_time_mean", 0)
        t_predict = ta.get("rollout_predict_mean", 0)
        t_critic = ta.get("critic_update_mean", 0)
        t_actor = ta.get("actor_update_mean", 0)
        t_wm = ta.get("wm_update_mean", 0)

        bar_agents.append(at); bar_wall.append(total_w)
        bar_predict.append(t_predict); bar_critic.append(t_critic)
        bar_actor.append(t_actor); bar_wm.append(t_wm)
        bar_other.append(max(total_w - t_predict - t_critic - t_actor - t_wm, 0))
        bar_params.append(d.get("wm_params", 0))
        per_step_predict_mean.append(d.get("avg_predict_ms", 0))
        per_step_predict_std.append(d.get("avg_predict_std_ms", 0))

    if bar_agents:
        x = np.arange(len(bar_agents)); width = 0.6
        ax4a.bar(x, bar_predict, width, label="predict", color="tab:red", alpha=0.85)
        ax4a.bar(x, bar_actor, width, bottom=np.array(bar_predict),
                 label="actor", color="tab:orange", alpha=0.85)
        ax4a.bar(x, bar_critic, width,
                 bottom=np.array(bar_predict)+np.array(bar_actor),
                 label="critic", color="tab:blue", alpha=0.85)
        ax4a.bar(x, bar_wm, width,
                 bottom=np.array(bar_predict)+np.array(bar_actor)+np.array(bar_critic),
                 label="WM update", color="tab:green", alpha=0.85)
        ax4a.bar(x, bar_other, width,
                 bottom=np.array(bar_predict)+np.array(bar_actor)+np.array(bar_critic)+np.array(bar_wm),
                 label="other", color="tab:gray", alpha=0.5)
        ax4a.set_xticks(x)
        ax4a.set_xticklabels(bar_agents, rotation=20, ha="right", fontsize=9)
        ax4a.set_ylabel("Time (s)")
        ax4a.set_title(f"Wall-Clock Breakdown (H={timing_horizon}, avg {len(seeds)} seeds)",
                       fontweight="bold")
        ax4a.legend(fontsize=8); ax4a.grid(True, alpha=0.3, axis="y")
        for i_bar, tw in enumerate(bar_wall):
            ax4a.text(i_bar, tw+0.5, f"{tw:.0f}s", ha="center", fontsize=9, fontweight="bold")

    # --- 4b: Per-call predict cost ---
    ax4b = axes4[0, 1]
    if bar_agents:
        bar_colors = [colors.get(at, "tab:gray") for at in bar_agents]
        ax4b.bar(x, per_step_predict_mean, width, yerr=per_step_predict_std,
                 color=bar_colors, alpha=0.85, capsize=4)
        ax4b.set_xticks(x)
        ax4b.set_xticklabels(bar_agents, rotation=20, ha="right", fontsize=9)
        ax4b.set_ylabel("Time (ms)")
        ax4b.set_title(f"Per-Call Predict Cost (H={timing_horizon})", fontweight="bold")
        ax4b.grid(True, alpha=0.3, axis="y")
        for i_bar, (m, s) in enumerate(zip(per_step_predict_mean, per_step_predict_std)):
            ax4b.text(i_bar, m + s + 0.1, f"{m:.2f}ms", ha="center", fontsize=8)

    # --- 4c: Parameter count comparison ---
    ax4c = axes4[1, 0]
    if bar_agents:
        param_colors = [colors.get(at, "tab:gray") for at in bar_agents]
        ax4c.bar(x, [p / 1000 for p in bar_params], width,
                 color=param_colors, alpha=0.85)
        ax4c.set_xticks(x)
        ax4c.set_xticklabels(bar_agents, rotation=20, ha="right", fontsize=9)
        ax4c.set_ylabel("Parameters (K)")
        ax4c.set_title("World Model Parameter Count", fontweight="bold")
        ax4c.grid(True, alpha=0.3, axis="y")
        for i_bar, p in enumerate(bar_params):
            ax4c.text(i_bar, p/1000 + 0.5, f"{p:,}", ha="center", fontsize=8)

    # --- 4d: Cost-efficiency (reward per second) ---
    ax4d = axes4[1, 1]
    if bar_agents:
        final_rewards = []
        for at in bar_agents:
            k = f"{at} H={timing_horizon}"
            if k in all_results:
                d = all_results[k]
                final_rewards.append(d["eval_rewards_mean"][-1] if d["eval_rewards_mean"] else 0)
            else:
                final_rewards.append(0)
        efficiency = [r / max(w, 1) for r, w in zip(final_rewards, bar_wall)]
        bar_colors = [colors.get(at, "tab:gray") for at in bar_agents]
        ax4d.bar(x, efficiency, width, color=bar_colors, alpha=0.85)
        ax4d.set_xticks(x)
        ax4d.set_xticklabels(bar_agents, rotation=20, ha="right", fontsize=9)
        ax4d.set_ylabel("Final Reward / Wall Time (reward/s)")
        ax4d.set_title("Cost-Efficiency", fontweight="bold")
        ax4d.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Computational Cost Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("minigrid_compute.png", dpi=200, bbox_inches="tight")
    print("Saved: minigrid_compute.png")

    # -------------------------------------------------------------------------
    # Plot 5: Wall-clock (reward vs wall time)
    # -------------------------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    ax5.set_title(f"Reward vs Wall Time (H={timing_horizon})", fontweight="bold")
    ax5.set_xlabel("Wall Time (s)"); ax5.set_ylabel("Eval Reward"); ax5.grid(True, alpha=0.3)
    for at in agent_types:
        k = f"{at} H={timing_horizon}"
        if k not in all_results: continue
        d = all_results[k]
        sr = d["seed_results"][0]
        wc = sr["timing"]["wall_clock"]
        if not wc or not sr["eval_steps"]: continue
        wc_steps = [w["step"] for w in wc]; wc_times = [w["elapsed"] for w in wc]
        if len(wc_steps) >= 2:
            eval_times = np.interp(sr["eval_steps"], wc_steps, wc_times)
            mean_rewards = np.array(d["eval_rewards_mean"])
            std_rewards = np.array(d["eval_rewards_std"])
            ax5.plot(eval_times, mean_rewards, label=at, color=colors[at], linewidth=2)
            ax5.fill_between(eval_times, mean_rewards - std_rewards,
                             mean_rewards + std_rewards, color=colors[at], alpha=0.15)
    ax5.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("minigrid_wallclock.png", dpi=200, bbox_inches="tight")
    print("Saved: minigrid_wallclock.png")

    # -------------------------------------------------------------------------
    # Print detailed compute table
    # -------------------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"DETAILED COMPUTATIONAL COST TABLE (H={timing_horizon}, {len(seeds)} seeds)")
    print(f"{'='*90}")
    print(f"{'Agent':<18} {'Params':>10} {'Wall(s)':>10} {'Predict(s)':>12} "
          f"{'WM(s)':>10} {'Critic(s)':>10} {'Actor(s)':>10} {'ms/pred':>10}")
    print("-" * 90)
    for at in agent_types:
        k = f"{at} H={timing_horizon}"
        if k not in all_results: continue
        d = all_results[k]
        ta = d.get("timing_agg", {})
        print(f"{at:<18} "
              f"{d.get('wm_params', 0):>10,} "
              f"{d.get('total_wall_time_mean', 0):>10.1f} "
              f"{ta.get('rollout_predict_mean', 0):>12.1f} "
              f"{ta.get('wm_update_mean', 0):>10.1f} "
              f"{ta.get('critic_update_mean', 0):>10.1f} "
              f"{ta.get('actor_update_mean', 0):>10.1f} "
              f"{d.get('avg_predict_ms', 0):>10.2f}")
    print(f"{'='*90}")

    np.save("minigrid_results.npy", all_results, allow_pickle=True)
    print("Saved: minigrid_results.npy\nDone!")


if __name__ == "__main__":
    main()