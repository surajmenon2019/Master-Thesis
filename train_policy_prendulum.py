"""
Policy Convergence Training — Pendulum-v1

STANDARD MBRL SETUP:
  World models are pretrained on random data, then trained ONLINE on
  REAL environment transitions (standard model-based RL).

  The actor and critic train SOLELY on IMAGINED rollouts through the
  current world model.

  This is the classic Dyna-style loop:
    1. Collect real transitions -> train world model
    2. Imagine rollouts through world model -> train actor + critic
    3. Repeat

  Real data is used for:
    - World model training (online, from real replay buffer)
    - Reward model training (needs ground-truth rewards)
    - Evaluation diagnostics

  Critic and Actor train on imagined rollouts only.

Agents:
  1. EBM (Langevin)  — Flow init + EBM Langevin refinement
  2. Flow            — RealNVP direct sampling
  3. MDN             — Mixture Density Network direct sampling

Outputs:
  1. pendulum_convergence.png  — eval reward vs steps, per horizon
  2. pendulum_mse.png          — 1-step prediction MSE over training
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy

from models import (
    Actor, ValueNetwork, TwinCritic, BilinearEBM, RealNVP,
    MixtureDensityNetwork, RewardModel
)
from utils_sampling import langevin_refine

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
    "LR_WORLD_MODEL": 1e-4,

    "ENTROPY_COEFF": 0.01,

    # World model online update schedule (on REAL data)
    "WM_UPDATE_EVERY": 50,
    "WM_UPDATE_BATCH": 128,

    # Langevin
    "LANGEVIN_STEPS": 15,
    "LANGEVIN_STEP_SIZE": 0.01,
    "LANGEVIN_NOISE_SCALE": 0.005,

    # InfoNCE (EBM online update)
    "NUM_NEGATIVES": 256,
    "INFONCE_TEMPERATURE": 0.1,

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
# REAL REPLAY BUFFER (Updated to track state statistics for TrustComputer)
# =============================================================================
class ReplayBuffer:
    """Stores REAL environment transitions and computes rolling stats."""
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
        
        # Tracking running statistics for Z-score Trust filtering
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
# TRUST COMPUTER (The Safeguard against Model Exploitation)
# =============================================================================
class TrustComputer:
    """Calculates trust weight based on Z-score distance from real states."""
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


# =============================================================================
# TD(lambda) RETURNS (Updated to penalize using Trust weights)
# =============================================================================
def compute_lambda_returns(rewards, values, trust, discount, lam):
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)
    for t in reversed(range(H)):
        # THE FIX: Trust exponentially decays the discount factor for OOD states
        eff_gamma = discount * trust[:, t] 
        if t == H - 1:
            returns[:, t] = rewards[:, t] + eff_gamma * values[:, t]
        else:
            td1 = rewards[:, t] + eff_gamma * values[:, t]
            cont = rewards[:, t] + eff_gamma * returns[:, t + 1]
            returns[:, t] = (1 - lam) * td1 + lam * cont
    return returns


# =============================================================================
# SYMLOG / SYMEXP
# =============================================================================
def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# =============================================================================
# PREDICTION ACCURACY MONITOR
# =============================================================================
@torch.no_grad()
def evaluate_prediction_accuracy(agent, buffer, num_samples=512):
    if buffer.size < num_samples:
        return None
    batch = buffer.sample(num_samples)
    pred = agent.predict_next_state(batch["states"], batch["actions"], differentiable=False)
    mse = F.mse_loss(pred, batch["next_states"]).item()
    per_dim = ((pred - batch["next_states"]) ** 2).mean(dim=0).cpu().numpy()
    return {"mse": mse, "per_dim_mse": per_dim}


# =============================================================================
# WORLD MODEL AGENT
# =============================================================================
class WorldModelAgent:
    def __init__(self, agent_type, state_dim, action_dim, device):
        self.agent_type = agent_type
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        hd = CONFIG["HIDDEN_DIM"]
        lr = CONFIG["LR_WORLD_MODEL"]

        self.ebm = None
        self.flow = None
        self.mdn = None

        if agent_type == "EBM (Langevin)":
            self.ebm = BilinearEBM(state_dim, action_dim, hidden_dim=hd).to(device)
            self.ebm.load_state_dict(torch.load(
                "pretrained_ebm_pendulum.pth",
                map_location=device, weights_only=True))

            self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=hd).to(device)
            self.flow.load_state_dict(torch.load(
                "pretrained_flow_pendulum.pth",
                map_location=device, weights_only=True))

            self.wm_optimizer = optim.Adam(
                list(self.ebm.parameters()) + list(self.flow.parameters()), lr=lr
            )

        elif agent_type == "Flow":
            self.flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=hd).to(device)
            self.flow.load_state_dict(torch.load(
                "pretrained_flow_pendulum.pth",
                map_location=device, weights_only=True))

            self.wm_optimizer = optim.Adam(self.flow.parameters(), lr=lr)

        elif agent_type == "MDN":
            self.mdn = MixtureDensityNetwork(
                state_dim, action_dim,
                num_gaussians=CONFIG["MDN_NUM_GAUSSIANS"],
                hidden_dim=hd
            ).to(device)
            self.mdn.load_state_dict(torch.load(
                "pretrained_mdn_pendulum.pth",
                map_location=device, weights_only=True))

            self.wm_optimizer = optim.Adam(self.mdn.parameters(), lr=lr)

    def predict_next_state(self, state, action, differentiable=True):
        if self.agent_type == "EBM (Langevin)":
            with torch.enable_grad():
                return langevin_refine(
                    self.ebm, state, action, self.flow,
                    config={
                        "LANGEVIN_STEPS": CONFIG["LANGEVIN_STEPS"],
                        "LANGEVIN_STEP_SIZE": CONFIG["LANGEVIN_STEP_SIZE"],
                        "LANGEVIN_NOISE_SCALE": CONFIG["LANGEVIN_NOISE_SCALE"],
                    },
                    differentiable=differentiable,
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

    def update_world_model(self, real_buffer):
        BS = CONFIG["WM_UPDATE_BATCH"]
        if real_buffer.size < BS:
            return {}

        batch = real_buffer.sample(BS)
        s = batch["states"]
        a = batch["actions"]
        ns = batch["next_states"]

        B = s.shape[0]
        metrics = {}

        self.wm_optimizer.zero_grad()
        loss = torch.tensor(0.0, device=self.device)

        if self.ebm is not None:
            self.ebm.train()
            E_pos = self.ebm(s, a, ns)
            neg_idx = np.random.randint(0, real_buffer.size, (B, CONFIG["NUM_NEGATIVES"]))
            neg_ns = torch.tensor(
                real_buffer.next_states[neg_idx], device=self.device
            )
            s_exp = s.unsqueeze(1).expand(B, CONFIG["NUM_NEGATIVES"], -1)
            a_exp = a.unsqueeze(1).expand(B, CONFIG["NUM_NEGATIVES"], -1)
            E_neg = self.ebm(s_exp, a_exp, neg_ns)

            logits = torch.cat([E_pos.unsqueeze(1), E_neg], dim=1) / CONFIG["INFONCE_TEMPERATURE"]
            labels = torch.zeros(B, dtype=torch.long, device=self.device)
            ebm_loss = F.cross_entropy(logits, labels)
            loss = loss + ebm_loss

            with torch.no_grad():
                metrics["ebm_loss"] = ebm_loss.item()
                metrics["E_gap"] = (E_pos.mean() - E_neg.mean()).item()
                metrics["ebm_acc"] = (logits.argmax(dim=1) == 0).float().mean().item()

        if self.flow is not None:
            self.flow.train()
            context = torch.cat([s, a], dim=1)
            log_prob = self.flow.log_prob(ns, context=context)
            flow_loss = -log_prob.mean() / self.state_dim
            loss = loss + flow_loss
            with torch.no_grad(): metrics["flow_loss"] = flow_loss.item()

        if self.mdn is not None:
            self.mdn.train()
            mdn_ll = self.mdn.log_prob(s, a, ns)
            mdn_loss = -mdn_ll.mean() / self.state_dim
            loss = loss + mdn_loss
            with torch.no_grad(): metrics["mdn_loss"] = mdn_loss.item()

        loss.backward()

        if self.ebm is not None: torch.nn.utils.clip_grad_norm_(self.ebm.parameters(), 1.0)
        if self.flow is not None: torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
        if self.mdn is not None: torch.nn.utils.clip_grad_norm_(self.mdn.parameters(), 1.0)

        self.wm_optimizer.step()

        if self.ebm is not None: self.ebm.eval()
        if self.flow is not None: self.flow.eval()
        if self.mdn is not None: self.mdn.eval()

        return metrics

    def freeze_for_rollout(self):
        if self.ebm is not None:
            if self.agent_type != "EBM (Langevin)":
                for p in self.ebm.parameters(): p.requires_grad = False
        if self.flow is not None:
            for p in self.flow.parameters(): p.requires_grad = False
        if self.mdn is not None:
            for p in self.mdn.parameters(): p.requires_grad = False

    def unfreeze_after_rollout(self):
        if self.ebm is not None:
            for p in self.ebm.parameters(): p.requires_grad = True
        if self.flow is not None:
            for p in self.flow.parameters(): p.requires_grad = True
        if self.mdn is not None:
            for p in self.mdn.parameters(): p.requires_grad = True
        self.wm_optimizer.zero_grad()


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

    agent = WorldModelAgent(agent_type, state_dim, action_dim, device)

    actor = Actor(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"],
                  action_scale=CONFIG["ACTION_SCALE"]).to(device)
    critic = TwinCritic(state_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)
    critic_target = deepcopy(critic)
    for p in critic_target.parameters():
        p.requires_grad = False

    reward_model = RewardModel(state_dim, action_dim, hidden_dim=CONFIG["HIDDEN_DIM"]).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=CONFIG["LR_ACTOR"])
    critic_opt = optim.Adam(critic.parameters(), lr=CONFIG["LR_CRITIC"])
    reward_opt = optim.Adam(reward_model.parameters(), lr=CONFIG["LR_REWARD"])

    real_buffer = ReplayBuffer(state_dim, action_dim)
    
    # Initialize TrustComputer
    trust_comp = TrustComputer(real_buffer, threshold=3.0, sharpness=2.0)

    print("Collecting 2000 random transitions for initial buffer...")
    state, _ = env.reset()
    for _ in range(2000):
        a = env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a)
        real_buffer.add(state, a, r, ns)
        state = ns if not (term or trunc) else env.reset()[0]

    eval_steps, eval_rewards = [], []
    diag_log = []

    state, _ = env.reset()
    total_steps = 0

    while total_steps < CONFIG["TOTAL_STEPS"]:
        # ==== 1. REAL ENVIRONMENT STEP ====
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = actor.sample(s_t).cpu().numpy()[0]
        action = np.clip(action, -CONFIG["ACTION_SCALE"], CONFIG["ACTION_SCALE"])
        next_state, reward, term, trunc, _ = env.step(action)
        real_buffer.add(state, action, reward, next_state)
        total_steps += 1
        state = next_state if not (term or trunc) else env.reset()[0]

        # ==== 2. WORLD MODEL UPDATE (REAL DATA) ====
        if total_steps % CONFIG["WM_UPDATE_EVERY"] == 0 and real_buffer.size >= CONFIG["WM_UPDATE_BATCH"]:
            wm_metrics = agent.update_world_model(real_buffer)
            if total_steps % CONFIG["LOG_INTERVAL"] == 0 and wm_metrics:
                parts = [f"{k}={v:.4f}" for k, v in wm_metrics.items()]
                print(f"  [WM Update (real)] {' | '.join(parts)}")

        # ==== 3. REWARD MODEL UPDATE ====
        if real_buffer.size >= CONFIG["BATCH_SIZE"] and total_steps % 5 == 0:
            batch = real_buffer.sample(CONFIG["BATCH_SIZE"])
            pred_r = reward_model(batch["states"], batch["actions"], batch["next_states"])
            r_loss = F.mse_loss(pred_r, batch["rewards"])
            reward_opt.zero_grad()
            r_loss.backward()
            reward_opt.step()

        # ==== 4. IMAGINED ROLLOUT + CRITIC + ACTOR (every 5 steps) ====
        if real_buffer.size >= CONFIG["BATCH_SIZE"] and total_steps % 5 == 0:
            B = CONFIG["BATCH_SIZE"]
            H = horizon

            agent.freeze_for_rollout()
            for p in reward_model.parameters():
                p.requires_grad = False

            # --- CRITIC ROLLOUT ---
            curr = real_buffer.sample_states(B)
            i_states, i_rewards, i_next, i_trust = [], [], [], []

            for t in range(H):
                a = actor.sample(curr)
                ns = agent.predict_next_state(curr, a)
                r = reward_model(curr, a, ns).squeeze(-1)
                
                # Compute trust for safeguarding
                w = trust_comp.compute_trust(ns).squeeze(-1)

                i_states.append(curr)
                i_rewards.append(r)
                i_next.append(ns)
                i_trust.append(w)
                curr = ns

            rew_t = torch.stack(i_rewards, dim=1)
            ns_t = torch.stack(i_next, dim=1)
            st_t = torch.stack(i_states, dim=1)
            tr_t = torch.stack(i_trust, dim=1)

            with torch.no_grad():
                nv = critic_target.min_value(
                    ns_t.reshape(B * H, -1)
                ).squeeze(-1).reshape(B, H)
                nv = symexp(nv)

                # Pass trust weights to correctly penalize the targets
                targets = compute_lambda_returns(
                    rew_t.detach(), nv, tr_t.detach(),
                    CONFIG["DISCOUNT"], CONFIG["LAMBDA"]
                )

            targets_norm = symlog(targets)

            v1, v2 = critic(st_t.reshape(B * H, -1).detach())
            v1 = v1.squeeze(-1).reshape(B, H)
            v2 = v2.squeeze(-1).reshape(B, H)

            critic_loss = F.mse_loss(v1, targets_norm) + F.mse_loss(v2, targets_norm)
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

            # --- ACTOR UPDATE ---
            for p in critic.parameters():
                p.requires_grad = False

            curr_a = real_buffer.sample_states(B)
            ret = torch.zeros(B, device=device)
            disc = torch.ones(B, device=device)

            for t in range(H):
                a = actor.sample(curr_a)
                ns = agent.predict_next_state(curr_a, a)
                r = reward_model(curr_a, a, ns).squeeze(-1)
                
                # Re-compute trust to cut off Actor's objective
                w = trust_comp.compute_trust(ns).squeeze(-1)

                mu, log_std = actor(curr_a)
                std = torch.exp(torch.clamp(log_std, -5, 0.5))
                ent = 0.5 * torch.log(2 * np.pi * np.e * std.pow(2)).sum(dim=-1)

                ret = ret + disc * (r + CONFIG["ENTROPY_COEFF"] * ent)
                
                # Penalty: discount is permanently zeroed if hallucination occurs
                disc = disc * CONFIG["DISCOUNT"] * w
                curr_a = ns

            v1_a, v2_a = critic(curr_a)
            v_boot = symexp(torch.min(v1_a, v2_a).squeeze(-1))
            ret = ret + disc * v_boot
            
            actor_loss = -symlog(ret).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()

            for p in critic.parameters():
                p.requires_grad = True
            for p in reward_model.parameters():
                p.requires_grad = True
            agent.unfreeze_after_rollout()

            with torch.no_grad():
                for p, tp in zip(critic.parameters(), critic_target.parameters()):
                    tp.data.copy_(CONFIG["TAU"] * p.data + (1 - CONFIG["TAU"]) * tp.data)

            if total_steps % CONFIG["LOG_INTERVAL"] == 0:
                print(f"  [Step {total_steps}] Critic: {critic_loss.item():.4f} | "
                      f"Actor: {actor_loss.item():.4f}")

        # ==== 5. PREDICTION DIAGNOSTIC ====
        if total_steps % CONFIG["DIAGNOSTIC_INTERVAL"] == 0:
            diag = evaluate_prediction_accuracy(agent, real_buffer)
            if diag:
                diag_log.append({"step": total_steps, **diag})
                print(f"  [Diag] 1-step MSE vs real: {diag['mse']:.6f}")

        # ==== 6. EVAL ====
        if total_steps % CONFIG["EVAL_INTERVAL"] == 0:
            em, es = evaluate_policy(actor, CONFIG["EVAL_EPISODES"])
            eval_steps.append(total_steps)
            eval_rewards.append(em)
            print(f"Step {total_steps}/{CONFIG['TOTAL_STEPS']} | "
                  f"Eval: {em:.1f} +/- {es:.1f}")

    env.close()
    return {
        "eval_steps": eval_steps,
        "eval_rewards": eval_rewards,
        "diagnostic_log": diag_log,
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"\n{'#'*60}")
    print(f"PENDULUM-v1 MBRL POLICY CONVERGENCE")
    print(f"World models train on REAL data online (standard MBRL)")
    print(f"Critic + Actor on imagined rollouts ONLY (Z-Score Safeguard)")
    print(f"{'#'*60}\n")

    agent_types = ["EBM (Langevin)", "Flow", "MDN"]
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
    colors = {"EBM (Langevin)": "tab:orange",
              "Flow": "tab:blue", "MDN": "tab:green"}

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
    plt.suptitle("Policy Convergence (WM on Real Data, Standard MBRL)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("pendulum_convergence.png", dpi=200, bbox_inches="tight")
    print("\nSaved: pendulum_convergence.png")

    # ===== PLOT 2: Prediction MSE =====
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.set_title("1-Step Prediction MSE (vs Real Data)",
                  fontsize=12, fontweight="bold")
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

    np.save("pendulum_results.npy", all_results, allow_pickle=True)
    print("Saved: pendulum_results.npy\nDone!")

if __name__ == "__main__":
    main()