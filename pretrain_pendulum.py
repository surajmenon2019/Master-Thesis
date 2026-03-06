"""
Pretrain world models (EBM, Flow, MDN) on random Pendulum-v1 data.

Single model per architecture (no ensemble).
Saves: pretrained_{ebm,flow,mdn}_pendulum.pth
       pretrained_config_pendulum.pth
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from models import BilinearEBM, RealNVP, MixtureDensityNetwork

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "COLLECT_STEPS": 10000,
    "TRAIN_STEPS": 5000,
    "BATCH_SIZE": 256,
    "LR_EBM": 3e-4,
    "LR_FLOW": 3e-4,
    "LR_MDN": 1e-3,
    "HIDDEN_DIM": 128,
    "NUM_NEGATIVES": 256,
    "TEMPERATURE": 0.1,
    "VAL_SPLIT": 0.1,
    "MDN_NUM_GAUSSIANS": 5,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# DATA COLLECTION
# =============================================================================
def collect_data(num_steps):
    """Collect random transitions. Returns numpy arrays."""
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]  # 3
    action_dim = env.action_space.shape[0]      # 1

    states = np.zeros((num_steps, state_dim), dtype=np.float32)
    actions = np.zeros((num_steps, action_dim), dtype=np.float32)
    next_states = np.zeros((num_steps, state_dim), dtype=np.float32)

    state, _ = env.reset()
    for i in range(num_steps):
        action = env.action_space.sample()
        next_state, _, terminated, truncated, _ = env.step(action)
        states[i] = state
        actions[i] = action
        next_states[i] = next_state
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    env.close()
    print(f"Collected {num_steps} transitions | state_dim={state_dim}, action_dim={action_dim}")
    return states, actions, next_states, state_dim, action_dim


# =============================================================================
# INFONCE LOSS
# =============================================================================
def infonce_loss(ebm, state, action, pos_ns, all_next_states,
                 num_negatives=256, temperature=0.1):
    B = state.shape[0]
    N = all_next_states.shape[0]
    device = state.device

    E_pos = ebm(state, action, pos_ns)
    neg_idx = torch.randint(0, N, (B, num_negatives), device=device)
    neg_ns = all_next_states[neg_idx]
    s_exp = state.unsqueeze(1).expand(B, num_negatives, -1)
    a_exp = action.unsqueeze(1).expand(B, num_negatives, -1)
    E_neg = ebm(s_exp, a_exp, neg_ns)

    logits = torch.cat([E_pos.unsqueeze(1), E_neg], dim=1) / temperature
    labels = torch.zeros(B, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels)

    acc = (logits.argmax(dim=1) == 0).float().mean().item()
    gap = (E_pos.mean() - E_neg.mean()).item()
    return loss, {"accuracy": acc, "E_gap": gap}


# =============================================================================
# VALIDATION
# =============================================================================
@torch.no_grad()
def validate_flow(flow, val_s, val_a, val_ns, state_dim):
    z = torch.randn_like(val_s)
    context = torch.cat([val_s, val_a], dim=1)
    pred = flow.sample(z, context=context)
    return F.mse_loss(pred, val_ns).item()

@torch.no_grad()
def validate_mdn(mdn, val_s, val_a, val_ns):
    pi, mu, sigma = mdn(val_s, val_a)
    best_k = pi.argmax(dim=1)
    pred = mu[torch.arange(mu.shape[0]), best_k]
    return F.mse_loss(pred, val_ns).item()


# =============================================================================
# TRAIN SINGLE MODELS
# =============================================================================
def train_models(train_s, train_a, train_ns, n_train,
                 val_s, val_a, val_ns, state_dim, action_dim, device):
    """Train one EBM + one Flow + one MDN."""
    torch.manual_seed(42)

    hd = CONFIG["HIDDEN_DIM"]

    ebm = BilinearEBM(state_dim, action_dim, hidden_dim=hd).to(device)
    flow = RealNVP(state_dim, context_dim=state_dim + action_dim, hidden_dim=hd).to(device)
    mdn = MixtureDensityNetwork(
        state_dim, action_dim,
        num_gaussians=CONFIG["MDN_NUM_GAUSSIANS"],
        hidden_dim=hd
    ).to(device)

    ebm_opt = optim.Adam(ebm.parameters(), lr=CONFIG["LR_EBM"])
    flow_opt = optim.Adam(flow.parameters(), lr=CONFIG["LR_FLOW"])
    mdn_opt = optim.Adam(mdn.parameters(), lr=CONFIG["LR_MDN"])

    BS = CONFIG["BATCH_SIZE"]

    for step in range(CONFIG["TRAIN_STEPS"]):
        idx = torch.randint(0, n_train, (BS,), device=device)
        s, a, ns = train_s[idx], train_a[idx], train_ns[idx]
        context = torch.cat([s, a], dim=1)

        # --- EBM (InfoNCE) ---
        ebm_loss, ebm_metrics = infonce_loss(
            ebm, s, a, ns, train_ns,
            num_negatives=CONFIG["NUM_NEGATIVES"],
            temperature=CONFIG["TEMPERATURE"]
        )
        ebm_opt.zero_grad()
        ebm_loss.backward()
        torch.nn.utils.clip_grad_norm_(ebm.parameters(), 1.0)
        ebm_opt.step()

        # --- Flow (Forward KL) ---
        log_prob = flow.log_prob(ns, context=context)
        flow_loss = -log_prob.mean() / state_dim
        flow_opt.zero_grad()
        flow_loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        flow_opt.step()

        # --- MDN (NLL) ---
        mdn_ll = mdn.log_prob(s, a, ns)
        mdn_loss = -mdn_ll.mean() / state_dim
        mdn_opt.zero_grad()
        mdn_loss.backward()
        torch.nn.utils.clip_grad_norm_(mdn.parameters(), 1.0)
        mdn_opt.step()

        # --- Logging ---
        if step % 1000 == 0:
            flow_mse = validate_flow(flow, val_s, val_a, val_ns, state_dim)
            mdn_mse = validate_mdn(mdn, val_s, val_a, val_ns)

            print(f"  Step {step:5d} | "
                  f"EBM acc={ebm_metrics['accuracy']:.2f} gap={ebm_metrics['E_gap']:.1f} | "
                  f"Flow val_mse={flow_mse:.6f} | "
                  f"MDN val_mse={mdn_mse:.6f}")

    # Final validation
    flow_mse = validate_flow(flow, val_s, val_a, val_ns, state_dim)
    mdn_mse = validate_mdn(mdn, val_s, val_a, val_ns)
    print(f"  FINAL | Flow MSE={flow_mse:.6f} | MDN MSE={mdn_mse:.6f} | "
          f"EBM acc={ebm_metrics['accuracy']:.3f}")

    # Save
    torch.save(ebm.state_dict(), "pretrained_ebm_pendulum.pth")
    torch.save(flow.state_dict(), "pretrained_flow_pendulum.pth")
    torch.save(mdn.state_dict(), "pretrained_mdn_pendulum.pth")

    return {
        "flow_mse": flow_mse,
        "mdn_mse": mdn_mse,
        "ebm_acc": ebm_metrics["accuracy"],
    }


# =============================================================================
# MAIN
# =============================================================================
def pretrain():
    device = CONFIG["DEVICE"]

    print(f"\n{'='*60}")
    print(f"PRETRAINING WORLD MODELS — Pendulum-v1")
    print(f"Single model per architecture")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Collect data
    np.random.seed(42)
    states, actions, next_states, state_dim, action_dim = collect_data(CONFIG["COLLECT_STEPS"])

    # Train/val split
    N = len(states)
    n_val = int(N * CONFIG["VAL_SPLIT"])
    n_train = N - n_val

    perm = np.random.permutation(N)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_s = torch.tensor(states[train_idx], device=device)
    train_a = torch.tensor(actions[train_idx], device=device)
    train_ns = torch.tensor(next_states[train_idx], device=device)

    val_s = torch.tensor(states[val_idx], device=device)
    val_a = torch.tensor(actions[val_idx], device=device)
    val_ns = torch.tensor(next_states[val_idx], device=device)

    print(f"Train: {n_train}, Val: {n_val}\n")

    results = train_models(
        train_s, train_a, train_ns, n_train,
        val_s, val_a, val_ns, state_dim, action_dim, device
    )

    # Save config
    torch.save({
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": CONFIG["HIDDEN_DIM"],
        "mdn_num_gaussians": CONFIG["MDN_NUM_GAUSSIANS"],
    }, "pretrained_config_pendulum.pth")

    # Summary
    print(f"\n{'='*60}")
    print(f"PRETRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Flow MSE={results['flow_mse']:.6f} | "
          f"MDN MSE={results['mdn_mse']:.6f} | EBM acc={results['ebm_acc']:.3f}")
    print(f"\nSaved pretrained checkpoints. Pretraining complete!")


if __name__ == "__main__":
    pretrain()