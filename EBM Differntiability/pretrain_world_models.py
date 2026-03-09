"""
Pretrain EBM with InfoNCE only.

InfoNCE: trains E(s,a,s'_true) > E(s,a,s'_random) -> good energy ranking.
No DSM — the energy gradient landscape must emerge naturally from InfoNCE
so that Experiment 1 can cleanly test differentiability regimes.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from models import BilinearEBM


def collect_random_data(env_name="Pendulum-v1", num_transitions=50000):
    env = gym.make(env_name)
    sd = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    states = np.zeros((num_transitions, sd), dtype=np.float32)
    actions = np.zeros((num_transitions, ad), dtype=np.float32)
    next_states = np.zeros((num_transitions, sd), dtype=np.float32)
    state, _ = env.reset()
    for i in range(num_transitions):
        a = env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a)
        states[i], actions[i], next_states[i] = state, a, ns
        state = ns if not (term or trunc) else env.reset()[0]
    env.close()
    print(f"Collected {num_transitions} transitions")
    return states, actions, next_states, sd, ad


def pretrain_ebm(states, actions, next_states, state_dim, action_dim,
                 device="cpu", hidden_dim=128, epochs=50, batch_size=256,
                 num_negatives=128, temperature=0.1, lr=1e-3):

    ebm = BilinearEBM(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(ebm.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    N = len(states)
    s_t = torch.tensor(states, device=device)
    a_t = torch.tensor(actions, device=device)
    ns_t = torch.tensor(next_states, device=device)

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total_infonce, total_acc, n_batches = 0, 0, 0

        for i in range(0, N - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            s, a, ns = s_t[idx], a_t[idx], ns_t[idx]
            B = s.shape[0]

            optimizer.zero_grad()

            # --- InfoNCE ---
            E_pos = ebm(s, a, ns)
            neg_idx = torch.randint(0, N, (B, num_negatives), device=device)
            neg_ns = ns_t[neg_idx]
            s_exp = s.unsqueeze(1).expand(B, num_negatives, -1)
            a_exp = a.unsqueeze(1).expand(B, num_negatives, -1)
            E_neg = ebm(s_exp, a_exp, neg_ns)

            logits = torch.cat([E_pos.unsqueeze(1), E_neg], dim=1) / temperature
            labels = torch.zeros(B, dtype=torch.long, device=device)
            infonce_loss = F.cross_entropy(logits, labels)

            loss = infonce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ebm.parameters(), 1.0)
            optimizer.step()

            total_infonce += infonce_loss.item()
            total_acc += (logits.argmax(1) == 0).float().mean().item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"InfoNCE: {total_infonce/n_batches:.4f} | "
                  f"Acc: {total_acc/n_batches:.3f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # --- Validate sampling quality ---
    print("\n  Validating Langevin sampling quality...")
    ebm.eval()
    from utils_sampling2 import langevin_sample
    val_batch = 500
    idx = np.random.randint(0, N, val_batch)
    vs, va, vns = s_t[idx], a_t[idx], ns_t[idx]

    pred_ns = langevin_sample(
        ebm, vs, va, diff_mode="detached",
        config={
            "LANGEVIN_STEPS": 50,
            "LANGEVIN_STEP_SIZE": 0.01,
            "LANGEVIN_NOISE_MAX": 0.01,
            "LANGEVIN_NOISE_MIN": 0.0001,
            "LANGEVIN_INIT_NOISE": 0.05,
            "LANGEVIN_GRAD_CLIP": 1.0,
        },
        use_persistent=False,
    )
    mse = F.mse_loss(pred_ns, vns).item()
    naive_mse = F.mse_loss(vs, vns).item()
    print(f"  Langevin prediction MSE: {mse:.6f}")
    print(f"  Naive baseline MSE (predict s'=s): {naive_mse:.6f}")
    print(f"  Improvement ratio: {naive_mse/max(mse, 1e-8):.2f}x")

    return ebm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    states, actions, next_states, sd, ad = collect_random_data()
    print("\n--- Pretraining BilinearEBM (InfoNCE only) ---")
    ebm = pretrain_ebm(states, actions, next_states, sd, ad, device=device)
    torch.save(ebm.state_dict(), "pretrained_ebm_pendulum.pth")
    print("\nSaved: pretrained_ebm_pendulum.pth")


if __name__ == "__main__":
    main()