"""
MiniGrid Dynamic-Obstacles Gymnasium Wrapper for MBRL Experiments.

Wraps MiniGrid-Dynamic-Obstacles into the continuous state/action interface
expected by the EBM/Flow/MDN world model codebase.

Requirements:
    pip install minigrid gymnasium

═══════════════════════════════════════════════════════════════════════
ENVIRONMENT DYNAMICS
═══════════════════════════════════════════════════════════════════════
  - 8×8 grid (6×6 walkable) with N moving obstacle balls
  - Obstacles move to a random adjacent empty cell EVERY step
    → genuine environment-side stochasticity (not just action slip)
  - Collision with obstacle → -1 reward, episode terminates
  - Reaching green goal → positive reward, episode terminates
  - Action space: Discrete(3) — {left, right, forward} (already restricted
    by DynamicObstaclesEnv itself)
  - Stochastic action slip applied on top (configurable)

═══════════════════════════════════════════════════════════════════════
STATE REPRESENTATION  (dim = 7 + 2 * N_OBSTACLES, default 15)
═══════════════════════════════════════════════════════════════════════

  Base (7):
    [x_norm, y_norm, dir_sin, dir_cos, goal_dx, goal_dy, goal_dist]

  Per obstacle, sorted nearest-first (2 * N_OBSTACLES):
    [obs0_dx, obs0_dy, obs1_dx, obs1_dy, ...]

  Sorted by distance → canonical ordering (permutation-invariant w.r.t.
  obstacle identity) so the world model doesn't need to learn that
  obstacle 1 and obstacle 3 are interchangeable.

═══════════════════════════════════════════════════════════════════════
REWARD (dense)
═══════════════════════════════════════════════════════════════════════
  + (prev_goal_dist - curr_goal_dist) * 5.0   # approach goal
  - step_penalty                                # efficiency
  + goal_bonus         (if goal reached)        # success
  - collision_penalty  (if hit obstacle)        # survival
═══════════════════════════════════════════════════════════════════════
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import minigrid
from minigrid.wrappers import FullyObsWrapper


class DynamicObstaclesContinuousWrapper(gym.Wrapper):
    """
    Wraps MiniGrid-Dynamic-Obstacles into a continuous-state interface.

    Observation: R^(7 + 2*n_obstacles) flat vector
    Action:      Discrete(3) {0: left, 1: right, 2: forward}
    Reward:      Dense distance-based + collision penalty + goal bonus
    """

    USEFUL_ACTIONS = 3

    def __init__(self, env, n_obstacles=4, slip_prob=0.1,
                 dense_reward=True, goal_bonus=10.0, step_penalty=0.01,
                 collision_penalty=5.0):
        super().__init__(env)

        self.n_obstacles = n_obstacles
        self.slip_prob = slip_prob
        self.dense_reward = dense_reward
        self.goal_bonus = goal_bonus
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty

        self.grid_w = self.unwrapped.width
        self.grid_h = self.unwrapped.height
        self.interior_w = self.grid_w - 2
        self.interior_h = self.grid_h - 2

        self.state_dim = 7 + 2 * n_obstacles
        self.action_dim = 3

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.USEFUL_ACTIONS)

        self._goal_pos = np.array([self.grid_w - 2, self.grid_h - 2],
                                   dtype=np.float32)
        self._step_count = 0
        self._prev_dist = None

    def _get_obstacle_positions(self):
        """Extract obstacle (Ball) positions from the env."""
        uw = self.unwrapped
        positions = []
        if hasattr(uw, 'obstacles'):
            for obj in uw.obstacles:
                if obj is not None and obj.cur_pos is not None:
                    positions.append(tuple(obj.cur_pos))
        if len(positions) == 0:
            for x in range(self.grid_w):
                for y in range(self.grid_h):
                    cell = uw.grid.get(x, y)
                    if cell is not None and cell.type == 'ball':
                        positions.append((x, y))
        return positions

    def _get_state(self):
        """Extract continuous state vector, shape (state_dim,)."""
        uw = self.unwrapped
        ax, ay = uw.agent_pos
        direction = uw.agent_dir

        x_norm = (ax - 1) / max(self.interior_w - 1, 1)
        y_norm = (ay - 1) / max(self.interior_h - 1, 1)

        angle = direction * (np.pi / 2)
        dir_sin = np.sin(angle)
        dir_cos = np.cos(angle)

        gx, gy = self._goal_pos
        goal_dx = (gx - ax) / self.interior_w
        goal_dy = (gy - ay) / self.interior_h
        goal_dist = np.sqrt(goal_dx**2 + goal_dy**2)

        base = [x_norm, y_norm, dir_sin, dir_cos, goal_dx, goal_dy, goal_dist]

        obs_positions = self._get_obstacle_positions()
        obs_feats = []
        for (ox, oy) in obs_positions:
            dx = (ox - ax) / self.interior_w
            dy = (oy - ay) / self.interior_h
            dist = np.sqrt(dx**2 + dy**2)
            obs_feats.append((dist, dx, dy))

        obs_feats.sort(key=lambda t: t[0])

        flat_obs = []
        for i in range(self.n_obstacles):
            if i < len(obs_feats):
                _, dx, dy = obs_feats[i]
                flat_obs.extend([dx, dy])
            else:
                flat_obs.extend([1.0, 1.0])

        return np.array(base + flat_obs, dtype=np.float32)

    def _compute_reward(self, terminated, truncated, orig_reward, state):
        goal_dist = state[6]
        reward = 0.0

        if self._prev_dist is not None:
            reward += (self._prev_dist - goal_dist) * 5.0
        self._prev_dist = goal_dist

        reward -= self.step_penalty

        if terminated and orig_reward < 0:
            reward -= self.collision_penalty
        elif terminated and orig_reward > 0:
            reward += self.goal_bonus

        return reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        state = self._get_state()
        self._prev_dist = state[6]
        info["agent_pos"] = tuple(self.unwrapped.agent_pos)
        info["agent_dir"] = self.unwrapped.agent_dir
        return state, info

    def step(self, action):
        action = int(action)
        assert 0 <= action < self.USEFUL_ACTIONS

        if self.slip_prob > 0 and self.np_random.random() < self.slip_prob:
            action = int(self.np_random.integers(0, self.USEFUL_ACTIONS))

        obs, orig_reward, terminated, truncated, info = self.env.step(action)

        self._step_count += 1
        state = self._get_state()

        if self.dense_reward:
            reward = self._compute_reward(terminated, truncated, orig_reward, state)
        else:
            reward = float(orig_reward)

        collision = terminated and orig_reward < 0
        goal_reached = terminated and orig_reward >= 0 and not truncated

        info["minigrid_reward"] = float(orig_reward)
        info["step_count"] = self._step_count
        info["agent_pos"] = tuple(self.unwrapped.agent_pos)
        info["agent_dir"] = self.unwrapped.agent_dir
        info["goal_dist"] = float(state[6])
        info["goal_reached"] = goal_reached
        info["collision"] = collision

        return state, reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────

def make_minigrid_env(size=8, n_obstacles=4, slip_prob=0.1,
                      dense_reward=True, goal_bonus=10.0, step_penalty=0.01,
                      collision_penalty=5.0, max_steps=100, random_start=True):
    """
    Create a wrapped MiniGrid-Dynamic-Obstacles environment.

    Returns:
        DynamicObstaclesContinuousWrapper with:
          state_dim  = 7 + 2*n_obstacles  (default 15)
          action_dim = 3
    """
    if random_start and size in (5, 6):
        env_id = f"MiniGrid-Dynamic-Obstacles-Random-{size}x{size}-v0"
    else:
        env_id = f"MiniGrid-Dynamic-Obstacles-{size}x{size}-v0"

    base_env = gym.make(env_id, n_obstacles=n_obstacles, max_steps=max_steps)
    full_env = FullyObsWrapper(base_env)

    return DynamicObstaclesContinuousWrapper(
        full_env,
        n_obstacles=n_obstacles,
        slip_prob=slip_prob,
        dense_reward=dense_reward,
        goal_bonus=goal_bonus,
        step_penalty=step_penalty,
        collision_penalty=collision_penalty,
    )


# ─────────────────────────────────────────────────────────────────────
# ACTION UTILITIES
# ─────────────────────────────────────────────────────────────────────

def continuous_to_discrete(action_logits):
    try:
        import torch
        if isinstance(action_logits, torch.Tensor):
            action_logits = action_logits.detach().cpu().numpy()
    except ImportError:
        pass
    if action_logits.ndim == 1:
        return int(np.argmax(action_logits))
    return np.argmax(action_logits, axis=1)


def discrete_to_onehot(action_int, n_actions=3):
    vec = np.zeros(n_actions, dtype=np.float32)
    vec[int(action_int)] = 1.0
    return vec


# ─────────────────────────────────────────────────────────────────────
# ANALYTICAL REWARD FOR IMAGINED ROLLOUTS
# ─────────────────────────────────────────────────────────────────────

def minigrid_analytical_reward(state, next_state, action):
    """
    Differentiable dense reward for imagined rollouts.

    Matches _compute_reward as closely as possible:
      + (prev_goal_dist - curr_goal_dist) * 5.0   # approach goal
      - 0.01                                        # step penalty
      - collision_proxy                             # soft collision penalty
      + goal_proxy                                  # soft goal bonus

    The env gives hard -5.0 on collision and +10.0 on goal reach, but
    imagined rollouts have no termination signal. We use smooth proxies:
      collision: steep penalty when nearest obstacle dist < ~0.15
      goal:      steep bonus when goal dist < ~0.1

    State layout:
      [6]    goal_dist
      [7,8]  nearest obstacle dx, dy  (sorted nearest-first)
    """
    import torch

    dist_curr = state[:, 6]
    dist_next = next_state[:, 6]

    # Distance-based shaping (same as env)
    reward = (dist_curr - dist_next) * 5.0 - 0.01

    # Collision proxy: smooth approximation of the -5.0 collision penalty.
    # Uses a steep sigmoid that ramps up sharply when nearest obstacle
    # is within ~0.15 normalized distance (roughly 1 grid cell).
    if next_state.shape[1] > 8:
        obs0_dx = next_state[:, 7]
        obs0_dy = next_state[:, 8]
        nearest_dist = torch.sqrt(obs0_dx**2 + obs0_dy**2 + 1e-8)
        # Sigmoid centered at 0.12, steep slope — outputs ~5.0 when dist→0
        collision_proxy = 5.0 * torch.sigmoid(8.0 * (0.12 - nearest_dist))
        reward = reward - collision_proxy

    # Goal bonus proxy: smooth approximation of +10.0 goal reward.
    # Fires when goal_dist is very small (agent near goal).
    goal_proxy = 10.0 * torch.sigmoid(15.0 * (0.08 - dist_next))
    reward = reward + goal_proxy

    return reward


# ─────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MiniGrid Dynamic-Obstacles Wrapper — Self Test")
    print("=" * 60)

    env = make_minigrid_env(size=8, n_obstacles=4, slip_prob=0.1)
    print(f"\nState dim:   {env.state_dim}")
    print(f"Action dim:  {env.action_dim}")
    print(f"N obstacles: {env.n_obstacles}")

    n_ep = 50
    rewards, lengths = [], []
    goals, collisions, timeouts = 0, 0, 0
    for ep in range(n_ep):
        state, info = env.reset(seed=ep)
        ep_r, done = 0.0, False
        while not done:
            a = env.action_space.sample()
            state, r, term, trunc, info = env.step(a)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
        lengths.append(info["step_count"])
        if info.get("goal_reached"): goals += 1
        elif info.get("collision"): collisions += 1
        else: timeouts += 1

    print(f"\n{n_ep} episodes (random policy):")
    print(f"  Avg reward:   {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Avg length:   {np.mean(lengths):.1f}")
    print(f"  Goals: {goals}  Collisions: {collisions}  Timeouts: {timeouts}")

    env.close()
    print("\nDone!")