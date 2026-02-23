"""
Multimodal Point Navigation Environment

A continuous 2D point environment with GENUINELY MULTIMODAL dynamics.

State:  [x, y, goal_x, goal_y, cos(heading), sin(heading),
         obs1_x, obs1_y, obs1_r, obs2_x, obs2_y, obs2_r,
         obs3_x, obs3_y, obs3_r]                              -> 15 dim
Action: [forward_speed, turn_rate]                             -> 2 dim (continuous)

Dynamics (Multimodal Transition):
  When the agent takes an action, one of 3 things happens:
    - With prob (1 - slip_prob):  action executed normally
    - With prob slip_prob / 2:    action heading rotated by +deflection_angle
    - With prob slip_prob / 2:    action heading rotated by -deflection_angle

  Obstacles add collision-bounce dynamics:
    - If the agent moves into an obstacle, it bounces off (reflected)
    - Combined with slip, this creates even richer multimodal transitions

  This creates a TRIMODAL (or more) transition distribution p(s'|s,a).
  An MDN will average these modes, predicting a location between them that
  is NEVER actually reached. An EBM can represent all three modes faithfully.

Arena:     [-1, 1] x [-1, 1] square
Obstacles: N_OBSTACLES circular obstacles with random positions and radii
Goal:      Random position, episode ends when agent is within threshold.

Compatible with the existing thesis codebase:
  - Gymnasium-style API: reset() -> state, step(action) -> (state, reward, done, info)
  - Continuous state & action spaces
  - Same interface as MiniGridAdapter / SafetyGymAdapter
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultimodalPointEnv(gym.Env):
    """
    Continuous 2D point navigation with multimodal stochastic dynamics
    and circular obstacles.

    The agent controls a point mass with heading. Actions are:
      - forward_speed: how fast to move in the heading direction
      - turn_rate: how much to rotate heading

    Stochasticity: with configurable probability, the agent's movement
    direction is deflected by ±deflection_angle, creating a trimodal
    transition distribution.

    Obstacles: circular obstacles that the agent bounces off of.
    Combined with slip, collisions create richer multimodal transitions
    (e.g., a slip near an obstacle can cause a bounce in an unexpected
    direction).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        arena_size=1.0,
        goal_threshold=0.15,
        max_steps=100,
        step_scale=0.15,
        slip_prob=0.3,
        deflection_angle=90.0,
        n_obstacles=3,
        obstacle_radius_range=(0.08, 0.15),
        render_mode=None,
    ):
        """
        Args:
            arena_size:             Half-width of the square arena ([-1,1]²)
            goal_threshold:         Distance to goal to count as "reached"
            max_steps:              Maximum steps per episode
            step_scale:             Scaling factor for movement speed
            slip_prob:              Probability of deflection (split equally ±)
            deflection_angle:       Deflection angle in degrees
            n_obstacles:            Number of circular obstacles
            obstacle_radius_range:  (min_radius, max_radius) for random obstacles
            render_mode:            'rgb_array' or None
        """
        super().__init__()
        self.arena_size = arena_size
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.step_scale = step_scale
        self.slip_prob = slip_prob
        self.deflection_rad = np.radians(deflection_angle)
        self.n_obstacles = n_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.render_mode = render_mode

        # State: [x, y, goal_x, goal_y, cos(heading), sin(heading),
        #         obs1_x, obs1_y, obs1_r, ..., obsN_x, obsN_y, obsN_r]
        self.state_dim = 6 + 3 * n_obstacles
        self.action_dim = 2

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # Internal state
        self._pos = None
        self._heading = None
        self._goal = None
        self._obstacles = []  # list of (cx, cy, radius)
        self._step_count = 0

    def _get_obs(self):
        obs = [
            self._pos[0], self._pos[1],
            self._goal[0], self._goal[1],
            np.cos(self._heading), np.sin(self._heading),
        ]
        for (cx, cy, r) in self._obstacles:
            obs.extend([cx, cy, r])
        return np.array(obs, dtype=np.float32)

    def _place_obstacles(self):
        """Generate N non-overlapping circular obstacles."""
        self._obstacles = []
        min_r, max_r = self.obstacle_radius_range
        agent_clearance = 0.2  # minimum dist from agent start
        goal_clearance = 0.2   # minimum dist from goal center

        for _ in range(self.n_obstacles):
            for attempt in range(200):
                r = self.np_random.uniform(min_r, max_r)
                cx = self.np_random.uniform(-0.7, 0.7)
                cy = self.np_random.uniform(-0.7, 0.7)

                # Check no overlap with existing obstacles
                valid = True
                for (ox, oy, orad) in self._obstacles:
                    if np.sqrt((cx - ox)**2 + (cy - oy)**2) < r + orad + 0.05:
                        valid = False
                        break

                # Check clearance from agent start
                if valid and np.sqrt((cx - self._pos[0])**2 + (cy - self._pos[1])**2) < r + agent_clearance:
                    valid = False

                # Check clearance from goal
                if valid and np.sqrt((cx - self._goal[0])**2 + (cy - self._goal[1])**2) < r + goal_clearance:
                    valid = False

                if valid:
                    self._obstacles.append((cx, cy, r))
                    break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random start position (avoid edges)
        self._pos = self.np_random.uniform(-0.8, 0.8, size=2)
        # Random heading
        self._heading = self.np_random.uniform(-np.pi, np.pi)
        # Random goal (ensure minimum distance from start)
        for _ in range(100):
            self._goal = self.np_random.uniform(-0.8, 0.8, size=2)
            if np.linalg.norm(self._goal - self._pos) > 0.4:
                break
        # Place obstacles (after agent and goal are set)
        self._place_obstacles()
        self._step_count = 0
        return self._get_obs(), {}

    def _check_obstacle_collision(self, old_pos, new_pos):
        """
        Check if movement from old_pos to new_pos collides with any obstacle.
        If collision, return the bounced position. Otherwise return new_pos.

        Bounce physics: reflect the movement vector off the obstacle surface.
        """
        agent_radius = 0.02  # small agent radius for collision
        bounced = False

        for (cx, cy, obs_r) in self._obstacles:
            center = np.array([cx, cy])
            dist = np.linalg.norm(new_pos - center)
            collision_dist = obs_r + agent_radius

            if dist < collision_dist:
                # Collision! Compute bounce.
                # Normal vector: from obstacle center to agent
                normal = new_pos - center
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-8:
                    # Agent is at obstacle center (shouldn't happen), push out
                    normal = old_pos - center
                    norm_len = np.linalg.norm(normal)
                    if norm_len < 1e-8:
                        normal = np.array([1.0, 0.0])
                        norm_len = 1.0
                normal = normal / norm_len

                # Push agent outside obstacle
                new_pos = center + normal * collision_dist

                # Reflect velocity (movement direction)
                movement = new_pos - old_pos
                reflected = movement - 2 * np.dot(movement, normal) * normal
                new_pos = old_pos + reflected * 0.5  # damped bounce

                # Ensure still outside obstacle after bounce
                dist_after = np.linalg.norm(new_pos - center)
                if dist_after < collision_dist:
                    new_pos = center + normal * collision_dist

                bounced = True

        return new_pos, bounced

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        forward_speed = action[0]
        turn_rate = action[1]

        # Update heading
        self._heading += turn_rate * 0.5
        self._heading = (self._heading + np.pi) % (2 * np.pi) - np.pi

        # Compute movement direction
        base_angle = self._heading

        # --- MULTIMODAL STOCHASTICITY ---
        slip_rand = self.np_random.random()
        if slip_rand < self.slip_prob / 2:
            move_angle = base_angle + self.deflection_rad
            slip_direction = 'positive'
        elif slip_rand < self.slip_prob:
            move_angle = base_angle - self.deflection_rad
            slip_direction = 'negative'
        else:
            move_angle = base_angle
            slip_direction = 'none'

        # Move
        old_pos = self._pos.copy()
        dx = forward_speed * self.step_scale * np.cos(move_angle)
        dy = forward_speed * self.step_scale * np.sin(move_angle)
        new_pos = self._pos + np.array([dx, dy])

        # --- OBSTACLE COLLISION ---
        new_pos, bounced = self._check_obstacle_collision(old_pos, new_pos)

        # Clamp to arena
        new_pos = np.clip(new_pos, -self.arena_size, self.arena_size)
        self._pos = new_pos

        # Reward
        dist_to_goal = np.linalg.norm(self._pos - self._goal)
        reward = -dist_to_goal

        # Penalty for bumping obstacles
        if bounced:
            reward -= 0.5

        # Check done
        self._step_count += 1
        reached = dist_to_goal < self.goal_threshold
        truncated = self._step_count >= self.max_steps
        terminated = reached

        if reached:
            reward += 10.0

        info = {
            "dist_to_goal": dist_to_goal,
            "reached": reached,
            "slip_occurred": slip_direction != 'none',
            "slip_direction": slip_direction,
            "bounced": bounced,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self._render_frame()

    def _render_frame(self, size=400):
        img = np.ones((size, size, 3), dtype=np.uint8) * 240

        def to_pixel(pos):
            px = int((pos[0] + self.arena_size) / (2 * self.arena_size) * size)
            py = int((1 - (pos[1] + self.arena_size) / (2 * self.arena_size)) * size)
            return np.clip(px, 0, size - 1), np.clip(py, 0, size - 1)

        def radius_to_pixels(r):
            return int(r / (2 * self.arena_size) * size)

        # Arena border
        img[0:3, :] = 0; img[-3:, :] = 0; img[:, 0:3] = 0; img[:, -3:] = 0

        # Draw obstacles (dark gray circles)
        for (cx, cy, obs_r) in self._obstacles:
            ox, oy = to_pixel(np.array([cx, cy]))
            r_px = radius_to_pixels(obs_r)
            for y in range(max(0, oy - r_px), min(size, oy + r_px + 1)):
                for x in range(max(0, ox - r_px), min(size, ox + r_px + 1)):
                    if (x - ox)**2 + (y - oy)**2 <= r_px**2:
                        img[y, x] = [80, 80, 80]

        # Draw goal (green circle)
        gx, gy = to_pixel(self._goal)
        r_goal = radius_to_pixels(self.goal_threshold)
        for y in range(max(0, gy - r_goal), min(size, gy + r_goal + 1)):
            for x in range(max(0, gx - r_goal), min(size, gx + r_goal + 1)):
                if (x - gx)**2 + (y - gy)**2 <= r_goal**2:
                    img[y, x] = [50, 200, 50]

        # Draw agent (blue circle with heading line)
        apx, apy = to_pixel(self._pos)
        r_agent = 8
        for y in range(max(0, apy - r_agent), min(size, apy + r_agent + 1)):
            for x in range(max(0, apx - r_agent), min(size, apx + r_agent + 1)):
                if (x - apx)**2 + (y - apy)**2 <= r_agent**2:
                    img[y, x] = [50, 50, 200]

        # Heading line
        hx = int(apx + 15 * np.cos(self._heading))
        hy = int(apy - 15 * np.sin(self._heading))
        for t in np.linspace(0, 1, 20):
            lx = int(apx + t * (hx - apx))
            ly = int(apy + t * (hy - apy))
            if 0 <= lx < size and 0 <= ly < size:
                img[ly, lx] = [200, 50, 50]

        return img


class MultimodalPointAdapter:
    """
    Adapter to match the interface used by the thesis codebase
    (same API as MiniGridAdapter / StochasticMiniGridAdapter).
    """

    def __init__(self, slip_prob=0.3, deflection_angle=90.0, n_obstacles=3, **kwargs):
        self.env = MultimodalPointEnv(
            slip_prob=slip_prob,
            deflection_angle=deflection_angle,
            n_obstacles=n_obstacles,
            **kwargs
        )
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.slip_prob = slip_prob
        self.deflection_angle = deflection_angle
        self.n_obstacles = n_obstacles
        print(
            f"MultimodalPointAdapter: state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, slip_prob={slip_prob}, "
            f"deflection_angle={deflection_angle}°, n_obstacles={n_obstacles}"
        )

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        """
        Args:
            action: numpy array of shape (action_dim,) or torch tensor
        Returns:
            next_state, reward, done, info
        """
        import torch
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if hasattr(action, 'shape') and len(action.shape) > 1:
            action = action.squeeze(0)

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
