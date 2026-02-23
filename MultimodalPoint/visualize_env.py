"""
Visualize the Multimodal Point Environment

Generates publication-quality figures showing:
  1. The environment layout with agent, goal, obstacles, and sample trajectories
  2. The multimodal transition distribution (scatter plot of next states
     from identical (s, a) pairs, showing the 3 modes)
  3. Comparison: deterministic vs multimodal stochastic trajectories

Usage:
    python visualize_env.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import os
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, SCRIPT_DIR)

from multimodal_point_env import MultimodalPointEnv


def draw_obstacles(ax, obstacles, arena_size=1.0):
    """Draw circular obstacles on axes."""
    for (cx, cy, r) in obstacles:
        circle = plt.Circle(
            (cx, cy), r, color='#555555', alpha=0.6, linewidth=1.5,
            edgecolor='#333333', zorder=4
        )
        ax.add_patch(circle)


def plot_environment_overview(save_dir="figures"):
    """
    Figure 1: Environment overview with a sample trajectory.
    Shows the arena, goal, obstacles, agent path, and slip events.
    """
    os.makedirs(save_dir, exist_ok=True)

    env = MultimodalPointEnv(slip_prob=0.3, deflection_angle=90.0,
                             n_obstacles=3, max_steps=80)
    obs, _ = env.reset(seed=42)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # Arena
    arena = patches.Rectangle(
        (-1, -1), 2, 2, linewidth=2, edgecolor='#333', facecolor='#F8F8F8'
    )
    ax.add_patch(arena)

    # Obstacles
    draw_obstacles(ax, env._obstacles)

    # Goal
    goal_circle = plt.Circle(
        env._goal, env.goal_threshold, color='#2ECC71', alpha=0.35, linewidth=0
    )
    ax.add_patch(goal_circle)
    ax.plot(*env._goal, 'o', color='#27AE60', markersize=12, zorder=10,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.annotate('Goal', xy=env._goal, xytext=(10, 10),
                textcoords='offset points', fontsize=11, color='#27AE60',
                fontweight='bold')

    # Run episode with a simple goal-seeking policy
    positions = [env._pos.copy()]
    slips = []
    bounces = []

    for step in range(80):
        to_goal = env._goal - env._pos
        angle_to_goal = np.arctan2(to_goal[1], to_goal[0])
        angle_diff = angle_to_goal - env._heading
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        action = np.array([0.8, np.clip(angle_diff * 2.0, -1, 1)])

        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(env._pos.copy())
        slips.append(info['slip_occurred'])
        bounces.append(info['bounced'])

        if terminated or truncated:
            break

    positions = np.array(positions)

    # Draw trajectory with color indicating event type
    for i in range(len(positions) - 1):
        if bounces[i]:
            color = '#F39C12'  # orange for bounce
        elif slips[i]:
            color = '#E74C3C'  # red for slip
        else:
            color = '#3498DB'  # blue for normal
        ax.plot(
            [positions[i, 0], positions[i + 1, 0]],
            [positions[i, 1], positions[i + 1, 1]],
            '-', color=color, linewidth=2.2, alpha=0.9, zorder=5
        )

    # Start marker
    ax.plot(*positions[0], 's', color='#2C3E50', markersize=10, zorder=11,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.annotate('Start', xy=positions[0], xytext=(-30, -15),
                textcoords='offset points', fontsize=11, color='#2C3E50',
                fontweight='bold')

    # End marker
    ax.plot(*positions[-1], 'D', color='#8E44AD', markersize=10, zorder=11,
            markeredgecolor='white', markeredgewidth=1.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='#3498DB', linewidth=2, label='Normal step'),
        Line2D([0], [0], color='#E74C3C', linewidth=2, label='Slip (deflected)'),
        Line2D([0], [0], color='#F39C12', linewidth=2, label='Bounce (obstacle)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2C3E50',
               markersize=10, label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60',
               markersize=10, label='Goal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555',
               markersize=10, label='Obstacle'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9.5,
              framealpha=0.9, edgecolor='#CCC')

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(
        'Multimodal Point Environment\n'
        '(slip_prob=0.3, deflection=90°, 3 obstacles)',
        fontsize=14, fontweight='bold'
    )
    ax.grid(True, alpha=0.15, linestyle='--')

    fig.tight_layout()
    path = os.path.join(save_dir, "env_overview.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)
    return path


def plot_multimodal_transitions(save_dir="figures"):
    """
    Figure 2: Multimodal transition distribution.
    From a FIXED (state, action), sample many next states to show the 3 modes.
    This is the key figure: WHY EBMs are needed.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    slip_probs = [0.0, 0.3, 0.5]
    titles = [
        "Deterministic\n(slip_prob=0.0)",
        "Moderate Stochasticity\n(slip_prob=0.3)",
        "High Stochasticity\n(slip_prob=0.5)"
    ]

    for idx, (sp, title) in enumerate(zip(slip_probs, titles)):
        ax = axes[idx]
        env = MultimodalPointEnv(slip_prob=sp, deflection_angle=90.0, n_obstacles=0)

        # Fixed starting state
        fixed_pos = np.array([0.0, 0.0])
        fixed_heading = 0.0  # pointing right
        fixed_goal = np.array([0.7, 0.0])
        fixed_action = np.array([1.0, 0.0])  # full forward, no turn

        next_positions = []

        for trial in range(600):
            env.reset(seed=trial + 1000)
            env._pos = fixed_pos.copy()
            env._heading = fixed_heading
            env._goal = fixed_goal.copy()

            obs, reward, term, trunc, info = env.step(fixed_action.copy())
            next_positions.append(env._pos.copy())

        next_positions = np.array(next_positions)

        # Classify modes by angle from start
        mode_colors = []
        for npos in next_positions:
            d = npos - fixed_pos
            angle = np.arctan2(d[1], d[0])
            if abs(angle) < 0.3:
                mode_colors.append('#3498DB')   # intended (forward)
            elif angle > 0:
                mode_colors.append('#E74C3C')   # deflected +90°
            else:
                mode_colors.append('#F39C12')   # deflected -90°

        ax.scatter(
            next_positions[:, 0], next_positions[:, 1],
            c=mode_colors, alpha=0.45, s=22, zorder=5, edgecolors='none'
        )

        # Start position
        ax.plot(*fixed_pos, 'ko', markersize=12, zorder=10,
                markeredgecolor='white', markeredgewidth=2)
        ax.annotate('(s, a)', xy=fixed_pos, xytext=(-28, -18),
                    textcoords='offset points', fontsize=12,
                    fontweight='bold')

        # MDN prediction (mean of all samples)
        mean_pred = next_positions.mean(axis=0)
        ax.plot(*mean_pred, 'X', color='#8E44AD', markersize=16, zorder=11,
                markeredgecolor='white', markeredgewidth=2)
        ax.annotate('MDN prediction\n(mode-averaged)', xy=mean_pred,
                    xytext=(8, 15), textcoords='offset points',
                    fontsize=9, color='#8E44AD', fontweight='bold',
                    ha='left')

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Y Position', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, linestyle='--')

        # Mode legend
        if sp > 0:
            mode_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
                       markersize=8, label='Intended'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
                       markersize=8, label='Deflected +90°'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12',
                       markersize=8, label='Deflected −90°'),
                Line2D([0], [0], marker='X', color='w', markerfacecolor='#8E44AD',
                       markersize=10, label='MDN (averaged)'),
            ]
            ax.legend(handles=mode_legend, fontsize=8.5, loc='upper left',
                      framealpha=0.9)

        ax.set_xlim(-0.05, 0.25)
        ax.set_ylim(-0.2, 0.2)

    fig.suptitle(
        "Transition Distribution p(s'|s, a): Multimodal Dynamics",
        fontsize=15, fontweight='bold', y=1.03
    )
    fig.tight_layout()
    path = os.path.join(save_dir, "multimodal_transitions.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)
    return path


def plot_trajectory_comparison(save_dir="figures"):
    """
    Figure 3: Multiple trajectories showing the effect of stochasticity + obstacles.
    Same start/goal/obstacles, different random outcomes.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    configs = [
        (0.0, "Deterministic (slip_prob=0.0)"),
        (0.3, "Multimodal Stochastic (slip_prob=0.3)")
    ]

    for idx, (sp, title) in enumerate(configs):
        ax = axes[idx]

        # Arena background
        arena = patches.Rectangle(
            (-1, -1), 2, 2, linewidth=2, edgecolor='#333', facecolor='#FAFAFA'
        )
        ax.add_patch(arena)

        cmap = plt.cm.viridis
        n_trajectories = 10

        # Use the same seed for reset (same start, goal, obstacles) across trajectories
        ref_env = MultimodalPointEnv(slip_prob=sp, deflection_angle=90.0,
                                     n_obstacles=3, max_steps=80)
        ref_env.reset(seed=42)
        fixed_pos = ref_env._pos.copy()
        fixed_goal = ref_env._goal.copy()
        fixed_heading = ref_env._heading
        fixed_obstacles = list(ref_env._obstacles)

        # Draw obstacles (same for all trajectories)
        draw_obstacles(ax, fixed_obstacles)

        for traj_i in range(n_trajectories):
            env = MultimodalPointEnv(slip_prob=sp, deflection_angle=90.0,
                                     n_obstacles=3, max_steps=80)
            env.reset(seed=42)  # same layout
            # Override rng for different stochastic outcomes
            env.np_random = np.random.default_rng(seed=traj_i * 777)

            positions = [env._pos.copy()]

            for step in range(80):
                to_goal = env._goal - env._pos
                angle_to_goal = np.arctan2(to_goal[1], to_goal[0])
                angle_diff = angle_to_goal - env._heading
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                action = np.array([0.7, np.clip(angle_diff * 2.0, -1, 1)])

                obs, reward, term, trunc, info = env.step(action)
                positions.append(env._pos.copy())

                if term or trunc:
                    break

            positions = np.array(positions)
            color = cmap(traj_i / n_trajectories)
            ax.plot(positions[:, 0], positions[:, 1], '-',
                    color=color, linewidth=1.5, alpha=0.7, zorder=5)

        # Goal
        goal_circle = plt.Circle(
            fixed_goal, ref_env.goal_threshold, color='#2ECC71',
            alpha=0.3, linewidth=0
        )
        ax.add_patch(goal_circle)
        ax.plot(*fixed_goal, 'o', color='#27AE60', markersize=12, zorder=10,
                markeredgecolor='white', markeredgewidth=1.5)

        # Start
        ax.plot(*fixed_pos, 's', color='#2C3E50', markersize=10, zorder=10,
                markeredgecolor='white', markeredgewidth=1.5)

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Y Position', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.15, linestyle='--')

    fig.suptitle(
        'Trajectory Diversity: Same Policy, Same Layout, Different Outcomes',
        fontsize=14, fontweight='bold', y=1.02
    )
    fig.tight_layout()
    path = os.path.join(save_dir, "trajectory_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)
    return path


if __name__ == "__main__":
    save_dir = os.path.join(SCRIPT_DIR, "figures")
    print("Generating environment visualizations...\n")

    p1 = plot_environment_overview(save_dir)
    p2 = plot_multimodal_transitions(save_dir)
    p3 = plot_trajectory_comparison(save_dir)

    print(f"\nAll figures saved to: {save_dir}/")
    print(f"  1. {p1}")
    print(f"  2. {p2}")
    print(f"  3. {p3}")
