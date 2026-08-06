"""
puddleworld.py — PuddleWorld + determinized-MDP generator.
=============================================================

Exported from the *implicitgoals* research repo. Builds on the shared
``GridWorld`` MDP and ``augment_mdp_to_deterministic`` helper defined in
``gridworld_core.py`` (the same core used by gridworld.py, rockworld.py,
and taxiworld.py — see that module for the shared plumbing).

PuddleWorld is a GridWorld where some empty cells become "puddles" (map value
``0.5``). Puddles only change the *reward* (a penalty for stepping in one); they
do **not** block movement, so the determinized transition array has the same
structure as a plain grid with the same obstacles.

Quick start
-----------
    from puddleworld import generate_determinized_models
    out = generate_determinized_models(size=4, num_humans=3,
                                       obstacles_percent=0.1, puddle_percent=0.2,
                                       seed=0)
    T_R, s0, g = out["robot"][:3]
    print(out["total_determinizing_time"])
"""

import time
import random

import numpy as np

from gridworld_core import GridWorld, augment_mdp_to_deterministic


# ─────────────────────────────────────────────────────────────────────────────
# PuddleWorld (ported from PuddleWorldClass.py)
# ─────────────────────────────────────────────────────────────────────────────

class PuddleWorld(GridWorld):
    """GridWorld with puddles (map value ``0.5``). Puddles penalize rewards but
    do not block movement, so transitions match a plain grid's."""

    def __init__(self, size=5, start=None, goal=None, obstacles_percent=0.1,
                 puddle_percent=0.2, puddle_penalty=-1, goal_reward=1,
                 slip_prob=0.1, discount=0.99, max_tries=100, obstacle_seed=1):
        super().__init__(size=size, start=start, goal=goal,
                         obstacles_percent=obstacles_percent,
                         slip_prob=slip_prob, discount=discount,
                         max_tries=max_tries, obstacle_seed=obstacle_seed)
        self.puddle_percent = puddle_percent
        self.puddle_penalty = puddle_penalty
        self.goal_reward = goal_reward
        self.place_puddles()
        self.reward_func = self.puddle_reward_func

    def place_puddles(self):
        total_puddles = int(self.size * self.size * self.puddle_percent)
        for _ in range(total_puddles):
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            if self.map[x, y] == 0:
                self.map[x, y] = 0.5

    def puddle_reward_func(self, state, action, next_state):
        x, y = next_state[0]
        if self.check_goal_reached(next_state):
            return self.goal_reward
        elif self.map[x, y] == 0.5:
            return self.puddle_penalty
        else:
            return 0

    def visualize(self):
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if self.map[i, j] == -1:
                    row += "# "
                elif self.map[i, j] == 0.5:
                    row += "~ "
                elif (i, j) == self.start_pos:
                    row += "S "
                elif (i, j) == self.goal_pos:
                    row += "G "
                else:
                    row += ". "
            print(row)


def generate_and_visualize_puddleworld(size, start, goal, obstacles_percent, puddle_percent,
                                       model_type="Model", obstacle_seed=None):
    """Generate a ``PuddleWorld`` (mirrors the function in experiments.py)."""
    return PuddleWorld(size=size, start=start, goal=goal,
                       obstacles_percent=obstacles_percent,
                       puddle_percent=puddle_percent,
                       obstacle_seed=obstacle_seed)


# ─────────────────────────────────────────────────────────────────────────────
# High-level driver — robot + N humans, with compute timing
# ─────────────────────────────────────────────────────────────────────────────

def _make_determinized(size, obstacles_percent, puddle_percent, model_type, visualize=False):
    """Generate one puddle world and determinize it; returns (next_states, s0, g, det_time).

    If ``visualize`` is True, print the generated map before determinizing.
    """
    mdp = generate_and_visualize_puddleworld(
        size=size, start=(0, 0), goal=(size - 1, size - 1),
        obstacles_percent=obstacles_percent, puddle_percent=puddle_percent,
        model_type=model_type, obstacle_seed=random.randint(1, 10000))
    if visualize:
        print(f"\n{model_type}:")
        mdp.visualize()
    t0 = time.time()
    next_states, start_idx, goal_idx = augment_mdp_to_deterministic(mdp)
    return next_states, start_idx, goal_idx, time.time() - t0


def generate_determinized_models(size=4, num_humans=3, obstacles_percent=0.1,
                                 puddle_percent=0.2, seed=None, verbose=True,
                                 visualize=False):
    """Build a robot model + ``num_humans`` human PuddleWorld models and determinize each.

    Parameters
    ----------
    size : int              grid side length
    num_humans : int        number of human models
    obstacles_percent : float   obstacle density in [0, 1]
    puddle_percent : float  puddle density in [0, 1]
    seed : int or None      seeds ``random``/``numpy`` for reproducibility
    verbose : bool          print a short timing summary
    visualize : bool        print each generated map (robot + humans)

    Returns
    -------
    dict: 'robot', 'humans', 'determinizing_times', 'total_determinizing_time'
    (each model is a tuple ``(next_states, start_idx, goal_idx, det_time)``)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    robot = _make_determinized(size, obstacles_percent, puddle_percent, "Robot Model", visualize)
    humans = [_make_determinized(size, obstacles_percent, puddle_percent, f"Human Model {i + 1}", visualize)
              for i in range(num_humans)]

    det_times = [robot[3]] + [h[3] for h in humans]
    total = float(sum(det_times))

    if verbose:
        print(f"[puddleworld] size={size} obstacles={obstacles_percent} "
              f"puddles={puddle_percent} humans={len(humans)}")
        print(f"  robot: {robot[0].shape[0]} states x {robot[0].shape[1]} actions "
              f"(determinized in {robot[3]:.4f}s)")
        print(f"  total determinizing time: {total:.4f}s")

    return {
        "robot": robot,
        "humans": humans,
        "determinizing_times": det_times,
        "total_determinizing_time": total,
    }


if __name__ == "__main__":
    out = generate_determinized_models(size=4, num_humans=3,
                                       obstacles_percent=0.1, puddle_percent=0.2, seed=0, visualize=True)
    T_R, s0, g, _ = out["robot"]
    print("\nRobot determinized transition array shape:", T_R.shape)
    print("start index:", s0, " goal index:", g)
    print("num human models:", len(out["humans"]))
    print("total determinizing time (s):", out["total_determinizing_time"])
