"""
rockworld.py — RockWorld + determinized-MDP generator.
==========================================================

Exported from the *implicitgoals* research repo. Builds on the shared
``GridWorld`` MDP and ``augment_mdp_to_deterministic`` helper defined in
``gridworld_core.py`` (the same core used by gridworld.py, puddleworld.py,
and taxiworld.py — see that module for the shared plumbing).

RockWorld is a GridWorld with valuable rocks (map value ``1``) and dangerous
rocks (``2``). Rocks only change the *reward*; they do **not** block movement,
so the determinized transition array has the same structure as a plain grid
with the same obstacles.

Quick start
-----------
    from rockworld import generate_determinized_models
    out = generate_determinized_models(size=4, num_humans=3,
                                       obstacles_percent=0.1, rock_percent=0.3,
                                       seed=0)
    T_R, s0, g = out["robot"][:3]
    print(out["total_determinizing_time"])
"""

import time
import random

import numpy as np

from gridworld_core import GridWorld, augment_mdp_to_deterministic


# ─────────────────────────────────────────────────────────────────────────────
# RockWorld (ported from RockWorldClass.py)
# ─────────────────────────────────────────────────────────────────────────────

class RockWorld(GridWorld):
    """GridWorld with valuable rocks (``1``) and dangerous rocks (``2``). Rocks
    change rewards only; they do not block movement, so transitions match a
    plain grid's."""

    def __init__(self, size=5, start=None, goal=None, obstacles_percent=0.1,
                 rock_percent=0.3, valuable_rock_ratio=0.4,
                 valuable_rock_reward=10, dangerous_rock_penalty=-5,
                 slip_prob=0.1, discount=0.99, max_tries=100, obstacle_seed=1):
        super().__init__(size=size, start=start, goal=goal,
                         obstacles_percent=obstacles_percent,
                         slip_prob=slip_prob, discount=discount,
                         max_tries=max_tries, obstacle_seed=obstacle_seed)
        self.rock_percent = rock_percent
        self.valuable_rock_ratio = valuable_rock_ratio
        self.valuable_rock_reward = valuable_rock_reward
        self.dangerous_rock_penalty = dangerous_rock_penalty
        self.place_rocks()
        self.reward_func = self.rock_reward_func

    def place_rocks(self):
        total_rocks = int(self.size * self.size * self.rock_percent)
        valuable_rocks = int(total_rocks * self.valuable_rock_ratio)
        dangerous_rocks = total_rocks - valuable_rocks
        for _ in range(valuable_rocks):
            self.place_rock(1)
        for _ in range(dangerous_rocks):
            self.place_rock(2)

    def place_rock(self, rock_type):
        while True:
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            if self.map[x, y] == 0:
                self.map[x, y] = rock_type
                break

    def rock_reward_func(self, state, action, next_state):
        x, y = next_state[0]
        if self.check_goal_reached(next_state):
            return self.valuable_rock_reward
        elif self.map[x, y] == 1:
            self.map[x, y] = 0
            return self.valuable_rock_reward
        elif self.map[x, y] == 2:
            return self.dangerous_rock_penalty
        else:
            return -1

    def visualize(self):
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if self.map[i, j] == -1:
                    row += "# "
                elif self.map[i, j] == 1:
                    row += "V "
                elif self.map[i, j] == 2:
                    row += "D "
                elif (i, j) == self.start_pos:
                    row += "S "
                elif (i, j) == self.goal_pos:
                    row += "G "
                else:
                    row += ". "
            print(row)


def generate_and_visualize_rockworld(size, start, goal, obstacles_percent, rock_percent,
                                     model_type="Model", obstacle_seed=None):
    """Generate a ``RockWorld`` (mirrors the function in experiments.py)."""
    return RockWorld(size=size, start=start, goal=goal,
                     obstacles_percent=obstacles_percent,
                     rock_percent=rock_percent,
                     obstacle_seed=obstacle_seed)


# ─────────────────────────────────────────────────────────────────────────────
# High-level driver — robot + N humans, with compute timing
# ─────────────────────────────────────────────────────────────────────────────

def _make_determinized(size, obstacles_percent, rock_percent, model_type, visualize=False):
    """Generate one rock world and determinize it; returns (next_states, s0, g, det_time).

    If ``visualize`` is True, print the generated map before determinizing.
    """
    mdp = generate_and_visualize_rockworld(
        size=size, start=(0, 0), goal=(size - 1, size - 1),
        obstacles_percent=obstacles_percent, rock_percent=rock_percent,
        model_type=model_type, obstacle_seed=random.randint(1, 10000))
    if visualize:
        print(f"\n{model_type}:")
        mdp.visualize()
    t0 = time.time()
    next_states, start_idx, goal_idx = augment_mdp_to_deterministic(mdp)
    return next_states, start_idx, goal_idx, time.time() - t0


def generate_determinized_models(size=4, num_humans=3, obstacles_percent=0.1,
                                 rock_percent=0.3, seed=None, verbose=True,
                                 visualize=False):
    """Build a robot model + ``num_humans`` human RockWorld models and determinize each.

    Parameters
    ----------
    size : int              grid side length
    num_humans : int        number of human models
    obstacles_percent : float   obstacle density in [0, 1]
    rock_percent : float    rock density in [0, 1]
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

    robot = _make_determinized(size, obstacles_percent, rock_percent, "Robot Model", visualize)
    humans = [_make_determinized(size, obstacles_percent, rock_percent, f"Human Model {i + 1}", visualize)
              for i in range(num_humans)]

    det_times = [robot[3]] + [h[3] for h in humans]
    total = float(sum(det_times))

    if verbose:
        print(f"[rockworld] size={size} obstacles={obstacles_percent} "
              f"rocks={rock_percent} humans={len(humans)}")
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
                                       obstacles_percent=0.1, rock_percent=0.3, seed=0, visualize=True)
    T_R, s0, g, _ = out["robot"]
    print("\nRobot determinized transition array shape:", T_R.shape)
    print("start index:", s0, " goal index:", g)
    print("num human models:", len(out["humans"]))
    print("total determinizing time (s):", out["total_determinizing_time"])
