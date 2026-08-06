"""
gridworld.py — GridWorld + determinized-MDP generator.
========================================================

Exported from the *implicitgoals* research repo. Builds on the shared
``GridWorld`` MDP and ``augment_mdp_to_deterministic`` helper defined in
``gridworld_core.py`` (the same core used by puddleworld.py, rockworld.py,
and taxiworld.py — see that module for the shared plumbing).

  1. generate random GridWorld (or four-rooms) environments, and
  2. turn them into the deterministic ``next_states[state, action] -> next_state``
     integer array **exactly** as ``parallel_experiments_2.py`` does
     (via :func:`augment_mdp_to_deterministic`).

Quick start
-----------
    from gridworld import generate_determinized_models
    out = generate_determinized_models(size=4, num_humans=3,
                                       obstacles_percent=0.1, seed=0)
    T_R, s0, g = out["robot"][:3]          # robot determinized transition array
    print(out["total_determinizing_time"]) # compute time (seconds)
"""

import time
import random

import numpy as np

from gridworld_core import GridWorld, augment_mdp_to_deterministic


def generate_and_visualize_gridworld(size, start, goal, obstacles_percent, divide_rooms,
                                     max_attempts=100, model_type="Model", obstacle_seed=None):
    """Generate a solvable ``GridWorld`` (retries up to ``max_attempts``).

    Mirrors the function of the same name in ``GridWorldClass.py``. Returns the
    ``GridWorld`` instance, or ``None`` if no solvable layout was found.
    """
    for _ in range(max_attempts):
        grid = GridWorld(size=size, start=start, goal=goal,
                         obstacles_percent=obstacles_percent,
                         divide_rooms=divide_rooms, obstacle_seed=obstacle_seed)
        if grid.check_for_path():
            return grid
    return None


# ─────────────────────────────────────────────────────────────────────────────
# High-level driver — robot + N humans, with compute timing
# ─────────────────────────────────────────────────────────────────────────────

def _make_determinized(size, obstacles_percent, divide_rooms, model_type, visualize=False):
    """Generate one grid and determinize it; returns (next_states, s0, g, det_time).

    If ``visualize`` is True, print the generated map before determinizing.
    """
    mdp = generate_and_visualize_gridworld(
        size=size, start=(0, 0), goal=(size - 1, size - 1),
        obstacles_percent=obstacles_percent, divide_rooms=divide_rooms,
        model_type=model_type, obstacle_seed=random.randint(1, 10000))
    if mdp is None:
        return None
    if visualize:
        print(f"\n{model_type}:")
        mdp.visualize()
    t0 = time.time()
    next_states, start_idx, goal_idx = augment_mdp_to_deterministic(mdp)
    return next_states, start_idx, goal_idx, time.time() - t0


def generate_determinized_models(size=4, num_humans=3, obstacles_percent=0.1,
                                 divide_rooms=False, seed=None, verbose=True,
                                 visualize=False):
    """Build a robot model + ``num_humans`` human models and determinize each.

    Parameters
    ----------
    size : int              grid side length
    num_humans : int        number of human models to generate
    obstacles_percent : float   obstacle density in [0, 1]
    divide_rooms : bool     True for a four-rooms layout
    seed : int or None      seeds ``random``/``numpy`` for reproducibility
                            (None -> fresh randomness each call)
    verbose : bool          print a short timing summary
    visualize : bool        print each generated map (robot + humans)

    Returns
    -------
    dict with keys:
      'robot'   : (next_states, start_idx, goal_idx, det_time)
      'humans'  : list of (next_states, start_idx, goal_idx, det_time)
      'determinizing_times'      : [robot_time, human1_time, ...]
      'total_determinizing_time' : float (seconds)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    robot = _make_determinized(size, obstacles_percent, divide_rooms, "Robot Model", visualize)
    if robot is None:
        raise RuntimeError("Failed to generate a solvable robot GridWorld.")

    humans = []
    for i in range(num_humans):
        h = _make_determinized(size, obstacles_percent, divide_rooms, f"Human Model {i + 1}", visualize)
        if h is not None:
            humans.append(h)

    det_times = [robot[3]] + [h[3] for h in humans]
    total = float(sum(det_times))

    if verbose:
        print(f"[gridworld] size={size} density={obstacles_percent} "
              f"four_rooms={divide_rooms} humans={len(humans)}")
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
                                       obstacles_percent=0.1, seed=0, visualize=True)
    T_R, s0, g, _ = out["robot"]
    print("\nRobot determinized transition array shape:", T_R.shape)
    print("start index:", s0, " goal index:", g)
    print("num human models:", len(out["humans"]))
    print("total determinizing time (s):", out["total_determinizing_time"])
