"""
gridworld.py — GridWorld + determinized-MDP generator.
========================================================

Builds on the shared ``GridWorld`` MDP and ``augment_mdp_to_deterministic``
helper defined in ``gridworld_core.py`` (the same core used by puddleworld.py,
rockworld.py, and taxiworld.py — see that module for the shared plumbing).

  1. generate random GridWorld environments, and
  2. turn them into the deterministic ``next_states[state, action] -> next_state``
     integer array (via :func:`augment_mdp_to_deterministic`).

Quick start
-----------
    from gridworld import generate_determinized_models
    out = generate_determinized_models(room_side=4, num_humans=3,
                                       obstacle_density=0.1, seed=0)
    T_R, s0, g = out["robot"][:3]          # robot determinized transition array
    print(out["total_determinizing_time"]) # compute time (seconds)
"""

import time
import random

import numpy as np

from gridworld_core import GridWorld, augment_mdp_to_deterministic, board_side


def generate_and_visualize_gridworld(start, goal, obstacle_density,
                                     max_attempts=100, model_type="Model", obstacle_seed=None,
                                     rooms_per_side=1, room_side=5, slip_prob=0.0):
    """Generate a solvable ``GridWorld`` (retries up to ``max_attempts``).

    Returns the ``GridWorld`` instance, or ``None`` if no solvable layout was
    found.
    """
    for _ in range(max_attempts):
        grid = GridWorld(start=start, goal=goal,
                         obstacle_density=obstacle_density,
                         rooms_per_side=rooms_per_side,
                         room_side=room_side, slip_prob=slip_prob,
                         obstacle_seed=obstacle_seed)
        if grid.check_for_path():
            return grid
    return None


# ─────────────────────────────────────────────────────────────────────────────
# High-level driver — robot + N humans, with compute timing
# ─────────────────────────────────────────────────────────────────────────────

def _make_determinized(obstacle_density, model_type, visualize=False,
                       rooms_per_side=1, room_side=5, slip_prob=0.0):
    """Generate one grid and determinize it; returns (next_states, s0, g, det_time).

    The two corners are derived from the same board the grid will build, so they
    cannot name a cell that is off it.  If ``visualize`` is True, print the
    generated map before determinizing.
    """
    n = board_side(rooms_per_side, room_side)
    mdp = generate_and_visualize_gridworld(
        start=(0, 0), goal=(n - 1, n - 1),
        obstacle_density=obstacle_density,
        rooms_per_side=rooms_per_side,
        room_side=room_side, slip_prob=slip_prob,
        model_type=model_type, obstacle_seed=random.randint(1, 10000))
    if mdp is None:
        return None
    if visualize:
        print(f"\n{model_type}:")
        mdp.visualize()
    t0 = time.time()
    next_states, start_idx, goal_idx = augment_mdp_to_deterministic(mdp)
    # The MDP itself is returned too: it carries the stochastic transition
    # probabilities that determinization discards, which Hypothesis 3 needs.
    return next_states, start_idx, goal_idx, time.time() - t0, mdp


def generate_determinized_models(num_humans=3, obstacle_density=0.1,
                                 seed=None, verbose=True,
                                 visualize=False, rooms_per_side=1, room_side=4,
                                 slip_prob=0.0):
    """Build a robot model + ``num_humans`` human models and determinize each.

    Parameters
    ----------
    num_humans : int        number of human models to generate
    obstacle_density : float   obstacle density in [0, 1]
    rooms_per_side : int    rooms along each side of the board; 1 is the open
                            board, one room and no walls
    room_side : int         cells along each side of one room, so the board is
                            ``rooms_per_side * room_side`` — walls are thin, and no
                            cell is spent on them.  Every model gets the same walls
                            *and* the same doors (each at the middle of its wall);
                            only the obstacles differ, so the humans disagree about
                            which route is forced rather than about where a wall
                            opens.
    seed : int or None      seeds ``random``/``numpy`` for reproducibility
                            (None -> fresh randomness each call)
    verbose : bool          print a short timing summary
    visualize : bool        print each generated map (robot + humans)

    Returns
    -------
    dict with keys:
      'robot'   : (next_states, start_idx, goal_idx, det_time)
      'robot_mdp': the robot's un-determinized MDP (stochastic model)
      'humans'  : list of (next_states, start_idx, goal_idx, det_time)
      'determinizing_times'      : [robot_time, human1_time, ...]
      'total_determinizing_time' : float (seconds)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    robot = _make_determinized(obstacle_density, "Robot Model", visualize,
                               rooms_per_side=rooms_per_side,
                               room_side=room_side, slip_prob=slip_prob)
    if robot is None:
        raise RuntimeError("Failed to generate a solvable robot GridWorld.")

    humans = []
    for i in range(num_humans):
        h = _make_determinized(obstacle_density, f"Human Model {i + 1}",
                               visualize, rooms_per_side=rooms_per_side,
                               room_side=room_side, slip_prob=slip_prob)
        if h is not None:
            humans.append(h)

    det_times = [robot[3]] + [h[3] for h in humans]
    total = float(sum(det_times))

    if verbose:
        size = board_side(rooms_per_side, room_side)
        geometry = (f"{size}x{size} open board" if rooms_per_side == 1 else
                    f"{size}x{size} board = {rooms_per_side}x{rooms_per_side} rooms "
                    f"of {room_side}x{room_side}")
        print(f"[gridworld] {geometry}, {size * size} cells, "
              f"density={obstacle_density} humans={len(humans)}")
        print(f"  robot: {robot[0].shape[0]} states x {robot[0].shape[1]} actions "
              f"(determinized in {robot[3]:.4f}s)")
        print(f"  total determinizing time: {total:.4f}s")

    return {
        "robot": robot[:4],
        "robot_mdp": robot[4],
        "humans": [h[:4] for h in humans],
        "determinizing_times": det_times,
        "total_determinizing_time": total,
    }


if __name__ == "__main__":
    out = generate_determinized_models(room_side=4, num_humans=3,
                                       obstacle_density=0.1, seed=0, visualize=True)
    T_R, s0, g, _ = out["robot"]
    print("\nRobot determinized transition array shape:", T_R.shape)
    print("start index:", s0, " goal index:", g)
    print("num human models:", len(out["humans"]))
    print("total determinizing time (s):", out["total_determinizing_time"])
