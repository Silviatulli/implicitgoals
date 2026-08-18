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


def build_gridworld(start, goal, obstacle_density,
                                     obstacle_seed=None,
                                     rooms_per_side=1, room_side=5, slip_prob=0.0):
    """Build one ``GridWorld``.  Returns a solvable grid, or raises.

    Solvability is settled inside ``GridWorld.__init__``, which redraws the
    layout up to ``DEFAULT_MAX_TRIES`` times and then raises
    ``UnsolvableLayout`` rather than substituting an easier board.  So there is
    nothing here for a caller to retry, and never an unsolvable grid handed back
    — the two outcomes are a good grid and an exception.
    """
    return GridWorld(start=start, goal=goal,
                     obstacle_density=obstacle_density,
                     rooms_per_side=rooms_per_side,
                     room_side=room_side, slip_prob=slip_prob,
                     obstacle_seed=obstacle_seed)


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
    mdp = build_gridworld(
        start=(0, 0), goal=(n - 1, n - 1),
        obstacle_density=obstacle_density,
        rooms_per_side=rooms_per_side,
        room_side=room_side, slip_prob=slip_prob,
        obstacle_seed=random.randint(1, 10000))
    if visualize:
        print(f"\n{model_type}:")
        mdp.visualize()
    t0 = time.time()
    next_states, start_idx, goal_idx = augment_mdp_to_deterministic(mdp)
    # The MDP object is handed back alongside the matrix, because the matrix
    # alone has forgotten the board: it is integer state IDs and nothing else.
    # Hypothesis 3 needs to know where each state *sits* — it ranks bottlenecks
    # by straight-line distance to the goal — and reads that off the MDP's state
    # space, where state[0] is the (row, col) cell.
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
    humans = [_make_determinized(obstacle_density, f"Human Model {i + 1}",
                                 visualize, rooms_per_side=rooms_per_side,
                                 room_side=room_side, slip_prob=slip_prob)
              for i in range(num_humans)]

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
