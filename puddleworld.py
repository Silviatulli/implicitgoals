"""
puddleworld.py — PuddleWorld + determinized-MDP generator.
=============================================================

Builds on the shared ``GridWorld`` MDP and ``augment_mdp_to_deterministic``
helper defined in ``gridworld_core.py`` (the same core used by gridworld.py, rockworld.py,
and taxiworld.py — see that module for the shared plumbing).

PuddleWorld is a GridWorld where some empty cells become "puddles" (map value
``0.5``). Puddles only change the *reward* (a penalty for stepping in one); they
do **not** block movement, so the determinized transition array has the same
structure as a plain grid with the same obstacles.

Quick start
-----------
    from puddleworld import generate_determinized_models
    out = generate_determinized_models(room_side=4, num_humans=3,
                                       obstacle_density=0.1, puddle_density=0.2,
                                       seed=0)
    T_R, s0, g = out["robot"][:3]
    print(out["total_determinizing_time"])
"""

import time
import random

import numpy as np

from gridworld_core import GridWorld, augment_mdp_to_deterministic, board_side


# ─────────────────────────────────────────────────────────────────────────────
# PuddleWorld
# ─────────────────────────────────────────────────────────────────────────────

class PuddleWorld(GridWorld):
    """GridWorld with puddles (map value ``0.5``). Puddles penalize rewards but
    do not block movement, so transitions match a plain grid's."""

    def __init__(self, start=None, goal=None, obstacle_density=0.1,
                 puddle_density=0.2, puddle_penalty=-1, goal_reward=10,
                 slip_prob=0.0, discount=0.99, max_tries=100, obstacle_seed=1,
                 rooms_per_side=1, room_side=5):
        super().__init__(start=start, goal=goal,
                         obstacle_density=obstacle_density,
                         slip_prob=slip_prob, discount=discount,
                         max_tries=max_tries, obstacle_seed=obstacle_seed,
                         rooms_per_side=rooms_per_side,
                         room_side=room_side)
        self.puddle_density = puddle_density
        self.puddle_penalty = puddle_penalty
        self.goal_reward = goal_reward
        self.place_puddles()
        self.reward_func = self.puddle_reward_func

    def place_puddles(self):
        """Puddles on free cells only, and never on the start or the goal.

        The goal exclusion is load-bearing, not tidiness.  The goal is absorbing,
        so a puddle there makes the sink pay puddle_penalty on its own self-loop
        for ever and V(goal) settles at puddle_penalty/(1-gamma) = -100 instead
        of 0 — the mirror image of the goal_reward leak puddle_reward_func
        guards against.  Reusing protected_cells() keeps this in step with the
        obstacle placement, which excludes exactly the same cells.
        """
        total_puddles = int(self.board_side * self.board_side * self.puddle_density)
        protected = self.protected_cells()
        for _ in range(total_puddles):
            x = self.rng.randint(self.board_side)
            y = self.rng.randint(self.board_side)
            if self.map[x, y] == 0 and (x, y) not in protected:
                self.map[x, y] = 0.5

    def puddle_reward_func(self, state, action, next_state):
        x, y = next_state[0]
        # "and not already there" matters: the goal is absorbing, so without it
        # the goal's own self-loop keeps paying goal_reward and V(goal) settles
        # at goal_reward/(1-gamma) = 100 instead of 0.  The reward belongs on the
        # transition that *enters* the goal, exactly as in GridWorld.
        if (self.check_goal_reached(next_state[0])
                and not self.check_goal_reached(state[0])):
            return self.goal_reward
        elif self.map[x, y] == 0.5:
            return self.puddle_penalty
        else:
            return 0

    CHAR_STYLE = {**GridWorld.CHAR_STYLE,
                   "~": ("#81d4fa", "#01579b")}      # puddle: shallow water

    def cell_char(self, i, j):
        """``~`` for a puddle; everything else renders as in a plain grid."""
        if self.map[i, j] == 0.5:
            return "~"
        return super().cell_char(i, j)


def generate_and_visualize_puddleworld(start, goal, obstacle_density, puddle_density,
                                       model_type="Model", obstacle_seed=None,
                                       puddle_penalty=-1, goal_reward=10,
                                       rooms_per_side=1, room_side=5):
    """Generate a ``PuddleWorld``.

    ``puddle_penalty`` and ``goal_reward`` are forwarded rather than dropped:
    their ratio is what decides whether V_R encodes distance-to-goal or merely
    local puddle density, and Hypothesis 3 ranks bottlenecks by V_R.  At a 1:1
    ratio the goal reward is the same size as the penalties accumulated on the
    way to it and corr(V_R, distance) averages only -0.56; the 10:1 default
    reaches -0.82.
    """
    return PuddleWorld(start=start, goal=goal,
                       obstacle_density=obstacle_density,
                       puddle_density=puddle_density,
                       puddle_penalty=puddle_penalty,
                       goal_reward=goal_reward,
                       obstacle_seed=obstacle_seed,
                       rooms_per_side=rooms_per_side,
                       room_side=room_side)


# ─────────────────────────────────────────────────────────────────────────────
# High-level driver — robot + N humans, with compute timing
# ─────────────────────────────────────────────────────────────────────────────

def _make_determinized(obstacle_density, puddle_density, model_type, visualize=False,
                       rooms_per_side=1, room_side=5):
    """Generate one puddle world and determinize it; returns (next_states, s0, g, det_time).

    The two corners are derived from the same board the grid will build, so they
    cannot name a cell that is off it.  If ``visualize`` is True, print the
    generated map before determinizing.
    """
    n = board_side(rooms_per_side, room_side)
    mdp = generate_and_visualize_puddleworld(
        start=(0, 0), goal=(n - 1, n - 1),
        obstacle_density=obstacle_density, puddle_density=puddle_density,
        rooms_per_side=rooms_per_side,
        room_side=room_side,
        model_type=model_type, obstacle_seed=random.randint(1, 10000))
    if visualize:
        print(f"\n{model_type}:")
        mdp.visualize()
    t0 = time.time()
    next_states, start_idx, goal_idx = augment_mdp_to_deterministic(mdp)
    # The MDP itself is returned too: it carries the stochastic transition
    # probabilities that determinization discards, which Hypothesis 3 needs.
    return next_states, start_idx, goal_idx, time.time() - t0, mdp


def generate_determinized_models(num_humans=3, obstacle_density=0.1,
                                 puddle_density=0.2, seed=None, verbose=True,
                                 visualize=False, rooms_per_side=1, room_side=4):
    """Build a robot model + ``num_humans`` human PuddleWorld models and determinize each.

    Parameters
    ----------
    num_humans : int        number of human models
    obstacle_density : float   obstacle density in [0, 1]
    puddle_density : float  puddle density in [0, 1]
    rooms_per_side : int    rooms along each side of the board; 1 is the open
                            board, one room and no walls
    room_side : int         cells along each side of one room, so the board is
                            ``rooms_per_side * room_side`` — walls are thin, and
                            no cell is spent on them
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

    robot = _make_determinized(obstacle_density, puddle_density, "Robot Model", visualize,
                               rooms_per_side=rooms_per_side,
                               room_side=room_side)
    humans = [_make_determinized(obstacle_density, puddle_density, f"Human Model {i + 1}",
                                 visualize, rooms_per_side=rooms_per_side,
                                 room_side=room_side)
              for i in range(num_humans)]

    det_times = [robot[3]] + [h[3] for h in humans]
    total = float(sum(det_times))

    if verbose:
        size = board_side(rooms_per_side, room_side)
        geometry = (f"{size}x{size} open board" if rooms_per_side == 1 else
                    f"{size}x{size} board = {rooms_per_side}x{rooms_per_side} rooms "
                    f"of {room_side}x{room_side}")
        print(f"[puddleworld] {geometry}, {size * size} cells, "
              f"obstacles={obstacle_density} puddles={puddle_density} "
              f"humans={len(humans)}")
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
                                       obstacle_density=0.1, puddle_density=0.2, seed=0, visualize=True)
    T_R, s0, g, _ = out["robot"]
    print("\nRobot determinized transition array shape:", T_R.shape)
    print("start index:", s0, " goal index:", g)
    print("num human models:", len(out["humans"]))
    print("total determinizing time (s):", out["total_determinizing_time"])
