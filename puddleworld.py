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
                                       obstacle_density=0.1, puddle_density=0.1,
                                       seed=0)
    T_R, s0, g = out["robot"][:3]
    print(out["total_determinizing_time"])
"""

import time
import random

import numpy as np

from gridworld_core import (GridWorld, augment_mdp_to_deterministic, board_side,
                            DEFAULT_MAX_TRIES)


# ─────────────────────────────────────────────────────────────────────────────
# PuddleWorld
# ─────────────────────────────────────────────────────────────────────────────

class PuddleWorld(GridWorld):
    """GridWorld with puddles (map value ``0.5``). Puddles penalize rewards but
    do not block movement, so transitions match a plain grid's."""

    def __init__(self, start=None, goal=None, obstacle_density=0.1,
                 puddle_density=0.1, puddle_penalty=-1, goal_reward=10,
                 slip_prob=0.0, gamma=0.99, max_tries=DEFAULT_MAX_TRIES,
                 obstacle_seed=1,
                 rooms_per_side=1, room_side=5, puddle_positions=None):
        self.puddle_density = puddle_density
        self.puddle_penalty = puddle_penalty
        self.goal_reward = goal_reward
        # Set *before* super().__init__() so protected_cells() sees them while the
        # obstacles are drawn — same rule as the start, the goal, the doors,
        # TaxiWorld's passenger and RockWorld's rocks: shared across the robot and
        # every human of an instance, with only the obstacles varying.
        self.puddle_positions = list(puddle_positions or [])
        self._shared_puddles = bool(puddle_positions)

        super().__init__(start=start, goal=goal,
                         obstacle_density=obstacle_density,
                         slip_prob=slip_prob, gamma=gamma,
                         max_tries=max_tries, obstacle_seed=obstacle_seed,
                         rooms_per_side=rooms_per_side,
                         room_side=room_side)

        if self._shared_puddles:
            # Obstacles avoided these cells, so painting cannot bury a puddle and
            # cannot overwrite an obstacle.
            self.paint_puddles()
        else:
            self.place_puddles()
        self.reward_func = self.puddle_reward_func

    def protected_cells(self):
        """Keep obstacles off every shared puddle, as well as start and goal.

        Needed because the shared layout is painted *after* the obstacles: without
        it, a puddle landing on an obstacle cell would silently erase that
        obstacle and the models would stop agreeing on the map.
        """
        return super().protected_cells() | set(self.puddle_positions)

    def paint_puddles(self):
        """Write the shared puddle layout onto the map."""
        for pos in self.puddle_positions:
            self.map[pos] = 0.5

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



def sample_puddle_layout(board, puddle_density, rng, protected=()):
    """One puddle layout for a whole instance, drawn once and shared.

    Mirrors PuddleWorld.place_puddles' count, but draws distinct cells rather
    than sampling with rejection, so the board carries exactly the requested
    number instead of however many survived collisions.

    `rng` is a module-level ``random``: the layout belongs to the instance, not
    to any one model.
    """
    total_puddles = int(board * board * puddle_density)
    free = [(i, j) for i in range(board) for j in range(board)
            if (i, j) not in set(protected)]
    return rng.sample(free, min(total_puddles, len(free)))

def build_puddleworld(start, goal, obstacle_density, puddle_density,
                                       obstacle_seed=None,
                                       puddle_penalty=-1, goal_reward=10,
                                       rooms_per_side=1, room_side=5, slip_prob=0.0,
                                       puddle_positions=None):
    """Generate a ``PuddleWorld``.

    ``puddle_penalty`` and ``goal_reward`` are forwarded rather than dropped, so
    that a caller solving one of these grids as an ordinary MDP gets a sensible
    reward.  Their *ratio* is what matters there: at 1:1 the goal is worth no
    more than the puddles crossed to reach it, so an optimal policy dodges water
    instead of finishing, while the 10:1 default makes reaching the goal
    dominate.

    The benchmark pipeline reads neither of them.  Bottlenecks come from
    reachability alone, so puddles change what a route *costs* without changing
    which routes exist — which is why puddleworld's numbers stay close to plain
    gridworld's.
    """
    return PuddleWorld(start=start, goal=goal,
                       obstacle_density=obstacle_density,
                       puddle_density=puddle_density,
                       puddle_penalty=puddle_penalty,
                       goal_reward=goal_reward,
                       obstacle_seed=obstacle_seed, slip_prob=slip_prob,
                       rooms_per_side=rooms_per_side,
                       room_side=room_side,
                       puddle_positions=puddle_positions)


# ─────────────────────────────────────────────────────────────────────────────
# High-level driver — robot + N humans, with compute timing
# ─────────────────────────────────────────────────────────────────────────────

def _make_determinized(obstacle_density, puddle_density, model_type,
                       obstacle_seed, visualize=False,
                       rooms_per_side=1, room_side=5, slip_prob=0.0,
                       puddle_positions=None):
    """Generate one puddle world and determinize it; returns (next_states, s0, g, det_time).

    The two corners are derived from the same board the grid will build, so they
    cannot name a cell that is off it.  If ``visualize`` is True, print the
    generated map before determinizing.
    """
    n = board_side(rooms_per_side, room_side)
    mdp = build_puddleworld(
        start=(0, 0), goal=(n - 1, n - 1),
        obstacle_density=obstacle_density, puddle_density=puddle_density,
        rooms_per_side=rooms_per_side,
        room_side=room_side, slip_prob=slip_prob,
        puddle_positions=puddle_positions,
        obstacle_seed=obstacle_seed)
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


def plan_determinized_models(num_humans=3, obstacle_density=0.1,
                             puddle_density=0.1, seed=None, visualize=False,
                             rooms_per_side=1, room_side=4, slip_prob=0.0):
    """Draw every random choice this instance makes, and build nothing yet.

    See gridworld.plan_determinized_models for why this split exists.  The order
    of the draws is the order generate_determinized_models made them: the shared
    puddle layout first, then one obstacle seed per model, robot first.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # One puddle layout for the whole instance, as with the start, the goal, the
    # doors, the taxi passenger and the rocks.  Only the obstacles vary.
    size = board_side(rooms_per_side, room_side)
    puddle_positions = sample_puddle_layout(
        size, puddle_density, random, protected=((0, 0), (size - 1, size - 1)))
    seeds = [random.randint(1, 10000) for _ in range(num_humans + 1)]

    def build(i):
        label = "Robot Model" if i == 0 else f"Human Model {i}"
        return _make_determinized(obstacle_density, puddle_density, label,
                                  seeds[i], visualize,
                                  rooms_per_side=rooms_per_side,
                                  room_side=room_side, slip_prob=slip_prob,
                                  puddle_positions=puddle_positions)

    return build, num_humans + 1


def generate_determinized_models(num_humans=3, obstacle_density=0.1,
                                 puddle_density=0.1, seed=None, verbose=True,
                                 visualize=False, rooms_per_side=1, room_side=4,
                                 slip_prob=0.0):
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
    build, n_models = plan_determinized_models(
        num_humans=num_humans, obstacle_density=obstacle_density,
        puddle_density=puddle_density, seed=seed, visualize=visualize,
        rooms_per_side=rooms_per_side, room_side=room_side, slip_prob=slip_prob)
    robot = build(0)
    humans = [build(i) for i in range(1, n_models)]

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
                                       obstacle_density=0.1, puddle_density=0.1, seed=0, visualize=True)
    T_R, s0, g, _ = out["robot"]
    print("\nRobot determinized transition array shape:", T_R.shape)
    print("start index:", s0, " goal index:", g)
    print("num human models:", len(out["humans"]))
    print("total determinizing time (s):", out["total_determinizing_time"])
