"""
rockworld.py — RockWorld + determinized-MDP generator.
==========================================================

Builds on the shared ``GridWorld`` MDP and ``augment_mdp_to_deterministic``
helper defined in ``gridworld_core.py`` (the same core used by gridworld.py, puddleworld.py,
and taxiworld.py — see that module for the shared plumbing).

RockWorld is a GridWorld with valuable rocks (map value ``1``) and dangerous
rocks (``2``). Rocks do **not** block movement — they only change the *reward*
and what the agent has collected.

A state is ``[position, collected]``, where ``collected`` is a single **boolean**
— "have I picked up a valuable rock yet?".  So the state space is exactly
``2 * board^2`` however many rocks the board carries, and ``rock_density`` is
free to generate the map without a cap on the rock count.

The task is *collect, then deliver*.  The goal absorbs only while carrying:
``[goal, True]`` is the one goal state, and ``[goal, False]`` is an ordinary
passable cell you may walk across before collecting anything.  That is the same
shape as TaxiWorld, whose sink is the ``delivered`` flag and not the destination
cell — and it means a solvable layout needs two legs, start -> some valuable
rock -> goal.

Note that "collect at least one rock" is a **disjunctive** requirement: with
several reachable rocks no individual rock lies on every path, so no rock is a
bottleneck on its own.  What the dominator analysis can see is whatever is
shared on the way to them — a door every route to the rocks must cross.

Rewards are cost-to-go: ``−1`` per step, ``−5`` more for a dangerous rock, and
``+10`` on first collection.  **The reward is known broken** — one boolean
cannot say which rocks were taken, so only the first pays — and is tracked in
todo.md.  Nothing in the benchmark reads it: bottlenecks come from reachability,
not from reward.

Quick start
-----------
    from rockworld import generate_determinized_models
    out = generate_determinized_models(rooms_per_side=3, room_side=3,
                                       num_humans=3, obstacle_density=0.1,
                                       seed=0)

The board has to be large enough for `rock_density` to yield at least one
*valuable* rock, or the goal can never absorb and RockWorld raises.  At the
default densities that means roughly 30 cells or more.
    T_R, s0, g = out["robot"][:3]
    print(out["total_determinizing_time"])
"""

import time
import random

import numpy as np

from gridworld_core import (GridWorld, augment_mdp_to_deterministic, board_side,
                            seeded_rng, DEFAULT_MAX_TRIES)


# ─────────────────────────────────────────────────────────────────────────────
# RockWorld
# ─────────────────────────────────────────────────────────────────────────────

# Share of the rocks that are valuable; the rest are dangerous.  A module
# constant rather than only a default argument so the instance-level sampler and
# a standalone grid agree on the split.
#
# There is no cap on how many rocks a board may carry.  There used to have to be
# one: `collected` was a bit per valuable rock, so every rock doubled the state
# space and rock_density could not be honoured.  `collected` is now a single
# boolean — "have I picked anything up yet" — so the state space is 2 x board^2
# whatever the density, and rock_density alone decides the map.
VALUABLE_ROCK_RATIO = 0.4


class RockWorld(GridWorld):
    """GridWorld with valuable rocks (``1``) and dangerous rocks (``2``). Rocks
    change rewards only; they do not block movement, so transitions match a
    plain grid's.

    A state is ``[position, collected]``, where ``collected`` is one bit per
    valuable rock (in ``valuable_positions`` order).  Carrying that set in the
    state is what makes "each rock pays once" Markov.  Recording it by mutating
    ``self.map`` from inside the reward would corrupt the map and make the reward
    depend on evaluation order.
    """

    def __init__(self, start=None, goal=None, obstacle_density=0.1,
                 rock_density=0.1, valuable_rock_ratio=VALUABLE_ROCK_RATIO,
                 valuable_rock_reward=10, dangerous_rock_penalty=-5,
                 slip_prob=0.0, gamma=0.99, max_tries=DEFAULT_MAX_TRIES,
                 obstacle_seed=1,
                 rooms_per_side=1, room_side=5,
                 valuable_positions=None, dangerous_positions=None):
        self.rock_density = rock_density
        self.valuable_rock_ratio = valuable_rock_ratio
        self.valuable_rock_reward = valuable_rock_reward
        self.dangerous_rock_penalty = dangerous_rock_penalty

        # The rocks have to exist *before* super().__init__() runs, for two
        # reasons: protected_cells() keeps obstacles off them while the layout is
        # drawn, and check_for_path() — which decides whether a layout is
        # accepted — now has to reach a valuable rock before the goal.
        #
        # Shared positions come from the caller (generate_determinized_models
        # draws one layout for the whole instance, so the robot and every human
        # agree on where the rocks are).  A standalone grid draws its own here,
        # through the same sampler, so there is one code path and not two.
        if valuable_positions is None and dangerous_positions is None:
            obstacle_seed, self.rng = seeded_rng(obstacle_seed)
            valuable_positions, dangerous_positions = sample_rock_layout(
                board_side(rooms_per_side, room_side), rock_density,
                random.Random(obstacle_seed),
                valuable_rock_ratio=valuable_rock_ratio,
                protected=(start, goal) if start and goal else ())
        self.valuable_positions = list(valuable_positions or [])
        self.dangerous_positions = list(dangerous_positions or [])
        # Membership is asked once per transition, so keep it a set.
        self._valuable_set = set(self.valuable_positions)

        if not self._valuable_set:
            # Without one collectable rock the goal can never absorb, so the
            # layout search would run its full budget and then raise.  Say why
            # here instead, where the cause is visible.
            raise ValueError(
                f"rock_density {rock_density} with valuable_rock_ratio "
                f"{valuable_rock_ratio} puts no valuable rock on a "
                f"{board_side(rooms_per_side, room_side)}x"
                f"{board_side(rooms_per_side, room_side)} board, and the goal "
                f"only absorbs once one has been collected — so no layout can "
                f"ever be solvable.  Raise rock_density.")

        super().__init__(start=start, goal=goal,
                         obstacle_density=obstacle_density,
                         slip_prob=slip_prob, gamma=gamma,
                         max_tries=max_tries, obstacle_seed=obstacle_seed,
                         rooms_per_side=rooms_per_side,
                         room_side=room_side)

        # Obstacles avoided these cells, so painting cannot bury a rock, and
        # every model of an instance paints the same ones.
        self.paint_rocks()
        self.reward_func = self.rock_reward_func

    def protected_cells(self):
        """Keep obstacles off every rock, as well as the start and the goal.

        A shared rock has to survive each model's independent obstacle draw, for
        the same reason TaxiWorld protects the passenger: burying it under one
        model's obstacles would silently break the sharing.  An obstacle *next*
        to a rock is fine and is how a model can end up unable to collect it —
        that is variation, not corruption, and unlike the passenger it never
        makes the board unsolvable, since reaching the goal never requires
        collecting anything.
        """
        return (super().protected_cells()
                | set(self.valuable_positions) | set(self.dangerous_positions))

    def paint_rocks(self):
        """Write the shared rock layout onto the map (1 valuable, 2 dangerous)."""
        for pos in self.valuable_positions:
            self.map[pos] = 1
        for pos in self.dangerous_positions:
            self.map[pos] = 2

    # ── State space ──────────────────────────────────────────────────────────
    def create_state_space(self):
        """``[position, collected]`` for every cell, with ``collected`` a bool.

        ``collected`` answers one question — "have I picked up a valuable rock
        yet?" — so the state space is exactly ``2 * board^2``, whatever the rock
        density.  It does not grow with the number of rocks, which is why there
        is no cap on how many a board may carry.

        Nothing collapses here.  ``[goal, False]`` is an ordinary passable cell:
        you may walk over the goal before collecting anything and carry on.
        ``[goal, True]`` is the single absorbing state, and the only one the
        pipeline calls the goal — see get_goal_states and is_absorbing_state.
        """
        self.state_space = [[(i, j), collected]
                            for i in range(self.board_side)
                            for j in range(self.board_side)
                            for collected in (False, True)]

    def _next_collected(self, collected, next_pos):
        """The flag after stepping onto ``next_pos``.

        Monotone and never reset: once a valuable rock has been entered the flag
        stays True for the rest of the episode, including at the goal.  That is
        what makes "reach the goal *having collected something*" a property of
        the state rather than of the history.
        """
        return bool(collected) or next_pos in self._valuable_set

    # ── Dynamics and reward ──────────────────────────────────────────────────
    def is_absorbing_state(self, state):
        """The sink is *the goal while carrying*, not the goal cell.

        Overriding this is what keeps the goal passable until a rock has been
        collected: ``[goal, False]`` behaves like any other cell, and only
        ``[goal, True]`` self-loops.  Same shape as TaxiWorld, whose sink is the
        ``delivered`` flag rather than the destination cell.
        """
        return self.check_goal_reached(state[0]) and bool(state[1])

    def check_for_path(self):
        """Is the *goal state* reachable — that is, goal reached while carrying?

        GridWorld walks to the goal cell, which was enough while arriving there
        ended the episode.  It no longer does: the taxi problem's shape now
        applies here too, and the trajectory needs two legs, start -> some
        valuable rock -> goal.  Both are real questions on a room-grid board,
        where a one-way door can let the agent reach a rock and then strand it.

        "Some" rock, not "every": collecting is disjunctive, so a layout is
        solvable as soon as *one* valuable rock can be picked up and the goal
        reached afterwards.

        Runs from inside ``GridWorld.__init__``, which is why the rocks are
        sampled before that call.
        """
        if self.start_pos is None or self.goal_pos is None:
            return False
        from_start = self._reachable_positions(self.start_pos)
        return any(self.goal_pos in self._reachable_positions(rock)
                   for rock in self.valuable_positions if rock in from_start)

    def get_transition_probability(self, state, action, state_prime):
        """Position moves exactly as in a plain grid; the collection flag is a
        deterministic function of where you land, so any inconsistent successor
        has probability 0."""
        if bool(state_prime[1]) != self._next_collected(state[1],
                                                        state_prime[0]):
            return 0
        return super().get_transition_probability(state, action, state_prime)

    def rock_reward_func(self, state, action, next_state):
        """Cost-to-go: −1 per step, −5 more for entering a dangerous rock, and
        +10 the *first* time a valuable rock is collected.

        KNOWN BROKEN, deliberately, and tracked in todo.md.  "Each valuable rock
        pays once" is no longer expressible: `collected` is one boolean, so the
        state cannot say *which* rocks have been taken, only that something has.
        The bonus is therefore paid on the single False→True flip and every
        later rock is worth nothing.  Any value function built on this reward is
        wrong.

        Left running rather than deleted because nothing in the benchmark reads
        it — bottlenecks come from reachability, not reward — so this is a
        placeholder to be redesigned alongside the value function, not a live
        defect.  Making each rock pay again means putting the identity of the
        collected rocks back into the state, which is exactly the 2^k blow-up
        the boolean flag was introduced to remove.
        """
        if self.is_absorbing_state(state):
            return 0                                    # the sink pays nothing
        reward = -1
        next_pos = next_state[0]
        if not state[1] and next_state[1]:              # the one False->True flip
            reward += self.valuable_rock_reward
        if self.map[next_pos] == 2:
            reward += self.dangerous_rock_penalty
        return reward

    def get_init_state(self):
        return [self.start_pos, False]

    def get_goal_states(self):
        """The one goal state: on the goal cell, carrying.

        Exactly one, which the pipeline requires — extract_bottlenecks walks the
        dominator tree from a single goal and augment_mdp_to_deterministic takes
        get_goal_states()[0].  ``[goal, False]`` is deliberately not here: it is
        a cell you can stand on, not the end of the task.
        """
        return [[self.goal_pos, True]]

    CHAR_STYLE = {**GridWorld.CHAR_STYLE,
                   "V": ("#fdd835", "#4e342e"),      # valuable rock: worth +10
                   "D": ("#e53935", "#ffffff")}      # dangerous rock: costs -5

    def cell_char(self, i, j):
        """``V`` valuable rock, ``D`` dangerous one; the rest as in a plain grid."""
        if self.map[i, j] == 1:
            return "V"
        if self.map[i, j] == 2:
            return "D"
        return super().cell_char(i, j)


def sample_rock_layout(board, rock_density, rng,
                       valuable_rock_ratio=VALUABLE_ROCK_RATIO,
                       protected=()):
    """One rock layout: (valuable_positions, dangerous_positions).

    ``rock_density`` decides the count outright — ``board^2 * rock_density``
    rocks, a ``valuable_rock_ratio`` share of them valuable.  Nothing is capped,
    because the state space no longer grows with the rock count.

    Called twice over: once per instance in generate_determinized_models, whose
    result is handed to every model so the robot and the humans see the same
    rocks; and once inside RockWorld.__init__ for a standalone grid that was
    given no layout.  One sampler, so the two cannot drift apart.

    `rng` is a ``random.Random`` — the layout belongs to the instance, not to any
    one model's numpy stream.
    """
    total_rocks = int(board * board * rock_density)
    n_valuable = int(total_rocks * valuable_rock_ratio)
    free = [(i, j) for i in range(board) for j in range(board)
            if (i, j) not in set(protected)]
    picked = rng.sample(free, min(total_rocks, len(free)))
    return picked[:n_valuable], picked[n_valuable:]


def build_rockworld(start, goal, obstacle_density, rock_density,
                                     obstacle_seed=None,
                                     rooms_per_side=1, room_side=5, slip_prob=0.0,
                                     valuable_positions=None,
                                     dangerous_positions=None):
    """Generate a ``RockWorld``, on a shared rock layout when one is given."""
    return RockWorld(start=start, goal=goal,
                     obstacle_density=obstacle_density,
                     rock_density=rock_density,
                     obstacle_seed=obstacle_seed, slip_prob=slip_prob,
                     rooms_per_side=rooms_per_side,
                     room_side=room_side,
                     valuable_positions=valuable_positions,
                     dangerous_positions=dangerous_positions)


# ─────────────────────────────────────────────────────────────────────────────
# High-level driver — robot + N humans, with compute timing
# ─────────────────────────────────────────────────────────────────────────────

def _make_determinized(obstacle_density, rock_density, model_type, visualize=False,
                       rooms_per_side=1, room_side=5, slip_prob=0.0,
                       valuable_positions=None, dangerous_positions=None):
    """Generate one rock world and determinize it; returns (next_states, s0, g, det_time).

    The two corners are derived from the same board the grid will build, so they
    cannot name a cell that is off it.  If ``visualize`` is True, print the
    generated map before determinizing.
    """
    n = board_side(rooms_per_side, room_side)
    mdp = build_rockworld(
        start=(0, 0), goal=(n - 1, n - 1),
        obstacle_density=obstacle_density, rock_density=rock_density,
        rooms_per_side=rooms_per_side,
        room_side=room_side, slip_prob=slip_prob,
        valuable_positions=valuable_positions,
        dangerous_positions=dangerous_positions,
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
                                 rock_density=0.1, seed=None, verbose=True,
                                 visualize=False, rooms_per_side=1, room_side=4,
                                 slip_prob=0.0):
    """Build a robot model + ``num_humans`` human RockWorld models and determinize each.

    Parameters
    ----------
    num_humans : int        number of human models
    obstacle_density : float   obstacle density in [0, 1]
    rock_density : float    rock density in [0, 1]
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

    # One rock layout for the whole instance, as with the start, the goal, the
    # doors and TaxiWorld's passenger; only the obstacles vary between models.
    # Drawing it per model would give each human a different set of collectable
    # cells, so "reach the goal carrying" would mean a different task in each
    # one and the bottleneck sets the pipeline unions would not be comparable.
    size = board_side(rooms_per_side, room_side)
    valuable_positions, dangerous_positions = sample_rock_layout(
        size, rock_density, random, protected=((0, 0), (size - 1, size - 1)))

    robot = _make_determinized(obstacle_density, rock_density, "Robot Model", visualize,
                               rooms_per_side=rooms_per_side,
                               room_side=room_side, slip_prob=slip_prob,
                               valuable_positions=valuable_positions,
                               dangerous_positions=dangerous_positions)
    humans = [_make_determinized(obstacle_density, rock_density, f"Human Model {i + 1}",
                                 visualize, rooms_per_side=rooms_per_side,
                                 room_side=room_side, slip_prob=slip_prob,
                                 valuable_positions=valuable_positions,
                                 dangerous_positions=dangerous_positions)
              for i in range(num_humans)]

    det_times = [robot[3]] + [h[3] for h in humans]
    total = float(sum(det_times))

    if verbose:
        size = board_side(rooms_per_side, room_side)
        geometry = (f"{size}x{size} open board" if rooms_per_side == 1 else
                    f"{size}x{size} board = {rooms_per_side}x{rooms_per_side} rooms "
                    f"of {room_side}x{room_side}")
        print(f"[rockworld] {geometry}, {size * size} cells, "
              f"obstacles={obstacle_density} rocks={rock_density} "
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
    out = generate_determinized_models(rooms_per_side=3, room_side=3, num_humans=3,
                                       obstacle_density=0.1, seed=0, visualize=True)
    T_R, s0, g, _ = out["robot"]
    print("\nRobot determinized transition array shape:", T_R.shape)
    print("start index:", s0, " goal index:", g)
    print("num human models:", len(out["humans"]))
    print("total determinizing time (s):", out["total_determinizing_time"])
