"""
rockworld.py — RockWorld + determinized-MDP generator.
==========================================================

Builds on the shared ``GridWorld`` MDP and ``augment_mdp_to_deterministic``
helper defined in ``gridworld_core.py`` (the same core used by gridworld.py, puddleworld.py,
and taxiworld.py — see that module for the shared plumbing).

RockWorld is a GridWorld with valuable rocks (map value ``1``) and dangerous
rocks (``2``). Rocks do **not** block movement — they only change the *reward*
and what the agent has collected.

A state is ``[position, collected]``: one bit per valuable rock, so that "each
rock pays once" is Markov.  That multiplies the state count by
``2 ** len(valuable_positions)``, which is why the number of valuable rocks is
capped at ``MAX_VALUABLE_ROCKS``.  Reaching the goal collapses every collection
set into the single canonical sink ``[goal_pos, (0,)*k]``, so the pipeline still
has exactly one goal state.

Rewards are additive cost-to-go: ``−1`` per step everywhere except the goal
(which pays ``0`` and is absorbing), ``+10`` the first time a valuable rock is
entered, ``−5`` for a dangerous one.

Quick start
-----------
    from rockworld import generate_determinized_models
    out = generate_determinized_models(room_side=4, num_humans=3,
                                       obstacle_density=0.1, rock_density=0.3,
                                       seed=0)
    T_R, s0, g = out["robot"][:3]
    print(out["total_determinizing_time"])
"""

import time
import random
from itertools import product

import numpy as np

from gridworld_core import (GridWorld, augment_mdp_to_deterministic, board_side,
                            DEFAULT_MAX_TRIES)


# ─────────────────────────────────────────────────────────────────────────────
# RockWorld
# ─────────────────────────────────────────────────────────────────────────────

# Collected rocks live in the state, so each valuable rock doubles the state
# space.  The cap keeps that exponential in check: the default 9x9 board carries
# 9 valuable rocks uncapped, which is 512x the states.  3 rocks is 8x.
MAX_VALUABLE_ROCKS = 3

# Share of the rocks that are valuable; the rest are dangerous.  A module
# constant rather than only a default argument because the instance-level
# sampler and the per-grid fallback must agree on the split, or the two paths
# would put different numbers of rocks on the same board.
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
                 rock_density=0.3, valuable_rock_ratio=VALUABLE_ROCK_RATIO,
                 valuable_rock_reward=10, dangerous_rock_penalty=-5,
                 slip_prob=0.0, discount=0.99, max_tries=DEFAULT_MAX_TRIES,
                 obstacle_seed=1,
                 max_valuable_rocks=MAX_VALUABLE_ROCKS,
                 rooms_per_side=1, room_side=5,
                 valuable_positions=None, dangerous_positions=None):
        self.rock_density = rock_density
        self.valuable_rock_ratio = valuable_rock_ratio
        self.valuable_rock_reward = valuable_rock_reward
        self.dangerous_rock_penalty = dangerous_rock_penalty
        self.max_valuable_rocks = max_valuable_rocks
        # Set *before* super().__init__() so protected_cells() can see them while
        # the obstacles are being drawn.  Shared rocks are the whole point: bit i
        # of a state's `collected` tuple names valuable_positions[i], so unless
        # every model of an instance lists the same cells in the same order, the
        # same state ID means a different rock in each model and the bottleneck
        # sets the pipeline unions are not comparable.  Same rule as the start,
        # the goal, the doors and TaxiWorld's passenger: only obstacles vary.
        self.valuable_positions = list(valuable_positions or [])
        self.dangerous_positions = list(dangerous_positions or [])
        self._shared_rocks = bool(valuable_positions or dangerous_positions)

        super().__init__(start=start, goal=goal,
                         obstacle_density=obstacle_density,
                         slip_prob=slip_prob, discount=discount,
                         max_tries=max_tries, obstacle_seed=obstacle_seed,
                         rooms_per_side=rooms_per_side,
                         room_side=room_side)

        if self._shared_rocks:
            # Obstacles avoided these cells, so painting them now cannot bury
            # one, and every model of the instance paints the same cells.
            self.paint_rocks()
        else:
            # Standalone use with no shared layout: draw this grid's own rocks
            # on whatever the obstacles left free, as before.
            self.place_rocks()
        self.reward_func = self.rock_reward_func
        # GridWorld.__init__ built the state space before the rocks existed, so
        # it has no collected bits yet.  Now that they are placed, rebuild it.
        self.state_space = None
        self.create_state_space()

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
        """``[position, collected]`` for every cell and every collection subset.

        The goal collapses: whatever you have picked up, entering the goal lands
        in the single canonical sink ``[goal_pos, (0,)*k]``.  That keeps one
        unambiguous goal state for the pipeline (which needs exactly one) while
        leaving collection optional and purely reward-driven.
        """
        k = len(getattr(self, "valuable_positions", []))
        self.state_space = []
        for i in range(self.board_side):
            for j in range(self.board_side):
                if (i, j) == self.goal_pos:
                    self.state_space.append([(i, j), (0,) * k])
                else:
                    for collected in product((0, 1), repeat=k):
                        self.state_space.append([(i, j), collected])

    def _next_collected(self, collected, next_pos):
        """The collection set after stepping onto ``next_pos`` — the only way it
        ever changes, and it only ever gains bits."""
        if next_pos == self.goal_pos:
            return (0,) * len(self.valuable_positions)      # canonical sink
        updated = list(collected)
        for i, pos in enumerate(self.valuable_positions):
            if pos == next_pos:
                updated[i] = 1
        return tuple(updated)

    # ── Rocks ────────────────────────────────────────────────────────────────
    def place_rocks(self):
        total_rocks = int(self.board_side * self.board_side * self.rock_density)
        valuable_rocks = min(int(total_rocks * self.valuable_rock_ratio),
                             self.max_valuable_rocks)
        dangerous_rocks = total_rocks - valuable_rocks
        self.valuable_positions = []
        for _ in range(valuable_rocks):
            pos = self.place_rock(1)
            if pos is not None:
                self.valuable_positions.append(pos)
        for _ in range(dangerous_rocks):
            self.place_rock(2)

    def place_rock(self, rock_type):
        """Drop a rock on a free cell, or return None when the board is full.

        Everything in ``protected_cells()`` is excluded — start and goal, plus
        whatever a subclass adds: a rock on the goal could never be collected,
        since entering the goal collapses the collection set.  Drawing from the
        free list, rather than retrying until a free cell turns up, is what makes
        a full board return None instead of looping for ever.
        """
        protected = self.protected_cells()
        free = [(x, y) for x in range(self.board_side) for y in range(self.board_side)
                if self.map[x, y] == 0 and (x, y) not in protected]
        if not free:
            return None
        x, y = free[self.rng.randint(len(free))]
        self.map[x, y] = rock_type
        return (x, y)

    # ── Dynamics and reward ──────────────────────────────────────────────────
    def get_transition_probability(self, state, action, state_prime):
        """Position moves exactly as in a plain grid; the collection set is a
        deterministic function of where you land, so any inconsistent successor
        has probability 0."""
        if tuple(state_prime[1]) != self._next_collected(tuple(state[1]),
                                                         state_prime[0]):
            return 0
        return super().get_transition_probability(state, action, state_prime)

    def rock_reward_func(self, state, action, next_state):
        """Cost-to-go: −1 per step everywhere except the goal, which pays 0.

        Rewards are additive, so stepping onto a fresh valuable rock nets
        −1 + 10 = +9 and onto a dangerous one −1 − 5 = −6.  The valuable bonus is
        paid only when the rock's bit actually flips 0→1, so revisiting a
        collected rock is just another −1 — enforced by the state, not by
        mutating the map.  There is no bonus for arriving at the goal: a reward
        paid at an absorbing state cannot change the policy and would only break
        V(goal) = 0.
        """
        if self.check_goal_reached(state[0]):
            return 0                                    # the sink pays nothing
        reward = -1
        next_pos = next_state[0]
        for i, pos in enumerate(self.valuable_positions):
            if pos == next_pos and not state[1][i] and next_state[1][i]:
                reward += self.valuable_rock_reward
        if self.map[next_pos] == 2:
            reward += self.dangerous_rock_penalty
        return reward

    def get_init_state(self):
        return [self.start_pos, (0,) * len(self.valuable_positions)]

    def get_goal_states(self):
        return [[self.goal_pos, (0,) * len(self.valuable_positions)]]

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
                       max_valuable_rocks=MAX_VALUABLE_ROCKS,
                       protected=()):
    """One rock layout for a whole instance: (valuable_positions, dangerous).

    Drawn once in generate_determinized_models and handed to every model, so the
    robot and the humans agree on which rock bit i of `collected` refers to.
    Counts mirror RockWorld.place_rocks exactly, so sharing does not change how
    many rocks a board carries — only that they land in the same places.

    `rng` is a module-level ``random`` (not a grid's own RandomState): the layout
    belongs to the instance, not to any one model.
    """
    total_rocks = int(board * board * rock_density)
    n_valuable = min(int(total_rocks * valuable_rock_ratio), max_valuable_rocks)
    free = [(i, j) for i in range(board) for j in range(board)
            if (i, j) not in set(protected)]
    picked = rng.sample(free, min(total_rocks, len(free)))
    return picked[:n_valuable], picked[n_valuable:]


def generate_and_visualize_rockworld(start, goal, obstacle_density, rock_density,
                                     model_type="Model", obstacle_seed=None,
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
    mdp = generate_and_visualize_rockworld(
        start=(0, 0), goal=(n - 1, n - 1),
        obstacle_density=obstacle_density, rock_density=rock_density,
        rooms_per_side=rooms_per_side,
        room_side=room_side, slip_prob=slip_prob,
        valuable_positions=valuable_positions,
        dangerous_positions=dangerous_positions,
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
                                 rock_density=0.3, seed=None, verbose=True,
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
    # doors and TaxiWorld's passenger.  Drawing it per model made bit i of
    # `collected` name a different cell in every model, so a bottleneck state ID
    # meant something different in T_R than in each T_H and the union the
    # pipeline takes over them was comparing unrelated labels.
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
    out = generate_determinized_models(room_side=4, num_humans=3,
                                       obstacle_density=0.1, rock_density=0.3, seed=0, visualize=True)
    T_R, s0, g, _ = out["robot"]
    print("\nRobot determinized transition array shape:", T_R.shape)
    print("start index:", s0, " goal index:", g)
    print("num human models:", len(out["humans"]))
    print("total determinizing time (s):", out["total_determinizing_time"])
