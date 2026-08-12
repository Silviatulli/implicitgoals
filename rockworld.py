"""
rockworld.py — RockWorld + determinized-MDP generator.
==========================================================

Exported from the *implicitgoals* research repo. Builds on the shared
``GridWorld`` MDP and ``augment_mdp_to_deterministic`` helper defined in
``gridworld_core.py`` (the same core used by gridworld.py, puddleworld.py,
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
    out = generate_determinized_models(size=4, num_humans=3,
                                       obstacles_percent=0.1, rock_percent=0.3,
                                       seed=0)
    T_R, s0, g = out["robot"][:3]
    print(out["total_determinizing_time"])
"""

import time
import random
from itertools import product

import numpy as np

from gridworld_core import GridWorld, augment_mdp_to_deterministic


# ─────────────────────────────────────────────────────────────────────────────
# RockWorld (ported from RockWorldClass.py)
# ─────────────────────────────────────────────────────────────────────────────

# Collected rocks live in the state, so each valuable rock doubles the state
# space.  The cap keeps that exponential in check: at the default sweep size 8 an
# uncapped board carries 7 valuable rocks (128x the states, ~86 s to determinize
# a single model), which makes the experiment infeasible.  3 rocks is 8x.
MAX_VALUABLE_ROCKS = 3


class RockWorld(GridWorld):
    """GridWorld with valuable rocks (``1``) and dangerous rocks (``2``). Rocks
    change rewards only; they do not block movement, so transitions match a
    plain grid's.

    A state is ``[position, collected]``, where ``collected`` is one bit per
    valuable rock (in ``valuable_positions`` order).  Carrying that set in the
    state is what makes "each rock pays once" Markov: the old implementation
    mutated ``self.map`` inside the reward to remember, which corrupted the map
    and made the reward depend on evaluation order.
    """

    def __init__(self, size=5, start=None, goal=None, obstacles_percent=0.1,
                 rock_percent=0.3, valuable_rock_ratio=0.4,
                 valuable_rock_reward=10, dangerous_rock_penalty=-5,
                 slip_prob=0.1, discount=0.99, max_tries=100, obstacle_seed=1,
                 max_valuable_rocks=MAX_VALUABLE_ROCKS):
        super().__init__(size=size, start=start, goal=goal,
                         obstacles_percent=obstacles_percent,
                         slip_prob=slip_prob, discount=discount,
                         max_tries=max_tries, obstacle_seed=obstacle_seed)
        self.rock_percent = rock_percent
        self.valuable_rock_ratio = valuable_rock_ratio
        self.valuable_rock_reward = valuable_rock_reward
        self.dangerous_rock_penalty = dangerous_rock_penalty
        self.max_valuable_rocks = max_valuable_rocks
        self.valuable_positions = []
        self.place_rocks()
        self.reward_func = self.rock_reward_func
        # GridWorld.__init__ built the state space before the rocks existed, so
        # it has no collected bits yet.  Now that they are placed, rebuild it.
        self.state_space = None
        self.create_state_space()

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
        for i in range(self.size):
            for j in range(self.size):
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
        total_rocks = int(self.size * self.size * self.rock_percent)
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

        Start and goal are excluded: a rock on the goal could never be collected
        (entering the goal collapses the collection set), and the old unbounded
        ``while True`` spun for ever on a board with no free cell left.
        """
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if self.map[x, y] == 0
                and (x, y) != self.start_pos and (x, y) != self.goal_pos]
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
    # The MDP itself is returned too: it carries the stochastic transition
    # probabilities that determinization discards, which Hypothesis 3 needs.
    return next_states, start_idx, goal_idx, time.time() - t0, mdp


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
        "robot": robot[:4],
        "robot_mdp": robot[4],
        "humans": [h[:4] for h in humans],
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
