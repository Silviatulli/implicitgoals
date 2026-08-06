"""
gridworld_core.py — shared GridWorld MDP base + determinization helper.
=========================================================================

Factored out of gridworld.py / puddleworld.py / rockworld.py / taxiworld.py,
which used to each carry a byte-for-byte copy of this code. This module holds
the plain stochastic 2D grid (``GridWorld``), its BFS/powerset helpers, and
the stochastic-to-deterministic MDP conversion (``augment_mdp_to_deterministic``)
that all four world types use identically.

Only dependency: ``numpy``.
"""

from queue import Queue
from itertools import chain, combinations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def powerset(iterable):
    """All subsets of ``iterable`` as tuples (includes the empty tuple)."""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def _bfs_reachable(start_state, goal_test, successor_generator):
    """Breadth-first search; returns the action path to a goal, or None."""
    fringe = Queue()
    closed = set()
    fringe.put((start_state, []))
    while not fringe.empty():
        state, path = fringe.get()
        if goal_test(state):
            return path
        state_hash = hash(tuple(state))
        if state_hash not in closed:
            closed.add(state_hash)
            for next_state, action in successor_generator(state):
                if hash(tuple(next_state)) not in closed:
                    fringe.put((next_state, path + [action]))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# The GridWorld game (stochastic 2D grid with slip)
# ─────────────────────────────────────────────────────────────────────────────

class GridWorld:
    """2D grid MDP. Ported from ``GridWorldClass.py`` (visualization / value
    iteration helpers dropped). A state is ``[(row, col), (), ()]``; the empty
    tuples are placeholders for optional agent features / locatables.

    Randomness is controlled by ``obstacle_seed`` (feeds ``np.random.seed``), so
    two calls with the same seed give the same map. ``obstacles_percent`` sets
    the obstacle density; ``divide_rooms=True`` gives a four-rooms layout.

    Used directly by gridworld.py; subclassed by PuddleWorld, RockWorld, and
    TaxiWorld to add rewards/actions specific to each world.
    """

    def __init__(self, size=5, start=None, goal=None, obstacles_percent=0.1,
                 divide_rooms=False, room_count=4, agent_features=None,
                 locatables=None, slip_prob=0.1, discount=0.99, max_tries=100,
                 obstacle_seed=1, starting_features=None):
        self.size = size
        self.start_pos = start
        self.goal_pos = goal
        self.obstacles_percent = obstacles_percent
        self.divide_rooms = divide_rooms
        self.room_count = room_count
        self.agent_features = agent_features if agent_features is not None else []
        self.locatables = locatables if locatables is not None else []
        self.slip_prob = slip_prob
        self.reward_func = self.goal_reward_func
        self.map = np.zeros((size, size))
        self.state_space = None
        self.discount = discount
        self.obstacle_seed = obstacle_seed if obstacle_seed is not None else np.random.randint(0, 10000)

        valid_config_found = False
        curr_tries = 0
        while not valid_config_found and curr_tries < max_tries:
            self.map = np.zeros((size, size))
            self.place_random_obstacles()
            if self.divide_rooms:
                self.divide_into_rooms()
            self.place_start_and_goal()
            if self.check_for_path():
                valid_config_found = True
            else:
                curr_tries += 1

        if not valid_config_found:
            self.map = np.zeros((size, size))
            self.place_start_and_goal()

        self.create_state_space()
        self.start_features = starting_features if starting_features is not None else []
        assert slip_prob >= 0 and slip_prob * 3 <= 1, \
            "Slip probability should be >= 0 and 3*slip_prob <= 1."

    # ── Map construction ─────────────────────────────────────────────────────
    def place_random_obstacles(self):
        self.state_space = None
        np.random.seed(self.obstacle_seed)
        total_obstacles = int(self.size * self.size * self.obstacles_percent)
        obstacles_placed = 0
        while obstacles_placed < total_obstacles:
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            if (x, y) != self.start_pos and (x, y) != self.goal_pos and self.map[x, y] != -1:
                self.map[x, y] = -1
                obstacles_placed += 1

    def divide_into_rooms(self):
        self.state_space = None
        assert self.room_count == 4, "Currently only supports 4 rooms."
        room_divider = self.size // 2
        self.map[room_divider, :] = -1
        self.map[:, room_divider] = -1
        x1 = np.random.randint(room_divider)
        self.map[x1, room_divider] = 0
        x2 = np.random.randint(room_divider + 1, self.size)
        self.map[x2, room_divider] = 0
        y1 = np.random.randint(room_divider)
        self.map[room_divider, y1] = 0
        y2 = np.random.randint(room_divider + 1, self.size)
        self.map[room_divider, y2] = 0

    def place_start_and_goal(self):
        if self.start_pos is None:
            self.start_pos = (np.random.randint(self.size), np.random.randint(self.size))
        if self.goal_pos is None:
            self.goal_pos = (np.random.randint(self.size), np.random.randint(self.size))

    # ── Connectivity ─────────────────────────────────────────────────────────
    def get_all_neighbors(self, state):
        x, y = state
        neighbors = []
        for dx, dy, action in [(0, -1, "left"), (0, 1, "right"), (-1, 0, "up"), (1, 0, "down")]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                if self.map[new_x, new_y] != -1:
                    neighbors.append(((new_x, new_y), action))
        return neighbors

    def check_goal_reached(self, state):
        return state == self.goal_pos

    def check_for_path(self):
        if self.start_pos is None or self.goal_pos is None:
            return False
        return _bfs_reachable(self.start_pos, self.check_goal_reached, self.get_all_neighbors) is not None

    # ── MDP interface ────────────────────────────────────────────────────────
    def get_actions(self):
        return ["up", "down", "left", "right"]

    def create_state_space(self):
        if self.state_space is not None:
            return None
        self.state_space = []
        for i in range(self.size):
            for j in range(self.size):
                current_state = [(i, j)]
                for agent_feature_set in powerset(self.agent_features):
                    for locatable_set in powerset(self.locatables):
                        current_state.append(agent_feature_set)
                        current_state.append(locatable_set)
                self.state_space.append(current_state)

    def get_state_space(self):
        if self.state_space is None:
            self.create_state_space()
        return self.state_space

    def get_transition_probability_for_move(self, state, action, state_prime):
        if self.map[state[0]] == -1:
            return 1 if state == state_prime else 0
        if self.map[state_prime[0]] == -1:
            return 0
        if self.check_goal_reached(state[0]):
            return 1 if state == state_prime else 0

        x, y = state[0]
        x_prime, y_prime = state_prime[0]
        if (x_prime, y_prime) not in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x, y)]:
            return 0

        up_free = x - 1 >= 0 and self.map[x - 1, y] != -1
        down_free = x + 1 < self.size and self.map[x + 1, y] != -1
        left_free = y - 1 >= 0 and self.map[x, y - 1] != -1
        right_free = y + 1 < self.size and self.map[x, y + 1] != -1

        if action == "up":
            total_prob = 1 + down_free + left_free + right_free
            if up_free:
                return (1 - total_prob * self.slip_prob) if (x_prime == x - 1 and y_prime == y) else self.slip_prob
            return (1 - (total_prob - 1) * self.slip_prob) if (x_prime == x and y_prime == y) else self.slip_prob
        elif action == "down":
            total_prob = 1 + up_free + left_free + right_free
            if down_free:
                return (1 - total_prob * self.slip_prob) if (x_prime == x + 1 and y_prime == y) else self.slip_prob
            return (1 - (total_prob - 1) * self.slip_prob) if (x_prime == x and y_prime == y) else self.slip_prob
        elif action == "left":
            total_prob = 1 + up_free + down_free + right_free
            if left_free:
                return (1 - total_prob * self.slip_prob) if (x_prime == x and y_prime == y - 1) else self.slip_prob
            return (1 - (total_prob - 1) * self.slip_prob) if (x_prime == x and y_prime == y) else self.slip_prob
        elif action == "right":
            total_prob = 1 + up_free + down_free + left_free
            if right_free:
                return (1 - total_prob * self.slip_prob) if (x_prime == x and y_prime == y + 1) else self.slip_prob
            return (1 - (total_prob - 1) * self.slip_prob) if (x_prime == x and y_prime == y) else self.slip_prob
        assert False, "Should never reach here."

    def get_transition_probability(self, state, action, state_prime):
        return self.get_transition_probability_for_move(state, action, state_prime)

    def goal_reward_func(self, state, action, next_state):
        if self.check_goal_reached(next_state) and not self.check_goal_reached(state):
            return 1
        return 0

    def get_state_hash(self, state):
        return str(state)

    def get_reward(self, state, action, next_state):
        return self.reward_func(state, action, next_state)

    def get_init_state(self):
        return [self.start_pos, tuple(self.start_features), tuple(self.locatables)]

    def get_goal_states(self):
        return [[self.goal_pos, tuple(self.start_features), tuple(self.locatables)]]

    def visualize(self):
        """ASCII render of the grid. Subclasses override this to show their
        own map symbols (puddles, rocks, passenger, ...)."""
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if (i, j) == self.start_pos:
                    row += "S "
                elif (i, j) == self.goal_pos:
                    row += "G "
                elif self.map[i, j] == -1:
                    row += "# "
                else:
                    row += ". "
            print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Determinization — identical to parallel_experiments_2._augment_mdp_to_deterministic
# ─────────────────────────────────────────────────────────────────────────────

def augment_mdp_to_deterministic(mdp):
    """Convert a stochastic MDP into a deterministic transition array.

    For each state, every ``(original_action, outcome)`` pair with positive
    probability becomes a deterministic augmented action ``act_0, act_1, ...``.
    The augmented action space is shared across states: its size is the max
    number of outcomes needed by any single state (not the sum over all states).

    Returns
    -------
    next_states : ndarray, shape (n_states, n_augmented_actions), int32
        ``next_states[state_idx, act_i]`` is the resulting state index, or
        ``state_idx`` itself (self-loop) when ``act_i`` is undefined there.
    start_idx : int   index of the initial state
    goal_idx  : int   index of the goal state
    """
    states = mdp.get_state_space()
    original_actions = mdp.get_actions()
    n_states = len(states)

    per_state_outcomes = []
    for state in states:
        outcomes = []
        for orig_action in original_actions:
            for next_state_idx, next_state in enumerate(states):
                if mdp.get_transition_probability(state, orig_action, next_state) > 1e-12:
                    outcomes.append(next_state_idx)
        per_state_outcomes.append(outcomes)

    n_augmented_actions = max((len(o) for o in per_state_outcomes), default=0)
    next_states = np.tile(np.arange(n_states, dtype=np.int32).reshape(-1, 1),
                          (1, n_augmented_actions))
    for state_idx, outcomes in enumerate(per_state_outcomes):
        for act_idx, next_state_idx in enumerate(outcomes):
            next_states[state_idx, act_idx] = next_state_idx

    state_hashes = [mdp.get_state_hash(s) for s in states]
    start_idx = state_hashes.index(mdp.get_state_hash(mdp.get_init_state()))
    goal_idx = state_hashes.index(mdp.get_state_hash(mdp.get_goal_states()[0]))

    return next_states, start_idx, goal_idx
