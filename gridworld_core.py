"""
gridworld_core.py — shared GridWorld MDP base + determinization helper.
=========================================================================

Factored out of gridworld.py / puddleworld.py / rockworld.py / taxiworld.py,
which used to each carry a byte-for-byte copy of this code. This module holds
the plain stochastic 2D grid (``GridWorld``), its BFS helper, and the
stochastic-to-deterministic MDP conversion (``augment_mdp_to_deterministic``)
that all four world types use identically.

Only dependency: ``numpy``.
"""

from queue import Queue
from collections import deque

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def seeded_rng(obstacle_seed=None):
    """(seed, generator) for one grid — the only randomness a world may use.

    ``np.random.RandomState``, not ``np.random.default_rng``: RandomState is the
    same MT19937 stream ``np.random.seed`` drove, so a given ``obstacle_seed``
    still produces the map it produced when this generator was the global one.
    Only the *sharing* changes, which is the whole point — a grid no longer
    resets the process-wide numpy RNG, so callers downstream (experiment.py's
    human draw, the random-order query baseline) keep the seed they were given.

    ``obstacle_seed=None`` draws one from the global RNG.  Reading it is fine;
    it is writing to it that this function exists to stop.

    Subclasses that need randomness *before* ``GridWorld.__init__`` runs — see
    TaxiWorld, whose passenger is placed first — call this themselves and pass
    the seed down; ``GridWorld.__init__`` then adopts the generator instead of
    building a second, differently-seeded one.
    """
    if obstacle_seed is None:
        obstacle_seed = np.random.randint(0, 10000)
    return obstacle_seed, np.random.RandomState(obstacle_seed)


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
    iteration helpers dropped). A state is ``[(row, col)]`` — a one-element list,
    so that ``state[0]`` is the position in every world.  Subclasses append their
    own slots: RockWorld carries a collected-rocks tuple, TaxiWorld a
    passenger/delivered pair.

    Randomness is controlled by ``obstacle_seed``, which seeds this grid's own
    ``self.rng`` (see :func:`seeded_rng`), so two calls with the same seed give
    the same map and building a grid leaves the global numpy RNG alone.
    ``obstacles_percent`` sets the obstacle density; ``divide_rooms=True`` gives
    a four-rooms layout.

    Used directly by gridworld.py; subclassed by PuddleWorld, RockWorld, and
    TaxiWorld to add rewards/actions specific to each world.
    """

    def __init__(self, size=5, start=None, goal=None, obstacles_percent=0.1,
                 divide_rooms=False, room_count=4,
                 slip_prob=0.1, discount=0.99, max_tries=100,
                 obstacle_seed=1):
        self.size = size
        self.start_pos = start
        self.goal_pos = goal
        self.obstacles_percent = obstacles_percent
        self.divide_rooms = divide_rooms
        self.room_count = room_count
        self.slip_prob = slip_prob
        self.reward_func = self.goal_reward_func
        self.map = np.zeros((size, size))
        self.state_space = None
        self.discount = discount
        # One generator per grid, seeded once and never reset.  A subclass may
        # have built it already (TaxiWorld places its passenger before calling
        # up); adopting that one keeps a single stream per grid rather than two
        # independently seeded halves.
        if hasattr(self, "rng"):
            self.obstacle_seed = obstacle_seed
        else:
            self.obstacle_seed, self.rng = seeded_rng(obstacle_seed)

        valid_config_found = False
        curr_tries = 0
        while not valid_config_found and curr_tries < max_tries:
            self.map = np.zeros((size, size))
            # Start and goal *first*.  protected_cells() is what keeps obstacles
            # off them, and it can only do that if they already exist — placed
            # afterwards, as they used to be, protected_cells() saw {None} and
            # the two cells were protected only when the caller happened to pass
            # them explicitly.  When both are passed (every experiment path)
            # this consumes no randomness, so generated maps are unchanged.
            self.place_start_and_goal()
            self.place_random_obstacles()
            if self.divide_rooms:
                self.divide_into_rooms()
            if self.check_for_path():
                valid_config_found = True
            else:
                curr_tries += 1

        if not valid_config_found:
            self.map = np.zeros((size, size))
            self.place_start_and_goal()

        self.create_state_space()
        assert slip_prob >= 0 and slip_prob * 3 <= 1, \
            "Slip probability should be >= 0 and 3*slip_prob <= 1."

    # ── Map construction ─────────────────────────────────────────────────────
    def protected_cells(self):
        """Cells an obstacle must never cover. Subclasses widen this.

        TaxiWorld adds the passenger: when one location is shared across the
        robot and every human, burying it under one model's obstacles would
        force that model to relocate and silently break the sharing.
        """
        return {self.start_pos, self.goal_pos}

    def place_random_obstacles(self):
        """Scatter obstacles on free, unprotected cells.

        Draws from ``self.rng``, which is seeded once in ``__init__`` and never
        reset here.  It used to call ``np.random.seed(self.obstacle_seed)`` on
        every entry, which had two costs: it clobbered the process-wide numpy
        RNG (so every caller downstream inherited this grid's obstacle seed),
        and — since this runs inside ``__init__``'s retry loop — it rewound the
        stream to the same point on each retry, redrawing the identical
        unsolvable layout ``max_tries`` times before giving up on the empty-map
        fallback.  With a persistent generator a retry actually retries.
        """
        self.state_space = None
        total_obstacles = int(self.size * self.size * self.obstacles_percent)
        protected = self.protected_cells()
        obstacles_placed = 0
        while obstacles_placed < total_obstacles:
            x = self.rng.randint(self.size)
            y = self.rng.randint(self.size)
            if (x, y) not in protected and self.map[x, y] != -1:
                self.map[x, y] = -1
                obstacles_placed += 1

    def divide_into_rooms(self):
        self.state_space = None
        assert self.room_count == 4, "Currently only supports 4 rooms."
        room_divider = self.size // 2
        self.map[room_divider, :] = -1
        self.map[:, room_divider] = -1
        x1 = self.rng.randint(room_divider)
        self.map[x1, room_divider] = 0
        x2 = self.rng.randint(room_divider + 1, self.size)
        self.map[x2, room_divider] = 0
        y1 = self.rng.randint(room_divider)
        self.map[room_divider, y1] = 0
        y2 = self.rng.randint(room_divider + 1, self.size)
        self.map[room_divider, y2] = 0
        # The dividers are drawn blind, so they can bury the start or the goal.
        # Reopen those cells: same invariant place_random_obstacles keeps.
        for cell in self.protected_cells():
            if cell is not None:
                self.map[cell] = 0

    def place_start_and_goal(self):
        if self.start_pos is None:
            self.start_pos = (self.rng.randint(self.size), self.rng.randint(self.size))
        if self.goal_pos is None:
            self.goal_pos = (self.rng.randint(self.size), self.rng.randint(self.size))

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

    def is_absorbing_state(self, state):
        """True when no action can leave `state` — the terminal sink.

        Takes a *full* state (unlike check_goal_reached, which takes a position).
        Defaults to "the agent stands on the goal cell", which is what the
        reach-the-goal games want.  TaxiWorld overrides it: its destination cell
        must stay passable until the passenger has actually been delivered, so
        there the sink is the `delivered` flag, not the position.
        """
        return self.check_goal_reached(state[0])

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
        self.state_space = [[(i, j)]
                            for i in range(self.size)
                            for j in range(self.size)]

    def get_state_space(self):
        if self.state_space is None:
            self.create_state_space()
        return self.state_space

    def get_transition_probability_for_move(self, state, action, state_prime):
        if self.map[state[0]] == -1:
            return 1 if state == state_prime else 0
        if self.map[state_prime[0]] == -1:
            return 0
        if self.is_absorbing_state(state):
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
        # check_goal_reached compares a bare position, so reward on the state's
        # *position* component: the whole state is a list and would never equal
        # goal_pos, which would silently make this reward always 0.
        if self.check_goal_reached(next_state[0]) and not self.check_goal_reached(state[0]):
            return 1
        return 0

    def get_state_hash(self, state):
        return str(state)

    def get_reward(self, state, action, next_state):
        return self.reward_func(state, action, next_state)

    def get_reward_function(self):
        """The callable R(state, action, next_state) scoring transitions.

        value_iteration() evaluates it over the pruned state space to build the
        per-(state, action) reward that drives V_R.  Subclasses override
        reward_func, so this stays generic across the grid games."""
        return self.reward_func

    def get_init_state(self):
        return [self.start_pos]

    def get_goal_states(self):
        return [[self.goal_pos]]

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


def build_stochastic_matrix(mdp):
    """The robot's stochastic model, pruned to the states reachable from the start.

    This is the matrix the determinization throws away.  augment_mdp_to_deterministic
    turns every (action, outcome) pair into its own action, which lets the agent
    choose its own slip outcome; values computed from that determinized array are
    therefore optimistic, not the values of M_R.

    Returns
    -------
    A 5-tuple.  The first two are the model; the last three are what
    bottlenecks.value_iteration needs to run in *reward* mode rather than
    probability mode, and are the only reason this returns more than a matrix.

    T_R_sto     : (n_reachable, n_actions, n_reachable) float64, P(s'|s,a) over
                  the *original* (un-augmented) actions, rows summing to 1.
    index       : dict {state ID in mdp.get_state_space() order -> row of
                  T_R_sto}.  The same full-space IDs
                  augment_mdp_to_deterministic produces, so a bottleneck ID from
                  the pipeline maps straight through.
    reward_func : callable(state, action, next_state) -> float, the MDP's own
                  reward.  Its presence is what selects reward mode.
    kept_states : list — kept_states[i] is the original state *object* at row i,
                  so the callable (which is keyed by objects, not row indices)
                  can be evaluated on the pruned, reindexed matrix.
    actions     : list — action labels in T_R_sto's action-axis order.

    Reachability is settled before the matrix is built, so the expensive
    O(|S|^2 |A|) probability scan only ever runs over the reachable states.

    Note for callers: Overcooked's build_stochastic_matrix returns only the
    first two.  `T, index, *rest = build_stochastic_matrix(...)` followed by
    `value_iteration(T, index[goal], *rest)` is correct for both.
    """
    states  = mdp.get_state_space()
    actions = mdp.get_actions()
    n_s     = len(states)

    hashes    = [mdp.get_state_hash(s) for s in states]
    start_idx = hashes.index(mdp.get_state_hash(mdp.get_init_state()))

    # Successors first (probabilities not needed yet), then BFS from the start.
    # This O(|S|^2 |A|) scan is the dominant cost of the whole H3 stage — far
    # more than the value iteration it feeds.
    successors = [set() for _ in range(n_s)]
    for si, s in enumerate(states):
        for a in actions:
            for sj, s2 in enumerate(states):
                if mdp.get_transition_probability(s, a, s2) > 1e-12:
                    successors[si].add(sj)

    seen     = {start_idx}
    frontier = deque([start_idx])
    while frontier:
        s = frontier.popleft()
        for s2 in successors[s]:
            if s2 not in seen:
                seen.add(s2)
                frontier.append(s2)

    kept  = sorted(seen)
    index = {s: i for i, s in enumerate(kept)}
    n_r   = len(kept)

    T = np.zeros((n_r, len(actions), n_r), dtype=np.float64)
    for i, si in enumerate(kept):
        for ai, a in enumerate(actions):
            for sj in successors[si]:
                p = mdp.get_transition_probability(states[si], a, states[sj])
                if p > 1e-12:
                    T[i, ai, index[sj]] = p

    # An action with no defined outcome becomes a self-loop, so every row is a
    # distribution and value iteration cannot leak probability mass.
    rowsum = T.sum(axis=2)
    dead   = rowsum <= 1e-12
    if dead.any():
        di, da = np.nonzero(dead)
        T[di, da, di] = 1.0
        rowsum = T.sum(axis=2)
    T /= rowsum[:, :, None]
    # kept_states[i] is the original state object at row i of T; together with
    # `actions` it lets value_iteration evaluate the reward callable on the
    # pruned, reindexed matrix (the callable is keyed by state objects).
    kept_states = [states[k] for k in kept]
    return T, index, mdp.get_reward_function(), kept_states, actions
