"""
gridworld_core.py — shared GridWorld MDP base + determinization helper.
=========================================================================

Shared by gridworld.py / puddleworld.py / rockworld.py / taxiworld.py: the plain
2D grid (``GridWorld``), its BFS helper, and the stochastic-to-deterministic MDP
conversion (``augment_mdp_to_deterministic``) that all four world types use
identically.

The grid is stochastic only when ``slip_prob > 0``.  It defaults to 0, which
makes every move deterministic: each unintended outcome has probability exactly
0.0, so the determinizer's ``> 1e-12`` test drops it and the augmented action set
stays the four real moves rather than growing to ~20.

Only dependency: ``numpy``.
"""

from queue import Queue
from collections import deque

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Retry budget and the transient status line
# ─────────────────────────────────────────────────────────────────────────────
# A retry loop allowed 100 000 attempts has to say so *while* it runs, without
# leaving 100 lines of scrollback behind and without fighting a tqdm bar over the
# terminal.  Both retry loops in the repo — this module's layout search and
# experiment.py's instance redraw — therefore report through :func:`status_line`,
# and whoever owns the terminal decides where the text lands:
#
#   * experiment.py points the sink at its tqdm bar's postfix, which already
#     redraws in place: one line, no scrolling, and no second writer competing
#     for "\r";
#   * a bare script (``python gridworld.py``) leaves the sink unset and gets a
#     "\r"-erased line on stdout instead.
#
# It is module state rather than a parameter because the alternative is threading
# a callback through four game modules, two wrapper functions each, and down into
# ``GridWorld.__init__``.

DEFAULT_MAX_TRIES  = 100_000    # attempts before a retry loop gives up
RETRY_REPORT_EVERY = 1_000      # ... and how often it says how far along it is


class UnsolvableLayout(RuntimeError):
    """No solvable board could be laid out at the requested densities.

    Raised by ``GridWorld.__init__`` when its layout search exhausts
    ``max_tries``.  It is deliberately an error rather than a fallback: the
    obvious fallback — hand back a board with no obstacles — silently swaps in a
    *different game* (obstacle density 0) under the density the caller asked
    for.  A benchmark that averages such a board in reports a mean over a
    population it never describes, and nothing in the output says so.

    experiment.py catches it and reports the repetition as NaN, counted under
    ``n_skipped``, so a configuration whose densities are too tight shows up as
    missing rather than as a suspiciously easy result.
    """

_status_sink  = None
_status_width = 0


def set_status_sink(sink):
    """Route :func:`status_line` elsewhere; ``None`` restores the stdout line.

    ``sink`` is called with the text to show, or with ``None`` meaning "clear".
    """
    global _status_sink
    _status_sink = sink


def status_line(text):
    """Show ``text`` as a transient one-line status, or clear it when ``None``.

    The stdout fallback pads to the widest line shown so far, so a shorter
    message cannot leave the tail of a longer one behind it on the terminal.
    """
    global _status_width
    if _status_sink is not None:
        _status_sink(text)
        return
    if text is None:
        if _status_width:
            print("\r" + " " * _status_width + "\r", end="", flush=True)
            _status_width = 0
        return
    print("\r" + text.ljust(_status_width), end="", flush=True)
    _status_width = max(_status_width, len(text))


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def seeded_rng(obstacle_seed=None):
    """(seed, generator) for one grid — the only randomness a world may use.

    ``RandomState``, not ``default_rng``: it is the same MT19937 stream
    ``np.random.seed`` drove, so a given ``obstacle_seed`` still produces the map
    it always did.  Only the *sharing* changes, which is the point — a grid no
    longer resets the process-wide numpy RNG, so callers downstream keep the seed
    they were given.  ``obstacle_seed=None`` reads one from the global RNG;
    reading it is fine, writing to it is what this exists to stop.

    Subclasses needing randomness *before* ``GridWorld.__init__`` (TaxiWorld's
    passenger) call this themselves and pass the seed down; ``__init__`` then
    adopts the generator rather than seeding a second one.
    """
    if obstacle_seed is None:
        obstacle_seed = np.random.randint(0, 10000)
    return obstacle_seed, np.random.RandomState(obstacle_seed)


def board_side(rooms_per_side, room_side):
    """Cells per side of the whole board.

    Walls are thin — they run between two adjacent cells and consume none of
    their own — so the board is exactly the product of the two room numbers.
    (Thick walls would make it ``rooms_per_side * room_side + rooms_per_side - 1``.)

    Deriving it, with no way to state a board size directly, is what removes the
    need to check that a board splits into equal rooms.
    """
    return rooms_per_side * room_side


def _room_midlines(rooms_per_side, room_side):
    """The middle row of each room band, in order — where the doors go.

    Everything is square, so the same indices serve the door rows of the vertical
    walls and the door columns of the horizontal ones.
    """
    return [i * room_side + room_side // 2 for i in range(rooms_per_side)]


def _room_boundaries(rooms_per_side, room_side):
    """Indices ``b`` such that a wall runs between cell ``b`` and cell ``b + 1``.

    One per pair of adjacent rooms, so ``rooms_per_side - 1`` of them.  A boundary
    names a *gap*, not a cell — nothing on the board is consumed by it.

    Same shape as :func:`_room_midlines`: the start of room ``i`` plus an offset,
    here the last cell of the room rather than its middle.
    """
    return [i * room_side + room_side - 1 for i in range(rooms_per_side - 1)]


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
# The GridWorld game (2D grid, with optional slip — off by default)
# ─────────────────────────────────────────────────────────────────────────────

class GridWorld:
    """2D grid MDP.  A state is ``[(row, col)]`` — a one-element list, so that
    ``state[0]`` is the position in every world.  Subclasses append their own
    slots: RockWorld a collected-rocks tuple, TaxiWorld a passenger/delivered
    pair.

    ``obstacle_seed`` seeds this grid's own ``self.rng`` (see :func:`seeded_rng`),
    so two calls with the same seed give the same map and building a grid leaves
    the global numpy RNG alone.

    The board is described by exactly two numbers: ``rooms_per_side`` rooms along
    each side of the board, each room ``room_side`` cells along each of its own
    sides, so the board is their product.  ``rooms_per_side=1`` is the open board.
    There is deliberately no way to state the board size directly, so no two
    numbers can contradict each other.

    Used directly by gridworld.py; subclassed by PuddleWorld, RockWorld, and
    TaxiWorld to add rewards/actions specific to each world.
    """

    def __init__(self, rooms_per_side=1, room_side=5, start=None, goal=None,
                 obstacle_density=0.1,
                 slip_prob=0.0, gamma=0.99, max_tries=DEFAULT_MAX_TRIES,
                 obstacle_seed=1):
        # Every game reaches the board through here, so this is the one place the
        # geometry is checked.  Unguarded, rooms_per_side=0 builds a 0x0 board in
        # silence, and a float fails later inside numpy rather than here.
        # np.integer is accepted: a number read out of an array is still an
        # integer, but `isinstance(np.int64(3), int)` is False.
        for name, value in (("rooms_per_side", rooms_per_side),
                            ("room_side", room_side)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        self.board_side = board_side(rooms_per_side, room_side)
        self.rooms_per_side = rooms_per_side
        self.room_side = room_side
        self.start_pos = start
        self.goal_pos = goal
        self.obstacle_density = obstacle_density
        self.slip_prob = slip_prob
        self.reward_func = self.goal_reward_func
        self.map = np.zeros((self.board_side, self.board_side))
        self.state_space = None
        # This grid's discount factor γ — the one in
        # V(s) = max_a [R(s,a) + γ·Σ P(s'|s,a)·V(s')], discounting *steps taken
        # on the board*.
        #
        # Not to be confused with the Query MDP's discount, which is a different
        # number over a different thing — it discounts *questions asked*, and is
        # named `q_gamma` in bottlenecks.py precisely so the two cannot be
        # mistaken for each other.  Both happen to default to 0.99.
        #
        # Stored here, and currently read by nothing.  That is not because it is
        # decorative: it used to be *the* γ of the value iteration, back when
        # that took the MDP object and could reach in for it.  The value
        # iteration now takes bare matrices — bottlenecks.value_iteration(
        # T_R_sto, goal_state, ..., gamma=0.99) — so it has no MDP to ask and
        # falls back on its own argument, and build_stochastic_matrix does not
        # carry this value across either.
        #
        # Consequence to know before trusting it: passing gamma=0.9 today
        # changes *nothing*, silently.  The four selection rules the benchmark
        # reports compute no value function at all, so no γ of either kind moves
        # a published number.
        #
        # Kept rather than deleted because this is the right home for it, and
        # re-attaching is one keyword argument: a future value-based rule would
        # read it in experiment.py, which already holds mdp_R and already pulls
        # `positions` out of it for H3, and pass gamma=mdp_R.gamma.
        self.gamma = gamma
        # One generator per grid, seeded once and never reset.  A subclass may
        # have built it already (TaxiWorld draws its passenger before calling up);
        # adopting that one keeps a single stream per grid, not two.
        if hasattr(self, "rng"):
            self.obstacle_seed = obstacle_seed
        else:
            self.obstacle_seed, self.rng = seeded_rng(obstacle_seed)

        valid_config_found = False
        curr_tries = 0
        while not valid_config_found and curr_tries < max_tries:
            # Report on every RETRY_REPORT_EVERY-th try — the modulo, not
            # "past the first thousand", which would report on every remaining
            # attempt.  status_line() overwrites itself rather than scrolling, so
            # 100 000 tries still cost one line — see the module header.
            if curr_tries and curr_tries % RETRY_REPORT_EVERY == 0:
                status_line(f"{type(self).__name__}: layout try {curr_tries:,}/"
                            f"{max_tries:,} before giving up on these densities")
            self._blank_board()
            # Start and goal first: protected_cells() can only keep obstacles off
            # them if they already exist.
            self.place_start_and_goal()
            # Walls before obstacles: built afterwards they would bury obstacles
            # already counted, quietly lowering the real density.  An obstacle may
            # land beyond a door and block it — allowed, and how a route gets
            # pruned.
            if self.rooms_per_side > 1:
                self.build_one_way_rooms()
            self.place_random_obstacles()
            # Reaching the goal is the only condition.  Traps — a room whose exits
            # are all blocked — are deliberately allowed: Algorithm 1 already
            # refuses to count a trapped waypoint as achievable.
            if self.check_for_path():
                valid_config_found = True
            else:
                curr_tries += 1
        status_line(None)

        if not valid_config_found:
            # No silent fallback.  See UnsolvableLayout: substituting an
            # obstacle-free board here would answer the caller with a different
            # game than the one it asked for, and every mean computed from it
            # would be wrong without saying so.
            raise UnsolvableLayout(
                f"{type(self).__name__}: no solvable {self.board_side}x"
                f"{self.board_side} layout in {max_tries:,} tries at "
                f"obstacle_density={self.obstacle_density} "
                f"({self.rooms_per_side}x{self.rooms_per_side} rooms of "
                f"{self.room_side}x{self.room_side}). Lower the density, or "
                f"widen the board.")

        self.create_state_space()
        assert slip_prob >= 0 and slip_prob * 3 <= 1, \
            "Slip probability should be >= 0 and 3*slip_prob <= 1."

    # ── Map construction ─────────────────────────────────────────────────────
    def _blank_board(self):
        """Empty map, no walls, no doors — the state every attempt starts from.

        Every attempt of the layout search starts from here, so no attempt can
        inherit the walls or obstacles of the one before it.  With a single room
        both collections stay empty, which is what reduces every movement test to
        "is this cell an obstacle?".
        """
        self.map = np.zeros((self.board_side, self.board_side))
        self.doors = {}
        self.blocked_edges = set()

    def protected_cells(self):
        """Cells an obstacle must never cover. Subclasses widen this.

        Doors cannot be in here: a door is an edge, not a cell, so there is
        nothing for an obstacle to land on.

        TaxiWorld adds the passenger: its location is shared across the robot and
        every human, so burying it under one model's obstacles would force that
        model to relocate and silently break the sharing.
        """
        return {self.start_pos, self.goal_pos}

    def build_one_way_rooms(self):
        """Split the board into ``rooms_per_side`` x ``rooms_per_side`` equal square
        rooms of ``room_side`` cells a side, and open exactly one **one-way**
        door in every wall between two neighbouring rooms.

        Walls are thin — they run *between* cells and take up none of the board —
        so a wall is a set of forbidden *crossings*, not a row of cells.  You
        never stand in a door; you stand west of it and one step right puts you in
        the next room.

        Every door faces east or south, so the room grid is a DAG from the start's
        room (top-left) to the goal's (bottom-right).  That orientation is the
        point: on a reversible board one trajectory can tour every bottleneck and
        come back, so Algorithm 1 finds a single maximally achievable subset
        however the walls fall.  Here a door *commits*, so the achievable subsets
        are the monotone routes through the room grid — ``C(2(r-1), r-1)`` of them
        for ``r`` rooms per side (6 for 3x3, 20 for 4x4).

        Doors sit at the **middle** of their wall with no randomness, so the robot
        and every human share them, as they already share the start, the goal and
        TaxiWorld's passenger.  Only the obstacles differ, deliberately: B is the
        *union* of the humans' bottleneck sets, so per-model doors made |B| grow
        with the human count against an Algorithm 1 costing 2^|B|.  Shared
        doors make the humans disagree about which *route* is forced instead.
        """
        self.state_space = None            # stale once the layout changes
        boundaries = _room_boundaries(self.rooms_per_side, self.room_side)
        midlines = _room_midlines(self.rooms_per_side, self.room_side)

        # Each wall is sealed down its whole length before its doors are punched,
        # which is why the door lines are known up front rather than band by band.
        for b in boundaries:
            for row in range(self.board_side):
                self.blocked_edges.add(((row, b + 1), (row, b)))      # westward: never
                if row not in midlines:
                    self.blocked_edges.add(((row, b), (row, b + 1)))  # eastward: only doors
            for row in midlines:
                self.doors[((row, b), (row, b + 1))] = "east"
        for b in boundaries:
            for col in range(self.board_side):
                self.blocked_edges.add(((b + 1, col), (b, col)))      # northward: never
                if col not in midlines:
                    self.blocked_edges.add(((b, col), (b + 1, col)))  # southward: only doors
            for col in midlines:
                self.doors[((b, col), (b + 1, col))] = "south"
        # Nothing to reopen afterwards: thin walls never overwrite a cell, so the
        # start and the goal survive the layout untouched.

    def _edge_free(self, frm, to):
        """True when a single step ``frm`` -> ``to`` is permitted.

        The one predicate every part of this class asks about movement: bounds,
        obstacles, and the directed door edges.  With a single room
        ``blocked_edges`` is empty and this is a plain bounds-and-obstacles test.
        """
        x, y = to
        if not (0 <= x < self.board_side and 0 <= y < self.board_side):
            return False
        if self.map[x, y] == -1:
            return False
        return (frm, to) not in self.blocked_edges

    def place_random_obstacles(self):
        """Scatter obstacles on free, unprotected cells.

        Draws from ``self.rng``, seeded once in ``__init__`` and never reset
        here.  Reseeding on entry would both clobber the process-wide numpy RNG
        and, since this runs inside the retry loop, rewind the stream so every
        retry redrew the identical unsolvable layout.
        """
        self.state_space = None
        total_obstacles = int(self.board_side * self.board_side * self.obstacle_density)
        protected = self.protected_cells()
        # Guard against an impossible request.  The loop below draws cells at
        # random and rejects the protected and the already-taken ones, so asking
        # for more obstacles than there are cells to hold them does not fail —
        # it spins for ever, inside a swallowed stdout and behind a frozen
        # progress bar, which is the least diagnosable failure this code has.
        #
        # The ceiling is well below 1.0 because protected_cells() grows: on a 9x9
        # RockWorld at the default rock density, protecting every rock brings the
        # highest workable obstacle_density down from 0.975 to about 0.68.
        #
        # Raised rather than clamped: silently placing fewer obstacles would make
        # the obstacle_density reported in the CSV a lie, and the request is
        # arithmetically impossible for every seed, so retrying cannot help.
        placeable = self.board_side * self.board_side - len(protected)
        if total_obstacles > placeable:
            raise ValueError(
                f"obstacle_density {self.obstacle_density} asks for "
                f"{total_obstacles} obstacles on a {self.board_side}x"
                f"{self.board_side} board, but only {placeable} cells are free "
                f"after protecting {len(protected)} (start, goal, and any shared "
                f"items such as rocks or the taxi passenger).")
        obstacles_placed = 0
        while obstacles_placed < total_obstacles:
            x = self.rng.randint(self.board_side)
            y = self.rng.randint(self.board_side)
            if (x, y) not in protected and self.map[x, y] != -1:
                self.map[x, y] = -1
                obstacles_placed += 1

    def place_start_and_goal(self):
        if self.start_pos is None:
            self.start_pos = (self.rng.randint(self.board_side), self.rng.randint(self.board_side))
        if self.goal_pos is None:
            self.goal_pos = (self.rng.randint(self.board_side), self.rng.randint(self.board_side))

    def successor_cells(self, state):
        """The cells a single step from ``state`` can land on, staying included.

        A *superset*: it names where a step could possibly end up, and says
        nothing about whether any action actually goes there — walls, obstacles
        and one-way doors are not consulted.  Deciding that stays the job of
        ``get_transition_probability``, which remains the only authority on what
        a transition is.

        This exists purely so ``augment_mdp_to_deterministic`` can stop scanning
        the whole state space.  Every action in the grid family either moves one
        cell or stays put — the movement branch rejects anything else outright,
        and TaxiWorld's pickup/dropoff only flip flags — so no reachable
        successor is ever outside this set.  Presence of this method is also the
        determinizer's signal that ``state[0]`` is the state's cell, which is the
        invariant the whole grid family is built on.
        """
        x, y = state[0]
        return ((x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))

    # ── Connectivity ─────────────────────────────────────────────────────────
    def get_all_neighbors(self, state):
        x, y = state
        neighbors = []
        for dx, dy, action in [(0, -1, "left"), (0, 1, "right"), (-1, 0, "up"), (1, 0, "down")]:
            new_x, new_y = x + dx, y + dy
            if self._edge_free((x, y), (new_x, new_y)):
                neighbors.append(((new_x, new_y), action))
        return neighbors

    def check_goal_reached(self, state):
        return state == self.goal_pos

    def is_absorbing_state(self, state):
        """True when no action can leave `state` — the terminal sink.

        Takes a *full* state, unlike check_goal_reached, which takes a position.
        TaxiWorld overrides it: its destination must stay passable until the
        passenger is delivered, so there the sink is the `delivered` flag.
        """
        return self.check_goal_reached(state[0])

    def _reachable_positions(self, origin=None):
        """Cells reachable from ``origin`` (the start by default) by moves.

        Obstacles block, and so does the wrong side of a one-way door: this walks
        ``get_all_neighbors``, the directed edge set.  Reachability is therefore
        not symmetric, which is why the caller says where it starts.

        Lives on the base class because two games need to ask the question in
        legs — TaxiWorld must reach the passenger *then* the destination,
        RockWorld a valuable rock *then* the goal — and a plain start-to-goal
        walk cannot answer either.
        """
        origin = self.start_pos if origin is None else origin
        seen, frontier = {origin}, [origin]
        while frontier:
            for nxt, _ in self.get_all_neighbors(frontier.pop()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen

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
                            for i in range(self.board_side)
                            for j in range(self.board_side)]

    def get_state_space(self):
        if self.state_space is None:
            self.create_state_space()
        return self.state_space

    def _transition_probability_for_move(self, state, action, state_prime):
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

        # A blocked edge is impossible as the outcome of *any* action, slip
        # included.  Without this the branches below would hand it slip_prob: the
        # cell beyond is free, and only the edge into it is blocked.
        if ((x_prime, y_prime) != (x, y)
                and not self._edge_free((x, y), (x_prime, y_prime))):
            return 0

        up_free = self._edge_free((x, y), (x - 1, y))
        down_free = self._edge_free((x, y), (x + 1, y))
        left_free = self._edge_free((x, y), (x, y - 1))
        right_free = self._edge_free((x, y), (x, y + 1))

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
        return self._transition_probability_for_move(state, action, state_prime)

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

        Subclasses swap in their own reward_func — puddles, rocks, the taxi fare
        — so this accessor stays generic across the grid games.

        Nothing in the benchmark pipeline reads it: bottlenecks are a property of
        *reachability*, not of reward, so the whole comparison runs on the
        transition structure alone.  It is here for a caller that wants to solve
        one of these grids as an ordinary MDP.
        """
        return self.reward_func

    def get_init_state(self):
        return [self.start_pos]

    def get_goal_states(self):
        return [[self.goal_pos]]

    def cell_char(self, i, j):
        """The one character standing for cell ``(i, j)``.

        Subclasses override *this* rather than ``visualize``: drawing thin walls
        is the same work whatever a cell contains, so overriding ``visualize``
        would fork it into four copies that then drift.
        """
        if (i, j) == self.start_pos:
            return "S"
        if (i, j) == self.goal_pos:
            return "G"
        if self.map[i, j] == -1:
            return "#"
        return "."

    def wall_char(self, a, b):
        """The one character standing for the gap between adjacent cells a and b.

        The counterpart of ``cell_char``: that one says what is *in* a cell, this
        one what is *between* two.  ``" "`` when both directions are open, the
        door character when only the forward one survives, a wall otherwise.
        """
        same_row = a[0] == b[0]          # side by side, so the gap runs vertically
        forward_open = (a, b) not in self.blocked_edges
        backward_open = (b, a) not in self.blocked_edges
        if forward_open and backward_open:
            return " "
        if forward_open:
            return ">" if same_row else "v"
        if backward_open:
            return "<" if same_row else "^"
        return "|" if same_row else "-"

    def visualize(self):
        """ASCII render of the board.

        With no walls this is the plain ``". . . G"`` grid.  With a room grid the
        gaps are drawn too: ``|`` and ``-`` are sealed, and a door shows the one
        direction it allows — ``>`` east, ``v`` south.  So ``. > .`` means "step
        right and you are in the next room, with no way back".
        """
        has_walls = bool(self.blocked_edges)
        for i in range(self.board_side):
            row = ""
            for j in range(self.board_side):
                row += self.cell_char(i, j)
                if j + 1 < self.board_side:
                    row += self.wall_char((i, j), (i, j + 1)) if has_walls else " "
            print(row)
            if has_walls and i + 1 < self.board_side:
                gap = ""
                for j in range(self.board_side):
                    gap += self.wall_char((i, j), (i + 1, j))
                    if j + 1 < self.board_side:
                        gap += " "
                # Only the row boundaries that carry a wall get a line.  Between
                # two rows inside the same room every crossing is open, so the
                # line would be blank and would only stretch the board.
                if gap.strip():
                    print(gap)

    # ── Image rendering ──────────────────────────────────────────────────────
    # One entry per character cell_char() can return: (fill colour, text colour).
    # Subclasses extend this dict rather than reimplementing render(): cell_char
    # says *what* is in a cell, CHAR_STYLE how it looks.
    CHAR_STYLE = {
        ".": ("#ffffff", "#000000"),      # free cell
        "#": ("#37474f", "#ffffff"),      # obstacle
        "S": ("#1e88e5", "#ffffff"),      # start
        "G": ("#43a047", "#ffffff"),      # goal
    }
    WALL_COLOR = "#212121"
    DOOR_COLOR = "#ef6c00"
    GRID_COLOR = "#e0e0e0"

    def render(self, cell_px=54, dpi=100, title=None, show_coords=True):
        """Draw the board and return it as a ``PIL.Image.Image``.

        A picture rather than a print: a sealed crossing is a bar *between* two
        cells and a door an arrow through it, which is the geometry the MDP
        implements and the thing ASCII can only hint at.

        A notebook shows the returned image inline, and saving is its own job::

            img = mdp.render()
            img.save("board.png")          # or .pdf, .jpg, ...

        matplotlib is imported here, not at module scope, so this module stays
        importable with numpy alone; the figure goes through the Agg canvas rather
        than pyplot so rendering never touches a notebook's current figure.
        """
        from PIL import Image
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        n = self.board_side
        pad = 0.6 if show_coords else 0.15
        fig = Figure(figsize=((n + 2 * pad) * cell_px / dpi,
                              (n + 2 * pad) * cell_px / dpi), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(-pad, n + pad)
        ax.set_ylim(n + pad, -pad)          # row 0 on top, like the ASCII render
        ax.set_aspect("equal")
        ax.axis("off")

        default = self.CHAR_STYLE["."]
        for i in range(n):
            for j in range(n):
                char = self.cell_char(i, j)
                fill, ink = self.CHAR_STYLE.get(char, default)
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fill,
                                       edgecolor=self.GRID_COLOR, linewidth=0.8))
                if char not in (".", "#"):
                    ax.text(j + 0.5, i + 0.5, char, color=ink, ha="center",
                            va="center", fontsize=cell_px * 0.30, fontweight="bold")

        # Walls live on the boundaries, so they are drawn after every cell.
        for i in range(n):
            for j in range(n):
                if j + 1 < n:
                    self._draw_boundary(ax, (i, j), (i, j + 1), cell_px)
                if i + 1 < n:
                    self._draw_boundary(ax, (i, j), (i + 1, j), cell_px)
        ax.add_patch(Rectangle((0, 0), n, n, fill=False,
                               edgecolor=self.WALL_COLOR, linewidth=2.2))

        if show_coords:
            for k in range(n):
                ax.text(k + 0.5, -0.28, str(k), ha="center", va="center",
                        fontsize=cell_px * 0.17, color="#9e9e9e")
                ax.text(-0.28, k + 0.5, str(k), ha="center", va="center",
                        fontsize=cell_px * 0.17, color="#9e9e9e")
        if title:
            ax.set_title(title, fontsize=cell_px * 0.22, pad=6)

        canvas.draw()
        return Image.frombytes("RGBA", canvas.get_width_height(),
                               bytes(canvas.buffer_rgba()))

    def _draw_boundary(self, ax, a, b, cell_px):
        """Draw the wall or door on the boundary between adjacent cells a and b."""
        sep = self.wall_char(a, b)
        if sep == " ":
            return
        same_row = a[0] == b[0]              # a and b side by side -> vertical wall
        if same_row:
            x, y = b[1], a[0] + 0.5          # the shared edge is the column x
            wall = ((x, x), (a[0], a[0] + 1))
            arrow = ((x - 0.42, y), (x + 0.42, y))
        else:
            x, y = a[1] + 0.5, b[0]
            wall = ((a[1], a[1] + 1), (y, y))
            arrow = ((x, y - 0.42), (x, y + 0.42))

        if sep in ("|", "-"):                # sealed both ways
            ax.plot(*wall, color=self.WALL_COLOR, linewidth=2.6,
                    solid_capstyle="butt")
            return
        # A door: an arrow through the gap, no bar.  "<" and "^" are reachable
        # only if a caller blocks edges by hand — build_one_way_rooms never does.
        (x0, y0), (x1, y1) = arrow
        if sep in ("<", "^"):
            (x0, y0), (x1, y1) = (x1, y1), (x0, y0)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=self.DOOR_COLOR,
                                    linewidth=2.2, mutation_scale=cell_px * 0.30,
                                    shrinkA=0, shrinkB=0))


# ─────────────────────────────────────────────────────────────────────────────
# Determinization
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

    # The scan below is the whole cost of this function, so it is worth narrowing.
    #
    # Done naively it is O(|S|^2 |A|): for every (state, action), ask the MDP
    # about every state in the game.  Nearly all of those questions can only be
    # answered 0 — _transition_probability_for_move rejects any non-adjacent cell
    # on its first line — so a 25x25 rockworld would make ~100M calls to find at
    # most 40 non-zero answers per row.
    #
    # An MDP that can name the cells one step can reach (successor_cells) lets us
    # ask only about the states sitting on those cells, which is roughly a 100x
    # saving on the larger boards.  It is a search-space reduction and nothing
    # more: every surviving candidate still goes through
    # get_transition_probability, so the matrix produced is identical to the
    # exhaustive one.
    #
    # `getattr` rather than a plain call, so an MDP *without* the method still
    # works — it silently gets the exhaustive scan.  Every game in the repo
    # defines it (all four inherit GridWorld.successor_cells), so that branch is
    # currently unexercised; it is the contract that lets a future MDP whose
    # state is not simply a cell be determinized here without being rewritten
    # first.
    successor_cells = getattr(mdp, "successor_cells", None)
    states_by_cell = {}
    if successor_cells is not None:
        for state_idx, state in enumerate(states):
            states_by_cell.setdefault(state[0], []).append(state_idx)

    per_state_outcomes = []
    for state in states:
        if successor_cells is None:
            candidates = range(n_states)
        else:
            # Sorted, so outcomes are visited in ascending state-index order,
            # exactly as the exhaustive scan would visit them.  That is what
            # makes the augmented action indices — and so the whole matrix —
            # identical whichever of the two paths was taken.
            candidates = sorted(
                idx for cell in successor_cells(state)
                for idx in states_by_cell.get(cell, ()))
        outcomes = []
        for orig_action in original_actions:
            for next_state_idx in candidates:
                if mdp.get_transition_probability(
                        state, orig_action, states[next_state_idx]) > 1e-12:
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

    UNUSED — no caller anywhere in the pipeline, and no test.  Its only consumer
    is bottlenecks.value_iteration, which is unused on the same terms: none of
    the four selection rules the benchmark reports needs a value function.  Kept
    as the starting point for a value-based rule, should one be wanted, and to be
    treated as untested code until then.

    This is the matrix the determinization throws away.  augment_mdp_to_deterministic
    turns every (action, outcome) pair into its own action, which lets the agent
    choose its own slip outcome; values computed from that determinized array are
    therefore optimistic, not the values of M_R.

    Returns
    -------
    A 5-tuple.  The first two are the model; the last three are what
    bottlenecks.value_iteration would need to run in *reward* mode rather than
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
    first two, so unpack with a star to handle either —
    `T, index, *rest = build_stochastic_matrix(...)`, then
    `value_iteration(T, index[goal], *rest)` is correct for both.
    """
    states  = mdp.get_state_space()
    actions = mdp.get_actions()
    n_s     = len(states)

    hashes    = [mdp.get_state_hash(s) for s in states]
    start_idx = hashes.index(mdp.get_state_hash(mdp.get_init_state()))

    # Successors first (probabilities not needed yet), then BFS from the start.
    # This scan is exhaustive and O(|S|^2 |A|) — it has no equivalent of
    # augment_mdp_to_deterministic's successor_cells narrowing — and it dominates
    # the cost of anything built on top of it.
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
