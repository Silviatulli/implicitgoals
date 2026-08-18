"""
taxiworld.py — TaxiWorld + determinized-MDP generator.
==========================================================

Builds on the shared ``GridWorld`` MDP and ``augment_mdp_to_deterministic``
helper defined in ``gridworld_core.py`` (the same core used by gridworld.py, puddleworld.py,
and rockworld.py — see that module for the shared plumbing).

TaxiWorld is a GridWorld where a taxi must pick up a passenger and drop it at a
destination. A state is ``[(row, col), passenger_in_taxi, delivered]`` and the
action set adds ``"pickup"`` / ``"dropoff"`` to the four moves. Moves slip with
probability ``slip_prob`` per unintended neighbour, 0 by default; pickup /
dropoff are always deterministic.

``delivered`` latches True only on a dropoff at the destination while carrying,
and never resets, so the goal ``[destination, False, True]`` means "task
complete" — not merely "standing on the destination", which a taxi that never
picked the passenger up could also satisfy.  It is also the sink: the
destination cell stays passable until the delivery actually happens.

Rewards are cost-to-go: ``−1`` per action until delivery, ``0`` for ever after,
and an extra ``−10`` for dropping the passenger anywhere else.

Quick start
-----------
    from taxiworld import generate_determinized_models
    out = generate_determinized_models(room_side=4, num_humans=3,
                                       obstacle_density=0.1, seed=0)
    T_R, s0, g = out["robot"][:3]
    print(out["total_determinizing_time"])
"""

import time
import random

import numpy as np

from gridworld_core import (GridWorld, augment_mdp_to_deterministic, board_side,
                            seeded_rng, DEFAULT_MAX_TRIES)


# ─────────────────────────────────────────────────────────────────────────────
# TaxiWorld
# ─────────────────────────────────────────────────────────────────────────────

class TaxiWorld(GridWorld):
    """GridWorld + a passenger to pick up and drop at a destination. State is
    ``[(row, col), passenger_in_taxi, delivered]``; actions add ``pickup`` /
    ``dropoff``.  The third slot latches on a successful delivery and makes the
    goal ``[destination, False, True]`` mean "task complete" rather than merely
    "standing on the destination"."""

    def __init__(self, start=None, passenger_loc=None, destination=None,
                 obstacle_density=0.1, slip_prob=0.0, gamma=0.99,
                 max_tries=DEFAULT_MAX_TRIES,
                 obstacle_seed=1, wrong_dropoff_penalty=-10,
                 rooms_per_side=1, room_side=5):
        # The passenger is drawn before GridWorld.__init__ runs, so this world has
        # to know the board a step earlier than the others do.
        self.board_side = board_side(rooms_per_side, room_side)
        # This world draws *before* GridWorld.__init__ runs, so it builds the
        # generator and GridWorld adopts it rather than seeding a second one.
        obstacle_seed, self.rng = seeded_rng(obstacle_seed)
        self.passenger_loc = passenger_loc if passenger_loc is not None else self.place_random_location()
        self.wrong_dropoff_penalty = wrong_dropoff_penalty
        super().__init__(start=start, goal=destination,
                         obstacle_density=obstacle_density, slip_prob=slip_prob,
                         gamma=gamma, max_tries=max_tries, obstacle_seed=obstacle_seed,
                         rooms_per_side=rooms_per_side,
                         room_side=room_side)
        self.destination = self.goal_pos  # reuse goal_pos as destination
        self.reward_func = self.taxi_reward_func
        # The passenger was drawn before the map existed, so it may sit under an
        # obstacle or in a walled-off pocket.  The delivered goal cannot be
        # reached without collecting it, so an unreachable fare is a dead board.
        self._ensure_passenger_reachable()

    def place_random_location(self):
        while True:
            x, y = self.rng.randint(self.board_side), self.rng.randint(self.board_side)
            if not hasattr(self, 'map') or self.map[x, y] != -1:
                return (x, y)

    def protected_cells(self):
        """Keep obstacles off the passenger as well as the start and goal.

        The passenger location is shared by the robot and every human of an
        instance, so it must survive each model's independent obstacle draw.
        """
        return super().protected_cells() | {self.passenger_loc}

    def _reachable_positions(self, origin=None):
        """Cells reachable from ``origin`` (the start by default) by moves.

        Obstacles block, and so does the wrong side of a one-way door: this walks
        ``get_all_neighbors``, the directed edge set.  Reachability is therefore
        not symmetric, which is why the caller says where it starts.
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
        """Is there a trajectory from the initial state to the *goal state*?

        GridWorld walks to the goal cell, which is enough when the goal is "stand
        here".  Here the goal state is ``[destination, False, True]`` and
        ``delivered`` latches only on a dropoff while carrying, so standing on the
        destination proves nothing — the taxi has to fetch the passenger first.

        The trajectory therefore needs two legs, start -> passenger and
        passenger -> destination, which on a room-grid board are genuinely two
        questions: a one-way door can let the taxi reach the fare and then strand
        it.  Overriding this is what makes the retry loop redraw such a board.

        Runs from inside ``GridWorld.__init__``, before ``self.destination``
        exists, so it reads ``goal_pos`` — the same cell under an earlier name.
        """
        if self.start_pos is None or self.goal_pos is None:
            return False
        if self.passenger_loc not in self._reachable_positions(self.start_pos):
            return False
        return self.goal_pos in self._reachable_positions(self.passenger_loc)

    def _ensure_passenger_reachable(self):
        """Relocate the passenger if it is unreachable from the start.

        Prefers a cell the taxi can actually drive to; the destination itself is
        the last-resort choice (pick up and drop off on the spot), since
        start-to-destination connectivity is already guaranteed.
        """
        reachable = self._reachable_positions()
        if self.passenger_loc in reachable:
            return
        candidates = sorted(reachable - {self.destination})
        self.passenger_loc = (candidates[self.rng.randint(len(candidates))]
                              if candidates else self.destination)

    def get_actions(self):
        return super().get_actions() + ["pickup", "dropoff"]

    def create_state_space(self):
        """States are ``[position, passenger_in_taxi, delivered]``.

        ``delivered`` latches True on a successful dropoff at the destination and
        never resets — that is what separates "delivered" from "never picked the
        passenger up".  Only three flag combinations are legal: (False, False)
        not started / wrongly dropped off, (True, False) carrying, (False, True)
        delivered.  You cannot hold a passenger you have already delivered, so
        (True, True) is never enumerated.
        """
        self.state_space = []
        for i in range(self.board_side):
            for j in range(self.board_side):
                for carrying, delivered in ((False, False), (True, False), (False, True)):
                    self.state_space.append([(i, j), carrying, delivered])

    def is_absorbing_state(self, state):
        """The sink is *delivery*, not the destination cell.

        Overriding this keeps the destination passable until the job is done: the
        taxi may drive across it while fetching the passenger, and only
        ``[destination, False, True]`` self-loops.
        """
        return state[2]

    def get_transition_probability(self, state, action, state_prime):
        pos, carrying, delivered = state[0], state[1], state[2]

        if delivered:                        # absorbing: the task is over
            return 1 if state == state_prime else 0

        if action in ["up", "down", "left", "right"]:
            # Moving changes the position only; both flags must carry over.
            if state_prime[1] != carrying or state_prime[2]:
                return 0
            return super().get_transition_probability(state, action, state_prime)
        elif action == "pickup":
            if pos != self.passenger_loc or carrying:
                return 1 if state == state_prime else 0
            return 1 if state_prime == [pos, True, False] else 0
        elif action == "dropoff":
            if not carrying:
                return 1 if state == state_prime else 0
            if pos == self.destination:      # the delivery — latch `delivered`
                return 1 if state_prime == [pos, False, True] else 0
            return 1 if state_prime == [pos, False, False] else 0    # wrong place
        return 0

    def taxi_reward_func(self, state, action, next_state):
        """Cost-to-go: −1 per action until delivery, 0 for ever after.

        The delivery itself is an ordinary step (−1): there is no bonus at the
        absorbing state, since a reward paid *at* a sink cannot change the policy
        and only breaks V(goal) = 0.  Dropping the passenger anywhere other than
        the destination still costs an extra −10 on top of the step.
        """
        if state[2]:                         # already delivered: nothing is paid
            return 0
        reward = -1
        if action == "dropoff" and state[1] and not next_state[2]:
            reward += self.wrong_dropoff_penalty
        return reward

    def get_init_state(self):
        return [self.start_pos, False, False]

    def get_goal_states(self):
        # Delivered *and* actually transported — unreachable by merely driving to
        # the destination, which is the whole point of the third slot.
        return [[self.destination, False, True]]

    # "D" is the destination here, not RockWorld's dangerous rock — the palette is
    # per class precisely so the same letter can mean different things per game.
    CHAR_STYLE = {**GridWorld.CHAR_STYLE,
                   "T": ("#1e88e5", "#ffffff"),      # the taxi's start
                   "P": ("#8e24aa", "#ffffff"),      # the passenger
                   "D": ("#43a047", "#ffffff")}      # the destination

    def cell_char(self, i, j):
        """``T`` taxi start, ``P`` passenger, ``D`` destination, ``#`` obstacle."""
        if (i, j) == self.start_pos:
            return "T"
        if (i, j) == self.passenger_loc:
            return "P"
        if (i, j) == self.destination:
            return "D"
        if self.map[i, j] == -1:
            return "#"
        return "."


def build_taxiworld(start, goal, obstacle_density,
                                     obstacle_seed=None,
                                     passenger_loc=None, destination=None,
                                     rooms_per_side=1, room_side=5, slip_prob=0.0):
    """Generate a single-passenger ``TaxiWorld``.

    ``destination`` defaults to ``goal`` (or the bottom-right corner); the
    passenger is placed at a random cell if ``passenger_loc`` is None.  Both
    fall-backs are measured on the board the two room numbers describe, which is
    the same board TaxiWorld will build.
    """
    n = board_side(rooms_per_side, room_side)
    if destination is None:
        destination = goal if goal is not None else (n - 1, n - 1)
    if passenger_loc is None:
        passenger_loc = (random.randint(0, n - 1), random.randint(0, n - 1))
    return TaxiWorld(start=start, passenger_loc=passenger_loc,
                     destination=destination, obstacle_density=obstacle_density,
                     obstacle_seed=obstacle_seed, rooms_per_side=rooms_per_side,
                     room_side=room_side, slip_prob=slip_prob)


# ─────────────────────────────────────────────────────────────────────────────
# High-level driver — robot + N humans, with compute timing
# ─────────────────────────────────────────────────────────────────────────────

def _make_determinized(obstacle_density, model_type, visualize=False,
                       passenger_loc=None, rooms_per_side=1, room_side=5,
                       slip_prob=0.0):
    """Generate one taxi world and determinize it; returns (next_states, s0, g, det_time).

    ``passenger_loc`` is passed down so every model of an instance shares it; see
    generate_determinized_models.  If ``visualize`` is True, print the generated
    map before determinizing.
    """
    n = board_side(rooms_per_side, room_side)
    mdp = build_taxiworld(
        start=(0, 0), goal=(n - 1, n - 1),
        obstacle_density=obstacle_density, passenger_loc=passenger_loc,
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
                                 seed=None, verbose=True, visualize=False,
                                 rooms_per_side=1, room_side=4, slip_prob=0.0):
    """Build a robot model + ``num_humans`` human TaxiWorld models and determinize each.

    Parameters
    ----------
    num_humans : int        number of human models
    obstacle_density : float   obstacle density in [0, 1]
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
    # The shared passenger cell below is drawn on this board, and every model
    # then builds the same one from the same two room numbers.
    size = board_side(rooms_per_side, room_side)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # One passenger for the whole instance, as with the start and the goal: only
    # the obstacle map varies between models.  This is load-bearing, not tidiness.
    # The pickup cell is a mandatory waypoint — no delivery happens without it —
    # so a passenger drawn per model would give every human a *different*
    # mandatory waypoint, and |B|, which is the union over the humans, would grow
    # with --humans until it blew past --max-bottlenecks and no taxi instance
    # could be drawn at all.
    passenger_loc = (random.randint(0, size - 1), random.randint(0, size - 1))

    robot = _make_determinized(obstacle_density, "Robot Model", visualize,
                               passenger_loc=passenger_loc, rooms_per_side=rooms_per_side,
                               room_side=room_side, slip_prob=slip_prob)
    humans = [_make_determinized(obstacle_density, f"Human Model {i + 1}",
                                 visualize, passenger_loc=passenger_loc,
                                 rooms_per_side=rooms_per_side,
                                 room_side=room_side, slip_prob=slip_prob)
              for i in range(num_humans)]

    det_times = [robot[3]] + [h[3] for h in humans]
    total = float(sum(det_times))

    if verbose:
        geometry = (f"{size}x{size} open board" if rooms_per_side == 1 else
                    f"{size}x{size} board = {rooms_per_side}x{rooms_per_side} rooms "
                    f"of {room_side}x{room_side}")
        print(f"[taxiworld] {geometry}, {size * size} cells, "
              f"obstacles={obstacle_density} humans={len(humans)}")
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
