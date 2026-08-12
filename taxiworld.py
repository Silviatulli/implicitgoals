"""
taxiworld.py — TaxiWorld + determinized-MDP generator.
==========================================================

Exported from the *implicitgoals* research repo. Builds on the shared
``GridWorld`` MDP and ``augment_mdp_to_deterministic`` helper defined in
``gridworld_core.py`` (the same core used by gridworld.py, puddleworld.py,
and rockworld.py — see that module for the shared plumbing).

TaxiWorld is a GridWorld where a taxi must pick up a passenger and drop it at a
destination. A state is ``[(row, col), passenger_in_taxi, delivered]`` and the
action set adds ``"pickup"`` / ``"dropoff"`` to the four moves. Moves slip
(10%); pickup / dropoff are deterministic.

``delivered`` latches True only on a dropoff at the destination while carrying,
and never resets, so the goal ``[destination, False, True]`` means "task
complete" — not merely "standing on the destination", which a taxi that never
picked the passenger up could also satisfy.  It is also the sink: the
destination cell stays passable until the delivery actually happens.

Rewards are cost-to-go: ``−1`` per action until delivery, ``0`` for ever after,
and an extra ``−10`` for dropping the passenger anywhere else.

NOTE: the ``generate_and_visualize_taxiworld`` in the repo's ``experiments.py``
was out of sync with the ``TaxiWorld`` constructor (it passed multi-passenger
args the class does not accept) and was never exercised, since taxi only runs
off-macOS. The version here is written to match the real single-passenger class.

Quick start
-----------
    from taxiworld import generate_determinized_models
    out = generate_determinized_models(size=4, num_humans=3,
                                       obstacles_percent=0.1, seed=0)
    T_R, s0, g = out["robot"][:3]
    print(out["total_determinizing_time"])
"""

import time
import random

import numpy as np

from gridworld_core import GridWorld, augment_mdp_to_deterministic, seeded_rng


# ─────────────────────────────────────────────────────────────────────────────
# TaxiWorld (ported from TaxiWorldClass.py)
# ─────────────────────────────────────────────────────────────────────────────

class TaxiWorld(GridWorld):
    """GridWorld + a passenger to pick up and drop at a destination. State is
    ``[(row, col), passenger_in_taxi, delivered]``; actions add ``pickup`` /
    ``dropoff``.  The third slot latches on a successful delivery and makes the
    goal ``[destination, False, True]`` mean "task complete" rather than merely
    "standing on the destination"."""

    def __init__(self, size=5, start=None, passenger_loc=None, destination=None,
                 obstacles_percent=0.1, slip_prob=0.1, discount=0.99, max_tries=100,
                 obstacle_seed=1, wrong_dropoff_penalty=-10):
        self.size = size  # needed before place_random_location
        # This world draws *before* GridWorld.__init__ runs, so it builds the
        # generator and GridWorld adopts it rather than seeding a second one.
        obstacle_seed, self.rng = seeded_rng(obstacle_seed)
        self.passenger_loc = passenger_loc if passenger_loc is not None else self.place_random_location()
        self.wrong_dropoff_penalty = wrong_dropoff_penalty
        super().__init__(size=size, start=start, goal=destination,
                         obstacles_percent=obstacles_percent, slip_prob=slip_prob,
                         discount=discount, max_tries=max_tries, obstacle_seed=obstacle_seed)
        self.destination = self.goal_pos  # reuse goal_pos as destination
        self.reward_func = self.taxi_reward_func
        # The passenger was drawn before the map existed, so it may have landed
        # under an obstacle or in a walled-off pocket.  That used to be
        # harmless — the old goal [destination, False] was reachable without
        # ever collecting the passenger — but the delivered goal is not, so an
        # unreachable passenger now means an unreachable goal.  Fix it here.
        self._ensure_passenger_reachable()

    def place_random_location(self):
        while True:
            x, y = self.rng.randint(self.size), self.rng.randint(self.size)
            if not hasattr(self, 'map') or self.map[x, y] != -1:
                return (x, y)

    def protected_cells(self):
        """Keep obstacles off the passenger as well as the start and goal.

        The passenger location is shared by the robot and every human of an
        instance, so it must survive each model's independent obstacle draw.
        """
        return super().protected_cells() | {self.passenger_loc}

    def _reachable_positions(self):
        """Cells reachable from the start by moves (obstacles block)."""
        seen, frontier = {self.start_pos}, [self.start_pos]
        while frontier:
            for nxt, _ in self.get_all_neighbors(frontier.pop()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen

    def _ensure_passenger_reachable(self):
        """Relocate the passenger if it is unreachable from the start.

        Prefers a cell the taxi can actually drive to; the destination itself is
        allowed (pick up and drop off on the spot) and is the last-resort choice,
        since start→destination connectivity is already guaranteed by
        GridWorld.check_for_path.
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
        for i in range(self.size):
            for j in range(self.size):
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

    def visualize(self):
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if (i, j) == self.start_pos:
                    row += "T "
                elif (i, j) == self.passenger_loc:
                    row += "P "
                elif (i, j) == self.destination:
                    row += "D "
                elif self.map[i, j] == -1:
                    row += "# "
                else:
                    row += ". "
            print(row)


def generate_and_visualize_taxiworld(size, start, goal, obstacles_percent,
                                     model_type="Model", obstacle_seed=None,
                                     passenger_loc=None, destination=None):
    """Generate a single-passenger ``TaxiWorld``.

    ``destination`` defaults to ``goal`` (or the bottom-right corner); the
    passenger is placed at a random cell if ``passenger_loc`` is None.
    """
    if destination is None:
        destination = goal if goal is not None else (size - 1, size - 1)
    if passenger_loc is None:
        passenger_loc = (random.randint(0, size - 1), random.randint(0, size - 1))
    return TaxiWorld(size=size, start=start, passenger_loc=passenger_loc,
                     destination=destination, obstacles_percent=obstacles_percent,
                     obstacle_seed=obstacle_seed)


# ─────────────────────────────────────────────────────────────────────────────
# High-level driver — robot + N humans, with compute timing
# ─────────────────────────────────────────────────────────────────────────────

def _make_determinized(size, obstacles_percent, model_type, visualize=False,
                       passenger_loc=None):
    """Generate one taxi world and determinize it; returns (next_states, s0, g, det_time).

    ``passenger_loc`` is passed down so every model of an instance shares it; see
    generate_determinized_models.  If ``visualize`` is True, print the generated
    map before determinizing.
    """
    mdp = generate_and_visualize_taxiworld(
        size=size, start=(0, 0), goal=(size - 1, size - 1),
        obstacles_percent=obstacles_percent, passenger_loc=passenger_loc,
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
                                 seed=None, verbose=True, visualize=False):
    """Build a robot model + ``num_humans`` human TaxiWorld models and determinize each.

    Parameters
    ----------
    size : int              grid side length
    num_humans : int        number of human models
    obstacles_percent : float   obstacle density in [0, 1]
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

    # One passenger for the whole instance: the robot and every human agree on
    # where the fare is, and only the obstacle map varies between them — the same
    # convention the other grid games follow for start and goal.  Drawing it per
    # model instead made each human's pickup its own bottleneck, so |B| (the
    # union over humans) grew with the human count and blew past
    # --max-bottlenecks, skipping every taxi repetition.
    passenger_loc = (random.randint(0, size - 1), random.randint(0, size - 1))

    robot = _make_determinized(size, obstacles_percent, "Robot Model", visualize,
                               passenger_loc=passenger_loc)
    humans = [_make_determinized(size, obstacles_percent, f"Human Model {i + 1}",
                                 visualize, passenger_loc=passenger_loc)
              for i in range(num_humans)]

    det_times = [robot[3]] + [h[3] for h in humans]
    total = float(sum(det_times))

    if verbose:
        print(f"[taxiworld] size={size} obstacles={obstacles_percent} humans={len(humans)}")
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
                                       obstacles_percent=0.1, seed=0, visualize=True)
    T_R, s0, g, _ = out["robot"]
    print("\nRobot determinized transition array shape:", T_R.shape)
    print("start index:", s0, " goal index:", g)
    print("num human models:", len(out["humans"]))
    print("total determinizing time (s):", out["total_determinizing_time"])
