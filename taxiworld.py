"""
taxiworld.py — TaxiWorld + determinized-MDP generator.
==========================================================

Exported from the *implicitgoals* research repo. Builds on the shared
``GridWorld`` MDP and ``augment_mdp_to_deterministic`` helper defined in
``gridworld_core.py`` (the same core used by gridworld.py, puddleworld.py,
and rockworld.py — see that module for the shared plumbing).

TaxiWorld is a GridWorld where a taxi must pick up a passenger and drop it at a
destination. A state is ``[(row, col), passenger_in_taxi]`` and the action set
adds ``"pickup"`` / ``"dropoff"`` to the four moves. Moves slip (10%); pickup /
dropoff are deterministic.

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

from gridworld_core import GridWorld, augment_mdp_to_deterministic


# ─────────────────────────────────────────────────────────────────────────────
# TaxiWorld (ported from TaxiWorldClass.py)
# ─────────────────────────────────────────────────────────────────────────────

class TaxiWorld(GridWorld):
    """GridWorld + a passenger to pick up and drop at a destination. State is
    ``[(row, col), passenger_in_taxi]``; actions add ``pickup`` / ``dropoff``."""

    def __init__(self, size=5, start=None, passenger_loc=None, destination=None,
                 obstacles_percent=0.1, slip_prob=0.1, discount=0.99, max_tries=100,
                 obstacle_seed=1):
        self.size = size  # needed before place_random_location
        self.passenger_loc = passenger_loc if passenger_loc is not None else self.place_random_location()
        super().__init__(size=size, start=start, goal=destination,
                         obstacles_percent=obstacles_percent, slip_prob=slip_prob,
                         discount=discount, max_tries=max_tries, obstacle_seed=obstacle_seed)
        self.destination = self.goal_pos  # reuse goal_pos as destination
        self.reward_func = self.taxi_reward_func

    def place_random_location(self):
        while True:
            x, y = np.random.randint(self.size), np.random.randint(self.size)
            if not hasattr(self, 'map') or self.map[x, y] != -1:
                return (x, y)

    def place_start_and_goal(self):
        super().place_start_and_goal()

    def get_actions(self):
        return super().get_actions() + ["pickup", "dropoff"]

    def create_state_space(self):
        self.state_space = []
        for i in range(self.size):
            for j in range(self.size):
                for passenger_in_taxi in [False, True]:
                    self.state_space.append([(i, j), passenger_in_taxi])

    def get_transition_probability(self, state, action, state_prime):
        x, y = state[0]
        passenger_in_taxi = state[1]
        passenger_in_taxi_prime = state_prime[1]

        if action in ["up", "down", "left", "right"]:
            move_prob = super().get_transition_probability(state, action, state_prime)
            return move_prob if passenger_in_taxi == passenger_in_taxi_prime else 0
        elif action == "pickup":
            if (x, y) != self.passenger_loc or passenger_in_taxi:
                return 1 if state == state_prime else 0
            else:
                return 1 if state_prime == [(x, y), True] else 0
        elif action == "dropoff":
            if not passenger_in_taxi:
                return 1 if state == state_prime else 0
            else:
                return 1 if state_prime == [(x, y), False] else 0
        return 0

    def taxi_reward_func(self, state, action, next_state):
        if action == "dropoff" and next_state[0] == self.destination and state[1] and not next_state[1]:
            return 20
        elif action == "pickup" and state[0] == self.passenger_loc and not state[1] and next_state[1]:
            return 0
        elif action == "dropoff" and state[1] and not next_state[1]:
            return -10
        else:
            return -1

    def get_init_state(self):
        return [self.start_pos, False]

    def get_goal_states(self):
        return [[self.destination, False]]

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

def _make_determinized(size, obstacles_percent, model_type, visualize=False):
    """Generate one taxi world and determinize it; returns (next_states, s0, g, det_time).

    If ``visualize`` is True, print the generated map before determinizing.
    """
    mdp = generate_and_visualize_taxiworld(
        size=size, start=(0, 0), goal=(size - 1, size - 1),
        obstacles_percent=obstacles_percent,
        model_type=model_type, obstacle_seed=random.randint(1, 10000))
    if visualize:
        print(f"\n{model_type}:")
        mdp.visualize()
    t0 = time.time()
    next_states, start_idx, goal_idx = augment_mdp_to_deterministic(mdp)
    return next_states, start_idx, goal_idx, time.time() - t0


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

    robot = _make_determinized(size, obstacles_percent, "Robot Model", visualize)
    humans = [_make_determinized(size, obstacles_percent, f"Human Model {i + 1}", visualize)
              for i in range(num_humans)]

    det_times = [robot[3]] + [h[3] for h in humans]
    total = float(sum(det_times))

    if verbose:
        print(f"[taxiworld] size={size} obstacles={obstacles_percent} humans={len(humans)}")
        print(f"  robot: {robot[0].shape[0]} states x {robot[0].shape[1]} actions "
              f"(determinized in {robot[3]:.4f}s)")
        print(f"  total determinizing time: {total:.4f}s")

    return {
        "robot": robot,
        "humans": humans,
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
