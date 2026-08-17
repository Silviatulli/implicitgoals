"""
minigridworld.py — MiniGrid-style Unlock / UnlockPickup + determinized-MDP generator.
=========================================================================================

Builds on ``augment_mdp_to_deterministic`` from ``gridworld_core.py`` (the same
core used by gridworld.py, puddleworld.py, rockworld.py, and taxiworld.py) —
but UnlockEnv/UnlockPickupEnv do NOT subclass GridWorld. Their state is
(position, facing direction, door/key/box flags), not just a grid cell, so
GridWorld's obstacle-map / BFS-validated-start-goal machinery doesn't apply.
They implement the same minimal duck-typed interface GridWorld does
(get_state_space / get_actions / get_transition_probability / get_state_hash /
get_init_state / get_goal_states) directly, with no base class — matching how
GridWorld itself has no base class in this repo.

Two tasks, selected via the ``task`` parameter:
  "unlock"        — UnlockEnv: reach the door cell and toggle it open.
                    State: ((x, y), direction, door_state). Fast to determinize.
  "unlock_pickup" — UnlockPickupEnv: pick up a key, unlock the door, get past
                    a movable ball, and pick up a box in the far room.
                    State: ((x, y), direction, door_state, has_key, has_box,
                    (ball_x, ball_y)).

WARNING: UnlockPickupEnv's state space is grid_size**2 times larger than the
other worlds' (the movable ball's position is part of the state) — about
30,000 states at grid_size=5, vs. 16-32 for gridworld/puddleworld/etc.
augment_mdp_to_deterministic is O(states**2 * actions), so determinizing
UnlockPickupEnv at grid_size=5 means billions of transition checks and will
not finish in practical time. Use a small grid_size (2-3) for "unlock_pickup".

NOTE: unlike the other four worlds, nothing here is randomized — door/key/box/
ball positions are fixed functions of grid_size, not sampled per obstacle_seed.
So "robot" and every "human" model produced by generate_determinized_models are
identical MDPs; ``seed`` is accepted for API consistency with the other worlds
but has no effect on the generated layout.

Quick start
-----------
    from minigridworld import generate_determinized_models
    out = generate_determinized_models(size=5, num_humans=3, task="unlock", seed=0)
    T_R, s0, g = out["robot"][:3]
    print(out["total_determinizing_time"])
"""

import time
import random

import numpy as np

from gridworld_core import augment_mdp_to_deterministic


# ─────────────────────────────────────────────────────────────────────────────
# UnlockEnv
# ─────────────────────────────────────────────────────────────────────────────

class UnlockEnv:
    """Agent must reach the door cell and toggle it open. State is
    ``((x, y), direction, door_state)`` with ``door_state`` 0=open, 1=closed,
    2=locked. Actions: left/right (rotate), forward (move), toggle (interact
    with the door when standing on it)."""

    def __init__(self, grid_size=5, max_steps=100, slip_prob=0.0,
                 init_state=None, goal_states=None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.actions = ["left", "right", "forward", "toggle"]
        self.door_states = [0, 1, 2]  # 0=open, 1=closed, 2=locked
        self.step_count = 0
        self.slip_prob = slip_prob

        self.door_pos = (grid_size - 1, grid_size - 1)

        self.state_space = self.get_state_space()

        self.init_state = init_state if init_state is not None else ((0, 0), 0, 2)
        self.current_state = self.init_state

        if goal_states is None:
            self.goal_states = [((x, y), direction, 0) for x in range(self.grid_size)
                                for y in range(self.grid_size) for direction in range(4)]
        else:
            self.goal_states = goal_states

        self.discount = 0.99
        self.reward_func = self.get_reward

    def get_state_space(self):
        state_space = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for direction in range(4):
                    for door_state in self.door_states:
                        state_space.append(((x, y), direction, door_state))
        return state_space

    def get_actions(self):
        return self.actions

    def get_transition_probability(self, state, action, next_state):
        transitions = self.return_transition_probabilities(state, action)
        return transitions.get(next_state, 0.0)

    def return_transition_probabilities(self, state, action):
        transitions = {}
        (x, y), direction, door_state = state

        if action == "left" or action == "right":
            intended_direction = (direction - 1) % 4 if action == "left" else (direction + 1) % 4
            transitions[((x, y), intended_direction, door_state)] = 1 - self.slip_prob
            transitions[((x, y), direction, door_state)] = self.slip_prob / 3
            transitions[((x, y), (direction + 1) % 4, door_state)] = self.slip_prob / 3
            transitions[((x, y), (direction - 1) % 4, door_state)] = self.slip_prob / 3

        elif action == "forward":
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][direction]
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                transitions[((new_x, new_y), direction, door_state)] = 1 - self.slip_prob
            else:
                transitions[((x, y), direction, door_state)] = 1 - self.slip_prob

            # Slip probabilities
            transitions[((x, y), (direction + 1) % 4, door_state)] = self.slip_prob / 3
            transitions[((x, y), (direction - 1) % 4, door_state)] = self.slip_prob / 3
            transitions[((x, y), direction, door_state)] = transitions.get(((x, y), direction, door_state), 0) + self.slip_prob / 3

        elif action == "toggle":
            if (x, y) == self.door_pos:
                if door_state == 2:  # locked
                    new_door_state = 1  # closed
                elif door_state == 1:  # closed
                    new_door_state = 0  # open
                else:
                    new_door_state = door_state  # already open
                transitions[((x, y), direction, new_door_state)] = 1 - self.slip_prob
                transitions[((x, y), direction, door_state)] = self.slip_prob
            else:
                transitions[((x, y), direction, door_state)] = 1.0

        # Ensure probabilities sum to 1
        total_prob = sum(transitions.values())
        if abs(total_prob - 1.0) > 1e-10:  # Allow for small floating-point errors
            for key in transitions:
                transitions[key] /= total_prob

        return transitions

    def get_reward(self, state, action, next_state):
        if next_state in self.goal_states and state not in self.goal_states:
            return 1 - 0.9 * (self.step_count / self.max_steps)
        return 0

    def get_init_state(self):
        return self.init_state

    def get_state_hash(self, state):
        return str(state)

    def get_goal_states(self):
        return self.goal_states

    def visualize(self):
        (x, y), direction, door_state = self.current_state
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[y][x] = '^>v<'[direction]
        door_x, door_y = self.door_pos
        grid[door_y][door_x] = 'D' if door_state > 0 else 'O'

        for row in grid:
            print(' '.join(row))
        print(f"Door state: {'Locked' if door_state == 2 else 'Closed' if door_state == 1 else 'Open'}")


# ─────────────────────────────────────────────────────────────────────────────
# UnlockPickupEnv
# ─────────────────────────────────────────────────────────────────────────────

class UnlockPickupEnv:
    """Agent must pick up a key, unlock/open the door with it, get past a
    movable ball, and pick up a box in the far room. State is ``((x, y),
    direction, door_state, has_key, has_box, (ball_x, ball_y))``. Actions:
    left/right (rotate), forward (move), toggle (unlock the door if carrying
    the key), pickup (grab the key/box when standing on it, or push the ball
    forward when standing on it)."""

    def __init__(self, grid_size=5, max_steps=100, slip_prob=0.0):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.actions = ["left", "right", "forward", "toggle", "pickup"]
        self.door_states = [0, 1, 2]  # 0=open, 1=closed, 2=locked
        self.step_count = 0
        self.slip_prob = slip_prob

        self.agent_pos = (0, 0)
        self.agent_dir = 0  # 0: North, 1: East, 2: South, 3: West
        self.door_pos = (grid_size // 2, grid_size - 1)
        self.ball_pos = (self.door_pos[0] - 1, self.door_pos[1])
        self.key_pos = (0, grid_size - 1)
        self.box_pos = (grid_size - 1, grid_size - 1)

        self.door_state = 2  # Start with locked door
        self.has_key = False
        self.has_box = False

        self.state_space = self.get_state_space()
        self.init_state = self.get_init_state()
        self.discount = 0.99
        self.reward_func = self.get_reward

    def get_state_space(self):
        state_space = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for direction in range(4):
                    for door_state in self.door_states:
                        for has_key in [False, True]:
                            for has_box in [False, True]:
                                for ball_x in range(self.grid_size):
                                    for ball_y in range(self.grid_size):
                                        state_space.append(((x, y), direction, door_state, has_key, has_box, (ball_x, ball_y)))
        return state_space

    def get_actions(self):
        return self.actions

    def get_transition_probability(self, state, action, next_state):
        transitions = self.return_transition_probabilities(state, action)
        return transitions.get(next_state, 0.0)

    def return_transition_probabilities(self, state, action):
        transitions = {}
        (x, y), direction, door_state, has_key, has_box, ball_pos = state

        if action in ["left", "right"]:
            intended_direction = (direction - 1) % 4 if action == "left" else (direction + 1) % 4
            transitions[((x, y), intended_direction, door_state, has_key, has_box, ball_pos)] = 1 - self.slip_prob
            for slip_dir in range(4):
                if slip_dir != intended_direction:
                    transitions[((x, y), slip_dir, door_state, has_key, has_box, ball_pos)] = self.slip_prob / 3

        elif action == "forward":
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][direction]
            new_x, new_y = x + dx, y + dy
            # Collides against the *current* ball position from `state`, not
            # self.ball_pos: the ball moves when pushed, so its starting cell
            # stops being where it is.
            if self.is_valid_position(new_x, new_y) and (new_x, new_y) != ball_pos:
                transitions[((new_x, new_y), direction, door_state, has_key, has_box, ball_pos)] = 1 - self.slip_prob
            else:
                transitions[((x, y), direction, door_state, has_key, has_box, ball_pos)] = 1 - self.slip_prob

            for slip_dir in range(4):
                if slip_dir != direction:
                    transitions[((x, y), slip_dir, door_state, has_key, has_box, ball_pos)] = self.slip_prob / 3

        elif action == "toggle":
            if (x, y) == self.door_pos and has_key:
                new_door_state = max(0, door_state - 1)
                transitions[((x, y), direction, new_door_state, has_key, has_box, ball_pos)] = 1
            else:
                transitions[((x, y), direction, door_state, has_key, has_box, ball_pos)] = 1

        elif action == "pickup":
            if (x, y) == self.key_pos and not has_key:
                transitions[((x, y), direction, door_state, True, has_box, ball_pos)] = 1
            elif (x, y) == self.box_pos and not has_box and door_state == 0:
                transitions[((x, y), direction, door_state, has_key, True, ball_pos)] = 1
            elif (x, y) == ball_pos:
                # NOTE: same fix as above — compare against the ball's
                # current position (`ball_pos`, from `state`), not its fixed
                # starting position, so the ball can be pushed more than once.
                dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][direction]
                new_ball_x, new_ball_y = ball_pos[0] + dx, ball_pos[1] + dy
                if self.is_valid_position(new_ball_x, new_ball_y) and (new_ball_x, new_ball_y) != (x, y):
                    transitions[((x, y), direction, door_state, has_key, has_box, (new_ball_x, new_ball_y))] = 1
                else:
                    transitions[((x, y), direction, door_state, has_key, has_box, ball_pos)] = 1
            else:
                transitions[((x, y), direction, door_state, has_key, has_box, ball_pos)] = 1

        return transitions

    def is_valid_position(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def get_reward(self, state, action, next_state):
        _, _, _, _, has_box, _ = next_state
        if has_box:
            return 1 - 0.9 * (self.step_count / self.max_steps)
        return 0

    def get_init_state(self):
        return ((0, 0), 0, 2, False, False, self.ball_pos)

    def get_state_hash(self, state):
        return str(state)

    def get_goal_states(self):
        return [((x, y), direction, door_state, has_key, True, ball_pos)
                for x in range(self.grid_size)
                for y in range(self.grid_size)
                for direction in range(4)
                for door_state in self.door_states
                for has_key in [False, True]
                for ball_pos in [(bx, by) for bx in range(self.grid_size) for by in range(self.grid_size)]]

    def visualize(self):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.agent_pos[1]][self.agent_pos[0]] = '^>v<'[self.agent_dir]
        grid[self.door_pos[1]][self.door_pos[0]] = 'D' if self.door_state > 0 else 'O'
        grid[self.key_pos[1]][self.key_pos[0]] = 'K' if not self.has_key else '.'
        grid[self.box_pos[1]][self.box_pos[0]] = 'B' if not self.has_box else '.'
        grid[self.ball_pos[1]][self.ball_pos[0]] = 'o'

        for row in grid:
            print(' '.join(row))
        print(f"Door state: {'Locked' if self.door_state == 2 else 'Closed' if self.door_state == 1 else 'Open'}")
        print(f"Has key: {'Yes' if self.has_key else 'No'}")
        print(f"Has box: {'Yes' if self.has_box else 'No'}")


def generate_and_visualize_minigridworld(size, task, max_steps=100, slip_prob=0.0, model_type="Model"):
    """Build an ``UnlockEnv`` or ``UnlockPickupEnv``, selected by ``task``."""
    if task == "unlock":
        return UnlockEnv(grid_size=size, max_steps=max_steps, slip_prob=slip_prob)
    elif task == "unlock_pickup":
        return UnlockPickupEnv(grid_size=size, max_steps=max_steps, slip_prob=slip_prob)
    raise ValueError(f"Unknown task {task!r}; expected 'unlock' or 'unlock_pickup'.")


# ─────────────────────────────────────────────────────────────────────────────
# High-level driver — robot + N humans, with compute timing
# ─────────────────────────────────────────────────────────────────────────────

def _make_determinized(size, task, max_steps, slip_prob, model_type, visualize=False):
    """Generate one minigrid world and determinize it; returns (next_states, s0, g, det_time).

    If ``visualize`` is True, print the generated map before determinizing.
    """
    mdp = generate_and_visualize_minigridworld(
        size, task, max_steps=max_steps, slip_prob=slip_prob, model_type=model_type)
    if visualize:
        print(f"\n{model_type}:")
        mdp.visualize()
    t0 = time.time()
    next_states, start_idx, goal_idx = augment_mdp_to_deterministic(mdp)
    return next_states, start_idx, goal_idx, time.time() - t0


def generate_determinized_models(size=5, num_humans=3, task="unlock", max_steps=100,
                                 slip_prob=0.0, seed=None, verbose=True, visualize=False):
    """Build a robot model + ``num_humans`` human MiniGrid models and determinize each.

    Parameters
    ----------
    size : int              grid side length
    num_humans : int        number of human models
    task : str               "unlock" (UnlockEnv) or "unlock_pickup" (UnlockPickupEnv).
                             See the module docstring for the state-space-size warning
                             on "unlock_pickup" — keep ``size`` small (2-3) for it.
    max_steps : int          episode length cap used by the reward decay term
    slip_prob : float        probability of a slipped rotation/move
    seed : int or None       seeds ``random``/``numpy`` for API consistency with the
                             other worlds; has no effect here since nothing about
                             UnlockEnv/UnlockPickupEnv is randomized
    verbose : bool           print a short timing summary
    visualize : bool         print each generated map (robot + humans)

    Returns
    -------
    dict: 'robot', 'humans', 'determinizing_times', 'total_determinizing_time'
    (each model is a tuple ``(next_states, start_idx, goal_idx, det_time)``)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    robot = _make_determinized(size, task, max_steps, slip_prob, "Robot Model", visualize)
    humans = [_make_determinized(size, task, max_steps, slip_prob, f"Human Model {i + 1}", visualize)
              for i in range(num_humans)]

    det_times = [robot[3]] + [h[3] for h in humans]
    total = float(sum(det_times))

    if verbose:
        print(f"[minigridworld] task={task} size={size} humans={len(humans)}")
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
    # UnlockPickupEnv's state space grows as grid_size**2 (the movable ball's
    # position), which makes augment_mdp_to_deterministic (O(states**2 * actions))
    # impractical at grid_size=5 — see the module docstring. This demo sticks to
    # the fast "unlock" task; pass task="unlock_pickup" with size=2 or 3 to try
    # the harder one.
    out = generate_determinized_models(size=5, num_humans=3, task="unlock", seed=0, visualize=True)
    T_R, s0, g, _ = out["robot"]
    print("\nRobot determinized transition array shape:", T_R.shape)
    print("start index:", s0, " goal index:", g)
    print("num human models:", len(out["humans"]))
    print("total determinizing time (s):", out["total_determinizing_time"])
