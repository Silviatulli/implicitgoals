from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Value
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, FrozenSet
import time
import os
import signal
import sys
import random
import itertools
import inspect
import threading
import queue
import logging
import gc
import multiprocessing
import platform

from experiments import (
    generate_and_visualize_gridworld,
    generate_and_visualize_puddleworld,
    generate_and_visualize_rockworld,
    generate_and_visualize_taxiworld
)
from maximal_achievable_subsets import (
    find_maximally_achievable_subsets,
    find_maximally_achievable_subsets_no_pruning,
    improved_find_maximally_achievable_subsets
)
from QueryMDP import QueryMDP, simulate_policy_unachievable, simulate_policy_query_all
from DeterminizedMDP import identify_bottlenecks
from overcooked_env import (
    build_transition_matrix_nomove as overcooked_build_transition_matrix,
    extract_bottlenecks_nomove as overcooked_extract_bottlenecks,
    remove_toboggan_redundancies as overcooked_remove_toboggan_redundancies,
    find_maximally_achievable_subsets as overcooked_find_maximally_achievable_subsets,
    decode_subsets_to_2d_nomove as overcooked_decode_subsets_to_2d_nomove,
    solve_query_mdp_exact as overcooked_solve_query_mdp_exact,
    get_human_bottleneck_mask,
    simulate_overcooked_vi,
    simulate_overcooked_info_gain,
    NUM_POT as OVERCOOKED_NUM_POT,
    SERVE_ACTION as OVERCOOKED_SERVE_ACTION,
    CLIENT_SERVED as OVERCOOKED_CLIENT_SERVED,
)

from overcooked_env import _dominator_bottlenecks
import networkx as nx

IS_MACOS = platform.system() == 'Darwin'

def get_safe_process_count():
    cpu_count = multiprocessing.cpu_count()
    if IS_MACOS:
        return max(1, cpu_count // 3)
    else:
        return max(1, cpu_count // 2)

# Note: on platforms using the 'spawn' start method (macOS, Windows), each worker process
# re-imports this module and gets its own independent copy of these, not a value truly shared
# with the main process or other workers. Thus, if a worker increments these counters, it
# won't be reflected in the main process.
pruning_counter = Value('i', 0)
no_pruning_counter = Value('i', 0)
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def init_worker():
    global inspect
    import inspect

# Minimal logging configuration
logging.basicConfig(level=logging.ERROR)

def _augment_mdp_to_deterministic(mdp) -> np.ndarray:
    """
    Convert stochastic MDP to deterministic by expanding the action space.

    For each state, every (original_action, outcome) pair with positive
    probability becomes a deterministic augmented action act_0, act_1, ...
    The augmented action space is shared across states: its size is the max
    number of outcomes needed by any single state (matching DeterminizedMDP's
    'act_i' convention), not the sum over all states.

    Returns:
      next_states[state_idx, act_i] = deterministic next state index,
      or state_idx itself (self-loop) if act_i isn't defined for that state.
    """
    states = mdp.get_state_space()
    original_actions = mdp.get_actions()
    n_states = len(states)

    state_hashes = [mdp.get_state_hash(s) for s in states]
    hash_to_idx  = {h: i for i, h in enumerate(state_hashes)}

    # Fast path for grid worlds: at most 5 cells (self + 4 neighbours) can
    # ever have P > 0 for any (state, action) pair, so skip the O(n²) scan.
    grid_size = getattr(mdp, 'size', None)

    def _local_candidates(state, state_idx):
        if grid_size is None:
            return list(range(n_states))          # non-grid fallback
        x, y = state[0]
        cands = [state_idx]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < grid_size and 0 <= ny_ < grid_size:
                h = mdp.get_state_hash([(nx_, ny_)] + list(state[1:]))
                idx = hash_to_idx.get(h)
                if idx is not None:
                    cands.append(idx)
        return cands

    per_state_outcomes = []
    for state_idx, state in enumerate(states):
        candidates = _local_candidates(state, state_idx)
        outcomes = []
        for orig_action in original_actions:
            for cand_idx in candidates:
                if mdp.get_transition_probability(state, orig_action, states[cand_idx]) > 0:
                    outcomes.append(cand_idx)
        per_state_outcomes.append(outcomes)

    n_augmented_actions = max((len(outcomes) for outcomes in per_state_outcomes), default=0)
    next_states = np.tile(np.arange(n_states, dtype=np.int32).reshape(-1, 1), (1, n_augmented_actions))
    for state_idx, outcomes in enumerate(per_state_outcomes):
        for act_idx, next_state_idx in enumerate(outcomes):
            next_states[state_idx, act_idx] = next_state_idx

    start_idx = hash_to_idx[mdp.get_state_hash(mdp.get_init_state())]
    goal_idx  = hash_to_idx[mdp.get_state_hash(mdp.get_goal_states()[0])]

    return next_states, start_idx, goal_idx

def extract_bottlenecks(T_R, T_H_list, start_state, goal_state, verbose=True):
    """Single-goal bottleneck extraction (legacy — used by Overcooked path)."""
    G = nx.DiGraph()
    for state in range(T_R.shape[0]):
        for action in range(T_R.shape[1]):
            nxt = int(T_R[state, action])
            if nxt != state:
                G.add_edge(state, nxt)
    bottlenecks = {goal_state} | _dominator_bottlenecks(G, start_state, [goal_state])
    for T_H in T_H_list:
        for state in range(T_H.shape[0]):
            for action in range(T_H.shape[1]):
                nxt = int(T_H[state, action])
                if nxt != state:
                    G.add_edge(state, nxt)
        bottlenecks |= _dominator_bottlenecks(G, start_state, [goal_state])
    result = sorted(bottlenecks)
    if verbose:
        print(f"Found {len(result)} bottleneck states via dominator tree.")
    return result


def extract_bottlenecks_multigol(T_R, T_H_list, human_goal_indices, start_state, verbose=True):
    """
    Multi-goal bottleneck extraction.

    For each human model with its own distinct goal, find the dominator states
    on the path from start to that goal.  The final bottleneck set is the union
    across all human goals — these are the states that discriminate between plans.

    Parameters
    ----------
    T_R               : ndarray (n_states, n_actions)  robot/environment dynamics
    T_H_list          : list[ndarray]  one transition matrix per human model
    human_goal_indices: list[int]      goal state index for each human model
    start_state       : int
    verbose           : bool

    Returns
    -------
    list[int]  sorted union of per-goal dominator states
    """
    # Base graph from robot dynamics
    G_base = nx.DiGraph()
    for state in range(T_R.shape[0]):
        for action in range(T_R.shape[1]):
            nxt = int(T_R[state, action])
            if nxt != state:
                G_base.add_edge(state, nxt)

    bottlenecks = set()
    for T_H, goal in zip(T_H_list, human_goal_indices):
        G = G_base.copy()
        for state in range(T_H.shape[0]):
            for action in range(T_H.shape[1]):
                nxt = int(T_H[state, action])
                if nxt != state:
                    G.add_edge(state, nxt)
        if not G.has_node(goal) or not nx.has_path(G, start_state, goal):
            continue
        bottlenecks.add(goal)
        bottlenecks |= _dominator_bottlenecks(G, start_state, [goal])

    result = sorted(bottlenecks)
    if verbose:
        print(f"Found {len(result)} bottleneck states (multi-goal union).")
    return result


def build_I_from_humans(M_H_list, human_goal_idxs, bottlenecks, start_state):
    """
    Build hypothesis space I from per-human reachability.

    For each human model, identify which subset of ``bottlenecks`` they must
    pass through on their way to their specific goal.  I is the list of
    distinct bottleneck subsets — one per unique human "type".

    This replaces ``find_maximally_achievable_subsets`` for multi-goal grids
    because T_R can reach all goals, so the MAS algorithm would trivially
    return one giant subset.

    Parameters
    ----------
    M_H_list         : list[ndarray]  per-human transition matrices
    human_goal_idxs  : list[int]      goal state index per human
    bottlenecks      : list[int]      ordered list of bottleneck state IDs
    start_state      : int

    Returns
    -------
    I_decoded : list[list[tuple]]   e.g. [[(2,), (3,)], [(12,), (15,)], …]
    """
    import networkx as nx_local

    bn_set = set(bottlenecks)
    seen   = set()
    I      = []

    for T_H, goal in zip(M_H_list, human_goal_idxs):
        G = nx_local.DiGraph()
        for state in range(T_H.shape[0]):
            for action in range(T_H.shape[1]):
                nxt = int(T_H[state, action])
                if nxt != state:
                    G.add_edge(state, nxt)

        if not G.has_node(goal) or not nx_local.has_path(G, start_state, goal):
            continue

        dominated = _dominator_bottlenecks(G, start_state, [goal])
        dominated.add(goal)
        subset = tuple(sorted(dominated & bn_set))

        if subset and subset not in seen:
            seen.add(subset)
            I.append([(s,) for s in subset])

    return I


def generate_possible_goals(grid_size):
    """
    Return K candidate goal positions spread across the grid.

    For small grids (size ≤ 8): 8 fixed positions (corners + edge midpoints + centre).
    For larger grids: a regular sub-grid of ~grid_size positions so there are
    enough distinct hypothesis types even with 100+ human models.
    Start (0,0) is always excluded.
    """
    # Always include the structural anchors
    mid = grid_size // 2
    anchors = [
        (0,            grid_size - 1),
        (grid_size-1,  0),
        (grid_size-1,  grid_size-1),
        (0,            mid),
        (mid,          0),
        (mid,          grid_size-1),
        (grid_size-1,  mid),
        (mid,          mid),
    ]

    seen = set()
    result = []
    for g in anchors:
        if g != (0, 0) and g not in seen:
            seen.add(g)
            result.append(g)
    return result


def generate_human_model(world_type, grid_size, obstacle_percent, puddle_percent, rock_percent, model_num, divide_rooms=False):
    M_H = None

    try:
        if world_type == 'grid' or world_type == 'four_rooms':
            mdp = generate_and_visualize_gridworld(
                size=grid_size,
                start=(0,0),
                goal=(grid_size-1,grid_size-1),
                obstacles_percent=obstacle_percent,
                divide_rooms=(world_type == 'four_rooms'),
                model_type=f"Human Model {model_num}",
                obstacle_seed=random.randint(1, 10000)
            )
            det_t0 = time.time()
            next_states, start_idx, goal_idx = _augment_mdp_to_deterministic(mdp)
            det_time = time.time() - det_t0
            M_H = (next_states, start_idx, goal_idx, det_time)
        elif world_type == 'puddle':
            mdp = generate_and_visualize_puddleworld(
                size=grid_size,
                start=(0,0),
                goal=(grid_size-1,grid_size-1),
                obstacles_percent=obstacle_percent,
                puddle_percent=puddle_percent,
                model_type=f"Human Model {model_num}",
                obstacle_seed=random.randint(1, 10000)
            )
            det_t0 = time.time()
            next_states, start_idx, goal_idx = _augment_mdp_to_deterministic(mdp)
            det_time = time.time() - det_t0
            M_H = (next_states, start_idx, goal_idx, det_time)
        elif world_type == 'rock':
            mdp = generate_and_visualize_rockworld(
                size=grid_size,
                start=(0,0),
                goal=(grid_size-1,grid_size-1),
                obstacles_percent=obstacle_percent,
                rock_percent=rock_percent,
                model_type=f"Human Model {model_num}",
                obstacle_seed=random.randint(1, 10000)
            )
            det_t0 = time.time()
            next_states, start_idx, goal_idx = _augment_mdp_to_deterministic(mdp)
            det_time = time.time() - det_t0
            M_H = (next_states, start_idx, goal_idx, det_time)
        elif world_type == 'taxi':
            mdp = generate_and_visualize_taxiworld(
                size=grid_size,
                start=(0,0),
                goal=(grid_size-1,grid_size-1),
                obstacles_percent=obstacle_percent,
                model_type=f"Human Model {model_num}",
                obstacle_seed=random.randint(1, 10000)
            )
            det_t0 = time.time()
            next_states, start_idx, goal_idx = _augment_mdp_to_deterministic(mdp)
            det_time = time.time() - det_t0
            M_H = (next_states, start_idx, goal_idx, det_time)
    except Exception:
        pass

    return M_H

def generate_human_model_with_goal(world_type, grid_size, obstacle_percent,
                                    puddle_percent, rock_percent, model_num, goal):
    """
    Like generate_human_model but with an explicit (row, col) goal position.
    Returns (next_states, start_idx, goal_idx, det_time) or (None, …) on failure.
    """
    M_H = None
    try:
        if world_type in ('grid', 'four_rooms'):
            mdp = generate_and_visualize_gridworld(
                size=grid_size, start=(0, 0), goal=goal,
                obstacles_percent=obstacle_percent,
                divide_rooms=(world_type == 'four_rooms'),
                model_type=f"Human Model {model_num}",
                obstacle_seed=random.randint(1, 10000),
            )
        elif world_type == 'puddle':
            mdp = generate_and_visualize_puddleworld(
                size=grid_size, start=(0, 0), goal=goal,
                obstacles_percent=obstacle_percent,
                puddle_percent=puddle_percent,
                model_type=f"Human Model {model_num}",
                obstacle_seed=random.randint(1, 10000),
            )
        elif world_type == 'rock':
            mdp = generate_and_visualize_rockworld(
                size=grid_size, start=(0, 0), goal=goal,
                obstacles_percent=obstacle_percent,
                rock_percent=rock_percent,
                model_type=f"Human Model {model_num}",
                obstacle_seed=random.randint(1, 10000),
            )
        else:
            return None, None, None, 0.0

        det_t0 = time.time()
        next_states, start_idx, goal_idx = _augment_mdp_to_deterministic(mdp)
        det_time = time.time() - det_t0
        M_H = (next_states, start_idx, goal_idx, det_time)
    except Exception:
        pass

    if M_H is None:
        return None, None, None, 0.0
    return M_H


def generate_robot_model(world_type, grid_size, obstacle_percent, puddle_percent, rock_percent, divide_rooms=False):
    M_R = None

    try:
        if world_type == 'grid' or world_type == 'four_rooms':
            mdp = generate_and_visualize_gridworld(
                size=grid_size,
                start=(0,0),
                goal=(grid_size-1,grid_size-1),
                obstacles_percent=obstacle_percent,
                divide_rooms=(world_type == 'four_rooms'),
                model_type="Robot Model",
                obstacle_seed=random.randint(1, 10000)
            )
            det_t0 = time.time()
            next_states, start_idx, goal_idx = _augment_mdp_to_deterministic(mdp)
            det_time = time.time() - det_t0
            M_R = (next_states, start_idx, goal_idx, det_time)
        elif world_type == 'puddle':
            mdp = generate_and_visualize_puddleworld(
                size=grid_size,
                start=(0,0),
                goal=(grid_size-1,grid_size-1),
                obstacles_percent=obstacle_percent,
                puddle_percent=puddle_percent,
                model_type="Robot Model",
                obstacle_seed=random.randint(1, 10000)
            )
            det_t0 = time.time()
            next_states, start_idx, goal_idx = _augment_mdp_to_deterministic(mdp)
            det_time = time.time() - det_t0
            M_R = (next_states, start_idx, goal_idx, det_time)
        elif world_type == 'rock':
            mdp = generate_and_visualize_rockworld(
                size=grid_size,
                start=(0,0),
                goal=(grid_size-1,grid_size-1),
                obstacles_percent=obstacle_percent,
                rock_percent=rock_percent,
                model_type="Robot Model",
                obstacle_seed=random.randint(1, 10000)
            )
            det_t0 = time.time()
            next_states, start_idx, goal_idx = _augment_mdp_to_deterministic(mdp)
            det_time = time.time() - det_t0
            M_R = (next_states, start_idx, goal_idx, det_time)
        elif world_type == 'taxi':
            mdp = generate_and_visualize_taxiworld(
                size=grid_size,
                start=(0,0),
                goal=(grid_size-1,grid_size-1),
                obstacles_percent=obstacle_percent,
                model_type="Robot Model",
                obstacle_seed=random.randint(1, 10000)
            )
            det_t0 = time.time()
            next_states, start_idx, goal_idx = _augment_mdp_to_deterministic(mdp)
            det_time = time.time() - det_t0
            M_R = (next_states, start_idx, goal_idx, det_time)
    except Exception:
        pass

    return M_R

def build_overcooked_models():
    """Robot model (all recipes servable) + one human model per recipe (only that recipe servable)."""
    T_base, RECIPES, _ = overcooked_build_transition_matrix(verbose=False)

    T_R = T_base.copy()
    for r in RECIPES:
        T_R[r * OVERCOOKED_NUM_POT, OVERCOOKED_SERVE_ACTION] = OVERCOOKED_CLIENT_SERVED

    T_H_list = []
    for r in RECIPES:
        T_H = T_base.copy()
        T_H[r * OVERCOOKED_NUM_POT, OVERCOOKED_SERVE_ACTION] = OVERCOOKED_CLIENT_SERVED
        T_H_list.append(T_H)

    return T_R, T_H_list


def run_overcooked_experiment(T_R, T_H_list, determinizing_mdp_time) -> Dict[str, Any]:
    """
    Overcooked (no-move) pipeline.

    The environment is deterministic so timing is measured once; query counts
    are averaged across human models (each T_H follows a different recipe).
    Query All = len(B_filter) — ask about every decision-point bottleneck.
    """
    results = {
        "determinizing_mdp_times": [],
        "bottleneck_finding_times": [],
        "maximal_achievable_pruning_times": [],
        "maximal_achievable_no_pruning_times": [],
        "policy_computation_pruning_times": [],
        "policy_computation_no_pruning_times": [],
        "pruning": {"times": [], "checks": [], "subsets": []},
        "no_pruning": {"times": [], "checks": [], "subsets": []},
        "query_counts": [],
        "information_gain_counts": [],
        "query_all_counts": [],
        "human_bottlenecks": [],
        "initial_mdp_state_space_sizes": [],
        "initial_mdp_action_space_sizes": [],
    }

    num_trials = 3 if IS_MACOS else 5

    # Build pipeline once (deterministic)
    t0 = time.time()
    B        = overcooked_extract_bottlenecks([T_R], verbose=False)
    B_filter = overcooked_remove_toboggan_redundancies(T_R, B)
    t1 = time.time()

    I        = overcooked_find_maximally_achievable_subsets(B_filter, T_R)
    I_decoded = overcooked_decode_subsets_to_2d_nomove(I)
    t2 = time.time()

    qnet = overcooked_solve_query_mdp_exact(I_decoded)
    t3 = time.time()

    bottleneck_time = t1 - t0
    maximal_time    = t2 - t1
    policy_time     = t3 - t2

    # Precompute human bottleneck masks (one per recipe / T_H)
    human_masks = [
        get_human_bottleneck_mask(T_H, qnet.unique_B)
        for T_H in T_H_list
    ]

    # Simulate query counts across human models
    vi_counts  = [simulate_overcooked_vi(qnet, hm) for hm in human_masks]
    ig_counts  = [simulate_overcooked_info_gain(I_decoded, hm) for hm in human_masks]
    all_count  = len(B_filter)   # Query All: ask about every decision-point

    # Replicate timing arrays across trials (only noise varies between trials)
    for _ in range(num_trials):
        results["determinizing_mdp_times"].append(determinizing_mdp_time)
        results["initial_mdp_state_space_sizes"].append(T_R.shape[0])
        results["initial_mdp_action_space_sizes"].append(T_R.shape[1])
        results["bottleneck_finding_times"].append(bottleneck_time)
        results["maximal_achievable_pruning_times"].append(maximal_time)
        results["maximal_achievable_no_pruning_times"].append(maximal_time)
        results["pruning"]["times"].append(maximal_time)
        results["pruning"]["subsets"].append(len(I))
        results["no_pruning"]["times"].append(maximal_time)
        results["no_pruning"]["subsets"].append(len(I))
        results["policy_computation_pruning_times"].append(policy_time)
        results["policy_computation_no_pruning_times"].append(policy_time)
        results["human_bottlenecks"].append(len(B_filter))
        # Query counts: average over human models, one entry per trial
        results["query_counts"].append(float(np.mean(vi_counts)))
        results["information_gain_counts"].append(float(np.mean(ig_counts)))
        results["query_all_counts"].append(float(all_count))

    gc.collect()
    return results


def run_single_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    trial_seed = params.get('seed', 0)
    np.random.seed(trial_seed)
    random.seed(trial_seed)

    try:
        world_type = params['world_type']

        if world_type == 'overcooked':
            T_R_overcooked, T_H_list_overcooked = build_overcooked_models()
            return run_overcooked_experiment(T_R_overcooked, T_H_list_overcooked, 0)

        grid_size = params['grid_size']
        num_models = params['num_models']
        query_threshold = params['query_threshold']
        obstacle_percent = params['obstacle_percent']
        puddle_percent = params.get('puddle_percent', 0)
        rock_percent = params.get('rock_percent', 0)


        num_trials = 3 if IS_MACOS else 5

        for trial in range(num_trials):

            M_R, start_state, goal_state, determinizing_time = generate_robot_model(
                world_type=world_type,
                grid_size=grid_size,
                obstacle_percent=obstacle_percent,
                puddle_percent=puddle_percent,
                rock_percent=rock_percent
            )

            if M_R is None:
                continue

            # Assign each human a random goal from K candidates.
            # Build T_H only once per unique goal position to avoid redundant
            # MDP constructions when num_models >> num_candidate_goals.
            possible_goals = generate_possible_goals(grid_size)
            assigned_goals  = [random.choice(possible_goals) for _ in range(num_models)]
            unique_goals    = list(dict.fromkeys(assigned_goals))   # preserves order, dedup

            goal_to_T_H   = {}   # (row, col) → (next_states, goal_idx)
            for goal_pos in unique_goals:
                next_states, start_idx, goal_idx, human_det_time = \
                    generate_human_model_with_goal(
                        world_type=world_type,
                        grid_size=grid_size,
                        obstacle_percent=obstacle_percent,
                        puddle_percent=puddle_percent,
                        rock_percent=rock_percent,
                        model_num=0,
                        goal=goal_pos,
                    )
                if next_states is not None:
                    goal_to_T_H[goal_pos] = (next_states, goal_idx)
                    determinizing_time += human_det_time

            # Expand back to the full num_models list (entries may be shared)
            M_H_list        = []
            human_goal_idxs = []
            for goal_pos in assigned_goals:
                if goal_pos in goal_to_T_H:
                    ns, gi = goal_to_T_H[goal_pos]
                    M_H_list.append(ns)
                    human_goal_idxs.append(gi)


            results = {
                "determinizing_mdp_times": [],
                "bottleneck_finding_times": [],
                "maximal_achievable_pruning_times": [],
                "maximal_achievable_no_pruning_times": [],
                "policy_computation_pruning_times": [],
                "policy_computation_no_pruning_times": [],
                "pruning": {"times": [], "checks": [], "subsets": []},
                "no_pruning": {"times": [], "checks": [], "subsets": []},
                "query_counts": [],
                "information_gain_counts": [],
                "query_all_counts": [],
                "human_bottlenecks": [],
                "initial_mdp_state_space_sizes": [],
                "initial_mdp_action_space_sizes": [],
            }

            results["determinizing_mdp_times"].append(determinizing_time)
            results["initial_mdp_state_space_sizes"].append(M_R.shape[0])
            results["initial_mdp_action_space_sizes"].append(M_R.shape[1])

            t0 = time.time()
            # For large state spaces, corridor-dominators are extremely rare
            # (many paths exist in an open grid, so only the goal itself dominates).
            # Skip the expensive DiGraph / dominator construction and use each
            # human's goal state directly as its sole bottleneck.
            LARGE_GRID_THRESHOLD = 400   # n_states > this → fast-path

            if M_R.shape[0] > LARGE_GRID_THRESHOLD:
                unique_goals_sorted = sorted(set(human_goal_idxs))
                B_filter  = unique_goals_sorted
                I_decoded = [[(g,)] for g in unique_goals_sorted]
            else:
                B_raw    = extract_bottlenecks_multigol(
                    M_R, M_H_list, human_goal_idxs, start_state, verbose=False)
                B_filter = overcooked_remove_toboggan_redundancies(M_R, B_raw)
                I_decoded = build_I_from_humans(
                    M_H_list, human_goal_idxs, B_filter, start_state)
            t1 = time.time()

            qnet      = overcooked_solve_query_mdp_exact(I_decoded)
            t2 = time.time()
            t3 = t2   # policy time included in t1-t2

            bottleneck_time = t1 - t0
            maximal_time    = t2 - t1
            policy_time     = t3 - t2   # 0 in fast-path (merged into maximal_time)

            # Simulate per-human, each human has their own specific goal.
            # For large grids, skip the DiGraph build in get_human_bottleneck_mask
            # and directly compute the mask from the goal index.
            if M_R.shape[0] > LARGE_GRID_THRESHOLD:
                bn_to_bit = {b[0]: i for i, b in enumerate(qnet.unique_B)}
                human_masks = [
                    (1 << bn_to_bit[gi]) if gi in bn_to_bit else 0
                    for gi in human_goal_idxs
                ]
            else:
                human_masks = [
                    get_human_bottleneck_mask(T_H, qnet.unique_B,
                                              target=goal_i, start=start_state)
                    for T_H, goal_i in zip(M_H_list, human_goal_idxs)
                ]
            vi_counts  = [simulate_overcooked_vi(qnet, hm)            for hm in human_masks]
            ig_counts  = [simulate_overcooked_info_gain(I_decoded, hm) for hm in human_masks]

            results["bottleneck_finding_times"].append(bottleneck_time)
            results["maximal_achievable_pruning_times"].append(maximal_time)
            results["maximal_achievable_no_pruning_times"].append(maximal_time)
            results["pruning"]["times"].append(maximal_time)
            results["pruning"]["subsets"].append(len(I_decoded))
            results["no_pruning"]["times"].append(maximal_time)
            results["no_pruning"]["subsets"].append(len(I_decoded))
            results["policy_computation_pruning_times"].append(policy_time)
            results["policy_computation_no_pruning_times"].append(policy_time)
            results["human_bottlenecks"].append(len(B_filter))
            results["query_counts"].append(float(np.mean(vi_counts)) if vi_counts else 0.0)
            results["information_gain_counts"].append(float(np.mean(ig_counts)) if ig_counts else 0.0)
            results["query_all_counts"].append(float(len(B_filter)))

            gc.collect()

        return results

    except Exception as e:
        logging.error(f"Error in run_single_experiment ({world_type}): {str(e)}")
        return None

def get_available_world_types():
    """Get available world types based on platform and dependencies"""
    base_worlds = ['grid', 'puddle', 'rock', 'overcooked']

    if not IS_MACOS:
        base_worlds.append('taxi')

    return base_worlds

def run_parallel_experiments_with_pybullet(num_runs: int, grid_sizes: list,
                                         human_model_counts: list,
                                         query_threshold: int,
                                         obstacle_percentages: list,
                                         max_workers: int = None):
    world_types = get_available_world_types()

    all_environments_results = {}
    experiment_params = []

    max_workers = max_workers or get_safe_process_count()
    batch_size = max(max_workers, 6 if PYBULLET_AVAILABLE else 10)

    # Overcooked is grid/obstacle-independent — run it only once
    overcooked_config = "overcooked_nomove"
    for _ in range(num_runs):
        experiment_params.append((overcooked_config, {
            'world_type': 'overcooked',
            'grid_size': 0,
            'num_models': 0,
            'query_threshold': query_threshold,
            'obstacle_percent': 0.0,
            'seed': random.randint(1, 10000),
        }))

    grid_world_types = [w for w in world_types if w != 'overcooked']

    for grid_size in grid_sizes:
        for num_models in human_model_counts:
            for world_type in grid_world_types:

                if world_type == 'four_rooms':
                    world_config = f"{world_type}_{grid_size}_{num_models}_models_0.0"
                    for _ in range(num_runs):
                        params = {
                            'world_type': world_type,
                            'grid_size': grid_size,
                            'num_models': num_models,
                            'query_threshold': query_threshold,
                            'obstacle_percent': 0.0,
                            'puddle_percent': 0.0,
                            'rock_percent': 0.0,
                            'seed': random.randint(1, 10000)
                        }
                        experiment_params.append((world_config, params))

                elif world_type == 'overcooked':
                    # No grid, no obstacles — grid_size/obstacle_percent don't apply.
                    world_config = f"{world_type}_{grid_size}_{num_models}_models_nomove"
                    for _ in range(num_runs):
                        params = {
                            'world_type': world_type,
                            'grid_size': grid_size,
                            'num_models': num_models,
                            'query_threshold': query_threshold,
                            'obstacle_percent': 0.0,
                            'puddle_percent': 0.0,
                            'rock_percent': 0.0,
                            'seed': random.randint(1, 10000)
                        }
                        experiment_params.append((world_config, params))

                elif world_type == 'pybullet':
                    world_config = f"{world_type}_{grid_size}_{num_models}_models_physics"
                    pybullet_runs = max(1, num_runs // 2) if IS_MACOS else num_runs
                    for _ in range(pybullet_runs):
                        params = {
                            'world_type': world_type,
                            'grid_size': min(grid_size, 4),
                            'num_models': min(num_models, 8),
                            'query_threshold': query_threshold,
                            'obstacle_percent': 0.1,
                            'puddle_percent': 0.0,
                            'rock_percent': 0.0,
                            'seed': random.randint(1, 10000)
                        }
                        experiment_params.append((world_config, params))

                else:
                    for obstacle_percent in obstacle_percentages:
                        world_config = f"{world_type}_{grid_size}_{num_models}_models_{obstacle_percent}"
                        for _ in range(num_runs):
                            params = {
                                'world_type': world_type,
                                'grid_size': grid_size,
                                'num_models': num_models,
                                'query_threshold': query_threshold,
                                'obstacle_percent': obstacle_percent,
                                'puddle_percent': obstacle_percent,
                                'rock_percent': obstacle_percent,
                                'seed': random.randint(1, 10000)
                            }
                            experiment_params.append((world_config, params))

    max_workers = min(max_workers or get_safe_process_count(), 3)

    batch_size = 6 if PYBULLET_AVAILABLE else 10

    for batch in [experiment_params[i:i+batch_size] for i in range(0, len(experiment_params), batch_size)]:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_experiment, params) for _, params in batch]

            for j, future in enumerate(futures):
                try:
                    result = future.result(timeout=300)
                    if result:
                        world_config = batch[j][0]
                        if world_config not in all_environments_results:
                            all_environments_results[world_config] = {
                                "determinizing_mdp_times": [],
                                "bottleneck_finding_times": [],
                                "maximal_achievable_pruning_times": [],
                                "maximal_achievable_no_pruning_times": [],
                                "policy_computation_pruning_times": [],
                                "policy_computation_no_pruning_times": [],
                                "pruning": {"times": [], "checks": [], "subsets": []},
                                "no_pruning": {"times": [], "checks": [], "subsets": []},
                                "query_counts": [],
                                "information_gain_counts": [],
                                "query_all_counts": [],
                                "human_bottlenecks": [],
                                "initial_mdp_state_space_sizes": [],
                                "initial_mdp_action_space_sizes": [],
                            }

                        # Aggregate results
                        for key in result:
                            if isinstance(result[key], dict):
                                for subkey in result[key]:
                                    all_environments_results[world_config][key][subkey].extend(result[key][subkey])
                            else:
                                all_environments_results[world_config][key].extend(result[key])

                except Exception:
                    continue

        gc.collect()

    return all_environments_results

def create_enhanced_results_table(all_environments_results, output_file="experiment_results_2/pybullet_comparison.csv"):
    """Create enhanced results table including PyBullet results"""

    os.makedirs("experiment_results_2", exist_ok=True)

    combined_data = {
        'Environment': [],
        'Grid Size': [],
        'Number of Human Models': [],
        'Obstacle Percentage': [],
        'Environment Type': [],
        'Determinizing MDP Time (s)': [],
        'Finding Bottlenecks Time (s)': [],
        'Finding Maximal Achievable With Pruning (s)': [],
        'Finding Maximal Achievable Without Pruning (s)': [],
        'Computing Policy With Pruning (s)': [],
        'Computing Policy Without Pruning (s)': [],
        'Total Runtime With Pruning (s)': [],
        'Total Runtime Without Pruning (s)': [],
        'Runtime Improvement (%)': [],
        'Query Count (Strategic VI)': [],
        'Query Count (Info Gain)': [],
        'Query Count (Query All)': [],
        'Human Bottlenecks': [],
        'Initial State Space': [],
        'Initial Actions': []
    }

    for env_type, results in all_environments_results.items():
        if not results["pruning"]["times"]:
            continue

        env_parts = env_type.split('_')

        if "pybullet" in env_type:
            environment_name = "PyBullet"
            env_category = "3D Physics"
            grid_size = int(env_parts[1])
            num_models = int(env_parts[2])
            obstacle_percent = "Physics"
        elif "four_rooms" in env_type:
            environment_name = "Four Rooms"
            env_category = "2D Grid"
            grid_size = int(env_parts[2])
            num_models = int(env_parts[3])
            obstacle_percent = 0.0
        elif "overcooked" in env_type:
            environment_name = "Overcooked"
            env_category = "Recipe Game"
            grid_size = "n/a"
            num_models = int(env_parts[2]) if len(env_parts) > 2 and env_parts[2].isdigit() else "n/a"
            obstacle_percent = "N/A"
        else:
            environment_name = env_parts[0].capitalize()
            env_category = "2D Grid"
            grid_size = int(env_parts[1])
            num_models = int(env_parts[2])
            obstacle_percent = float(env_parts[-1])

        combined_data['Environment'].append(environment_name)
        combined_data['Environment Type'].append(env_category)
        combined_data['Grid Size'].append(grid_size)
        combined_data['Number of Human Models'].append(num_models)
        combined_data['Obstacle Percentage'].append(obstacle_percent)

        try:
            determinizing_times = np.array(results.get('determinizing_mdp_times', []))
            bottleneck_times = np.array(results.get('bottleneck_finding_times', []))
            pruning_times = np.array(results["pruning"]["times"])
            policy_pruning_times = np.array(results.get('policy_computation_pruning_times', []))
            no_pruning_times = np.array(results["no_pruning"]["times"]) if results["no_pruning"]["times"] else []
            policy_no_pruning_times = np.array(results.get('policy_computation_no_pruning_times', []))

            min_len = min(len(arr) for arr in [bottleneck_times, pruning_times, policy_pruning_times] if len(arr) > 0)

            if min_len == 0:
                continue

            determinizing_times = determinizing_times[:min_len] if len(determinizing_times) > 0 else np.zeros(min_len)
            bottleneck_times = bottleneck_times[:min_len]
            pruning_times = pruning_times[:min_len]
            policy_pruning_times = policy_pruning_times[:min_len]

            combined_data['Determinizing MDP Time (s)'].append(
                f"{np.mean(determinizing_times):.3f} ± {np.std(determinizing_times):.3f}")
            combined_data['Finding Bottlenecks Time (s)'].append(
                f"{np.mean(bottleneck_times):.3f} ± {np.std(bottleneck_times):.3f}")
            combined_data['Finding Maximal Achievable With Pruning (s)'].append(
                f"{np.mean(pruning_times):.3f} ± {np.std(pruning_times):.3f}")

            if len(no_pruning_times) > 0:
                no_pruning_times = no_pruning_times[:min_len]
                combined_data['Finding Maximal Achievable Without Pruning (s)'].append(
                    f"{np.mean(no_pruning_times):.3f} ± {np.std(no_pruning_times):.3f}")
            else:
                combined_data['Finding Maximal Achievable Without Pruning (s)'].append("N/A")

            if len(policy_pruning_times) > 0 and np.any(policy_pruning_times > 0):
                combined_data['Computing Policy With Pruning (s)'].append(
                    f"{np.mean(policy_pruning_times):.3f} ± {np.std(policy_pruning_times):.3f}")
            else:
                combined_data['Computing Policy With Pruning (s)'].append("N/A")

            if len(policy_no_pruning_times) > 0 and np.any(policy_no_pruning_times > 0):
                policy_no_pruning_times = policy_no_pruning_times[:min_len]
                combined_data['Computing Policy Without Pruning (s)'].append(
                    f"{np.mean(policy_no_pruning_times):.3f} ± {np.std(policy_no_pruning_times):.3f}")
            else:
                combined_data['Computing Policy Without Pruning (s)'].append("N/A")

            total_pruning = determinizing_times + bottleneck_times + pruning_times + policy_pruning_times
            combined_data['Total Runtime With Pruning (s)'].append(
                f"{np.mean(total_pruning):.3f} ± {np.std(total_pruning):.3f}")

            if len(no_pruning_times) > 0 and len(policy_no_pruning_times) > 0:
                total_no_pruning = determinizing_times + bottleneck_times + no_pruning_times + policy_no_pruning_times
                combined_data['Total Runtime Without Pruning (s)'].append(
                    f"{np.mean(total_no_pruning):.3f} ± {np.std(total_no_pruning):.3f}")

                improvement = ((np.mean(total_no_pruning) - np.mean(total_pruning)) /
                             np.mean(total_no_pruning) * 100)
                improvement_std = np.std([(n - p)/n * 100 for n, p in zip(total_no_pruning, total_pruning)])
                combined_data['Runtime Improvement (%)'].append(
                    f"{improvement:.1f} ± {improvement_std:.1f}")
            else:
                combined_data['Total Runtime Without Pruning (s)'].append("N/A")
                combined_data['Runtime Improvement (%)'].append("N/A")

            for col, key in [
                ('Query Count (Strategic VI)', 'query_counts'),
                ('Query Count (Info Gain)',    'information_gain_counts'),
                ('Query Count (Query All)',    'query_all_counts'),
            ]:
                vals = results.get(key, [])
                if vals:
                    arr = np.array(vals[:min_len], dtype=float)
                    combined_data[col].append(f"{np.mean(arr):.2f} ± {np.std(arr):.2f}")
                else:
                    combined_data[col].append("N/A")

            bottlenecks = np.array(results['human_bottlenecks'][:min_len])
            combined_data['Human Bottlenecks'].append(
                f"{np.mean(bottlenecks):.1f} ± {np.std(bottlenecks):.1f}")
            combined_data['Initial State Space'].append(
                f"{np.mean(results['initial_mdp_state_space_sizes'][:min_len]):.0f}")
            combined_data['Initial Actions'].append(
                f"{np.mean(results['initial_mdp_action_space_sizes'][:min_len]):.0f}")

        except Exception as e:
            logging.error(f"Error processing results for {env_type}: {str(e)}")
            continue

    df = pd.DataFrame(combined_data)
    df.to_csv(output_file, index=False)

    pybullet_results = df[df['Environment Type'] == '3D Physics']
    if not pybullet_results.empty:
        pybullet_file = output_file.replace('.csv', '_pybullet_only.csv')
        pybullet_results.to_csv(pybullet_file, index=False)

    backup_file = output_file.replace('.csv', '_detailed_backup.csv')
    detailed_data = []
    for env_type, results in all_environments_results.items():
        for i in range(len(results.get('pruning', {}).get('times', []))):
            row = {'Environment': env_type}
            for key, values in results.items():
                if isinstance(values, dict):
                    for subkey, subvalues in values.items():
                        if i < len(subvalues):
                            row[f"{key}_{subkey}"] = subvalues[i]
                else:
                    if i < len(values):
                        row[key] = values[i]
            detailed_data.append(row)

    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(backup_file, index=False)

    return df

def main():
    num_runs = 5
    grid_sizes = [4, 10, 20, 50, 100]
    human_model_counts = [10, 30, 60, 100]
    obstacle_percentages = [0.1, 0.15, 0.2]
    max_workers = 4
    query_threshold = 1000

    try:
        # Create results directory
        os.makedirs("experiment_results_2", exist_ok=True)

        # Run experiments
        start_time = time.time()
        results = run_parallel_experiments_with_pybullet(
            num_runs=num_runs,
            grid_sizes=grid_sizes,
            human_model_counts=human_model_counts,
            query_threshold=query_threshold,
            obstacle_percentages=obstacle_percentages,
            max_workers=max_workers
        )
        total_time = time.time() - start_time

        if results:
            combined_df = create_enhanced_results_table(
                results,
                "experiment_results_2/enhanced_pybullet_comparison.csv"
            )

            print(f"\nResults saved to experiment_results_2/")
            print(f"Environments tested: {len(results)}")
            print(f"Total runtime: {total_time/60:.1f} minutes")

            if not combined_df.empty:
                print(f"\nSummary:")
                for env_type in combined_df['Environment Type'].unique():
                    env_data = combined_df[combined_df['Environment Type'] == env_type]
                    print(f"  {env_type}: {len(env_data)} configurations")

            print("Experiment completed successfully!")

        else:
            print("No results were generated.")

    except Exception as e:
        print(f"Error during experiment execution: {str(e)}")
        if 'results' in locals() and results:
            try:
                create_enhanced_results_table(results, "experiment_results_2/partial_results.csv")
                print("Partial results saved to experiment_results_2/partial_results.csv")
            except:
                pass

if __name__ == "__main__":
    PYBULLET_AVAILABLE = False
    main()
