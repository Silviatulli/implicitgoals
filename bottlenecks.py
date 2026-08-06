"""
bottlenecks.py
==============
Everything the pipeline computes *from* transition matrices: bottleneck
extraction, Algorithm 1, and the Query MDP solvers.

Nothing in this module knows what a recipe, an inventory or a kitchen is.  A
problem instance is fully described by four things:

    T_R          the robot's transition matrix — T[state, action] → next_state
    T_H_list     one transition matrix per candidate human
    start_state  where every trajectory begins
    goal_state   the absorbing state every successful trajectory ends in

The game module builds those four things (for Overcooked, that is
Overcooked/overcooked_env.py, one configuration at a time) and the game's viz
module decodes state IDs back into something readable.  This module is shared
across games and depends on neither.

Pipeline
--------
  1.  extract_bottlenecks_union([T_R], start, goal)      → B          mandatory waypoints
  2.  remove_toboggan_redundancies(T_R, B, goal|None)    → B_filter   true decision nodes
  3.  find_maximally_achievable_subsets(B_filter, T_R, start, goal)
                                                         → I          [Algorithm 1]
  3b. toboggan_downstream_map + expand_subsets           → I over the raw B  [optional]
  4.  decode_subsets(I, decode)                          → I_decoded  [presentation only]
  5.  subsets_to_array(I, columns)                       → I_array    bool (len_I_array, n)
  6.  solve_query_mdp_exact(I)                           → ExactQNet  (3^n flat arrays)
  7.  simulate_exact_trajectories_real(mdp, I_array, …)  → step counts under the exact policy

build_bottleneck_problem(T_R, T_H_list, start_state, goal_state) runs steps 1–5
in one call and returns them bundled in a BottleneckProblem.

Notation
--------
B            bottleneck state IDs — mandatory waypoints on every path to a goal state.
B_filter     B minus the "toboggan" states that offer no choice (see §2).
I            list of maximally achievable bottleneck subsets (Algorithm 1 output);
             each element is a list of bottleneck state IDs.  One of them is the
             subset the human actually pursues.
I_array      bool matrix, shape (len(I), n): row k is subset I[k], columns ordered
             like the bottleneck list it was built against.
K_I          bitmask of bottlenecks confirmed to belong to the human's subset (oracle YES).
K_not_I      bitmask of bottlenecks confirmed not to belong to it (oracle NO).
I_hat        = K_I | (unqueried bits) — current upper-bound on the human's subset.
mdp          ExactQNet returned by solve_query_mdp_exact: V, policy, absorbing masks.
"""

import time
import numpy as np
import networkx as nx
from collections import deque
from dataclasses import dataclass, field
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Reachability primitives
# ─────────────────────────────────────────────────────────────────────────────

def get_reachable_states(T, start_state):
    """Vectorized BFS using a boolean frontier mask instead of a growing Python set."""
    num_states = T.shape[0]
    reachable = np.zeros(num_states, dtype=bool)
    reachable[start_state] = True
    frontier = np.array([start_state])

    while frontier.size > 0:
        neighbours = np.unique(T[frontier].ravel())
        neighbours = neighbours[~reachable[neighbours]]
        reachable[neighbours] = True
        frontier = neighbours

    return reachable


def goal_predecessors(T, goal_state):
    """
    States with at least one action leading into `goal_state` — the default
    terminal set when the caller does not supply an explicit one.

    `goal_state` itself is excluded: it is absorbing, so its self-loop is not a
    way of *reaching* the goal.
    """
    hits = np.any(np.asarray(T) == goal_state, axis=1)
    hits[goal_state] = False
    return np.nonzero(hits)[0].tolist()


def _build_adjacency(possible_bottlenecks, T, start_state=0):
    """
    Collapse per-state reachability into a compact bottleneck-to-bottleneck
    adjacency matrix — the only structure Algorithm 1 needs during its search.

    Returns
    -------
    from_start      : bool array, shape (n,)    from_start[j]    = start can reach j
    from_bottleneck : bool array, shape (n, n)  from_bottleneck[i, j] = i can reach j
    """
    masks = np.stack(
        [get_reachable_states(T, start_state)] +
        [get_reachable_states(T, b) for b in possible_bottlenecks]
    )
    idx = np.array(possible_bottlenecks)
    adj = masks[:, idx]           # shape (n+1, n)
    return adj[0], adj[1:]        # from_start (n,), from_bottleneck (n, n)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Bottleneck extraction  →  B
# ─────────────────────────────────────────────────────────────────────────────

def _dominator_bottlenecks(G, start_state, goal_state):
    """
    Walk the immediate-dominator tree backward from each goal state, collecting
    every mandatory waypoint between start_state and that goal.
    """
    idoms = nx.immediate_dominators(G, start_state)
    result = set()
    result.add(goal_state)
    current = goal_state
    while current != start_state:
        current = idoms.get(current, start_state)
        if current != start_state:
                result.add(current)
    return result


def _transition_graph(T):
    """Directed graph of T, self-loops dropped (they carry no reachability)."""
    T = np.asarray(T)
    src, act = np.nonzero(T != np.arange(T.shape[0], dtype=T.dtype)[:, None])
    G = nx.DiGraph()
    G.add_edges_from(zip(src.tolist(), T[src, act].tolist()))
    return G


def extract_bottlenecks(T, start_state, goal_state, goals=None, verbose=False):
    """Extract the mandatory bottleneck states of a single transition matrix.

    Builds a directed graph from T, runs the dominator tree from start_state, and
    collects every state that lies on every path to any goal state.

    Parameters
    ----------
    T           : ndarray, shape (n_states, n_actions)
    start_state : int  root of the dominator tree
    goal_state  : int  universal absorbing state, always included in the result
    goals       : list[int] or None
        States to trace back from.  None means "every predecessor of goal_state",
        which is the right answer whenever reaching the goal is what identifies a
        trajectory.  Pass an explicit list when the interesting terminal states sit
        earlier than the goal (e.g. move_goals in Overcooked/overcooked_env.py).
    verbose     : bool

    Returns
    -------
    list[int]  sorted bottleneck state IDs, always includes goal_state
    """
    if goals is None:
        goals = goal_predecessors(T, goal_state)
    bottlenecks = {goal_state} | _dominator_bottlenecks(
        _transition_graph(T), start_state, goal_state
    )
    result = sorted(bottlenecks)
    if verbose:
        print(f"Found {len(result)} bottleneck states via dominator tree.")
    return result


def extract_bottlenecks_union(T_list, start_state, goal_state, verbose=True):
    """
    Union of the bottlenecks of every matrix in T_list.

    Called with the single robot matrix [T_R] this is just extract_bottlenecks;
    called with a list of candidate humans it returns the universe of bottlenecks
    any of them could care about.  Matrices whose goal set is empty (no path to
    the goal at all) contribute nothing.
    """
    all_bottlenecks = set()
    for T in T_list:
        tg =  goal_predecessors(T, goal_state) 
        if len(tg):
            all_bottlenecks |= set(
                extract_bottlenecks(T, start_state, goal_state, tg, verbose=False)
            )
    result = sorted(all_bottlenecks)
    if verbose:
        n = len(T_list)
        print(f"Found {len(result)} bottleneck states "
              f"(union over {n} {'matrix' if n == 1 else 'matrices'}).")
    return result


def compute_bottlenecks_per_matrix(T_list, start_state, goal_state,
                                   goals_list=None, verbose=False) -> list:
    """
    One frozenset of raw bottleneck IDs per matrix — the ensemble the Oracle
    samples from.

    goals_list : None, or one goal-state list per matrix (same order as T_list)
                 for the case where each matrix has its own terminal states.
    """
    return [
        frozenset(extract_bottlenecks(
            T, start_state, goal_state, verbose=verbose,
        ))
        for i, T in enumerate(T_list)
    ]


def shuffle_bottlenecks(B, seed=None):
    """Return a randomly permuted copy of the bottleneck list B."""
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.permutation(len(B))
    return [B[i] for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Toboggan filtering  →  B_filter   [optional preprocessing before Algorithm 1]
# ─────────────────────────────────────────────────────────────────────────────

def remove_toboggan_redundancies(T_matrix, B_list, goal_state):
    """
    Remove linear, non-branching "toboggan" sequences from the bottleneck set.

    From each bottleneck, BFS to its immediate downstream bottleneck neighbours.
    A node with exactly one downstream bottleneck offers no real choice and is
    discarded.  Terminal nodes (0 successors) and true decision points (≥2
    successors) are kept.

    This step compresses 2^|B| to 2^|B_filter| before Algorithm 1, which is
    the key that makes the search tractable on larger bottleneck sets.

    Parameters
    ----------
    start_state : int
        The starting state for the BFS traversal.
    goal_state : int or None
        When given, the goal is held out of the analysis and appended back at the
        end.  This protects the last state of each branch — whose only downstream
        bottleneck is the shared goal, so it would otherwise be discarded as a
        toboggan even though it is exactly what tells the branches apart.
        None analyses every bottleneck, goal included; the goal itself always
        survives (it has no downstream at all), but those last per-branch states
        do not.  There is no default: which one you want is a modelling choice.

    Returns
    -------
    cleaned : list[int]  sorted bottlenecks after toboggan removal
    """

    if goal_state not in B_list:
        print("Warning: goal_state is not in B_list!")

    regular  = [b for b in B_list if b != goal_state]

    B_set = set(regular)
    cleaned = []
    num_actions = T_matrix.shape[1]

    for b in regular: # (code tres malin jadore)
        immediate_next = set()
        queue = deque([b])
        visited = {b}
        while queue:
            state = queue.popleft()
            for action in range(num_actions):
                nxt = int(T_matrix[state, action])
                if nxt not in visited:
                    visited.add(nxt)
                    if nxt in B_set:
                        immediate_next.add(nxt)
                    else:
                        queue.append(nxt)
        if len(immediate_next) != 1:   # 0 = terminal, ≥2 = branching point; 1 = toboggan
            cleaned.append(b)


    cleaned.append(goal_state)
    return sorted(cleaned)


def toboggan_downstream_map(possible_bottlenecks, cleaned, T_matrix):
    """For each toboggan, BFS to find its unique downstream cleaned bottleneck."""
    cleaned_set = set(cleaned)
    num_actions = T_matrix.shape[1]
    mapping = {}
    for b in possible_bottlenecks:
        if b in cleaned_set:
            continue
        immediate_next = set()
        queue = deque([b])
        visited = {b}
        while queue:
            state = queue.popleft()
            for action in range(num_actions):
                nxt = int(T_matrix[state, action])
                if nxt not in visited:
                    visited.add(nxt)
                    if nxt in cleaned_set:
                        immediate_next.add(nxt)
                    else:
                        queue.append(nxt)
        if len(immediate_next) == 1:
            mapping[b] = next(iter(immediate_next))
    return mapping


def expand_subsets(I_compact, toboggan_map, all_bottlenecks, T_matrix, start_state=0):
    """
    Re-include toboggan nodes into each maximal subset found on the filtered set.

    A toboggan t is a *candidate* for I_j whenever its unique downstream
    cleaned bottleneck toboggan_map[t] is a member of I_j. That condition alone is
    not sufficient: toboggan_downstream_map() is many-to-one whenever several
    distinct toboggans funnel into the same cleaned bottleneck — most notably the
    goal state, which every trajectory ends in and which therefore collects one
    toboggan per branch. Blindly adding every candidate would then glue
    mutually-exclusive raw states onto every subset, since *every* I_j contains
    that shared terminal.

    Each candidate is therefore checked for *chronological compatibility* against
    every other member already in I_j: t may be added only if, for every other
    member m of I_j, at least one of them can reach the other (t reaches m, or m
    reaches t) in the raw graph. Two raw states that can reach neither each other
    are necessarily on mutually-exclusive branches (committing to one branch makes
    the other's states unreachable for the rest of that trajectory) and cannot both
    have occurred on the one trajectory that achieves I_j.

    This is a pairwise, not a full joint-achievability, check: it is cheap (O(1)
    lookups against a precomputed reachability matrix, no recursion), and it
    exactly resolves the failure mode above, since I_j's own pre-existing members
    already carry whatever branch commitment is needed to rule out every candidate
    but the correct one. It is not a full re-proof that the *entire* expanded set
    remains sequentially achievable (that would require re-running
    CheckAchievability, which is expensive at the scale of the raw bottleneck set
    and unnecessary here).

    Parameters
    ----------
    I_compact       : list[iterable[int]]  maximal subsets over the filtered set
    toboggan_map    : dict {toboggan: downstream cleaned bottleneck}, from toboggan_downstream_map()
    all_bottlenecks : list[int]  the raw (unfiltered) bottleneck set B
    T_matrix        : ndarray    the same transition matrix used to build toboggan_map
    start_state     : int

    Returns
    -------
    expanded : list[list[int]]  sorted subsets over the raw bottleneck set
    """
    all_bottlenecks = list(all_bottlenecks)
    b_to_idx = {b: i for i, b in enumerate(all_bottlenecks)}
    _, from_bottleneck = _build_adjacency(all_bottlenecks, T_matrix, start_state)

    # Group candidate toboggans by shared downstream once, up front.
    by_downstream = {}
    for t, c in toboggan_map.items():
        by_downstream.setdefault(c, []).append(t)

    expanded = []
    for subset in I_compact:
        subset_set = set(subset)
        member_idx = [b_to_idx[b] for b in subset_set]

        extras = []
        for c in subset_set:
            for t in by_downstream.get(c, ()):
                t_idx = b_to_idx[t]
                compatible = all(
                    m_idx == t_idx
                    or from_bottleneck[t_idx, m_idx]
                    or from_bottleneck[m_idx, t_idx]
                    for m_idx in member_idx
                )
                if compatible:
                    extras.append(t)

        expanded.append(sorted(subset_set | set(extras)))
    return expanded


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Algorithm 1 — Include / Exclude DFS  (subset search)  →  I
# ─────────────────────────────────────────────────────────────────────────────

def _np_popcount_u64(arr):
    """Population count for a numpy uint64 array via parallel bit manipulation."""
    x = arr.copy()
    x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
    x  = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x  = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def filter_maximal_subsets(masks):
    """
    Maximality filter: discard any mask m that is a strict subset of another mask M
    (i.e. (m & M) == m and m != M).

    Supports n <= 63 (int64), n == 64 (uint64), n > 64 (pure-Python fallback).
    For large inputs (e.g. 18 M masks at n=64) uses a batched numpy approach so
    the per-mask Python loop only runs on the small set of surviving candidates.
    """
    if not masks:
        return []

    max_val = max(masks)
    if max_val <= np.iinfo(np.int64).max:
        dtype = np.int64
    elif max_val < 2 ** 64:
        dtype = np.uint64
    else:
        print("val_max =", max_val, "too big, filter_maximal_subsets() falling back to pure-Python loop")
        # n > 64: pure-Python fallback (arbitrary-precision ints)
        masks_sorted = sorted(masks, key=lambda m: bin(m).count("1"), reverse=True)
        kept = []
        for m in masks_sorted:
            if not any((m & k) == m for k in kept):
                kept.append(m)
        return kept

    arr = np.array(list(masks), dtype=dtype)

    # Sort by popcount descending so we process supersets before subsets
    arr_u64 = arr.view(np.uint64)
    pc = _np_popcount_u64(arr_u64).astype(np.intp)
    arr = arr[np.argsort(-pc, kind='stable')]

    # Batched filter: vectorized subsumption check against kept, then sequential
    # within the few surviving candidates per batch.
    BATCH = 100_000
    kept_list = []

    for b_start in range(0, len(arr), BATCH):
        batch = arr[b_start:b_start + BATCH]

        if kept_list:
            kept_np = np.array(kept_list, dtype=dtype)
            B, K = len(batch), len(kept_np)
            # subsumed[i] = any j: (batch[i] & kept[j]) == batch[i]
            subsumed = np.any(
                (batch.reshape(B, 1) & kept_np.reshape(1, K)) == batch.reshape(B, 1),
                axis=1,
            )
            candidates = batch[~subsumed]
        else:
            candidates = batch

        # Sequential pass on the (few) surviving candidates to handle intra-batch subsumption
        for m in candidates:
            m_int = int(m)
            if not any((m_int & k) == m_int for k in kept_list):
                kept_list.append(m_int)

    return kept_list


def check_sequential_achievability(mask, from_start, from_bottleneck, n, memo):
    """
    Paper's CheckAchievability test — does some visiting order for the subset
    encoded by `mask` exist?

    Works recursively: a subset {b1…bk} is achievable iff removing any one
    element bi leaves an achievable subset AND bi is reachable from whatever
    was visited last in that sub-order.

    Memoized on `mask` so each subset is solved at most once, regardless of
    how many DFS branches query it (the paper's "caching" step).
    memo[mask] stores a bitmask of valid "last-visited" bottlenecks,
    or -1 as a sentinel for the empty set (achievable vacuously).

    Uses Python int arithmetic for bitmasks so n > 63 bottlenecks are safe.

    memo is a dict {int: int}.
    "memo[set1] = set2" means "each elt of set2 is a valid last-visited node
    for the achievable subset set1"  (by construction, each element of set2
    is in set1. set 2 tell where a trajectory covering all set1 can end)

    il est malin ce claude !
    """

    if mask in memo:
        return memo[mask] != 0

    if mask == 0:
        memo[0] = -1    # empty set: achievable, no "last visited" node
        return True

    ends = 0
    for b in (i for i in range(n) if (mask >> i) & 1):
        prev_mask = mask & ~(1 << b)
        if not check_sequential_achievability(prev_mask, from_start, from_bottleneck, n, memo):
            continue
        prev_ends = memo[prev_mask]
        if prev_ends == -1:         # this implies that mask = {b}, and thus prev_mask = {}
            if from_start[b]:       # it thus suffices to check adjacency.
                ends |= 1 << b
        else:
            prev_ends_bool = np.array([(prev_ends >> i) & 1 for i in range(n)], dtype=bool)
            if np.any(prev_ends_bool & from_bottleneck[:, b]):
                ends |= 1 << b

    memo[mask] = ends
    return ends != 0


def find_maximally_achievable_subsets(possible_bottlenecks, T_R, start_state, goal_state, verbose=True):
    """
    Algorithm 1 — find all maximally achievable subsets of bottlenecks.

    Parameters
    ----------
    possible_bottlenecks : list[int]   bottleneck state IDs (B or B_filter)
    T_R                    : ndarray     clean transition matrix
    start_state          : int
    goal_state           : int or None
        When given and present in `possible_bottlenecks`, asserts the
        universal-bottleneck invariant (see below).

    Returns
    -------
    I : list[list[int]]   maximally achievable subsets (each is a list of state IDs)
    """
    n = len(possible_bottlenecks)

    if verbose:
        print(f"Building bottleneck adjacency for {n} bottlenecks...")
    from_start, from_bottleneck = _build_adjacency(possible_bottlenecks, T_R, start_state)
    memo             = {}
    achievable_masks = []
    _counter         = [0]

    def generate_subsets(index, current_mask):
        _counter[0] += 1
        if verbose and _counter[0] % 20_000 == 0:
            print(f"  {_counter[0]:7d} calls, depth {index}/{n}", end='\r', flush=True)
        if index == n:
            achievable_masks.append(current_mask)
            return
        generate_subsets(index + 1, current_mask)
        new_mask = current_mask | (1 << index)
        if check_sequential_achievability(new_mask, from_start, from_bottleneck, n, memo):
            generate_subsets(index + 1, new_mask)

    if verbose:
        print(f"Running Algorithm 1 (include/exclude DFS, {n} bottlenecks, "
              f"2^{n} = {1 << n} max subsets)...")
    generate_subsets(0, 0)
    if verbose:
        print(f"{len(achievable_masks)} achievable subsets at leaves "
              f"({len(memo)} subsets solved by CheckAchievability).")

    maximal_masks = filter_maximal_subsets(achievable_masks)
    I = [[possible_bottlenecks[i] for i in range(n) if (m >> i) & 1] for m in maximal_masks]

    # Universal-bottleneck invariant: if the goal state is in the bottleneck set it
    # must appear in every maximal achievable subset (any trajectory that can win
    # must pass through it).
    if goal_state in set(possible_bottlenecks):
        assert all(goal_state in subset for subset in I), (
            "Universal-bottleneck invariant violated: the goal state is absent from "
            "at least one maximal achievable subset.  Every trajectory through a "
            "matrix that can reach the goal must end there."
        )

    return I


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Subset encodings  →  I_decoded, I_array
# ─────────────────────────────────────────────────────────────────────────────

def bottleneck_index(columns):
    """{state ID: column index} for the bottleneck list indexing I_array's columns."""
    return {int(b): j for j, b in enumerate(columns)}


def subsets_to_array(I, columns):
    """
    Pack the subsets I into the bool matrix I_array.

    Parameters
    ----------
    I       : list[iterable[int]]  maximally achievable subsets, raw state IDs
    columns : list[int]            bottleneck list defining the column order
                                   (B_filter, or B when I was expanded back onto it)

    Returns
    -------
    I_array : bool ndarray, shape (len(I), len(columns))
              I_array[k, j] is True iff columns[j] belongs to subset I[k].
    """
    col_of = bottleneck_index(columns)
    I_array = np.zeros((len(I), len(columns)), dtype=bool)
    for k, subset in enumerate(I):
        for b in subset:
            I_array[k, col_of[int(b)]] = True
    return I_array


def decode_subsets(I, decode=None):
    """
    Map every state ID in every subset through `decode` — presentation only.

    `decode` is any callable int → whatever is readable in this configuration
    (e.g. decode_subsets_to_2d_nomove in Overcooked/overcooked_viz.py).  None returns
    plain copies of the raw subsets.
    """
    if decode is None:
        return [list(subset) for subset in I]
    return [[decode(s) for s in subset] for subset in I]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Trajectory ordering
# ─────────────────────────────────────────────────────────────────────────────

def order_trajectory(state_ids, T_matrix, start_state=0, max_full_order_size=10):
    """
    Reconstruct one valid chronological visiting order for `state_ids`.

    Tries exhaustive backtracking DFS when |state_ids| ≤ max_full_order_size.
    Falls back to BFS-depth ranking when no total order exists (e.g. when
    state_ids spans mutually-exclusive branches).

    Returns
    -------
    order    : list[int]  state IDs in visiting order
    is_total : bool       True if a true sequential order was found
    """
    state_ids  = list(state_ids)
    reach_cache = {}

    def reachable_from(s):
        if s not in reach_cache:
            reach_cache[s] = get_reachable_states(T_matrix, s)
        return reach_cache[s]

    def backtrack(current, remaining, path):
        if not remaining:
            return path
        for nxt in remaining:
            if reachable_from(current)[nxt]:
                result = backtrack(nxt, remaining - {nxt}, path + [nxt])
                if result is not None:
                    return result
        return None

    if len(state_ids) <= max_full_order_size:
        order = backtrack(start_state, frozenset(state_ids), [])
        if order is not None:
            return order, True

    # Fallback: rank by BFS depth from start_state
    depth = {start_state: 0}
    queue = deque([start_state])
    while queue:
        cur = queue.popleft()
        for nxt in T_matrix[cur]:
            nxt = int(nxt)
            if nxt not in depth:
                depth[nxt] = depth[cur] + 1
                queue.append(nxt)

    return sorted(state_ids, key=lambda s: depth.get(s, float("inf"))), False


# ─────────────────────────────────────────────────────────────────────────────
# 6.  One-shot driver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BottleneckProblem:
    """Everything build_bottleneck_problem() computes, in one place.

    B           : list[int]        raw bottlenecks of T_R
    B_filter    : list[int]        B after toboggan removal
    I           : list[list[int]]  maximally achievable subsets
    I_decoded   : list             I mapped through `decode` (== I when decode is None)
    I_array     : bool ndarray     (len(I), len(columns))
    columns     : list[int]        bottleneck list indexing I_array's columns
    b_to_int    : dict             {state ID: column index}
    oracle_sets : list[frozenset]  one bottleneck set per candidate human
    """
    B:           list
    B_filter:    list
    I:           list
    I_decoded:   list
    I_array:     np.ndarray
    columns:     list
    b_to_int:    dict
    oracle_sets: list = field(default_factory=list)


def build_bottleneck_problem(T_R, T_H_list, start_state, goal_state, *,
                             mode="filter", goals=None, h_goals=None,
                             protect_goal=True, decode=None,
                             verbose=True) -> BottleneckProblem:
    """
    Run the whole bottleneck pipeline on one problem instance.

    Parameters
    ----------
    T_R         : ndarray        robot transition matrix — B and I come from this one
    T_H_list    : list[ndarray]  candidate humans — used only for oracle_sets
    start_state : int
    goal_state  : int
    mode        : {'filter', 'raw', 'expand'}
        'filter'  Algorithm 1 on B_filter, I_array columns = B_filter.
        'raw'     Algorithm 1 on the full B, I_array columns = B (2^|B| — only
                  tractable on small bottleneck sets).
        'expand'  Algorithm 1 on B_filter, then expand_subsets() puts the
                  toboggans back; I_array columns = B.
    goals        : list[int] or None   explicit terminal states for T_R (see extract_bottlenecks)
    h_goals      : list[list[int]] or None  per-human terminal states, same order as T_H_list
    protect_goal : bool  hold the goal out of the toboggan filter
                   (see remove_toboggan_redundancies)
    decode       : callable int → readable, for I_decoded

    Returns
    -------
    BottleneckProblem
    """
    if mode not in ("filter", "raw", "expand"):
        raise ValueError(f"mode must be 'filter', 'raw' or 'expand', got {mode!r}")

    B        = extract_bottlenecks_union([T_R], start_state, goal_state, goals, verbose)
    B_filter = remove_toboggan_redundancies(T_R, B, goal_state if protect_goal else None)
    if verbose:
        print(f"Compression: {len(B)} -> {len(B_filter)} true decision nodes.")

    search  = B if mode == "raw" else B_filter
    columns = B_filter if mode == "filter" else B

    I = find_maximally_achievable_subsets(search, T_R, start_state, goal_state, verbose)
    if mode == "expand":
        I = expand_subsets(I, toboggan_downstream_map(B, B_filter, T_R),
                           B, T_R, start_state)

    return BottleneckProblem(
        B=B, B_filter=B_filter, I=I,
        I_decoded=decode_subsets(I, decode),
        I_array=subsets_to_array(I, columns),
        columns=list(columns),
        b_to_int=bottleneck_index(columns),
        oracle_sets=compute_bottlenecks_per_matrix(
            T_H_list, start_state, goal_state, h_goals
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Oracle  (non-uniform prior over bottleneck membership)
# ─────────────────────────────────────────────────────────────────────────────

class Oracle:
    """
    Query oracle backed by an ensemble of bottleneck sets.

    At each call to query(), the oracle independently draws one set from the
    ensemble and returns whether the queried state belongs to it.  Equivalently,
    P(YES | b) = |{S in ensemble : b in S}| / |ensemble|.

    All probabilities are precomputed at __init__ time into a flat float32 array
    indexed by raw state ID, so individual queries run in O(1) after construction.

    Special cases
    -------------
    Empty ensemble  → uniform 50/50 for every bottleneck.
    Single set      → deterministic (0 or 1) based on set membership.
    """

    def __init__(self, bottleneck_sets: list, n_states: int, seed: int = None):
        """
        Parameters
        ----------
        bottleneck_sets : list of sets / frozensets of int (raw state IDs),
                          e.g. BottleneckProblem.oracle_sets
        n_states        : state-space size; sets the length of the internal array
                          (use T.shape[0], or goal_state + 1)
        seed            : RNG seed for reproducible queries
        """
        self.rng = np.random.default_rng(seed)
        self._sets = list(bottleneck_sets)
        n = len(self._sets)

        if n == 0:
            self._p = np.full(n_states, 0.5, dtype=np.float32)
        else:
            counts = np.zeros(n_states, dtype=np.int32)
            for s in self._sets:
                for b in s:
                    counts[b] += 1
            self._p = counts.astype(np.float32) / n

        self._n_states = n_states

    def query(self, bottleneck: int) -> bool:
        """Sample YES/NO for a single bottleneck (raw state ID)."""
        return bool(self.rng.random() < self._p[bottleneck])

    def p_yes(self, bottleneck: int) -> float:
        """Return P(YES) for a single bottleneck (raw state ID)."""
        return float(self._p[bottleneck])

    # ── batch helpers (used by the MDP solver and training env) ─────────────

    def probs_for_raw_ids(self, raw_ids) -> np.ndarray:
        """
        Return a float32 array of P(YES) values.
        raw_ids : array-like of int, raw state IDs in any order.
        """
        return self._p[np.asarray(raw_ids, dtype=np.intp)]

    def bottleneck_matrix(self, raw_ids) -> np.ndarray:
        """
        Return a bool array of shape (n_sets, len(raw_ids)).
        Entry [k, j] is True iff raw_ids[j] belongs to bottleneck set k.
        Used to feed the per-episode deterministic oracle into QueryMDPVecEnv.
        """
        raw_ids = list(raw_ids)
        return np.array([[b in s for b in raw_ids] for s in self._sets], dtype=bool)


def min_queries_lower_bound(I_array, T_H_list, start_state, goal_state, columns,
                            goals_list=None, verbose=True):
    r"""
    Queries an omniscient agent would need, one number per candidate human.

    Hand the agent the identity of the human it faces. Human i's transition matrix implies
    a bottleneck set S_i, extracted with the same dominator-tree pass the rest of the
    pipeline uses. Absorption needs I_hat = B \ K_not to sit inside some I_k, and I_hat
    shrinks only through NO answers, so the agent asks exactly the bottlenecks the human
    does NOT own and hears NO to every one of them:

        cost(i) = |columns \ S_i|

    S_i needs no pruning to `columns` beforehand -- the set difference already ignores
    anything outside it.

    The count is exact whenever no I_k STRICTLY contains S_i, since the smallest
    admissible I_hat is then S_i itself. Where some I_k does strictly contain it,
    the agent could stop |I_k| - |S_i| queries earlier and this over-estimates;
    `I_array` is used for exactly that check and it is reported when verbose.

    It is a floor, not a goal. A real agent does not know i, and every query it spends
    working that out lands on a bottleneck the human DOES own, comes back YES, leaves
    K_not unchanged and so buys no progress towards absorption at all.

    Parameters
    ----------
    I_array      : (len_I_array, n) bool -- row k is a maximally achievable subset, columns
                   ordered like `columns`. Used only for the tightness check.
    T_H_list     : candidate human transition matrices, one per human model.
    start_state  : root of the dominator tree, as passed to extract_bottlenecks.
    goal_state   : the universal absorbing state.
    columns      : ordered bottleneck universe indexing the columns of I_array.
    goals_list   : None, or one explicit goal-state list per human (same order as T_H_list).
    verbose      : print the per-human table.

    Returns
    -------
    (len(T_H_list),) int64 array -- cost(i) for each human, in the order of T_H_list.
    """
    columns = list(columns)
    n = len(columns)
    I_array = np.asarray(I_array, dtype=bool)
    if I_array.ndim != 2 or I_array.shape[1] != n:
        raise ValueError(f"I_array must be (len_I_array, {n}) to match the column list, "
                         f"got {I_array.shape}")

    n_h = len(T_H_list)
    owned = np.zeros((n_h, n), dtype=bool)
    cost = np.empty(n_h, dtype=np.int64)
    for i, T_i in enumerate(T_H_list):
        tg  = goal_predecessors(T_i, goal_state) if goals_list is None else goals_list[i]
        raw = set(extract_bottlenecks(T_i, start_state, goal_state, tg, verbose=False)) if len(tg) else set()
        owned[i] = [b in raw for b in columns]
        cost[i] = n - int(owned[i].sum())                      # |columns \ S_i|

    # Tightened floor: absorption only needs I_hat inside SOME I_k, so when an I_k
    # strictly contains S_i the agent stops |I_k| - |S_i| queries early. Equal to `cost`
    # wherever no I_k strictly contains S_i.
    sizes = I_array.sum(axis=1)
    tight = cost.copy()
    slack = np.zeros(n_h, dtype=np.int64)
    for i in range(n_h):
        inside = ~(owned[i][None, :] & ~I_array).any(axis=1)        # S_i subset of I_k
        if inside.any():
            slack[i] = int(sizes[inside].max()) - int(owned[i].sum())
            tight[i] = n - int(sizes[inside].max())

    if verbose:
        print(f"{'human':>5} {'|S_i|':>6} {'queries':>8}  {'tight?':>9}")
        for i in range(n_h):
            tag = "yes" if slack[i] == 0 else f"over by {slack[i]}"
            print(f"{i:5d} {int(owned[i].sum()):6d} {int(cost[i]):8d}  {tag:>9}")
        n_loose = int((slack > 0).sum())
        print(f"\nn = {n} bottlenecks, {n_h} humans")
        print(f"mean |B \\ S_i|                    : {cost.mean():.2f}")
        if n_loose:
            print(f"mean floor (largest containing I_k): {tight.mean():.2f}   <- the real bound")
            print(f"note: {n_loose} human(s) sit strictly inside a larger I_k, so |B \\ S_i| "
                  f"over-estimates their cost by the amount shown")
    return cost


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Query MDP solvers  (Definition 6)
# ─────────────────────────────────────────────────────────────────────────────

class ExactQNet(nn.Module):
    """Exact Query MDP policy with the same callable interface as QNet.

    Not a neural network — backed by precomputed backward-induction arrays.
    Built by solve_query_mdp_exact(); do not instantiate directly.
    forward(x) takes a (batch, 2n) observation tensor and returns (batch, n) Q-values:
    0.0 for every tied-optimal action, −1e9 for all others.
    """

    def __init__(self, n, best_action_mask, POW3, V, failure, success,
                 unique_B, B_to_idx, p_I):
        super().__init__()
        self.n                = n
        self.best_action_mask = best_action_mask
        self.POW3             = POW3
        self.V                = V
        self.failure          = failure
        self.success          = success
        self.unique_B         = unique_B
        self.B_to_idx         = B_to_idx
        self.p_I              = p_I

    def forward(self, x):
        """(batch, 2n) float tensor → (batch, n) Q-value tensor."""
        n    = self.n
        pow3 = np.asarray(self.POW3, dtype=np.int64)
        best = np.asarray(self.best_action_mask, dtype=np.int64)
        obs    = (x.detach().cpu().numpy() > 0.5).astype(np.int64)  # (batch, 2n)
        KI     = obs[:, :n]
        KN     = obs[:, n:]
        states = (KI + 2 * KN) @ pow3
        masks  = best[states]
        bits   = np.arange(n, dtype=np.int64)
        q_np   = np.where(
            (masks[:, None] >> bits[None, :]) & 1, 0.0, -1e9
        ).astype(np.float32)
        return torch.from_numpy(q_np).to(x.device)


def _as_label(b):
    """Hashable, sortable key for one bottleneck: a raw state ID, or a decoded tuple."""
    if isinstance(b, (list, tuple, np.ndarray)):
        return tuple(int(x) for x in b)
    return int(b)


def solve_query_mdp_exact(I, C_Q=-10.0, p_I=1.0, gamma=0.99, p_F=0.0,
                          oracle: "Oracle | None" = None, alphabet=None):
    """
    Solve the Query MDP via vectorized backward induction.

    Encoding: each unique bottleneck gets a bit index.  A knowledge state
    (K_I, K_not_I) is encoded as a base-3 integer: digit i ∈
    {0=unqueried, 1=oracle_yes, 2=oracle_no}.  Total state space: 3^n.

    Absorbing states:
      failure — K_I is not a subset of any I_k mask (impossible to succeed)
      success — I_hat = K_I ∪ {all unqueried bits} covers some I_k entirely

    Backward induction sweeps from q = n−1 down to q = 0 queried bits,
    since a state with q bits queried only depends on states with q+1 bits.
    Ties in expected value are stored as bitmasks so the full tied-action set
    is available for analysis.

    Parameters
    ----------
    I : list of subsets of bottlenecks — raw state IDs (decoded tuples also work,
        but then `oracle` cannot be used since it is indexed by raw ID).
    C_Q, p_I, gamma : MDP cost/reward/discount parameters
    p_F : float (default 0.0)
        Terminal value of failure states.
    oracle : Oracle or None
        Per-bottleneck P(YES).  None means uniform 50/50.
    alphabet : iterable of bottlenecks, or None
        What the robot may query — the action set, and the bit order of the
        returned policy.  None (default) infers it from the labels occurring in
        I, which is the right choice when I already spans everything queryable.

        Pass it explicitly when the queryable set is *wider* than I, e.g. the
        full B_filter: a bottleneck belonging to no I_k is still worth asking
        about, since YES proves the human matches no hypothesis (failure) and NO
        is required before any hypothesis can be certified (success).  Inferring
        the alphabet would drop exactly those bottlenecks, and would also let the
        bit order drift from a caller-built I_array.  Must contain every label
        occurring in I.

    Returns
    -------
    ExactQNet with attributes: V, best_action_mask, failure, success,
                               unique_B, B_to_idx, n, POW3, p_I
    """
    if alphabet is None:
        unique_B = sorted({_as_label(b) for subset in I for b in subset})
    else:
        unique_B = sorted({_as_label(b) for b in alphabet})
        missing  = {_as_label(b) for subset in I for b in subset} - set(unique_B)
        if missing:
            raise ValueError(
                f"alphabet is missing {len(missing)} bottleneck(s) present in I: "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    B_to_idx = {b: i for i, b in enumerate(unique_B)}
    n        = len(unique_B)
    FULL_MASK = (1 << n) - 1

    if oracle is not None:
        probs = oracle.probs_for_raw_ids(unique_B)           # shape (n,), float32
    else:
        probs = np.full(n, 0.5, dtype=np.float32)

    I_masks = np.zeros(len(I), dtype=np.int32)
    for i, subset in enumerate(I):
        mask = 0
        for b in subset:
            mask |= 1 << B_to_idx[_as_label(b)]
        I_masks[i] = mask

    N3 = 3 ** n
    print(f"State space: 3^{n} = {N3:,} states "
          f"(~{N3 * 13 / 1e9:.2f} GB of working arrays)")
    t0 = time.time()

    POW3 = (3 ** np.arange(n)).astype(np.int64)

    K_I     = np.zeros(N3, dtype=np.int32)
    K_not_I = np.zeros(N3, dtype=np.int32)
    q_count = np.zeros(N3, dtype=np.int8)

    rem = np.arange(N3, dtype=np.int64)
    for i in range(n):
        digit    = rem % 3
        rem    //= 3
        K_I     |= (digit == 1).astype(np.int32) << i
        K_not_I |= (digit == 2).astype(np.int32) << i
        q_count += (digit != 0).astype(np.int8)
    del rem
    print(f"States decoded in {time.time() - t0:.1f}s")

    failure = np.ones(N3, dtype=bool)
    for t in I_masks:
        failure &= (K_I & t) != K_I   # True only if K_I ⊄ every I_k

    I_hat   = K_I | (FULL_MASK & ~K_not_I) # "K_I | " is useless by contruction, but makes the intent clear
    success = np.zeros(N3, dtype=bool)
    for t in I_masks:
        success |= (I_hat & ~t) == 0
    success &= ~failure
    failure |= (q_count == n) & ~success   # fully queried with no match → failure

    V                = np.zeros(N3, dtype=np.float32)
    V[success]       = p_I
    V[failure]       = p_F
    best_action_mask = np.zeros(N3, dtype=np.int32)
    non_absorbing    = ~failure & ~success
    print(f"Absorbing states in {time.time() - t0:.1f}s "
          f"({failure.sum():,} failures, {success.sum():,} successes)")

    for q in range(n - 1, -1, -1):
        level = np.nonzero((q_count == q) & non_absorbing)[0]
        if level.size == 0:
            continue

        used      = K_I[level] | K_not_I[level]
        best_val  = np.full(level.size, -np.inf, dtype=np.float32)
        best_mask = np.zeros(level.size, dtype=np.int32)

        for i in range(n):
            candidate = ((used >> i) & 1) == 0
            if not np.any(candidate):
                continue
            positions = np.nonzero(candidate)[0]
            idx       = level[positions]
            p_b       = probs[i]
            expected  = C_Q + gamma * (p_b * V[idx + POW3[i]] + (1.0 - p_b) * V[idx + 2 * POW3[i]])
            better = expected > best_val[positions]
            equal  = expected == best_val[positions]
            best_val [positions[better]] = expected[better]
            best_mask[positions[better]] = 1 << i
            best_mask[positions[equal ]] |= 1 << i

        V[level]                = best_val
        best_action_mask[level] = best_mask

    print(f"Backward induction done in {time.time() - t0:.1f}s total")

    return ExactQNet(
        n=n, best_action_mask=best_action_mask, POW3=POW3,
        V=V, failure=failure, success=success,
        unique_B=unique_B, B_to_idx=B_to_idx, p_I=p_I,
    )


def encode_base3(K_I, K_not_I, POW3, n):
    """Pack (K_I, K_not_I) bitmasks into the base-3 state index used by solve_query_mdp_exact."""
    state = 0
    for i in range(n):
        if (K_I >> i) & 1:
            state += int(POW3[i])
        elif (K_not_I >> i) & 1:
            state += 2 * int(POW3[i])
    return state


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MDP simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_exact_trajectories_real(mdp, I_array, num_tries: int) -> np.ndarray:
    """
    Evaluate the exact MDP policy under a deterministic oracle (real-human scenario).

    For each episode, a row I_k is sampled uniformly from I_array and the
    oracle answers YES iff the queried bottleneck belongs to I_k.  The exact policy
    follows the same optimal action sequence as in the theoretical game, but failure
    is now impossible (K_I ⊆ I_k by construction) — every episode ends in success.

    This is the correct counterpart to compare_with_query_all: same deterministic-oracle
    protocol, different (exact) policy.  A random 50/50 oracle instead causes fast
    failures and yields a misleadingly low average (~4 queries) that is not comparable
    with real-world evaluation.
    """
    
    if np.asarray(I_array).dtype != bool:
        print(f"Warning: I_array dtype is {np.asarray(I_array).dtype}, expected bool")

    best_action_mask = mdp.best_action_mask
    POW3             = mdp.POW3
    n                = mdp.n
    len_I_array       = I_array.shape[0]

    steps = np.zeros(num_tries, dtype=int)

    for run in range(num_tries):
        t_idx        = np.random.randint(len_I_array)
        true_bits    = set(np.where(I_array[t_idx])[0])
        s            = 0
        k_not        = set()
        outside_bits = set(range(n)) - true_bits
        for _ in range(n):
            mask         = int(best_action_mask[s])
            isolated_bit = mask & (-mask)
            action       = int(np.log2(isolated_bit))
            if action in true_bits:
                s += int(POW3[action])
            else:
                s += 2 * int(POW3[action])
                k_not.add(action)
            steps[run] += 1
            if outside_bits <= k_not:   # all outside bits ruled out → I_hat ⊆ T_k
                break

    return steps
