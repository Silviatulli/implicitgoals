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
  1.  compute_bottlenecks_per_matrix(T_H_list, start, goal) → one bottleneck set per
                                                         human; union them → B
  2.  remove_toboggan_redundancies(T_R, B, goal|None)    → B_filter   true decision nodes
  3.  find_maximally_achievable_subsets(B_filter, T_R, start, goal)
                                                         → I          [Algorithm 1]
  4.  subsets_to_array(I, columns)                       → I_array    bool (len_I_array, n)
  5.  solve_query_mdp_exact(I, alphabet=columns, oracle) → ExactQNet  (3^n flat arrays)

The exact policy is evaluated against a deterministic real-human oracle by
query_mdp_nn.evaluate_policy_on_real_human, which also provides the random-order
"query all" baseline.

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
# 4.  Subset encodings  →  I_array
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
                          e.g. compute_bottlenecks_per_matrix(...) output
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


def solve_query_mdp_exact(I, B, C_Q=-10.0, p_I=1.0, gamma=0.99, p_F=0.0,
                          oracle: "Oracle | None" = None):
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
    B : iterable of bottlenecks, or None
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
    if B is None:
        unique_B = sorted({_as_label(b) for subset in I for b in subset})
    else:
        unique_B = sorted({_as_label(b) for b in B  if b in B})
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


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Hypothesis conditions — H1, H2, H3, H4
#
# Each condition answers one question: *which bottleneck to query next*.  None
# of them owns the knowledge state and none of them decides when to stop —
# QueryMDPVecEnv does that, with the same I_hat ⊆ I_k test solve_query_mdp_exact
# uses (§7).  So every condition is comparable to the VI baseline by
# construction, and a condition is just a scoring rule over the alphabet.
#
# The exception is H2, which is not a selection rule at all: it is an
# answer-preserving reduction that grants extra NOs for free after each oracle
# NO.  It therefore lives on the knowledge state (the env), not in a score, and
# composes with any of the others.
# ─────────────────────────────────────────────────────────────────────────────

def _alphabet(I, B):
    """Bit/column order shared by every solver here — see solve_query_mdp_exact."""
    if B is None:
        unique_B = sorted({_as_label(b) for subset in I for b in subset})
    else:
        unique_B = sorted({_as_label(b) for b in B})
        missing  = {_as_label(b) for subset in I for b in subset} - set(unique_B)
        if missing:
            raise ValueError(
                f"alphabet is missing {len(missing)} bottleneck(s) present in I: "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    return unique_B, {b: i for i, b in enumerate(unique_B)}


def _hypothesis_matrix(I, B_to_idx):
    """Φ as a bool matrix, shape (len(I), n): row k is I[k] in alphabet order."""
    T = np.zeros((len(I), len(B_to_idx)), dtype=bool)
    for k, subset in enumerate(I):
        for b in subset:
            T[k, B_to_idx[_as_label(b)]] = True
    return T


def _xlog2x(v):
    """v·log2(v) with the 0·log2(0) = 0 convention."""
    out = np.zeros_like(v, dtype=np.float64)
    nz  = v > 0
    out[nz] = v[nz] * np.log2(v[nz])
    return out


def value_iteration_goal_probability(T_R_sto, goal_state,
                                     gamma=0.99, tol=1e-10, max_iter=10_000):
    """V_R: expected discounted probability of reaching goal_state under M_R.

    Value iteration on the robot's *stochastic* model, with the goal absorbing
    and worth 1:

        V(goal) = 1
        V(s)    = γ · max_a Σ_s' P(s'|s,a) · V(s')

    On a deterministic model this collapses to V(s) = γ^d(s) for d = shortest
    path length to the goal, and to 0 for states that cannot reach it.

    Parameters
    ----------
    T_R_sto    : (n, n_actions, n) float — P(s'|s,a), rows summing to 1, as
                 returned by gridworld_core.build_stochastic_matrix or
                 overcooked_env.build_stochastic_matrix.  Those builders have
                 already pruned it to the robot-reachable states and returned
                 the index that maps state IDs into it; nothing is pruned here.
    goal_state : int — the absorbing goal's row in T_R_sto, i.e. index[goal].
    gamma      : discount.

    Returns
    -------
    (n,) float64 — V_R over the rows of T_R_sto.
    """
    T = np.asarray(T_R_sto, dtype=np.float64)
    if T.ndim != 3:
        raise ValueError(
            "T_R_sto must be (n_states, n_actions, n_states) probabilities; got "
            f"{T.shape}.  Build it with build_stochastic_matrix().")

    V = np.zeros(T.shape[0], dtype=np.float64)
    V[goal_state] = 1.0
    for _ in range(max_iter):
        V_next = gamma * np.max(T @ V, axis=1)
        V_next[goal_state] = 1.0
        converged = np.max(np.abs(V_next - V)) < tol
        V = V_next
        if converged:
            break
    return V


def _dominance_closure(T):
    """H2(ii): dom[b2, b1] is True iff b1 ⪯ b2, i.e. ∀ϕ ∈ Φ: b1 ∈ ϕ ⇒ b2 ∈ ϕ.

    An oracle NO on b2 then entails NO on every such b1 at no query cost.
    Transitively closed here so the env only needs one propagation step.
    """
    n = T.shape[1]
    # prec[b1, b2] — no hypothesis holds b1 without also holding b2.
    prec = ~((T[:, :, None] & ~T[:, None, :]).any(0))
    dom  = np.ascontiguousarray(prec.T)          # dom[b2, b1] = b1 ⪯ b2
    np.fill_diagonal(dom, False)
    for k in range(n):                            # Warshall — n is small
        dom |= dom[:, k][:, None] & dom[k][None, :]
    np.fill_diagonal(dom, False)
    return dom


class GreedyQNet(nn.Module):
    """Greedy query policy with the same callable interface as ExactQNet.

    Scores are recomputed from (K_I, K_not) on every forward(); there is no 3^n
    table, which is why these run on instances the exact solver cannot be built
    for.  forward(x) takes (batch, 2n) and returns (batch, n) — the caller masks
    already-queried actions and takes the argmax.

    rule
    ----
    "entropy"  H1 — argmin_s Σ_± (n_±/N)·log2 n_±  over consistent hypotheses.
    "marginal" H4 — argmax_s |{ϕ ∈ Φ(B,K_I) : s ∈ ϕ}|.
    "static"   H3 — argmax_s score[s], a ranking fixed before the episode.
                    Unlike H1 and H4 this ignores Φ entirely.
    """

    def __init__(self, n, T, unique_B, B_to_idx, rule, static_score=None):
        super().__init__()
        self.n, self.T = n, T
        self.unique_B, self.B_to_idx = unique_B, B_to_idx
        self.rule = rule
        self.static_score = static_score

    def _consistent(self, KI, KN):
        """(batch, len(I)) bool — hypotheses compatible with (K_I, K_not)."""
        return (~(KI[:, None, :] & ~self.T[None, :, :]).any(2)      # K_I ⊆ ϕ
                & ~(KN[:, None, :] &  self.T[None, :, :]).any(2))   # K_not ∩ ϕ = ∅

    def forward(self, x):
        obs    = x.detach().cpu().numpy() > 0.5
        n      = self.n
        KI, KN = obs[:, :n], obs[:, n:]

        if self.rule == "static":
            score = np.tile(self.static_score, (obs.shape[0], 1))
        else:
            C      = self._consistent(KI, KN).astype(np.float64)
            n_plus = C @ self.T                                   # (batch, n)
            if self.rule == "marginal":
                score = n_plus
            else:
                N      = C.sum(1, keepdims=True)
                n_min  = N - n_plus
                score  = -(_xlog2x(n_plus) + _xlog2x(n_min)) / np.maximum(N, 1.0)

        return torch.from_numpy(np.ascontiguousarray(score, dtype=np.float32)).to(x.device)


# The four conditions.  C_Q / p_I / gamma / p_F / oracle are accepted for
# signature parity with solve_query_mdp_exact and are unused by the greedy
# rules, which never evaluate the query MDP's rewards.

def solve_query_mdp_info_gain(I, B, C_Q=-10.0, p_I=1.0, gamma=0.99, p_F=0.0,
                              oracle: "Oracle | None" = None):
    """Hypothesis 1 — one-step weighted-entropy minimisation over Φ(B, K_I)."""
    unique_B, B_to_idx = _alphabet(I, B)
    return GreedyQNet(len(unique_B), _hypothesis_matrix(I, B_to_idx),
                      unique_B, B_to_idx, rule="entropy")


def solve_query_mdp_frequency(I, B, C_Q=-10.0, p_I=1.0, gamma=0.99, p_F=0.0,
                              oracle: "Oracle | None" = None):
    """Hypothesis 4 — the bottleneck in the most currently consistent hypotheses."""
    unique_B, B_to_idx = _alphabet(I, B)
    return GreedyQNet(len(unique_B), _hypothesis_matrix(I, B_to_idx),
                      unique_B, B_to_idx, rule="marginal")


def solve_query_mdp_proximity(I, B, C_Q=-10.0, p_I=1.0, gamma=0.99, p_F=0.0,
                              oracle: "Oracle | None" = None,
                              V_R=None, state_index=None):
    """Hypothesis 3 — query bottlenecks in decreasing V_R, the robot's expected
    discounted probability of reaching the goal.

    V_R and state_index are the pair returned by
    value_iteration_goal_probability(): V_R is indexed by pruned row, and
    state_index maps a raw state ID to that row.  Time both calls together —
    the value iteration is part of H3's cost, not a free precomputation.

    A bottleneck missing from state_index is unreachable under M_R, so it
    cannot lie on any path to the goal; it scores 0 and is asked last.
    """
    unique_B, B_to_idx = _alphabet(I, B)
    if V_R is None or state_index is None:
        raise ValueError(
            "solve_query_mdp_proximity needs both V_R and state_index — they are "
            "the pair returned by value_iteration_goal_probability(T_R_sto, "
            "start_state, goal_state).")
    V_R = np.asarray(V_R, dtype=np.float64)
    if not all(isinstance(b, (int, np.integer)) for b in unique_B):
        raise ValueError("state_index is keyed by raw state ID, so the alphabet "
                         "must be raw state IDs, not decoded tuples.")
    score = np.array([V_R[state_index[b]] if b in state_index else 0.0
                      for b in unique_B])
    return GreedyQNet(len(unique_B), _hypothesis_matrix(I, B_to_idx),
                      unique_B, B_to_idx, rule="static", static_score=score)


def solve_query_mdp_transition(I, B, C_Q=-10.0, p_I=1.0, gamma=0.99, p_F=0.0,
                               oracle: "Oracle | None" = None, base=None):
    """Hypothesis 2(ii) — the dominance entailment b2 ∉ I_G ⇒ b1 ∉ I_G.

    Not a selection rule: `base` (default: the VI policy) still chooses every
    query.  The reduction rides along as `.dominance`, which QueryMDPVecEnv
    applies after each NO to rule out dominated bottlenecks for free.  Attach it
    to any other policy to layer H2 on that condition instead.
    """
    unique_B, B_to_idx = _alphabet(I, B)
    policy = base if base is not None else solve_query_mdp_exact(
        I, B, C_Q=C_Q, p_I=p_I, gamma=gamma, p_F=p_F, oracle=oracle)
    policy.dominance = _dominance_closure(_hypothesis_matrix(I, B_to_idx))
    return policy
