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

The game module builds those four things — for Overcooked, overcooked_env.py,
one configuration at a time.  This module is shared across games and knows about
none of them.

Pipeline
--------
  1.  compute_bottlenecks_per_matrix(T_H_list, start, goal) → one bottleneck set per
                                                         human; union them → B_nofilter
  2.  remove_toboggan_redundancies(T_R, B_nofilter, goal|None)
                                                         → B          true decision nodes
  3.  find_maximally_achievable_subsets(B, T_R, start, goal)
                                                         → I          [Algorithm 1]
  4.  subsets_to_array(I, B)                             → I_array    bool (len_I_array, n)
  5.  one policy per condition — solve_query_mdp_exact (VI), _info_gain (H1),
      _proximity (H3), _frequency (H4); build_dominance (H2) returns a mask, not
      a policy, because H2 selects nothing                → §7
  6.  evaluate_policy_on_real_human(...)                 → query count per episode

B is what the robot may ask about, and the bit order every policy and I_array
agree on.  It is sorted at step 1 and that order is kept to the end of the
pipeline; it is never derived from I — see _bit_order().

Step 6 also provides the random-order "query all" baseline (policy_network=None)
and applies H2 (dominance=...), which is why the "X" and "X + H2" columns can
share one policy object.

Notation
--------
B_nofilter   bottleneck state IDs — mandatory waypoints on every path to a goal
             state, unioned over the candidate humans.
B            the query set: B_nofilter minus the "toboggan" states that offer no
             choice (see §2).  What every function below means by `B`, and equal
             to B_nofilter wherever the filter is not run.
I            list of maximally achievable bottleneck subsets (Algorithm 1 output);
             each element is a list of bottleneck state IDs.  One of them is the
             subset the human actually pursues.
I_array      bool matrix, shape (len(I), n): row k is subset I[k], columns ordered
             like the bottleneck list it was built against.
I_G          the evaluated human's own subgoal set — the bottlenecks it answers YES
             to, restricted to B.  Unknown to the robot; the queries are
             what narrow it down.  The Query MDP assumes I_G ∈ I ("the hypothesis
             space contains the truth"); when it does not, the episode is a
             failure and Hypothesis 2's entailment stops being valid — see
             build_dominance.
K_I          bitmask of bottlenecks confirmed to belong to I_G (oracle YES).
K_not_I      bitmask of bottlenecks confirmed not to belong to it (oracle NO).
I_hat        = K_I | (unqueried bits) — current upper-bound on I_G.
             The invariant every condition maintains is K_I ⊆ I_G ⊆ I_hat.
ExactQNet    what solve_query_mdp_exact returns: V, best_action_mask, absorbing masks.
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


def extract_bottlenecks(T, start_state, goal_state, verbose=False):
    """Extract the mandatory bottleneck states of a single transition matrix.

    Builds a directed graph from T, runs the dominator tree from start_state, and
    collects every state that lies on every path to goal_state.

    The walk starts at goal_state itself, so the terminal set is not a parameter:
    every state it returns dominates the goal.  A model whose interesting
    terminal states sit *earlier* than the goal (a movement MDP stopping at the
    post-scoop state, say) needs a second dominator walk, not a keyword — the
    tree would have to be re-rooted, which is why there is no `goals` argument
    to pass one in.

    Parameters
    ----------
    T           : ndarray, shape (n_states, n_actions)
    start_state : int  root of the dominator tree
    goal_state  : int  universal absorbing state, always included in the result
    verbose     : bool

    Returns
    -------
    list[int]  sorted bottleneck state IDs, always includes goal_state
    """
    bottlenecks = {goal_state} | _dominator_bottlenecks(
        _transition_graph(T), start_state, goal_state
    )
    result = sorted(bottlenecks)
    if verbose:
        print(f"Found {len(result)} bottleneck states via dominator tree.")
    return result


def compute_bottlenecks_per_matrix(T_list, start_state, goal_state,
                                   verbose=False) -> list:
    """
    One frozenset of raw bottleneck IDs per matrix — the ensemble the Oracle
    samples from, and the pool the evaluated human is drawn from.
    """
    return [
        frozenset(extract_bottlenecks(T, start_state, goal_state, verbose=verbose))
        for T in T_list
    ]


def shuffle_bottlenecks(B, seed=None):
    """Return a randomly permuted copy of the bottleneck list B."""
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.permutation(len(B))
    return [B[i] for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Toboggan filtering  →  B   [optional preprocessing before Algorithm 1]
# ─────────────────────────────────────────────────────────────────────────────

def remove_toboggan_redundancies(T_matrix, B_list, goal_state):
    """
    Remove linear, non-branching "toboggan" sequences from the bottleneck set.

    From each bottleneck, BFS to its immediate downstream bottleneck neighbours.
    A node with exactly one downstream bottleneck offers no real choice and is
    discarded.  Terminal nodes (0 successors) and true decision points (≥2
    successors) are kept.

    This step compresses 2^|B_nofilter| to 2^|B| before Algorithm 1, which is
    the key that makes the search tractable on larger bottleneck sets.

    Parameters
    ----------
    T_matrix : ndarray, shape (n_states, n_actions)
    B_list : list[int]  bottlenecks to filter
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
    possible_bottlenecks : list[int]   bottleneck state IDs (B_nofilter or B)
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

    # A trajectory has to *finish*.  One-way doors make the transition graph not
    # strongly connected — a room whose exits are all blocked is a trap — so
    # visiting order alone does not settle achievability: a subset counts only if
    # some order covering it can still reach the goal afterwards.  Bottlenecks
    # inside a trap then belong to no I_k at all: still queryable, not achievable.
    # It is also what keeps the universal-bottleneck invariant below true rather
    # than merely asserted.
    goal_bit = (possible_bottlenecks.index(goal_state)
                if goal_state in possible_bottlenecks else None)

    def completable(mask):
        """Can a visiting order covering ``mask`` still end at the goal?

        ``mask == 0`` is handled apart from the memo: the DFS descends the
        exclude branch first, so the empty subset reaches here before any call has
        recorded ``memo[0]``.
        """
        if goal_bit is None:            # no goal in the set: nothing to require
            return True
        if mask == 0:                   # visit nothing, head straight for the goal
            return bool(from_start[goal_bit])
        end_bits = memo[mask]           # which bottlenecks an order can finish on
        # A subset containing the goal needs no special case: the goal is
        # reachable from the goal, so it is its own valid continuation.
        return any((end_bits >> i) & 1 and from_bottleneck[i, goal_bit]
                   for i in range(n))

    def generate_subsets(index, current_mask):
        _counter[0] += 1
        if verbose and _counter[0] % 20_000 == 0:
            print(f"  {_counter[0]:7d} calls, depth {index}/{n}", end='\r', flush=True)
        if index == n:
            if completable(current_mask):
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

def bottleneck_index(B):
    """{bottleneck: position} for the bottleneck list B — I_array's columns, and
    equally the bit order of every policy built on the same B.

    Goes through _as_label so a decoded tuple label works as well as a raw state
    ID; this is the one place the map is built.
    """
    return {_as_label(b): j for j, b in enumerate(B)}


def subsets_to_array(I, B):
    """
    Pack the subsets I into the bool matrix I_array.

    Parameters
    ----------
    I : list[iterable[int]]  maximally achievable subsets, raw state IDs
    B : list[int]            bottleneck list defining the column order — the query
                             set, in the order it was sorted into when it was built

    Returns
    -------
    I_array : bool ndarray, shape (len(I), len(B))
              I_array[k, j] is True iff B[j] belongs to subset I[k].
    """
    col_of = bottleneck_index(B)
    I_array = np.zeros((len(I), len(B)), dtype=bool)
    for k, subset in enumerate(I):
        for b in subset:
            I_array[k, col_of[int(b)]] = True
    return I_array


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Oracle  (non-uniform prior over bottleneck membership)
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
# 6.  Query MDP solvers  (Definition 6)
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
    B : iterable of bottlenecks — mandatory
        What the robot may query: the action set, and the bit order of the
        returned policy.  It is the query set — the toboggan filter's output
        wherever that stage runs — and it is deliberately *not* derivable
        from I — a bottleneck belonging to no I_k is still worth asking about,
        since YES proves the human matches no hypothesis (failure) and NO is
        required before any hypothesis can be certified (success).  Deriving it
        from I would drop exactly those bottlenecks and would let the bit order
        drift from a caller-built I_array, both silently.  Must contain every
        label occurring in I.

    Returns
    -------
    ExactQNet with attributes: V, best_action_mask, failure, success,
                               unique_B, B_to_idx, n, POW3, p_I
    """
    unique_B, B_to_idx = _bit_order(I, B)
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
# 7.  Hypothesis conditions — H1, H2, H3, H4
#
# Each condition answers one question: *which bottleneck to query next*.  None
# of them owns the knowledge state and none of them decides when to stop —
# §8 does that, with the same I_hat ⊆ I_k test solve_query_mdp_exact
# uses (§6).  So every condition is comparable to the VI baseline by
# construction, and a condition is just a scoring rule over B.
#
# The exception is H2, which is not a selection rule at all: it grants extra NOs
# for free after each oracle NO, so it lives on the knowledge state (the env),
# not in a score, and composes with any of the others.
#
# H2 is also **UNSOUND** — it can rule out a bottleneck the human actually wants
# without ever asking about it, and the robot then commits to a plan that skips
# that subgoal.  It has been dropped from the paper.  The code keeps it behind
# --h2, off by default, so the effect stays reproducible; see build_dominance
# for why it breaks and how often.  Do not turn it on to report a query count.
# ─────────────────────────────────────────────────────────────────────────────

def _bit_order(I, B):
    """Check B, and return it with the {bottleneck: bit} map every solver shares.

    Bit i means B[i] — in the policy's action index, in I_array's column i, and
    in the env's K_I / K_not bit i.  Those three must agree or the robot asks
    about one bottleneck and files the answer under another, silently.

    B is *validated*, never repaired.  It arrives sorted and unique because it is
    sorted once where it is built and that order is kept for the rest of the
    pipeline; re-sorting it here would be worse than useless, since a caller who
    built I_array from an unsorted B would get columns that no longer line up
    with the bits.  An order that is wrong should stop the run, not be quietly
    corrected underneath the caller.

    B is likewise mandatory, with no fallback to the labels occurring in I: those
    are a strict subset of B, so deriving it would drop the bottlenecks
    belonging to no I_k and change the answer without failing.
    """
    if B is None:
        raise ValueError(
            "B is mandatory — pass the query set. It cannot be derived from I: the "
            "bottlenecks belonging to no I_k are exactly the ones I does not "
            "mention, and dropping them changes the answer.")
    B = [_as_label(b) for b in B]
    if not B:
        raise ValueError("B is empty — there is nothing the robot may query.")
    if len(set(B)) != len(B):
        raise ValueError("B has duplicates — two bits would mean one bottleneck.")
    if B != sorted(B):
        raise ValueError(
            "B is not sorted.  It is sorted where it is built and that order is "
            "the bit order for the whole pipeline; re-sorting it here would "
            "desynchronise the bits from an I_array built on the order given.")
    missing = {_as_label(b) for subset in I for b in subset} - set(B)
    if missing:
        raise ValueError(
            f"B is missing {len(missing)} bottleneck(s) present in I: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    return B, bottleneck_index(B)


def _hypothesis_matrix(I, B_to_idx):
    """Φ as a bool matrix, shape (len(I), n): row k is I[k] in B's order."""
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


def value_iteration(T_R_sto, goal_state, reward_function=None,
                    states=None, actions=None,
                    gamma=0.99, tol=1e-10, max_iter=10_000):
    """V_R: the robot's optimal state value under M_R.

    DEAD CODE — nothing in the pipeline calls this any more.  It fed H3, which
    now ranks bottlenecks by Euclidean distance to the goal and needs no value at
    all (see solve_query_mdp_proximity).  It is kept rather than deleted because
    it still works and is the obvious starting point if a value-based condition
    is ever wanted again; its partner builders, gridworld_core and
    overcooked_env's build_stochastic_matrix, are kept for the same reason.
    Nothing exercises it, so treat it as untested from here on.

    Two modes, chosen by whether a reward function is supplied.

    Reward mode (reward_function given).  Standard value iteration driven by the
    MDP's own reward, with the goal treated as absorbing:

        V(s) = max_a [ R(s,a) + γ · Σ_s' P(s'|s,a) · V(s') ]

    where R(s,a) = Σ_s' P(s'|s,a) · reward_function(s, a, s') is the expected
    immediate reward, built once from the callable over the pruned state space.
    The goal is absorbing at reward 0 in M_R, so V(goal) converges to 0 and
    value cannot leak out of it.  For the grid's reach-the-goal reward this
    gives V(s) = γ^{d(s)-1} for d = shortest path length to the goal — monotone
    in distance, which is all H3's ranking needs — and 0 where the goal is
    unreachable.

    Probability mode (reward_function is None).  The legacy behaviour, kept for
    models with no reward object (e.g. Overcooked): the goal is pinned worth 1
    and V is the expected discounted probability of reaching it,

        V(goal) = 1 ,   V(s) = γ · max_a Σ_s' P(s'|s,a) · V(s').

    Parameters
    ----------
    T_R_sto         : (n, n_actions, n) float — P(s'|s,a), rows summing to 1, as
                      returned by gridworld_core.build_stochastic_matrix or
                      overcooked_env.build_stochastic_matrix.  Already pruned to
                      the robot-reachable states; nothing is pruned here.
    goal_state      : int — the absorbing goal's row in T_R_sto, i.e.
                      index[goal].  Used to pin V in probability mode.
    reward_function : callable(state, action, next_state) -> float, or None.
                      When given, `states` and `actions` must be too.
    states          : list — states[i] is the original state object at row i of
                      T_R_sto (build_stochastic_matrix's kept_states), so the
                      callable can be evaluated on the pruned matrix.
    actions         : list — action labels in T_R_sto's action-axis order.
    gamma           : discount.

    Returns
    -------
    (n,) float64 — V_R over the rows of T_R_sto.
    """
    T = np.asarray(T_R_sto, dtype=np.float64)
    if T.ndim != 3:
        raise ValueError(
            "T_R_sto must be (n_states, n_actions, n_states) probabilities; got "
            f"{T.shape}.  Build it with build_stochastic_matrix().")

    if reward_function is None:
        # Probability mode: goal absorbing and worth 1, no reward term.
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

    if states is None or actions is None:
        raise ValueError(
            "reward mode needs `states` and `actions` to evaluate the reward "
            "callable on the pruned matrix; pass the kept_states and actions "
            "returned by build_stochastic_matrix().")

    n, n_a, _ = T.shape
    # Expected immediate reward R(s,a) = Σ_s' P(s'|s,a)·reward(s,a,s'), built
    # once from the callable over the reachable successors of each row.
    R_sa = np.zeros((n, n_a), dtype=np.float64)
    for i in range(n):
        si = states[i]
        for ai in range(n_a):
            a = actions[ai]
            for j in np.nonzero(T[i, ai])[0]:
                R_sa[i, ai] += T[i, ai, j] * reward_function(si, a, states[j])

    V = np.zeros(n, dtype=np.float64)
    for _ in range(max_iter):
        V_next = np.max(R_sa + gamma * (T @ V), axis=1)
        converged = np.max(np.abs(V_next - V)) < tol
        V = V_next
        if converged:
            break
    return V


def _dominance_closure(T):
    """H2(ii): dom[b2, b1] is True iff b1 ⪯ b2, i.e. ∀ϕ ∈ Φ: b1 ∈ ϕ ⇒ b2 ∈ ϕ.

    An oracle NO on b2 then entails NO on every such b1 at no query cost.
    Transitively closed here so the caller only needs one propagation step.

    A bottleneck occurring in *no* hypothesis is excluded as a b1.  The
    implication is vacuously true for it against every b2, so it would come out
    dominated by the whole of B and the first NO anywhere would rule it out —
    which is a quantifier artifact, not a structural redundancy.  It is also
    exactly the wrong bit to discard for free: a bottleneck in no ϕ is the one
    whose YES proves the human matches nothing, so ruling it out unasked turns a
    failure into a success.

    Dropping those rows makes the mask *less* wrong, not correct: H2 is unsound
    with or without them, because the entailment needs I_G ∈ Φ and nothing
    guarantees it.  See build_dominance.
    """
    n = T.shape[1]
    # prec[b1, b2] — no hypothesis holds b1 without also holding b2.
    prec = ~((T[:, :, None] & ~T[:, None, :]).any(0))
    prec &= T.any(0)[:, None]                     # drop the vacuously-true rows
    dom  = np.ascontiguousarray(prec.T)           # dom[b2, b1] = b1 ⪯ b2
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
    unique_B, B_to_idx = _bit_order(I, B)
    return GreedyQNet(len(unique_B), _hypothesis_matrix(I, B_to_idx),
                      unique_B, B_to_idx, rule="entropy")


def solve_query_mdp_frequency(I, B, C_Q=-10.0, p_I=1.0, gamma=0.99, p_F=0.0,
                              oracle: "Oracle | None" = None):
    """Hypothesis 4 — the bottleneck in the most currently consistent hypotheses."""
    unique_B, B_to_idx = _bit_order(I, B)
    return GreedyQNet(len(unique_B), _hypothesis_matrix(I, B_to_idx),
                      unique_B, B_to_idx, rule="marginal")


def solve_query_mdp_proximity(I, B, C_Q=-10.0, p_I=1.0, gamma=0.99, p_F=0.0,
                              oracle: "Oracle | None" = None,
                              positions=None, goal_state=None):
    """Hypothesis 3 — three tiers, with Euclidean proximity inside the middle one.

    Distance alone is not the rule, because most bottlenecks are not worth asking
    about at all.  Split B by how many hypotheses hold a bottleneck:

      1. **in no I_k** — asked first.  It belongs to no achievable subset, so a
         YES proves the human's subgoal set is outside I: failure, found at once
         instead of after the whole budget.  A NO is needed before any hypothesis
         can be certified, so the query is never wasted either way.
      2. **in some but not all** — the only tier where the answer discriminates,
         and the only place the geometry is used: nearest the goal first, by
         straight-line distance.
      3. **in every I_k** — asked last, which in practice means never.  The oracle
         cannot say anything that moves either terminal test: a YES leaves I_hat
         untouched, and K_I stays inside every I_k so no failure can trigger.
         goal_state lives here, being a bottleneck of every model — which is what
         stops H3 from spending its first query on the goal, as a pure distance
         ranking does (distance 0 is the smallest there is).

    Tier 3 is never reached because the episode always ends first: once every
    bottleneck outside tier 3 is answered, either the YES bits all sit in one I_k
    (success, since the tier-3 bits are in that I_k by definition) or they do not
    (failure).  Ranking rather than removing keeps the action set equal to B, so
    the bit order still matches I_array and every other condition.

    Straight-line means straight-line: walls, one-way doors and obstacles are
    ignored inside tier 2, so two cells either side of a sealed wall can rank as
    neighbours.  That is the definition, not an oversight — a graph distance
    would respect them, and is a different hypothesis.

    The tiers come from I, the full hypothesis space, and are fixed for the
    episode; H3 does not re-derive them from the consistent set as H1 and H4 do.

    Parameters
    ----------
    positions : dict {raw state ID -> coordinate tuple}
        Where each state sits.  For the grid games that is
        ``{i: tuple(s[0]) for i, s in enumerate(mdp.get_state_space())}``, since
        ``state[0]`` is the (row, col) position in every grid world.  Any number
        of dimensions works; the norm does not care.
    goal_state : int
        Raw state ID of the goal, so its coordinate can be looked up here.

    A game with no geometry — Overcooked, whose states are bit-packed inventory /
    pot pairs — has no `positions` to pass and therefore no H3.  The caller skips
    the condition rather than inventing coordinates for it.
    """
    unique_B, B_to_idx = _bit_order(I, B)
    if positions is None or goal_state is None:
        raise ValueError(
            "solve_query_mdp_proximity needs `positions` and `goal_state`: a map "
            "from raw state ID to coordinate, and the goal's ID.  For a grid "
            "world, positions = {i: tuple(s[0]) for i, s in "
            "enumerate(mdp.get_state_space())}.")
    if not all(isinstance(b, (int, np.integer)) for b in unique_B):
        raise ValueError("positions is keyed by raw state ID, so B must be raw "
                         "state IDs, not decoded tuples.")
    if int(goal_state) not in positions:
        raise ValueError(f"goal_state {int(goal_state)} has no position.")
    missing = [b for b in unique_B if b not in positions]
    if missing:
        # Every state of a grid world owns a cell, so this means `positions` was
        # built against a different state space than B was.  Raise instead of
        # ranking the odd ones out by a sentinel, which would hide the mismatch.
        raise ValueError(
            f"{len(missing)} bottleneck(s) have no position: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    T = _hypothesis_matrix(I, B_to_idx)          # (len(I), n) bool
    in_none = ~T.any(0)
    # `and len(I)` matters: with no hypotheses at all, all(0) over zero rows is
    # vacuously True everywhere, which would put every bottleneck in tier 3 and
    # ask about none of them.  With I empty they are all in no hypothesis.
    in_all = T.all(0) & bool(len(I))

    goal_pos = np.asarray(positions[int(goal_state)], dtype=np.float64)
    dist = np.array([
        float(np.linalg.norm(np.asarray(positions[b], dtype=np.float64) - goal_pos))
        for b in unique_B])

    # Tier offsets, sized off the data rather than fixed: one span is wider than
    # any distance on this board, so the three bands cannot overlap however large
    # the board gets.  Finite on purpose — evaluate_policy_on_real_human masks
    # already-queried actions to -1e9, so a -inf tier would rank *below* them and
    # the argmax would re-query a spent bottleneck.
    span = float(dist.max()) + 1.0 if dist.size else 1.0
    tier = np.where(in_none, 1.0, np.where(in_all, -1.0, 0.0))
    # Negated distance so that, inside a tier, nearest the goal is asked first.
    score = span * tier - dist
    return GreedyQNet(len(unique_B), T, unique_B, B_to_idx,
                      rule="static", static_score=score)

def build_dominance(I, B):
    """Hypothesis 2(ii) — the dominance entailment b2 ∉ I_G ⇒ b1 ∉ I_G, where
    I_G is the evaluated human's own subgoal set (see the module Notation).

    Deliberately not a solver.  H2 is not a selection rule: it never chooses a
    query, it only widens K_not after an oracle NO, so it is a *layer* that any
    condition can wear.  What it returns is the mask, handed to
    evaluate_policy_on_real_human(dominance=...) at inference time; no policy is
    computed here and no policy is modified.

    That separation is what lets one policy serve both the "H1" and "H1 + H2"
    columns — the two runs differ only in whether this mask is passed.

    UNSOUND.  This hypothesis has been dropped from the paper; --h2 is off by
    default and the code is kept only so the effect stays reproducible.

    The entailment needs I_G ∈ Φ: the rule holds for every ϕ ∈ Φ, so it carries to
    the human's own subgoal set only if that set is one of them.  Nothing in the
    pipeline makes it one — Φ is what the *robot* can achieve maximally, I_G is
    the bottleneck set of a *human* matrix — so in general only I_G ⊆ ϕ for some
    ϕ, and the contraposition does not go through.  The vacuous-row exclusion in
    _dominance_closure narrows the hole but does not close it.

    Do not read "the mask never changes the answer" as evidence for it.  The mask
    *cannot* change the answer: _grant_entailed_nos only ever adds bits to K_not,
    which shrinks I_hat and so makes the success test I_hat ⊆ ϕ easier, while a
    bit granted a NO is never queried again and so keeps K_I smaller, making the
    failure test K_I ⊄ every ϕ harder.  Both effects point the same way, so a
    measured 0 is a property of that metric, not a result about H2.

    The invariant that does catch it is the one in the module Notation,
    K_I ⊆ I_G ⊆ I_hat.  Measured over 330 (instance x human) gridworld episodes
    under H1, 3x3 rooms of 3 cells, 3 humans, default density: without the mask a
    NO is granted on a bottleneck actually in I_G 0 times; with the mask, 165
    times.  In 10 of the 225 successful episodes the robot can then certify a ϕ
    that omits one of the human's real subgoals, having never asked about it.

    Returns (n, n) bool: dom[b2, b1] is True iff b1 ⪯ b2, i.e. ∀ϕ ∈ Φ,
    b1 ∈ ϕ ⇒ b2 ∈ ϕ.  Transitively closed, so one propagation pass suffices.
    """
    _, B_to_idx = _bit_order(I, B)
    return _dominance_closure(_hypothesis_matrix(I, B_to_idx))


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Evaluation against a real human
#
# One episode = one knowledge state (K_I, K_not) walked forward until it is
# absorbing.  The policy only ever answers "which bottleneck next"; whether the
# episode is over is decided here, by the same test solve_query_mdp_exact builds
# its failure / success masks from (§6).  That is what makes every condition
# comparable: they share a stopping rule none of them owns.
# ─────────────────────────────────────────────────────────────────────────────

def _terminal(K_I, K_not, I_array):
    """(done, success) for one knowledge state.

    failure : K_I ⊆ no hypothesis — the human matches nothing in I.
    success : I_hat ⊆ some hypothesis, with I_hat = K_I ∪ unqueried = ~K_not.
              Every bottleneck still in play is covered by one achievable
              subset, so the robot can satisfy the human without asking further.

    K_I, K_not : (n,) bool
    I_array    : (len(I), n) bool — row k is I[k] in B's bit order.
    """
    failure = not (~(K_I & ~I_array).any(1)).any()
    success = (not failure) and bool((~(~K_not & ~I_array).any(1)).any())
    return failure or success, success


def _grant_entailed_nos(K_I, K_not, dominance):
    """Hypothesis 2(ii), in place on K_not: the NOs the current K_not entails.

    b1 ⪯ b2 and b2 ∉ I_G ⇒ b1 ∉ I_G, so those bits are ruled out without
    spending a query.  `dominance` is transitively closed, so one pass reaches
    the fixpoint.  Bits already answered YES are left alone: overwriting one
    would forge an answer the oracle never gave.

    This is the line where H2's unsoundness happens: the implication is only
    valid when I_G ∈ Φ, and when it is not, a bit that belongs to I_G can be set
    in K_not here — breaking the K_I ⊆ I_G ⊆ I_hat invariant the rest of the
    module maintains.  Measured at 165 of 330 gridworld episodes.  See
    build_dominance.

    No-op when dominance is None, which is every condition except the "+ H2"
    ones — this is the *only* thing that separates a pair of columns.
    """
    if dominance is None:
        return
    K_not |= (K_not[:, None] & dominance).any(0) & ~K_I


def _run_query_episode(K_I, K_not, I_array, true_bits, choose_action, dominance):
    """Walk one episode to absorption; returns (n_queries, success).

    The empty knowledge state is tested *before* the loop.  It can already be
    absorbing: I_hat starts as the whole of B, so if B is contained in some I_k
    the instance is solved with zero questions.  Entering the loop regardless
    would bill one query for a problem that never posed a question — and would
    pick that query from an all-zero best_action_mask, since a policy has no
    meaningful action at an absorbing state.
    """
    done, success = _terminal(K_I, K_not, I_array)
    if done:
        return 0, success
    for n_q in range(1, I_array.shape[1] + 1):
        action = choose_action(K_I, K_not)
        if action in true_bits:
            K_I[action] = True
        else:
            K_not[action] = True
        _grant_entailed_nos(K_I, K_not, dominance)
        done, success = _terminal(K_I, K_not, I_array)
        if done:
            return n_q, success
    # Unreachable: once every bit is queried I_hat == K_I, so failure and
    # success partition the state.  Kept so the loop has no silent fall-through.
    raise RuntimeError("query episode ended without an absorbing state")


def evaluate_policy_on_real_human(
    true_bottlenecks,
    policy_network,       # ExactQNet / GreedyQNet, or None for the random-order baseline
    n_runs,
    I_array,
    b_to_int,
    c_q=-10.0,
    p_i=1.0,
    p_f=0.0,
    gamma=0.99,
    device="cpu",
    dominance=None,
):
    """Evaluate a query policy against a real human with known implicit subgoals.

    The oracle answers deterministically: YES iff the queried bottleneck is in
    true_bottlenecks.

    `dominance` is Hypothesis 2(ii), from build_dominance().  It is applied here,
    at inference, and never reaches the policy: the same policy_network run with
    and without it gives the "X" and "X + H2" columns.  H2 is unsound and has
    been dropped from the paper — passing a mask here lowers the query count by
    granting answers the oracle never gave, so the counts it produces are not
    comparable with the others.  Leave it None unless reproducing that effect.

    Parameters
    ----------
    true_bottlenecks : bottleneck identifiers the human would answer YES to.
        Those absent from b_to_int are dropped — they are outside the query set,
        so the robot can never ask about them.
    policy_network   : ExactQNet / GreedyQNet → guided policy;
                       None → "query all", a fresh random order each run.
    n_runs           : int  independent episodes
    I_array          : (len(I), n) bool
    b_to_int         : {bottleneck: column index} — bottleneck_index(B)
    c_q, p_i, p_f, gamma : MDP parameters, for the reported reward only; the
        query *count* does not depend on them, since the stopping rule is
        structural.  They must match the ones the policy was solved with.
    device           : torch device (ignored when policy_network is None)

    Returns
    -------
    dict with 'n_queries' (int), 'total_reward' (float), 'success' (bool),
    each an array of length n_runs.
    """
    n = I_array.shape[1]
    true_bits = frozenset(
        b_to_int[_as_label(b)] for b in true_bottlenecks if _as_label(b) in b_to_int
    )

    if policy_network is None:
        def choose_action(K_I, K_not):
            return next(order)
    else:
        policy_network.eval()
        def choose_action(K_I, K_not):
            obs   = torch.as_tensor(np.concatenate([K_I, K_not])[None, :],
                                    dtype=torch.float32, device=device)
            vmask = torch.as_tensor(~(K_I | K_not)[None, :],
                                    dtype=torch.bool, device=device)
            with torch.no_grad():
                q = policy_network(obs).masked_fill(~vmask, -1e9)
            return int(q.argmax(1).cpu().item())

    n_queries_arr    = np.zeros(n_runs, dtype=int)
    total_reward_arr = np.zeros(n_runs, dtype=float)
    success_arr      = np.zeros(n_runs, dtype=bool)

    for run in range(n_runs):
        K_I, K_not = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        if policy_network is None:
            order = iter(shuffle_bottlenecks(list(range(n))))
        n_q, success = _run_query_episode(K_I, K_not, I_array, true_bits,
                                          choose_action, dominance)
        n_queries_arr[run]    = n_q
        total_reward_arr[run] = n_q * c_q + gamma * (p_i if success else p_f)
        success_arr[run]      = success

    return {
        "n_queries":    n_queries_arr,
        "total_reward": total_reward_arr,
        "success":      success_arr,
    }
