"""
overcooked_env.py
=================
No-movement Overcooked game backend, bottleneck extraction (Algorithm 1), and
the exact Query MDP solver. Trimmed to only what parallel_experiments.py's
'overcooked' world type uses.

All functions operate on encoded integer state IDs — no decoding, no plotting.

Pipeline
--------
  1. build_transition_matrix_nomove()               → T, RECIPES, {}
  2. extract_bottlenecks_nomove([T])                 → possible_bottlenecks
  3. remove_toboggan_redundancies(T, B)               → B_cleaned        [optional]
  4. find_maximally_achievable_subsets(B_cleaned, T)  → I                [Algorithm 1]
  5. decode_subsets_to_2d_nomove(I)                   → I_decoded
  6. solve_query_mdp_exact(I_decoded)                 → ExactQNet

Notation
--------
T            transition matrix: T[state, action] → next_state  (ints throughout).
B            bottleneck state IDs — mandatory waypoints on every path to a terminal.
I            list of maximally achievable bottleneck subsets (Algorithm 1 output);
             each element is a list of bottleneck state IDs.
K_I          bitmask of bottlenecks confirmed to belong to the human's target (oracle YES).
K_not_I      bitmask of bottlenecks confirmed not to belong to it (oracle NO).
I_hat        = K_I | (unqueried bits) — current upper-bound on the target.
mdp          ExactQNet returned by solve_query_mdp_exact: V, policy, absorbing masks.
inv          bit-packed int: inventory state.  Bit layout (8 bits):
             bit 0 = plate flag, bit 1 = cooked flag,
             bits 2-3 = onion count, bits 4-5 = tomato count, bits 6-7 = mushroom count.
pot          bit-packed int: pot state.  Same bit layout as inv
             (bit 0 = served flag, bit 1 = cooking flag).
"""

import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import deque
import time
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NUM_INV     = 196
NUM_POT     = 196
NUM_STATES  = NUM_INV * NUM_POT
NUM_ACTIONS = 5

# action index → bit-packed inventory value for each raw ingredient
ITEM_MAP = [4, 16, 64]   # 0=onion, 1=tomato, 2=mushroom

# Serving extension (appended indices — nothing before them shifts)
SERVE_ACTION  = NUM_ACTIONS         # 5: interact_with_serving_desk
CLIENT_SERVED = NUM_INV * NUM_POT   # 38416: shared absorbing state (all recipes end here)

# ── No-movement MDP action indices ────────────────────────────────────────────
ACTION_PICK_ONION  = 0
ACTION_PICK_TOMATO = 1
ACTION_PICK_MUSH   = 2
ACTION_GRAB_PLATE  = 3
ACTION_INTERACT    = 4
PICKUP_ACTIONS     = (ACTION_PICK_ONION, ACTION_PICK_TOMATO, ACTION_PICK_MUSH)

# ── Bit-field accessors for packed inv / pot integers ────────────────────────
# Bit layout (shared by inv and pot):
#   bit 0 = plate flag (inv) / served flag (pot)
#   bit 1 = cooked / cooking flag
#   bits 2-3 = onion count, bits 4-5 = tomato count, bits 6-7 = mushroom count

def is_cooked(x):
    """True if the cooked/cooking flag (bit 1) is set."""
    return (x & 2) != 0

def is_served_or_plated(x):
    """True if the served/plate flag (bit 0) is set."""
    return (x & 1) != 0

def ingredient_counts(x):
    """Return (onions, tomatoes, mushrooms) counts packed in bits 2–7."""
    return (x >> 2) & 3, (x >> 4) & 3, (x >> 6) & 3


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Transition-matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def bfs_reachable_set(T_matrix, start, num_actions):
    """Plain set-BFS reachability over a transition matrix — used by the T builders."""
    reachable = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for action in range(num_actions):
            nxt = int(T_matrix[cur, action])
            if nxt not in reachable:
                reachable.add(nxt)
                queue.append(nxt)
    return reachable


def build_transition_matrix_nomove(allow_drop: bool = False, verbose: bool = True):
    """
    Build the clean, reachable-only transition matrix T.

    Step 1 — T_not_clean: exhaustive rule mapping over all (inv, pot) pairs.
      State encoding: s = inv * NUM_POT + pot
        inv bit layout: bit 0 = plate flag, bit 1 = cooked flag,
                        bits 2-3 = onion count, 4-5 = tomato, 6-7 = mushroom
        pot bit layout: bit 0 = served flag, bit 1 = cooking flag, same ingredient bits.

    Step 2 — BFS reachability filter: walk every state reachable from state 0
      (empty hands, empty pot); lock all ghost states into self-loops so they
      never pollute downstream computations.

    The returned matrix has shape (CLIENT_SERVED + 1, SERVE_ACTION + 1): the
    extra row is the CLIENT_SERVED absorbing state (self-loop on all actions) and
    the extra column is the SERVE_ACTION slot (self-loop everywhere in this base
    matrix — serving edges are added by the caller).

    When allow_drop=True, T_not_clean gains extra edges (cycles), but the BFS
    filter is still sound: it only promotes states reachable from state 0, so
    ghost states remain locked as self-loops in the final T regardless.

    Parameters
    ----------
    allow_drop : bool
        If True, the agent can put a held item back into its dispenser:
          - holding ingredient X + action i (pick-up action for X) → drop X back
          - holding plate     + action 3 (grab-plate)              → drop plate back
    verbose : bool
        If False, suppress tqdm bars and print statements (used for the silent
        module-level build that runs at import time).

    Returns
    -------
    T : ndarray, shape (CLIENT_SERVED + 1, SERVE_ACTION + 1), dtype int32
        Base matrix — SERVE_ACTION column is all self-loops (no serving edges).
    RECIPES : list[int]   raw pot encodings of every valid cooked-soup state
    """
    RECIPES = []
    T_not_clean = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.int32)
    for state in range(NUM_STATES):
        T_not_clean[state, :] = state   # default: self-loop everywhere

    for inv in tqdm(range(NUM_INV), desc="Building T_not_clean", disable=not verbose):
        for pot in range(NUM_POT):
            pot_cooked                       = is_cooked(pot)
            pot_onions, pot_tomato, pot_mush = ingredient_counts(pot)
            pot_total_items                  = pot_onions + pot_tomato + pot_mush
            inv_cooked                       = is_cooked(inv)
            s                                = inv * NUM_POT + pot

            for action in range(NUM_ACTIONS):

                if inv == 0:
                    if action in PICKUP_ACTIONS:
                        T_not_clean[s, action] = ITEM_MAP[action] * NUM_POT + pot
                    elif action == ACTION_GRAB_PLATE:
                        T_not_clean[s, action] = 1 * NUM_POT + pot
                    elif action == ACTION_INTERACT:
                        if pot_total_items == 3 and not pot_cooked:
                            T_not_clean[s, action] = (0 * NUM_POT) + (pot + 2)

                elif inv == 1:
                    if (action == ACTION_INTERACT and pot_cooked
                            and pot_total_items == 3 and not is_served_or_plated(pot)):
                        T_not_clean[s, action] = (pot + 1) * NUM_POT + 0
                        if (pot + 1) not in RECIPES:
                            RECIPES.append(pot + 1)
                    elif allow_drop and action == ACTION_GRAB_PLATE:
                        T_not_clean[s, action] = 0 * NUM_POT + pot

                elif not inv_cooked:
                    if inv in ITEM_MAP and action == ACTION_INTERACT:
                        if (pot_total_items < 3 and not pot_cooked
                                and not is_served_or_plated(pot)):
                            T_not_clean[s, action] = 0 * NUM_POT + (pot + inv)
                    elif allow_drop and action in PICKUP_ACTIONS and inv == ITEM_MAP[action]:
                        T_not_clean[s, action] = 0 * NUM_POT + pot

    reachable = bfs_reachable_set(T_not_clean, 0, NUM_ACTIONS)

    T = np.zeros((CLIENT_SERVED + 1, SERVE_ACTION + 1), dtype=np.int32)
    for state in range(CLIENT_SERVED + 1):
        T[state, :] = state
    for state in reachable:
        T[state, :NUM_ACTIONS] = T_not_clean[state, :]

    if verbose:
        print(f"T built — {len(reachable)} reachable states, "
              f"{NUM_STATES - len(reachable)} ghost states locked into self-loops.")
        print(f"Cookable recipes ({len(RECIPES)}): {RECIPES}")
    return T, RECIPES, {}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Graph & bottleneck extraction
# ─────────────────────────────────────────────────────────────────────────────

def _dominator_bottlenecks(G, start_state, targets):
    """
    Walk the immediate-dominator tree backward from each target, collecting
    every mandatory waypoint between start_state and that target.
    """
    idoms = nx.immediate_dominators(G, start_state)
    result = set()
    for target in targets:
        result.add(target)
        current = target
        while current != start_state:
            current = idoms.get(current, start_state)
            if current != start_state:
                result.add(current)
    return result


def extract_bottlenecks(T, targets, absorbing_state, start_state=0, verbose=True):
    """Extract mandatory bottleneck states from a single transition matrix.

    Builds a directed graph from T, runs the dominator tree from start_state,
    and collects every state that lies on every path to any target state.

    Parameters
    ----------
    T              : ndarray, shape (n_states, n_actions)
    targets        : list[int]  terminal states to trace back from
    absorbing_state: int  universal terminal always included in the result
    start_state    : int  root of the dominator tree
    verbose        : bool

    Returns
    -------
    list[int]  sorted bottleneck state IDs, always includes absorbing_state
    """
    G = nx.DiGraph()
    for state in range(T.shape[0]):
        for action in range(T.shape[1]):
            nxt = int(T[state, action])
            if nxt != state:
                G.add_edge(state, nxt)
    bottlenecks = {absorbing_state} | _dominator_bottlenecks(G, start_state, targets)
    result = sorted(bottlenecks)
    if verbose:
        print(f"Found {len(result)} bottleneck states via dominator tree.")
    return result


def _nomove_targets(T):
    """Terminal states for one no-movement transition matrix (states with a serving edge)."""
    if T.shape[0] <= NUM_INV * NUM_POT or T.shape[1] <= NUM_ACTIONS:
        return []
    return [s for s in range(NUM_INV * NUM_POT)
            if int(T[s, SERVE_ACTION]) == CLIENT_SERVED]


def extract_bottlenecks_nomove(T_list, start_state=0, verbose=True):
    """Union of bottlenecks over a list of no-movement transition matrices."""
    all_bottlenecks = set()
    for T in T_list:
        targets = _nomove_targets(T)
        if targets:
            all_bottlenecks |= set(
                extract_bottlenecks(T, targets, CLIENT_SERVED, start_state, verbose=False)
            )
    result = sorted(all_bottlenecks)
    if verbose:
        n = len(T_list)
        print(f"Found {len(result)} bottleneck states "
              f"(union over {n} {'matrix' if n == 1 else 'matrices'}).")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Toboggan filtering  [optional preprocessing before Algorithm 1]
# ─────────────────────────────────────────────────────────────────────────────

def remove_toboggan_redundancies(T_matrix, B_list):
    """
    Remove linear, non-branching "toboggan" sequences from the bottleneck set.

    From each bottleneck, BFS to its immediate downstream bottleneck neighbours.
    A node with exactly one downstream bottleneck offers no real choice and is
    discarded.  Terminal nodes (0 successors) and true decision points (≥2
    successors) are kept.

    This step compresses 2^|B| to 2^|B_cleaned| before Algorithm 1, which is
    the key that makes the search tractable on larger bottleneck sets.

    Returns
    -------
    cleaned : list[int]  sorted bottlenecks after toboggan removal
    """
    # CLIENT_SERVED is excluded from toboggan analysis: [1,X] states (which only
    # have CLIENT_SERVED as their T_R successor) would be wrongly classified as
    # toboggans otherwise.  It is always kept and appended at the end.
    has_client_served = CLIENT_SERVED in set(B_list)
    regular = [b for b in B_list if b != CLIENT_SERVED]

    B_set = set(regular)
    cleaned = []
    num_actions = T_matrix.shape[1]

    for b in regular:
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

    if has_client_served:
        cleaned.append(CLIENT_SERVED)

    return sorted(cleaned)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Algorithm 1 — Include / Exclude DFS  (subset search)
# ─────────────────────────────────────────────────────────────────────────────

def get_reachable_states(T_matrix, start_state):
    """Vectorized BFS using a boolean frontier mask instead of a growing Python set."""
    num_states = T_matrix.shape[0]
    reachable = np.zeros(num_states, dtype=bool)
    reachable[start_state] = True
    frontier = np.array([start_state])

    while frontier.size > 0:
        neighbours = np.unique(T_matrix[frontier].ravel())
        neighbours = neighbours[~reachable[neighbours]]
        reachable[neighbours] = True
        frontier = neighbours

    return reachable


def _build_adjacency(possible_bottlenecks, T_matrix, start_state=0):
    """
    Collapse per-state reachability into a compact bottleneck-to-bottleneck
    adjacency matrix — the only structure Algorithm 1 needs during its search.

    Returns
    -------
    from_start      : bool array, shape (n,)    from_start[j]    = start can reach j
    from_bottleneck : bool array, shape (n, n)  from_bottleneck[i, j] = i can reach j
    """
    masks = np.stack(
        [get_reachable_states(T_matrix, start_state)] +
        [get_reachable_states(T_matrix, b) for b in possible_bottlenecks]
    )
    idx = np.array(possible_bottlenecks)
    adj = masks[:, idx]           # shape (n+1, n)
    return adj[0], adj[1:]        # from_start (n,), from_bottleneck (n, n)


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
        if prev_ends == -1:                         # prev subset was empty: b must reach from start
            if from_start[b]:
                ends |= 1 << b
        else:
            prev_ends_bool = np.array([(prev_ends >> i) & 1 for i in range(n)], dtype=bool)
            if np.any(prev_ends_bool & from_bottleneck[:, b]):
                ends |= 1 << b

    memo[mask] = ends
    return ends != 0


def find_maximally_achievable_subsets(possible_bottlenecks, T_R, start_state=0):
    """
    Algorithm 1 — find all maximally achievable subsets of bottlenecks.

    Parameters
    ----------
    possible_bottlenecks : list[int]   bottleneck state IDs
    T_R : ndarray                      clean transition matrix

    Returns
    -------
    I : list[list[int]]   maximally achievable subsets (each is a list of state IDs)
    """
    n = len(possible_bottlenecks)

    print(f"Building bottleneck adjacency for {n} bottlenecks...")
    from_start, from_bottleneck = _build_adjacency(possible_bottlenecks, T_R, start_state)
    memo             = {}
    achievable_masks = []
    _counter         = [0]

    def generate_subsets(index, current_mask):
        _counter[0] += 1
        if _counter[0] % 20_000 == 0:
            print(f"  {_counter[0]:7d} calls, depth {index}/{n}", end='\r', flush=True)
        if index == n:
            achievable_masks.append(current_mask)
            return
        generate_subsets(index + 1, current_mask)
        new_mask = current_mask | (1 << index)
        if check_sequential_achievability(new_mask, from_start, from_bottleneck, n, memo):
            generate_subsets(index + 1, new_mask)

    print(f"Running Algorithm 1 (include/exclude DFS, {n} bottlenecks, "
          f"2^{n} = {1 << n} max subsets)...")
    generate_subsets(0, 0)
    print(f"{len(achievable_masks)} achievable subsets at leaves "
          f"({len(memo)} subsets solved by CheckAchievability).")

    maximal_masks = filter_maximal_subsets(achievable_masks)
    I = [[possible_bottlenecks[i] for i in range(n) if (m >> i) & 1] for m in maximal_masks]

    # Universal-bottleneck invariant: if CLIENT_SERVED is in the bottleneck set it
    # must appear in every maximal achievable subset (any trajectory that can win
    # must pass through it).
    if CLIENT_SERVED in set(possible_bottlenecks):
        assert all(CLIENT_SERVED in subset for subset in I), (
            "Universal-bottleneck invariant violated: CLIENT_SERVED is absent from "
            "at least one maximal achievable subset.  Every trajectory through a "
            "matrix with serving edges must end at CLIENT_SERVED."
        )

    return I


def decode_subsets_to_2d_nomove(I):
    """
    Decode raw no-movement state IDs (s = inv * NUM_POT + pot) back into
    [inv_id, pot_id] pairs, the format solve_query_mdp_exact expects.
    """
    return [[[s // NUM_POT, s % NUM_POT] for s in subset] for subset in I]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Query MDP solver  (Definition 6)
# ─────────────────────────────────────────────────────────────────────────────

class ExactQNet(nn.Module):
    """Exact Query MDP policy with the same callable interface as QNet.

    Not a neural network — backed by precomputed backward-induction arrays.
    Built by solve_query_mdp_exact(); do not instantiate directly.
    forward(x) takes a (batch, 2n) observation tensor and returns (batch, n) Q-values:
    0.0 for every tied-optimal action, −1e9 for all others.
    """

    def __init__(self, n, best_action_mask, POW3, V, failure, success,
                 unique_B, B_to_idx, p_I, target_masks=None):
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
        self.target_masks     = target_masks if target_masks is not None else []

    def forward(self, x):
        """(batch, 2n) float tensor → (batch, n) Q-value tensor."""
        n    = self.n
        pow3 = self.POW3            # already int64 (solve_query_mdp_exact) -- no cast needed
        best = self.best_action_mask  # int32; fancy-indexing with int64 `states` is fine as-is
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


def solve_query_mdp_exact(I_decoded, C_Q=-10.0, p_I=1.0, gamma=0.99, p_F=0.0, oracle=None):
    """
    Solve the Query MDP via vectorized backward induction.

    Encoding: each unique bottleneck gets a bit index.  A knowledge state
    (K_I, K_not_I) is encoded as a base-3 integer: digit i ∈
    {0=unqueried, 1=oracle_yes, 2=oracle_no}.  Total state space: 3^n.

    Absorbing states:
      failure — K_I is not a subset of any target mask (impossible to succeed)
      success — I_hat = K_I ∪ {all unqueried bits} covers some target entirely

    Backward induction sweeps from q = n−1 down to q = 0 queried bits,
    since a state with q bits queried only depends on states with q+1 bits.
    Ties in expected value are stored as bitmasks so the full tied-action set
    is available for analysis.

    Parameters
    ----------
    I_decoded : list of lists of [inv_id, pot_id] (from decode_subsets_to_2d_nomove)
    C_Q, p_I, gamma : MDP cost/reward/discount parameters
    p_F : float (default 0.0)
        Terminal value of failure states.
    oracle : optional object with a probs_for_raw_ids(raw_ids) method returning
        per-bottleneck P(YES) floats; defaults to uniform 0.5 for every bottleneck.

    Returns
    -------
    ExactQNet with attributes: V, best_action_mask, failure, success,
                               unique_B, B_to_idx, n, POW3, p_I
    """
    unique_B = sorted(set(tuple(b) for subset in I_decoded for b in subset))
    B_to_idx = {b: i for i, b in enumerate(unique_B)}
    n        = len(unique_B)
    FULL_MASK = (1 << n) - 1

    if oracle is not None:
        raw_ids = np.array([b[0] * NUM_POT + b[1] for b in unique_B], dtype=np.intp)
        probs   = oracle.probs_for_raw_ids(raw_ids)          # shape (n,), float32
    else:
        probs = np.full(n, 0.5, dtype=np.float32)

    target_masks = np.zeros(len(I_decoded), dtype=np.int32)
    for i, subset in enumerate(I_decoded):
        mask = 0
        for b in subset:
            mask |= 1 << B_to_idx[tuple(b)]
        target_masks[i] = mask

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
    for t in target_masks:
        failure &= (K_I & t) != K_I   # True only if K_I ⊄ every target

    I_hat   = K_I | (FULL_MASK & ~K_not_I)
    success = np.zeros(N3, dtype=bool)
    for t in target_masks:
        success |= I_hat == t
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
        target_masks=target_masks.tolist(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Policy simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_human_bottleneck_mask(
    T_H: np.ndarray,
    unique_B: list,
    target: int = None,
    start: int = 0,
) -> int:
    """
    Determine which bottlenecks in unique_B the human must pass through.

    Works for two bottleneck formats:
      - Overcooked: unique_B contains [inv_id, pot_id] pairs
                    → raw_id = inv_id * NUM_POT + pot_id
                    → target defaults to CLIENT_SERVED
      - Grid:       unique_B contains (state_id,) 1-tuples
                    → raw_id = state_id
                    → target must be provided (= goal_state index)

    Parameters
    ----------
    T_H     : ndarray, human's transition matrix
    unique_B: list of bottleneck descriptors (see above)
    target  : absorbing/goal state integer ID (None → Overcooked CLIENT_SERVED)
    start   : start state index (default 0)

    Returns
    -------
    int  bitmask over unique_B (bit i set ↔ human passes through unique_B[i])
    """
    if target is None:
        target = CLIENT_SERVED
        def to_raw(b): return b[0] * NUM_POT + b[1]
    else:
        def to_raw(b): return b[0]

    G = nx.DiGraph()
    for state in range(T_H.shape[0]):
        for action in range(T_H.shape[1]):
            nxt = int(T_H[state, action])
            if nxt != state:
                G.add_edge(state, nxt)

    if not G.has_node(target) or not nx.has_path(G, start, target):
        return 0

    human_bn_raw = _dominator_bottlenecks(G, start, [target])
    human_bn_raw.add(target)

    mask = 0
    for i, b in enumerate(unique_B):
        if to_raw(b) in human_bn_raw:
            mask |= (1 << i)
    return mask


def simulate_overcooked_vi(qnet: "ExactQNet", human_mask: int) -> int:
    """
    Simulate the Strategic VI policy (ExactQNet) against one human model,
    counting queries until the hypothesis space is uniquely determined.

    Termination criterion: at most one achievable subset remains consistent
    with the queries answered so far (same criterion as simulate_overcooked_info_gain).
    The qnet's policy selects WHICH bottleneck to query at each step.

    Parameters
    ----------
    qnet        : ExactQNet returned by solve_query_mdp_exact
    human_mask  : bitmask from get_human_bottleneck_mask

    Returns
    -------
    int  number of queries asked before the goal is uniquely determined
    """
    n            = qnet.n
    POW3         = qnet.POW3
    target_masks = qnet.target_masks
    FULL_MASK    = (1 << n) - 1
    K_I          = 0
    K_not        = 0
    count        = 0

    for _ in range(n + 1):
        # Stop when hypothesis space is uniquely determined
        consistent = [t for t in target_masks
                      if (K_I & t) == K_I and (K_not & t) == 0]
        if len(consistent) <= 1:
            break

        state_idx = int(np.dot(
            np.array([(((K_I >> i) & 1) + 2 * ((K_not >> i) & 1)) for i in range(n)],
                     dtype=np.int64),
            POW3
        ))
        if qnet.failure[state_idx]:
            break

        action_bitmask = int(qnet.best_action_mask[state_idx])
        if action_bitmask == 0:
            # No policy action — fall back to first unqueried bottleneck
            used = K_I | K_not
            unqueried = [i for i in range(n) if not ((used >> i) & 1)]
            if not unqueried:
                break
            action_idx = unqueried[0]
        else:
            action_idx = (action_bitmask & -action_bitmask).bit_length() - 1

        if (human_mask >> action_idx) & 1:
            K_I   |= (1 << action_idx)
        else:
            K_not |= (1 << action_idx)
        count += 1

    return count


def simulate_overcooked_info_gain(I_decoded: list, human_mask: int) -> int:
    """
    Simulate the greedy maximum-information-gain policy against one human model.

    At each step pick the unqueried bottleneck that maximises Shannon entropy
    reduction (uniform prior over consistent hypotheses).

    Parameters
    ----------
    I_decoded  : list of lists of [inv_id, pot_id] (from decode_subsets_to_2d_nomove)
    human_mask : bitmask from get_human_bottleneck_mask

    Returns
    -------
    int  number of queries asked before termination
    """
    unique_B   = sorted(set(tuple(b) for subset in I_decoded for b in subset))
    n          = len(unique_B)
    FULL_MASK  = (1 << n) - 1

    target_masks = []
    for subset in I_decoded:
        m = 0
        for b in subset:
            m |= 1 << unique_B.index(tuple(b))
        target_masks.append(m)

    K_I   = 0
    K_not = 0
    count = 0

    for _ in range(n + 1):
        used      = K_I | K_not
        unqueried = FULL_MASK & ~used
        I_hat     = K_I | unqueried

        # success: I_hat exactly equals one target
        if any(I_hat == t for t in target_masks):
            break

        consistent = [t for t in target_masks
                      if (K_I & t) == K_I and (K_not & t) == 0]
        if len(consistent) <= 1:
            break

        N = len(consistent)

        best_ig  = -1.0
        best_bit = -1
        for i in range(n):
            if (used >> i) & 1:
                continue
            yes_h = [t for t in consistent if (t >> i) & 1]
            no_h  = [t for t in consistent if not ((t >> i) & 1)]
            p_yes = len(yes_h) / N
            p_no  = len(no_h)  / N

            def _h(k):
                return -np.log2(1.0 / k) if k > 0 else 0.0

            ig = np.log2(N) - (p_yes * _h(len(yes_h)) + p_no * _h(len(no_h)))
            if ig > best_ig:
                best_ig  = ig
                best_bit = i

        if best_bit == -1:
            break

        if (human_mask >> best_bit) & 1:
            K_I   |= (1 << best_bit)
        else:
            K_not |= (1 << best_bit)
        count += 1

    return count


def _build_target_masks(I_decoded):
    """Helper: convert I_decoded list of bottleneck-tuples into (unique_B, target_masks, n)."""
    unique_B     = sorted(set(tuple(b) for subset in I_decoded for b in subset))
    n            = len(unique_B)
    idx          = {b: i for i, b in enumerate(unique_B)}
    target_masks = []
    for subset in I_decoded:
        m = 0
        for b in subset:
            m |= 1 << idx[tuple(b)]
        target_masks.append(m)
    return unique_B, target_masks, n


def _build_dominance(target_masks, n):
    """
    Build the dominance relation for Hypothesis 2 (Structural Redundancy).

    dominates[b2] = list of b1 such that b1 ⪯ b2, i.e.
        ∀ϕ ∈ Φ : b1 ∈ ϕ  ⟹  b2 ∈ ϕ
    Semantics: if b2 ∉ IG (oracle answers No), then b1 ∉ IG too — for free.
    """
    dominates: dict[int, list[int]] = {b2: [] for b2 in range(n)}
    for b1 in range(n):
        for b2 in range(n):
            if b1 == b2:
                continue
            # b1 ⪯ b2: no target has b1=1 and b2=0
            if not any(((m >> b1) & 1) and not ((m >> b2) & 1) for m in target_masks):
                dominates[b2].append(b1)
    return dominates


def _propagate_dominance(K_not: int, dominates: dict, n: int) -> int:
    """Transitively propagate negative responses via dominance entailments."""
    changed = True
    while changed:
        changed = False
        for b2 in range(n):
            if (K_not >> b2) & 1:
                for b1 in dominates[b2]:
                    if not ((K_not >> b1) & 1):
                        K_not |= (1 << b1)
                        changed = True
    return K_not


def simulate_overcooked_transition(
        qnet: "ExactQNet",
        I_decoded: list,
        human_mask: int) -> int:
    """
    Simulate the Transition condition (Hypothesis 2: Structural Redundancy).

    Uses the same Strategic VI policy as the baseline for query *selection*,
    but propagates the bottleneck-dominance entailment
        b2 ∉ IG  ⟹  b1 ∉ IG  (whenever  ∀ϕ, b1 ∈ ϕ ⟹ b2 ∈ ϕ)
    after every negative oracle response, ruling out dominated bottlenecks
    for *free* (without incrementing the query counter).

    Parameters
    ----------
    qnet       : ExactQNet from solve_query_mdp_exact
    I_decoded  : list of bottleneck-subset lists (for dominance graph)
    human_mask : bitmask from get_human_bottleneck_mask

    Returns
    -------
    int  number of paid queries until unique disambiguation
    """
    _, target_masks, n = _build_target_masks(I_decoded)
    dominates = _build_dominance(target_masks, n)

    POW3         = qnet.POW3
    qnet_targets = qnet.target_masks
    K_I          = 0
    K_not        = 0
    count        = 0

    for _ in range(n + 1):
        consistent = [t for t in qnet_targets
                      if (K_I & t) == K_I and (K_not & t) == 0]
        if len(consistent) <= 1:
            break

        state_idx = int(np.dot(
            np.array([(((K_I >> i) & 1) + 2 * ((K_not >> i) & 1)) for i in range(n)],
                     dtype=np.int64),
            POW3
        ))
        if qnet.failure[state_idx]:
            break

        action_bitmask = int(qnet.best_action_mask[state_idx])
        if action_bitmask == 0:
            used      = K_I | K_not
            unqueried = [i for i in range(n) if not ((used >> i) & 1)]
            if not unqueried:
                break
            action_idx = unqueried[0]
        else:
            action_idx = (action_bitmask & -action_bitmask).bit_length() - 1

        if (human_mask >> action_idx) & 1:
            K_I   |= (1 << action_idx)
        else:
            K_not |= (1 << action_idx)
            # Propagate dominance entailments — no additional query cost
            K_not = _propagate_dominance(K_not, dominates, n)
        count += 1

    return count


def simulate_overcooked_proximity(
        I_decoded: list,
        human_mask: int,
        v_star_per_bn: "np.ndarray") -> int:
    """
    Simulate the Proximity condition (Hypothesis 3: Goal Proximity).

    Rather than a uniform prior over consistent hypotheses, uses a
    temperature-τ=1 softmax prior weighted by the size-normalised
    average of V*_MR over each hypothesis's bottlenecks:

        ψ(ϕ) = (1/|ϕ|) Σ_{b∈ϕ} V*_MR(b)
        p_τ(ϕ) ∝ exp(ψ(ϕ)/τ),  τ = 1

    Query selection: pick s⋆ = argmax_s  P_τ(s ∈ IG | consistent).

    Parameters
    ----------
    I_decoded      : list of bottleneck-subset lists
    human_mask     : bitmask from get_human_bottleneck_mask
    v_star_per_bn  : 1-D array of length len(unique_B); v_star_per_bn[i]
                     = V*_MR evaluated at the i-th bottleneck in sorted(unique_B)

    Returns
    -------
    int  number of queries until unique disambiguation
    """
    _, target_masks, n = _build_target_masks(I_decoded)
    FULL_MASK = (1 << n) - 1

    K_I   = 0
    K_not = 0
    count = 0

    for _ in range(n + 1):
        used      = K_I | K_not
        unqueried = FULL_MASK & ~used
        I_hat     = K_I | unqueried

        if any(I_hat == t for t in target_masks):
            break

        consistent = [t for t in target_masks
                      if (K_I & t) == K_I and (K_not & t) == 0]
        if len(consistent) <= 1:
            break

        # ψ(ϕ) for each consistent hypothesis
        psi = np.array([
            np.mean([v_star_per_bn[i] for i in range(n) if (t >> i) & 1])
            if any((t >> i) & 1 for i in range(n)) else 0.0
            for t in consistent
        ])
        psi -= psi.max()                       # numerical stability
        weights = np.exp(psi)
        weights /= weights.sum()

        # P_τ(s ∈ IG)
        p_in_ig = np.zeros(n)
        for wi, t in zip(weights, consistent):
            for i in range(n):
                if (t >> i) & 1:
                    p_in_ig[i] += wi

        best_bit = -1
        best_p   = -1.0
        for i in range(n):
            if not ((used >> i) & 1) and p_in_ig[i] > best_p:
                best_p   = p_in_ig[i]
                best_bit = i

        if best_bit == -1:
            break

        if (human_mask >> best_bit) & 1:
            K_I   |= (1 << best_bit)
        else:
            K_not |= (1 << best_bit)
        count += 1

    return count


def simulate_overcooked_frequency(I_decoded: list, human_mask: int) -> int:
    """
    Simulate Hypothesis 4 (Query Frequency / greedy maximum overlap).

    At each step selects  s⋆ = argmax_s |{ϕ ∈ Φ(B, KI) : s ∈ ϕ}|,
    i.e.\ the unqueried bottleneck appearing in the most consistent
    hypotheses — equal to argmax_s P(s ∈ IG) under a uniform prior.

    Parameters
    ----------
    I_decoded  : list of bottleneck-subset lists
    human_mask : bitmask from get_human_bottleneck_mask

    Returns
    -------
    int  number of queries until unique disambiguation
    """
    _, target_masks, n = _build_target_masks(I_decoded)
    FULL_MASK = (1 << n) - 1

    K_I   = 0
    K_not = 0
    count = 0

    for _ in range(n + 1):
        used      = K_I | K_not
        unqueried = FULL_MASK & ~used
        I_hat     = K_I | unqueried

        if any(I_hat == t for t in target_masks):
            break

        consistent = [t for t in target_masks
                      if (K_I & t) == K_I and (K_not & t) == 0]
        if len(consistent) <= 1:
            break

        # Count occurrences of each unqueried bottleneck in consistent hypotheses
        freq = np.zeros(n, dtype=int)
        for t in consistent:
            for i in range(n):
                if ((t >> i) & 1) and not ((used >> i) & 1):
                    freq[i] += 1

        best_bit = int(np.argmax(freq))
        if freq[best_bit] == 0:
            break

        if (human_mask >> best_bit) & 1:
            K_I   |= (1 << best_bit)
        else:
            K_not |= (1 << best_bit)
        count += 1

    return count


def compute_v_star_grid(next_states: "np.ndarray", goal_idx: int,
                        gamma: float = 0.99) -> "np.ndarray":
    """
    Compute V*_MR for a deterministic grid MDP via reverse BFS from goal_idx.

    V*(s) = γ^d(s) where d(s) = shortest-path distance from s to goal_idx.
    Unreachable states get V*(s) = 0.

    Parameters
    ----------
    next_states : (n_states, n_actions) int array — next_states[s,a] = s'
    goal_idx    : absorbing goal state index
    gamma       : discount factor

    Returns
    -------
    np.ndarray of shape (n_states,) with V*(s) values in [0, 1]
    """
    from collections import deque
    n_states, n_actions = next_states.shape
    # Build reverse adjacency
    rev: list[list[int]] = [[] for _ in range(n_states)]
    for s in range(n_states):
        for a in range(n_actions):
            s_next = int(next_states[s, a])
            if s_next != s:
                rev[s_next].append(s)

    dist = np.full(n_states, np.inf)
    dist[goal_idx] = 0
    queue = deque([goal_idx])
    while queue:
        s = queue.popleft()
        for s_prev in rev[s]:
            if dist[s_prev] == np.inf:
                dist[s_prev] = dist[s] + 1
                queue.append(s_prev)

    return np.where(np.isfinite(dist), np.power(gamma, dist), 0.0)


def simulate_overcooked_random(I_decoded: list, human_mask: int,
                                rng: "np.random.Generator") -> int:
    """
    Simulate a uniformly-random query policy: at each step pick an unqueried
    bottleneck at random.  Uses the same termination criterion as Info~Gain.

    Parameters
    ----------
    I_decoded  : list of lists of bottleneck tuples
    human_mask : bitmask from get_human_bottleneck_mask
    rng        : numpy random Generator (for reproducibility)

    Returns
    -------
    int  number of queries asked before termination
    """
    unique_B = sorted(set(tuple(b) for subset in I_decoded for b in subset))
    n        = len(unique_B)
    FULL_MASK = (1 << n) - 1

    target_masks = []
    for subset in I_decoded:
        m = 0
        for b in subset:
            m |= 1 << unique_B.index(tuple(b))
        target_masks.append(m)

    K_I   = 0
    K_not = 0
    count = 0

    for _ in range(n + 1):
        used      = K_I | K_not
        unqueried = FULL_MASK & ~used
        I_hat     = K_I | unqueried

        if any(I_hat == t for t in target_masks):
            break

        consistent = [t for t in target_masks
                      if (K_I & t) == K_I and (K_not & t) == 0]
        if len(consistent) <= 1:
            break

        candidates = [i for i in range(n) if not ((used >> i) & 1)]
        if not candidates:
            break
        bit = int(rng.choice(candidates))

        if (human_mask >> bit) & 1:
            K_I   |= (1 << bit)
        else:
            K_not |= (1 << bit)
        count += 1

    return count
