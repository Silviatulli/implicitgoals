"""
overcooked_env.py
=================
Game backend: state encodings and transition-matrix builders.

This module is the only place where the *rules of the game* live — recipes,
the inventory / pot bit-packing, the kitchen grid, and which states carry a
serving edge.  It produces transition matrices and nothing else.

Everything computed *from* a transition matrix (the raw bottleneck union
B_nofilter, the filtered query set B, the maximally achievable subsets I,
I_array, the Query MDP) lives in
the shared, game-agnostic bottlenecks.py at the repo root, which works on the
matrices alone and has no notion of a recipe.

Configuration
-------------
One MDP, with an `allow_drop` variant.  The agent does not move: the kitchen is
abstracted away and a state is just what the agent holds and what is in the pot,
packed as inv * NUM_POT + pot.

    T_base, RECIPES = build_transition_matrix_nomove(allow_drop)
    T_R, T_H_list   = serving_matrices_nomove(T_base, RECIPES)
    start_state = 0
    goal_state  = CLIENT_SERVED

build_stochastic_matrix lifts T_R into the P(s'|s,a) form Hypothesis 3's value
iteration needs — one-hot, since this game is deterministic by construction —
pruned to the states reachable from the start.

Robot vs. human matrices
------------------------
T_R       robot matrix — a serving edge from *every* completed recipe.
T_H_list  one matrix per candidate human — a serving edge for that human's
          recipe only.  This is the only place recipes enter the pipeline: from
          here on, a "human" is just another transition matrix.

Both are handed to bottlenecks.py together with `start_state` and `goal_state`.

Notation
--------
T            transition matrix: T[state, action] → next_state  (ints throughout).
inv          bit-packed int: inventory state.  Bit layout (8 bits):
             bit 0 = plate flag, bit 1 = cooked flag,
             bits 2-3 = onion count, bits 4-5 = tomato count, bits 6-7 = mushroom count.
pot          bit-packed int: pot state.  Same bit layout as inv
             (bit 0 = served flag, bit 1 = cooking flag).
"""

import numpy as np
from tqdm import tqdm
from collections import deque


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

# ── Action indices ────────────────────────────────────────────────────────────
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
# 1.  No-movement MDP
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
    matrix — serving edges are added by serving_matrices_nomove).

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
    return T, RECIPES


def serving_matrices_nomove(T_base, recipes):
    """
    Add serving edges to a no-movement base matrix, one robot + one human each.

    T_R    : SERVE_ACTION leads to CLIENT_SERVED from *any* of the recipe-done
             states — the robot is happy to deliver whatever soup is ready.
    T_H[i] : only recipes[i]'s done-state has a serving edge — candidate human i
             accepts that soup and no other.

    All matrices are independent copies of T_base (shape unchanged).

    Returns
    -------
    T_R      : ndarray
    T_H_list : list[ndarray]   one per entry of `recipes`, in the same order
    """
    T_R = T_base.copy()
    for r in recipes:
        T_R[r * NUM_POT, SERVE_ACTION] = CLIENT_SERVED

    T_H_list = []
    for r in recipes:
        T_h = T_base.copy()
        T_h[r * NUM_POT, SERVE_ACTION] = CLIENT_SERVED
        T_H_list.append(T_h)

    return T_R, T_H_list


# ─────────────────────────────────────────────────────────────────────────────
# 2.  The robot's stochastic model  (Hypothesis 3)
# ─────────────────────────────────────────────────────────────────────────────

def build_stochastic_matrix(next_states, start_state=0):
    """Overcooked's T_R_sto — the deterministic counterpart of
    gridworld_core.build_stochastic_matrix.

    DEAD CODE — nothing in the pipeline calls this any more.  It existed to feed
    bottlenecks.value_iteration, which fed H3; H3 is now Euclidean distance to
    the goal, which Overcooked has no geometry for, so the game has no H3 column
    at all.  Kept rather than deleted because it still works; nothing exercises
    it, so treat it as untested from here on.

    Overcooked is deterministic by construction: build_transition_matrix_nomove
    emits T[state, action] -> next_state, with no slip and no MDP object behind
    it.  Its transition "distribution" is a point mass, so T_R_sto has entries
    in {0, 1} only.

    Pruning is what makes that matrix affordable.  The kitchen enumerates ~38k
    states but only a few hundred are reachable from the start, so the dense
    lift drops from tens of GB to about one MB.

    Returns
    -------
    T_R_sto : (n_reachable, n_actions, n_reachable) float64, one-hot.
    index   : dict {full state ID -> row of T_R_sto}, the bijection back to the
              38 417-state space the bottleneck pipeline names states in.
    """
    T = np.asarray(next_states)
    if T.ndim != 2:
        raise ValueError(f"expected (n_states, n_actions); got {T.shape}")
    if not np.issubdtype(T.dtype, np.integer):
        raise ValueError(f"expected integer state indices; got dtype {T.dtype}")
    n_s, n_a = T.shape
    if T.min() < 0 or T.max() >= n_s:
        raise ValueError(
            f"successor indices out of range for {n_s} states "
            f"[{T.min()}, {T.max()}]")
    if not 0 <= start_state < n_s:
        raise ValueError(f"start_state {start_state} outside [0, {n_s})")

    seen     = {int(start_state)}
    frontier = deque([int(start_state)])
    while frontier:
        s = frontier.popleft()
        for s2 in np.unique(T[s]):
            if int(s2) not in seen:
                seen.add(int(s2))
                frontier.append(int(s2))

    kept  = sorted(seen)
    index = {s: i for i, s in enumerate(kept)}
    n_r   = len(kept)

    relabel        = np.full(n_s, -1, dtype=np.int64)
    relabel[kept]  = np.arange(n_r)
    succ           = relabel[T[kept]]          # (n_r, n_a) — successors are reachable too

    T_sto = np.zeros((n_r, n_a, n_r), dtype=np.float64)
    T_sto[np.arange(n_r)[:, None], np.arange(n_a)[None, :], succ] = 1.0
    return T_sto, index
