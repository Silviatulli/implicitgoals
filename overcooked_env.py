"""
overcooked_env.py
=================
Game backend: state encodings and transition-matrix builders.

This module is the only place where the *rules of the game* live — recipes,
the inventory / pot bit-packing, the kitchen grid, and which states carry a
serving edge.  It produces transition matrices and nothing else.

Everything computed *from* a transition matrix (bottlenecks B, the filtered set
B_filter, the maximally achievable subsets I, I_array, the Query MDP) lives in
the shared, game-agnostic bottlenecks.py at the repo root, which works on the
matrices alone and has no notion of a recipe.  Decoding and plotting live in
overcooked_viz.py, DQN training in the shared query_mdp_nn.py.

Configurations
--------------
Two MDPs, each with an `allow_drop` variant:

No-movement MDP — abstract state = inv * NUM_POT + pot
    T_base, RECIPES, _ = build_transition_matrix_nomove(allow_drop)
    T_R, T_H_list      = serving_matrices_nomove(T_base, RECIPES)
    start_state = 0
    goal_state  = CLIENT_SERVED

Movement MDP — state = pos * STATE_STRIDE + facing * FACING_STRIDE
                     + inv * NUM_POT + pot
    T_R, RECIPES, grid_info = build_transition_matrix_move(grid_string, allow_drop)
    T_R, T_H_list, RECIPES, grid_info = serving_matrices_move(grid_string, allow_drop)
    start_state = grid_info['start_state']
    goal_state  = grid_info['CLIENT_SERVED_MOVE']

Robot vs. human matrices
------------------------
T_R       robot matrix — a serving edge from *every* completed recipe.
T_H_list  one matrix per candidate human — a serving edge for that human's
          recipe only.  This is the only place recipes enter the pipeline: from
          here on, a "human" is just another transition matrix.

Both are handed to bottlenecks.py together with `start_state` and
`goal_state`; the movement MDP additionally supplies an explicit `I_array`
(see move_goals) because its terminal states are the post-scoop states rather
than the direct predecessors of the absorbing state.

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

# ── Movement-MDP constants ────────────────────────────────────────────────────
# Facing directions (also used as the move-action indices 0-3)
DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT = 0, 1, 2, 3
DIR_DELTA = {DIR_UP: (-1, 0), DIR_DOWN: (1, 0), DIR_LEFT: (0, -1), DIR_RIGHT: (0, 1)}
NUM_FACING      = 4
NUM_ACTIONS_MOVE = 5  # 0=up 1=down 2=left 3=right 4=interact

# Default kitchen layout — hardcoded, passed as a default parameter to
# build_transition_matrix_move() so callers can drop in any other grid string.
# Legend: W=wall  0/1/2=onion/tomato/mushroom dispenser
#         B=bowl(plate) pile  P=pot  X=serving counter  A=agent start
DEFAULT_GRID_STR = """\
W012BPW
W     W
W A   W
W     W
WWWXWWW"""

# ── No-movement MDP action indices ────────────────────────────────────────────
ACTION_PICK_ONION  = 0
ACTION_PICK_TOMATO = 1
ACTION_PICK_MUSH   = 2
ACTION_GRAB_PLATE  = 3
ACTION_INTERACT    = 4   # also the interact index in the movement MDP
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
    return T, RECIPES, {}


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
# 1b.  Module-level no-movement matrices (built once at import — silent)
#
#  T_R         : robot matrix — SERVE_ACTION transitions to CLIENT_SERVED from
#                ANY of the 10 recipe-done states.
#  T_<OTM>     : human matrix for one specific recipe (OTM = onion-tomato-mush
#                count string, e.g. T_300, T_111, T_003).  Only that recipe's
#                done-state has a serving edge.
#
#  All 11 matrices have shape (CLIENT_SERVED+1, SERVE_ACTION+1).
#
#  RECIPES is also exported here for callers that just need the recipe list
#  without re-running build_transition_matrix_nomove.
# ─────────────────────────────────────────────────────────────────────────────

_T_BASE, RECIPES, _ = build_transition_matrix_nomove(verbose=False)

T_R, _T_H_LIST = serving_matrices_nomove(_T_BASE, RECIPES)

# recipe_id → no-movement T_H matrix (used by compare_with_query_all)
T_H_BY_RECIPE: dict = dict(zip(RECIPES, _T_H_LIST))

_T_HUMAN = {f"T_{(_r >> 2) & 3}{(_r >> 4) & 3}{(_r >> 6) & 3}": _T_h
            for _r, _T_h in zip(RECIPES, _T_H_LIST)}

T_300 = _T_HUMAN["T_300"]
T_210 = _T_HUMAN["T_210"]
T_120 = _T_HUMAN["T_120"]
T_030 = _T_HUMAN["T_030"]
T_201 = _T_HUMAN["T_201"]
T_111 = _T_HUMAN["T_111"]
T_021 = _T_HUMAN["T_021"]
T_102 = _T_HUMAN["T_102"]
T_012 = _T_HUMAN["T_012"]
T_003 = _T_HUMAN["T_003"]

del _T_BASE, _T_HUMAN


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Movement MDP
# ─────────────────────────────────────────────────────────────────────────────

def build_transition_matrix_move(grid_string=DEFAULT_GRID_STR,
                                  allow_drop: bool = False,
                                  verbose: bool = True,
                                  recipes=None):
    """
    Build the transition matrix for the movement MDP.

    State encoding
    --------------
    A state is a 4-tuple (pos_idx, facing, inv, pot) flattened as:
        state = pos_idx * (NUM_FACING * NUM_INV * NUM_POT)
              + facing  * (NUM_INV * NUM_POT)
              + inv     * NUM_POT
              + pot
    where
      pos_idx : index into the sorted list of walkable floor cells
      facing  : last movement direction — 0=up 1=down 2=left 3=right
      inv     : same bit-packed inventory encoding as the no-movement MDP
      pot     : same bit-packed pot encoding as the no-movement MDP

    Actions
    -------
    0=move_up  1=move_down  2=move_left  3=move_right  4=interact

    Movement (actions 0-3)
        New facing = action.
        If the cell in front is a walkable floor cell: move there.
        Otherwise (wall or counter): stay in place but face that direction.

    Interact (action 4)
        The agent acts on the cell in front (current facing direction):
          - Ingredient dispenser (0/1/2) : pick up if empty-handed
                                           (allow_drop) put back if holding that ingredient
          - Bowl dispenser (B)           : pick up plate if empty-handed
                                           (allow_drop) put plate back if holding one
          - Pot (P)                      : drop ingredient / turn stove on / scoop soup
          - Serving counter (X)          : deliver cooked soup → CLIENT_SERVED_MOVE

    Grid legend
    -----------
    W=wall  0/1/2=onion/tomato/mushroom dispenser
    B=bowl(plate) pile  P=pot  X=serving counter  A=agent start  ' '=walkable floor

    Parameters
    ----------
    recipes : list[int] or None
        Restrict the serving counter to these recipe ids — this is what turns the
        robot matrix into a candidate-human matrix.  None means "serve anything".

    Returns
    -------
    T_move      : ndarray, shape (CLIENT_SERVED_MOVE + 1, NUM_ACTIONS_MOVE), dtype int32
    RECIPES     : list[int]  same pot+1 encoding as no-movement MDP
    grid_info   : dict with grid layout, position index, and derived constants:
                    walkable        list of (row,col) for each pos_idx
                    pos_to_idx      dict (row,col) → pos_idx
                    NUM_POS         number of walkable cells
                    STATE_STRIDE    NUM_FACING * NUM_INV * NUM_POT
                    FACING_STRIDE   NUM_INV * NUM_POT
                    CLIENT_SERVED_MOVE  absorbing terminal state index
                    start_state     encoded initial state (agent at A, facing down)
                    objects         dict cell_char → (row, col)
                    grid            dict (row,col) → cell_char
    """
    human_recipe_ids = set(recipes) if recipes else set()
    rows = grid_string.strip().split('\n')
    grid = {}        # (row, col) → cell char
    walkable = []    # sorted list of walkable (row, col)
    agent_start_rc = None
    objects = {}     # cell_char → (row, col)

    for r, row_str in enumerate(rows):
        for c, ch in enumerate(row_str):
            if ch in (' ', 'A'):
                grid[(r, c)] = ' '
                walkable.append((r, c))
                if ch == 'A':
                    agent_start_rc = (r, c)
            elif ch == 'W':
                grid[(r, c)] = 'W'
            else:  # counter object: 0,1,2,B,P,X
                grid[(r, c)] = ch
                objects[ch] = (r, c)

    walkable.sort()
    pos_to_idx = {pos: idx for idx, pos in enumerate(walkable)}
    NUM_POS = len(walkable)

    if agent_start_rc is None:
        raise ValueError("Grid must contain an agent start cell 'A'.")

    FACING_STRIDE       = NUM_INV * NUM_POT
    STATE_STRIDE        = NUM_FACING * FACING_STRIDE
    NUM_STATES_MOVE     = NUM_POS * STATE_STRIDE
    CLIENT_SERVED_MOVE  = NUM_STATES_MOVE

    start_pos_idx  = pos_to_idx[agent_start_rc]
    start_state    = start_pos_idx * STATE_STRIDE + DIR_DOWN * FACING_STRIDE  # inv=0, pot=0

    T = np.empty((NUM_STATES_MOVE + 1, NUM_ACTIONS_MOVE), dtype=np.int32)
    for s in range(NUM_STATES_MOVE + 1):
        T[s] = s

    RECIPES = []
    item_set = set(ITEM_MAP)

    for state in range(NUM_STATES_MOVE):
        pos_idx  = state // STATE_STRIDE
        rem      = state  % STATE_STRIDE
        facing   = rem    // FACING_STRIDE
        rem2     = rem    %  FACING_STRIDE
        inv      = rem2   // NUM_POT
        pot      = rem2   %  NUM_POT

        r, c = walkable[pos_idx]

        pot_cooked                       = is_cooked(pot)
        pot_onions, pot_tomato, pot_mush = ingredient_counts(pot)
        pot_total_items                  = pot_onions + pot_tomato + pot_mush
        inv_cooked                       = is_cooked(inv)

        for action in range(4):
            new_facing = action
            dr, dc = DIR_DELTA[action]
            nr, nc = r + dr, c + dc
            front = grid.get((nr, nc), 'W')
            new_pos_idx = pos_to_idx[(nr, nc)] if front == ' ' else pos_idx
            T[state, action] = (new_pos_idx * STATE_STRIDE
                                + new_facing * FACING_STRIDE
                                + inv * NUM_POT + pot)

        dr, dc  = DIR_DELTA[facing]
        fr, fc  = r + dr, c + dc
        front   = grid.get((fr, fc), 'W')

        new_inv, new_pot = inv, pot
        served = False

        if front in ('0', '1', '2'):
            item = ITEM_MAP[int(front)]
            if inv == 0:
                new_inv = item
            elif allow_drop and inv == item:
                new_inv = 0

        elif front == 'B':
            if inv == 0:
                new_inv = 1
            elif allow_drop and inv == 1:
                new_inv = 0

        elif front == 'P':
            if (inv in item_set and not inv_cooked
                    and pot_total_items < 3 and not pot_cooked
                    and not is_served_or_plated(pot)):
                new_inv = 0
                new_pot = pot + inv
            elif inv == 0 and pot_total_items == 3 and not pot_cooked:
                new_pot = pot + 2
            elif (inv == 1 and pot_cooked
                      and pot_total_items == 3 and not is_served_or_plated(pot)):
                recipe_id = pot + 1
                new_inv   = recipe_id
                new_pot   = 0
                if recipe_id not in RECIPES and (not human_recipe_ids or recipe_id in human_recipe_ids):
                    RECIPES.append(recipe_id)

        elif front == 'X':
            if inv_cooked and (not human_recipe_ids or inv in human_recipe_ids):
                served = True

        if served:
            T[state, 4] = CLIENT_SERVED_MOVE
        else:
            T[state, 4] = (pos_idx * STATE_STRIDE
                           + facing * FACING_STRIDE
                           + new_inv * NUM_POT + new_pot)

    reachable = bfs_reachable_set(T, start_state, NUM_ACTIONS_MOVE)
    for state in range(NUM_STATES_MOVE):
        if state not in reachable:
            T[state] = state

    if verbose:
        print(f"T_move built — {len(reachable)} reachable states, "
              f"{NUM_STATES_MOVE - len(reachable)} ghost states locked.")
        print(f"Cookable recipes ({len(RECIPES)}): {RECIPES}")

    grid_info = {
        'grid':              grid,
        'walkable':          walkable,
        'pos_to_idx':        pos_to_idx,
        'NUM_POS':           NUM_POS,
        'NUM_FACING':        NUM_FACING,
        'STATE_STRIDE':      STATE_STRIDE,
        'FACING_STRIDE':     FACING_STRIDE,
        'CLIENT_SERVED_MOVE': CLIENT_SERVED_MOVE,
        'start_state':       start_state,
        'agent_start':       agent_start_rc,
        'objects':           objects,
        'grid_str':          grid_string,
        'allow_drop':        allow_drop,
    }
    return T, RECIPES, grid_info


def move_goals(recipes, grid_info):
    """
    Terminal states of the movement MDP: the post-scoop state of each recipe —
    agent standing next to the pot, facing it, holding the finished dish.

    These are handed to the bottleneck extractor as its explicit `I_array`.
    They are *not* the direct predecessors of CLIENT_SERVED_MOVE (those would be
    "standing in front of the serving counter holding a dish"): the walk back up
    the dominator tree from the post-scoop state stops exactly where the recipe
    is decided, which is the granularity the rest of the pipeline expects.

    Returns
    -------
    list[int]  one state ID per recipe, in the order of `recipes`
    """
    STATE_STRIDE  = grid_info['STATE_STRIDE']
    FACING_STRIDE = grid_info['FACING_STRIDE']
    pot_r, pot_c  = grid_info['objects']['P']
    scoop_pos_idx, scoop_facing = None, None
    for (r, c), idx in grid_info['pos_to_idx'].items():
        for facing, (dr, dc) in DIR_DELTA.items():
            if (r + dr, c + dc) == (pot_r, pot_c):
                scoop_pos_idx, scoop_facing = idx, facing
                break
        if scoop_pos_idx is not None:
            break
    return [
        scoop_pos_idx * STATE_STRIDE + scoop_facing * FACING_STRIDE + recipe * NUM_POT
        for recipe in recipes
    ]


def serving_matrices_move(grid_string=DEFAULT_GRID_STR, allow_drop: bool = False,
                          verbose: bool = False):
    """
    Movement-MDP counterpart of serving_matrices_nomove.

    The serving edge cannot be patched in after the fact here (it depends on the
    agent's position and facing), so each candidate human needs its own full
    build with `recipes=[r]` — the serving counter then only accepts recipe r.

    Returns
    -------
    T_R      : ndarray   robot matrix (serving counter accepts every recipe)
    T_H_list : list[ndarray]  one per recipe, in the order of RECIPES
    RECIPES  : list[int]
    grid_info: dict      from the robot build (identical for every human)
    """
    T_R, RECIPES, grid_info = build_transition_matrix_move(
        grid_string, allow_drop, verbose=verbose
    )
    T_H_list = [
        build_transition_matrix_move(grid_string, allow_drop, verbose=False, recipes=[r])[0]
        for r in RECIPES
    ]
    return T_R, T_H_list, RECIPES, grid_info
