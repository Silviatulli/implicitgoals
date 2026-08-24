# Algorithm 1 — from subset search to clique enumeration

Three algorithms compute the same object: the paper's **Algorithm 1**, the
**implementation** in `bottlenecks.py` (a pruned subset search over a Held–Karp
achievability test), and a **proposed rewrite** as maximal-clique enumeration.

This document states what they compute, walks each one step by step on a single
real instance, and compares their costs. Every number, table and trace below is
printed by the code, not illustrative.

---

## Contents

1. [The question](#1-the-question)
2. [The worked example](#2-the-worked-example)
3. [Three vocabularies](#3-three-vocabularies)
4. [Algorithm A — the paper's Algorithm 1](#4-algorithm-a--the-papers-algorithm-1)
5. [Algorithm B — the implementation](#5-algorithm-b--the-implementation)
6. [The observation](#6-the-observation)
7. [Algorithm C — comparability graph and cliques](#7-algorithm-c--comparability-graph-and-cliques)
8. [Complexity — the three side by side](#8-complexity--the-three-side-by-side)
9. [Validation](#9-validation)
10. [Where this stops being true](#10-where-this-stops-being-true)
11. [Reading](#11-reading)

---

## 1. The question

Algorithm 1 is handed four things:

| | |
|---|---|
| `B` | the bottlenecks — mandatory waypoints, unioned over the candidate humans |
| `T_R` | the robot's determinized transition matrix (`M^R` in the paper) |
| `start_state` | where every trajectory begins |
| `goal_state` | the absorbing state every successful trajectory ends in |

A subset `S ⊆ B` is **achievable** when some ordering of it can actually be walked:

> there is an order `s₁, s₂, …, s_k` of `S` such that the start reaches `s₁`, each
> `s_i` reaches `s_{i+1}`, and the last one still reaches the goal.

The travel *between* two consecutive bottlenecks is unrestricted — the robot may
wander anywhere, including across bottlenecks it was not asked to visit. **This
permissiveness is the hinge everything in §6 turns on.**

Algorithm 1 returns `I`, the **maximally** achievable subsets: those no further
bottleneck can be added to. Each one is a hypothesis about what the human wants.

### Notation used throughout

| symbol | meaning |
|---|---|
| `n` | `|B|`, the number of bottlenecks |
| `ℓ` | how many of them the start can reach at all ("live") |
| `A` | the number of **achievable** subsets |
| `k` | `|I|`, the number of **maximally** achievable subsets — the answer size |

`A ≥ k` always, and §8 shows `A/k` reaching `2^ℓ`.

---

## 2. The worked example

A 9×9 board, 3×3 rooms of 3×3 cells, joined by **one-way** doors (`>` east,
`v` south; `|` and `-` are sealed). `#` is an obstacle.

```
S . .|. . .|. # .
. . .>. . .>. . .
. . .|. . .|. # .
- v - - v - - v -
. . .|. . .|. . .
. # .>. . .>. . .
. . .|. . .|. . .
- v - - v - - v -
. . .|. . .|. . .
. . .>. . .>. . .
. . .|. . .|. . G
```

The dominator analysis returns three bottleneck sets, one per candidate human,
and `B` is their **union**:

```
human 1: [(1,2), (1,3), (8,8)]
human 2: [(8,8)]
human 3: [(2,1), (3,1), (8,8)]
─────────────────────────────────
union B: [(1,2), (1,3), (2,1), (3,1), (8,8)]     ids [11, 12, 19, 28, 80]
```

Human 2 has no mandatory waypoint but the goal — its route is unconstrained. The
other two each commit to one door. `B` is the union, so the robot may ask about
any waypoint any human might care about.

Reading the five off the board:

- `(1,2)` and `(2,1)` are both **inside the top-left room** — the near sides of its
  two exits.
- `(1,3)` is the far side of the **east** door.
- `(3,1)` is the far side of the **south** door.
- `(8,8)` is the goal.

The reachability table (`reaches[i][j]` = *i can reach j*):

```
          (1,2)   (1,3)   (2,1)   (3,1)   (8,8)
  (1,2)       1       1       1       1       1
  (1,3)       0       1       0       0       1
  (2,1)       1       1       1       1       1
  (3,1)       0       0       0       1       1
  (8,8)       0       0       0       0       1
```

Two facts are visible in that table, and they are the entire instance:

1. `(1,2)` and `(2,1)` reach **each other** — they share a room, and movement
   inside a room is reversible. Reachability here is a *preorder*, not a partial
   order.
2. `(1,3)` and `(3,1)` reach **neither** each other. Stepping through the east door
   commits you: you cannot come back west, so the south door is gone for ever, and
   vice versa.

So there are exactly two ways to play this board — **go east** or **go south** —
and that is what all three algorithms have to discover.

---

## 3. Three vocabularies

The same question gets asked in three languages below. They are worth separating
up front.

**Orderings.** "Is there a sequence visiting all of `S`?" This is the definition in
§1 and what `CheckAchievability` is named after.

**Hamiltonian path.** Build a graph whose vertices are `S` and whose arrows are
"can reach". An ordering is then a route through that graph visiting **every
vertex exactly once** — a *Hamiltonian path*. On a general graph, finding one is
NP-complete: there is no local rule for which vertex to take next, a choice that
looks fine can strand you later, and you are searching `n!` orderings. (Contrast
the *Eulerian* path, which visits every **edge** once — Euler solved that in 1736
with a rule you can check in linear time. Two problems that sound alike; one is
trivial, one is intractable.)

**Cliques.** A **clique** is a set of vertices in an *undirected* graph where
**every pair** is joined by an edge — the social sense of the word: a group where
everyone knows everyone.

```
      B                 A───B
     ╱ ╲                │╲ ╱│         A───B
    A───C               │ ╳ │         │   │
                        │╱ ╲│         │   │
  clique of 3           C───D         C───D
  (all 3 pairs)      clique of 4    NOT a clique
                     (all 6 pairs)  (A–D, B–C missing)
```

Two refinements matter here:

- **maximal** — no vertex can be added without breaking it;
- **maximum** — the largest clique in the whole graph.

Algorithm 1 wants **all the maximal ones**, not the maximum one:

```
      B
     ╱ ╲          {A,B,C}  size 3 — maximal and maximum
    A───C
    │               {A,D}  size 2 — maximal (D's only neighbour is A),
    D                              but not maximum
```

`{A,D}` cannot be extended, so it is a genuine maximal clique even though a bigger
one exists elsewhere. That is exactly right for this problem: a hypothesis with
fewer subgoals is not an inferior version of a larger hypothesis, it is a
**different** hypothesis — a human whose route forces three waypoints and one whose
route forces five are two different people to tell apart.

§6 shows the third vocabulary is the correct one, and §7 uses it.

---

## 4. Algorithm A — the paper's Algorithm 1

```
 1: Input: M^R, B
 2: Output: Set of maximal achievable subsets I
 3: function FindMaximalAchievableSubsets(M^R, B)
 4:     return GenerateSubsets(0, ∅, B, M^R)
 5: end function
 6: function GenerateSubsets(index, current_subset, B, M^R)
 7:     if index = |B| then
 8:         if ¬CheckAchievability(current_subset, M^R) then
 9:             return ∅
10:         end if
11:         maximal_subset ← current_subset
12:         for i from |current_subset| to |B| - 1 do
13:             new_subset ← maximal_subset ∪ B[i]
14:             if CheckAchievability(new_subset, M^R) then
15:                 maximal_subset ← new_subset
16:             end if
17:         end for
18:         return maximal_subset
19:     end if
20:     result ← result ∪ GenerateSubsets(index + 1, current_subset ∪ B[index], B, M^R)
21:     return result
22: end function
```

### What it does

Two phases. **Descend** to a leaf, building a subset one bottleneck at a time
(lines 20–21). **At the leaf**, test the subset; if it is achievable, extend it
**greedily** — walk the remaining bottlenecks in order, keeping each one that
still leaves the set achievable (lines 11–17). Return the completed set; union
across leaves.

The greedy completion is the pseudocode's distinguishing feature: it returns sets
that are already maximal, so **no separate maximality filter is needed**.

### What the code on `main` actually does

The pseudocode and the implementation
(`maximal_achievable_subsets.optimized_find_maximally_achievable_subsets` on the
`main` branch) differ, in ways that change the output. Three findings, verified by
running that branch.

**(i) The include/exclude enumeration is there.** Printed line 20 has only the
include branch, so the descent would reach a single leaf — the whole of `B`. The
code has both calls, so that is a transcription slip, not a design one.

**(ii) The leaf logic is not a greedy completion — it is a no-op.**

```python
if check_achievability_cached(frozenset(current_subset)):
    for i in range(len(current_subset), n):
        new_subset = current_subset + (B_list[i],)
        if not check_achievability_cached(frozenset(new_subset)):
            yield frozenset(current_subset)      # <- yields
            return
    yield frozenset(current_subset)              # <- yields the SAME thing
return
```

Both exits yield `frozenset(current_subset)`. Nothing is ever accumulated into a
running `maximal_subset` as pseudocode line 15 specifies, so the loop only ever
causes an early `return` and never changes what is emitted. **The result is every
achievable subset, not the maximal ones.** On the §2 board (with a stand-in
achievability oracle) it returns all **24** achievable subsets — `{}`, all five
singletons, every achievable pair — where the answer is **2**.

The loop also indexes with a cardinality (`range(len(current_subset), n)`, matching
pseudocode line 12), so the candidates it walks are positions
`|current_subset| … n−1` rather than the bottlenecks not already chosen.

**(iii) `check_achievability` rejects every non-empty subset.** This one dominates
the others. In `DeterminizedMDP`:

```python
def reward_function_for_goingthrough_all_bottleneck(self, state, action, next_state):
    if next_state[0] in self.bottleneck_states:
        return 1000
    return 0
```

`next_state[0]` is the **position**; `bottleneck_states` holds **full state
tuples** (`identify_bottlenecks` appends `tuple(state)`, and `B` is built from
those). Measured on a 5×5 board:

```
a bottleneck, as stored in B:  ((1, 0), (), ())
a state from the MDP        :  [(0, 0), (), ()]
state[0] — what is tested   :  (0, 0)

tests:   (0, 0)  in  {((1, 0), (), ())}     ->  False
V[init] = 0.0        max V over all states = 0.0
```

The 1000 reward is never paid anywhere, so `V ≡ 0`, and the gate

```python
if V_det[initial_state_hash] <= (len(I_prime)-1)*1000:
    return False
```

rejects `|I′| = 1` (`0 ≤ 0`) and every larger subset (`0 ≤ (m−1)·1000`), while
`|I′| = 0` passes (`0 ≤ −1000` is false). Directly confirmed:

```
check_achievability(single bottleneck) = False
check_achievability(empty set)         = True
```

**So the branch returns `I = {∅}` on every instance** — reproduced at |B| = 1 and
|B| = 4, both yielding just the empty set.

The two defects layer: (iii) masks (ii), and fixing (iii) alone would expose (ii)
— every achievable subset instead of the maximal ones. The fix for (iii) is to
compare like with like (store bottlenecks as positions, or test
`tuple(next_state)`); which convention to adopt depends on the rest of that branch.

The `yacine_exp` rewrite is unaffected: it replaced this path with dominator
extraction plus the reachability-ordering test of §5, and its `B` and its states
share a single integer ID space, so the mismatch has no analogue.

### Cost

`CheckAchievability` is the cost centre, and it is **not** a subset DP. It runs
three value iterations, the middle one over `BottleneckMDP`, whose constructor is

```python
possible_bottleneck_sets = powerset(self.bottlenecks)
for I in possible_bottleneck_sets:
    for J in self.original_mdp.get_state_space():
        self.state_space.append((J, set(I)))
```

— a state space of **2^|I′| × |S|**. One call on a subset of size `m` is therefore
a value iteration over `2^m · |S|` states: the call itself is exponential in the
subset size.

- **Leaves**: `2ⁿ`.
- **Distinct subsets tested**: `2ⁿ` (the `lru_cache` removes repeats, not the work).
- **Summed cost**: `Σ_{S ⊆ B} 2^|S| · |S|  =  3ⁿ · |S|`, since `Σ C(n,m)·2^m = 3ⁿ`.

**Algorithm A is `Θ(3ⁿ · |S| · VI-iterations)`.** The `lru_cache` is what keeps it
from being worse still: without it the greedy loop would re-run those value
iterations `O(n)` times per leaf. This is also why the 6×6 run in testing was
killed by the OOM killer rather than merely being slow — the intermediate MDP is
materialised as Python tuples.

For contrast, had `CheckAchievability` been the memoized Held–Karp DP of §5, the
same enumeration would cost `O(2ⁿ · n)`. The gap between `3ⁿ·|S|` and `2ⁿ·n` is the
price of answering a reachability question with value iteration over an augmented
MDP.

## 5. Algorithm B — the implementation

`bottlenecks.py` as it stands. It differs from §4 in three ways: the
achievability test is a memoized Held–Karp DP; the descent **prunes** unachievable
branches; and maximality is a **filter at the end** rather than a greedy completion
at each leaf.

### 5.1 How information flows

```
  T_R
   │  reachability BFS, once per bottleneck
   ▼
  from_start[b] , predecessors[b]          ← facts about SINGLE bottlenecks
   │
   ▼
  ┌──────────────────────────────────────────────────┐
  │  CheckAchievability                              │
  │  memo[S] = bitmask of endpoints of subset S      │   ← facts about SUBSETS
  │      ▲                          │                │
  │      └──────── recurses ────────┘                │      up to 2^n of them
  └──────────────────────────────────────────────────┘
   │
   ▼
  include / exclude DFS over all 2^n subsets
   │   achievable_masks
   ▼
  filter_maximal_subsets      (discard every S ⊂ some S')
   │
   ▼
  I
```

The exponential is in the **middle band**: every fact the algorithm derives is
indexed by a *subset*, so the number of facts is the number of subsets.

### 5.2 Step by step

#### Step 0 — the input `T_R`

```
shape (81, 4)  dtype int32
T_R[0] = [0, 9, 0, 1]
```

A **determinized** transition matrix: `T_R[state, action] → next_state`, plain
integers, no probabilities. 81 states (the 9×9 board), 4 actions.

Row 0 is the corner cell `(0,0)`: action 0 (up) gives state 0 — a **self-loop**,
meaning "undefined here", since you cannot go up from the top row. Action 1 (down)
gives state 9 = `(1,0)`, action 3 (right) gives state 1 = `(0,1)`. Self-loops for
undefined actions keep the array rectangular.

Everything downstream reads only this matrix — the board, the walls and the doors
have already been compiled away into these integers.

#### Step 1 — reachability BFS, once per bottleneck

`get_reachable_states(T, start)` is a vectorised breadth-first search: a boolean
frontier over states, repeatedly gathering `T[frontier]` and keeping what has not
been seen. It runs **n + 1 times** — once from the start, once from each bottleneck:

```
from start (0,0):  77 of 81 states reachable
from (1,2)      :  77 states, bottlenecks reachable: (1,2) (1,3) (2,1) (3,1) (8,8)
from (1,3)      :  51 states, bottlenecks reachable: (1,3) (8,8)
from (2,1)      :  77 states, bottlenecks reachable: (1,2) (1,3) (2,1) (3,1) (8,8)
from (3,1)      :  52 states, bottlenecks reachable: (3,1) (8,8)
from (8,8)      :   1 state,  bottlenecks reachable: (8,8)
```

The instance is already fully determined by these five rows:

- `(1,2)` and `(2,1)` each reach **77** states — they sit in the top-left room,
  before any commitment, so the whole board is still open.
- `(1,3)` reaches **51** — through the east door, the left column of rooms is gone
  for ever.
- `(3,1)` reaches **52** — symmetric, through the south door.
- `(8,8)` reaches **1** — itself. The goal absorbs.
- 81 − 77 = **4 states** are unreachable from the start: obstacles and cells sealed
  behind them.

Cost `O(n · |S| · |Act|)`. **Unchanged by the rewrite** — all three algorithms need
exactly this.

#### Step 2 — `_build_adjacency` compresses it

The 81-state reachability vectors are discarded; only the bottleneck-to-bottleneck
part is kept, as **bitmasks over `B`'s index order**:

```
from_start = [True, True, True, True, True]

predecessors[(1,2)] =  5   bits 10100   reached by: (1,2) (2,1)
predecessors[(1,3)] =  7   bits 11100   reached by: (1,2) (1,3) (2,1)
predecessors[(2,1)] =  5   bits 10100   reached by: (1,2) (2,1)
predecessors[(3,1)] = 13   bits 10110   reached by: (1,2) (2,1) (3,1)
predecessors[(8,8)] = 31   bits 11111   reached by: everything
```

`predecessors[j]` is a Python int whose bit `i` is set iff **bottleneck `i` can
reach bottleneck `j`**. It is stored by *destination* deliberately: the question
asked a million times downstream is "of the places I could be standing, does any
of them reach `j`?", and with both sides as bitmasks that is one `&`.

`predecessors[(8,8)] = 31 = 0b11111` — every bottleneck reaches the goal. That is
the universal-bottleneck property.

**From here on the board no longer exists.** Five integers and a boolean list are
the entire problem.

#### Step 3 — CheckAchievability, the memo

This is the Held–Karp DP. `memo[S]` is not a yes/no; it is the bitmask of **every
bottleneck at which some valid ordering of `S` can finish**:

```
memo[S] = { j ∈ S : some valid ordering of S ends at j }
```

built by asking where you were one step earlier:

> `S` can end at `b` **iff** `S \ {b}` can end at some `v`, and `v` reaches `b`.

```python
prev_ends = memo[prev_mask]
if prev_ends == -1:                        # S was a singleton
    if from_start[b]:
        ends |= low
elif prev_ends & predecessors[b]:          # someone I could be standing on reaches b
    ends |= low
```

The complete recurrence, base case included:

```
memo(∅)    = ⊥                                       sentinel: no endpoint yet
memo({j})  = {j}  if  start → j,  else ∅
memo(S)    = { j ∈ S : ∃v ∈ memo(S\{j}),  v → j }    for |S| ≥ 2

S achievable  ⟺  memo(S) ≠ ∅
```

The base case is load-bearing. The first bottleneck of an order has no predecessor
*bottleneck* to be reached from — it is reached from the **start**, a different
relation (`from_start`, not `predecessors`). Drop it and `memo({j})` asks for a
`v ∈ memo(∅)`, which is vacuously false, so every singleton comes back empty and by
induction the whole table collapses to ∅. Measured on this instance: **23 of 31
subsets wrong** without the base case, 0 with it.

That is also why the code carries a `-1` sentinel: `memo[∅] = -1` and `memo[S] = 0`
must be distinguishable — **0 means "no valid endpoint, `S` is unachievable"**,
while **∅ means "coverable, but there is no last-visited node yet"**.

Asked for every subset, the memo fills all 32 entries. The interesting ones are the
**8 that come back 0**:

```
{(1,3),(3,1)}                        {(1,2),(1,3),(2,1),(3,1)}
{(1,2),(1,3),(3,1)}                  {(1,2),(1,3),(3,1),(8,8)}
{(1,3),(2,1),(3,1)}                  {(1,3),(2,1),(3,1),(8,8)}
{(1,3),(3,1),(8,8)}                  {(1,2),(1,3),(2,1),(3,1),(8,8)}
```

**All eight contain both `(1,3)` and `(3,1)`** — and `8 = 2³` is exactly the number
of subsets containing a fixed pair out of five elements. There is one incomparable
pair on this board, and the DP rediscovers that same fact eight separate times,
once per superset. **That redundancy is the inefficiency, made visible.**

##### Why this is Held–Karp

A *state* in a dynamic program is one subproblem — one cell you compute and store.
The design question is always: what is the smallest description of "where I am"
that suffices to decide what comes next? For sequencing, the answer is two things:

| | | count |
|---|---|---|
| **which** nodes are visited | a subset `S ⊆ B` | `2ⁿ` |
| **where** you are standing | an endpoint `j ∈ S` | up to `n` |

giving `2ⁿ × n` cells. The saving over brute force is that `S` is a **set, not a
sequence**: `A→B→C` and `B→A→C` land in the same cell `({A,B,C}, C)` and are
computed once. Against the `(n−1)!` orderings:

| n | orderings `(n−1)!` | DP states `2ⁿ·n` |
|---|---|---|
| 7 | 720 | 896 |
| 10 | 362,880 | 10,240 |
| 17 | 2.1 × 10¹³ | 2.2 × 10⁶ |
| 20 | 1.2 × 10¹⁷ | 2.1 × 10⁷ |
| 25 | 6.2 × 10²³ | 8.4 × 10⁸ |

(The DP is *worse* below about n = 8 — factorials start small. It wins by ten
orders of magnitude by n = 20.)

`bottlenecks.py` does not allocate `2ⁿ × n` separate cells: it stores **one entry
per subset**, holding all `n` endpoint answers packed as bits — the same
`2ⁿ · n` bits of information, one Python integer per subset. That is why testing
all `n` candidate predecessors is the single `&` above. **The `2ⁿ` is the part that
cannot be packed away**, and it is the entry count that grows: a run touching most
subsets at n = 17 holds ~131,000 entries. That is exactly what
`--max-bottlenecks 17` was protecting.

#### Step 4 — the include/exclude DFS

```python
def generate_subsets(index, current_mask):
    if index == n:
        if completable(current_mask):
            achievable_masks.append(current_mask)
        return
    generate_subsets(index + 1, current_mask)                  # exclude bit `index`
    new_mask = current_mask | (1 << index)
    if check_sequential_achievability(new_mask, ...):          # include, if it survives
        generate_subsets(index + 1, new_mask)
```

A binary tree over bit positions: at each level, either leave bottleneck `index`
out or put it in. The **exclude branch is always taken**; the **include branch only
if the enlarged set is still achievable**. That single `if` is the only pruning in
the algorithm, and it is what §4 lacks.

On this instance: **51 calls, 28 memo entries, 4 prunes**.

Note 28 < 32 — pruning does spare a few subsets. But it prunes *late*: it stops
descending only once a set has already gone unachievable, so it still visits every
achievable subset, and here 24 of 32 are achievable.

#### Step 5 — `completable()` at each leaf

Reaching depth `n` means a full subset has been decided. One more test before it
counts:

```python
return bool(memo[mask] & predecessors[goal_bit])
```

*Some ordering of `mask` finishes at a bottleneck that reaches the goal.* A set you
can tour but not escape from is not a hypothesis — it is a trap. Here
`predecessors[(8,8)] = 31`, so every achievable set is completable and nothing is
filtered out at this step; on a board with a sealed room it would be.

`achievable_masks` ends with **24** entries:

| size | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| count | 1 | 5 | 9 | 7 | 2 |

#### Step 6 — `filter_maximal_subsets`

24 sets in, **2** out.

```python
# sort by popcount descending so supersets are processed first
arr = arr[np.argsort(-pc, kind='stable')]
...
subsumed = np.any((batch.reshape(B,1) & kept_np.reshape(1,K)) == batch.reshape(B,1), axis=1)
```

Sort largest-first, then keep a set only if no already-kept set contains it —
`(m & k) == m` is the subset test in one operation. Batched through numpy because
at scale this list is millions long.

```
kept: {(1,2),(2,1),(3,1),(8,8)}     and     {(1,2),(1,3),(2,1),(8,8)}
```

**22 of the 24 discarded.** And they had to be generated first. This is structural,
not bad luck: the achievable family is **closed downwards** — every subset of an
achievable set is achievable, since dropping a waypoint from a valid order leaves a
valid order (transitivity again) — so the DFS is *guaranteed* to enumerate the
entire downward closure of its own output before throwing it away. 24 for 2 here;
millions for 3 at |B| = 27.

#### Step 7 — `I`

```
I = [ [(1,2), (2,1), (3,1), (8,8)],      go south
      [(1,2), (1,3), (2,1), (8,8)] ]     go east
```

`(1,2)` and `(2,1)` appear in both — they are in the top-left room, which every
route crosses, so asking about them tells the robot nothing. The bit that
discriminates is `(1,3)` versus `(3,1)`.

---

## 6. The observation

Look again at what `CheckAchievability` asks, and what it asks it *about*.

Hamiltonian path is hard because the relation between vertices can be arbitrary:
knowing `a → b` and `b → c` tells you nothing about `a` and `c`, so the visiting
order genuinely matters and has to be searched.

But the relation here is **graph reachability**, and reachability is
**transitive**: if the robot can get from `a` to `b`, and from `b` to `c`, then it
can get from `a` to `c` — concatenate the two walks. This is exactly where "travel
between bottlenecks is unrestricted" earns its keep: the concatenated walk is legal
because nothing forbids passing through whatever lies in between.

Call two bottlenecks **comparable** when one reaches the other. Then:

> **A subset is achievable ⟺ every pair in it is comparable** (and every element is
> reachable from the start).

**(⟹)** Given a valid order `s₁ … s_k` and any `i < j`, chain the hops
`s_i ⇝ s_{i+1} ⇝ … ⇝ s_j` and compose: `s_i ⇝ s_j`. Every pair is comparable.
*This direction is the one that needs transitivity.*

**(⟸)** Given that all pairs are comparable, build the order: quotient by mutual
reachability (the strongly connected components — here, the top-left room holding
`(1,2)` and `(2,1)`), topologically sort the components, and list each component's
members in any order. Consecutive elements are either in the same component
(mutually reachable ✓) or in adjacent components in the right direction ✓.
*This direction needs no transitivity at all: it is Rédei's theorem (1934), that
every tournament has a Hamiltonian path — applied to a semicomplete digraph by
keeping one arc per pair.*

Drop transitivity and (⟹) fails at once: with `a → b`, `b → c` and no relation
between `a` and `c`, the order `a, b, c` is valid while `{a, c}` is incomparable.
That is the case a general Hamiltonian-path instance can present, and a
reachability relation cannot.

**Achievability was never a property of subsets. It is a property of pairs.**

And "every pair is joined" is the definition of a clique — so the achievable
subsets are the cliques of the comparability graph, and the maximally achievable
ones are its **maximal cliques**.

---

## 7. Algorithm C — comparability graph and cliques

### 7.1 How information flows now

```
  T_R
   │  reachability BFS, once per bottleneck      ← unchanged
   ▼
  from_start[b] , predecessors[b]
   │
   ▼
  comparability graph:  adj[i] ∋ j  iff  i ⇝ j  or  j ⇝ i      ← facts about PAIRS
   │                                                              n²/2 of them
   ▼
  Bron–Kerbosch (maximal cliques, with pivoting)
   │                                              ← recursion over the OUTPUT
   ▼
  keep the cliques containing the goal
   │
   ▼
  I
```

The middle band is gone. Nothing is indexed by a subset. The recursion that remains
explores the answer, not the search space — and no maximality filter is needed,
because Bron–Kerbosch emits only maximal cliques in the first place.

### 7.2 Step by step

#### Steps 1–2 — reachability BFS and `_build_adjacency`

**Unchanged.** Byte for byte the same code, producing the same `from_start` and
`predecessors`. This matters more than it looks: **the rewrite adds no new
information.** Everything it needs was already in `predecessors` — the old
algorithm just was not reading it the right way round.

#### Step 3 — the comparability graph

```python
live = sum(1 << i for i in range(n) if from_start[i])
for i in range(n):
    for j in range(i + 1, n):
        if (live >> j) & 1 and ((predecessors[j] >> i & 1)
                                or (predecessors[i] >> j & 1)):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
```

Two things happen here, and both are the point.

**`live` drops the unreachable.** A bottleneck the start cannot reach belongs to no
achievable subset at all. Here all five survive; on a board with a trap, the
trapped waypoints are excluded — still queryable, never achievable.

**The direction is thrown away.** `adj[i]` gets bit `j` if `i ⇝ j` **or** `j ⇝ i`.
The graph is *undirected*:

```
adj[(1,2)] = 30  bits 01111  ->  (1,3) (2,1) (3,1) (8,8)
adj[(1,3)] = 21  bits 10101  ->  (1,2) (2,1) (8,8)
adj[(2,1)] = 27  bits 11011  ->  (1,2) (1,3) (3,1) (8,8)
adj[(3,1)] = 21  bits 10101  ->  (1,2) (2,1) (8,8)
adj[(8,8)] = 15  bits 11110  ->  (1,2) (1,3) (2,1) (3,1)

missing edges: [ (1,3) — (3,1) ]
```

Discarding direction looks like losing information, and it is — but it is
information needed only to *reconstruct* the order afterwards, which a topological
sort does for free, not to *decide* achievability. That discard is what turns a
sequencing problem into a graph problem.

Ten pairs tested, nine edges, one missing: `K₅` minus an edge. Cost `n²/2`
lookups — for the 402-bottleneck instance, ~80,000, done in microseconds.

#### Step 4 — Bron–Kerbosch

**The three sets.** The recursion carries three disjoint bitmasks:

| | meaning |
|---|---|
| **`R`** | the clique built so far |
| **`P`** | candidates adjacent to *everything* in `R` — still legal extensions |
| **`X`** | vertices that would *also* extend `R`, but have already been explored |

The invariant: `R` is always a clique, and `R ∪ {v}` is a clique for every `v` in
`P` or `X`.

**`X` is what makes the output maximal.** When `P` empties, `R` cannot be extended
by anything new — but if `X` is non-empty, some *already-visited* vertex still
extends it, so `R` sits inside a clique that was already reported:

```
P empty and X empty  ->  R is maximal, emit it
P empty, X non-empty ->  R is not maximal, abandon silently
```

That test is why no `filter_maximal_subsets` is needed: non-maximal sets are never
emitted, rather than emitted and filtered.

**The pivot.**

```python
pivot = max((v for v in range(n) if (P | X) >> v & 1),
            key=lambda v: (P & adj[v]).bit_count())
cand = P & ~adj[pivot]
```

Without a pivot you branch on every vertex of `P` and rediscover the same clique
once per member. The pivot rule rests on one observation:

> Pick any `u ∈ P ∪ X`. Every maximal clique containing `R` either **excludes `u`**,
> or **contains a vertex that is not a neighbour of `u`** — because if it contained
> only neighbours of `u`, then `u` could be added, contradicting maximality.

So it suffices to branch on `P \ N(u)`, and choosing `u` to *maximise*
`|P ∩ N(u)|` makes that set as small as possible. This is what buys the
`O(3^(n/3))` bound.

**The trace.**

```
R={}                          P={(1,2),(1,3),(2,1),(3,1),(8,8)}   X={}
  pivot=(1,2) covers 4 of P  ->  branch only on {(1,2)}
  branch (1,2)
    R={(1,2)}                 P={(1,3),(2,1),(3,1),(8,8)}         X={}
      pivot=(2,1) covers 3   ->  branch only on {(2,1)}
      branch (2,1)
        R={(1,2),(2,1)}       P={(1,3),(3,1),(8,8)}               X={}
          pivot=(8,8) covers 2 -> branch only on {(8,8)}
          branch (8,8)
            R={(1,2),(2,1),(8,8)}   P={(1,3),(3,1)}               X={}
              pivot=(1,3) covers 0 -> branch on BOTH {(1,3),(3,1)}
              branch (1,3)
                R={(1,2),(1,3),(2,1),(8,8)}  P={}  X={}   MAXIMAL
              branch (3,1)
                R={(1,2),(2,1),(3,1),(8,8)}  P={}  X={}   MAXIMAL
```

**Six calls, no backtracking, no discarded work.**

Follow the pivot column. At the first three levels the pivot covers 4, then 3, then
2 of `P` — it is adjacent to everything still in play, so `P \ N(pivot)` is the
pivot alone and there is **exactly one branch**. The recursion walks straight down,
accumulating the vertices every hypothesis shares.

Then at `R = {(1,2),(2,1),(8,8)}` the pivot covers **0 of P**: `(1,3)` is adjacent
to neither itself nor `(3,1)`, so `P \ N(pivot)` is all of `P` and the recursion
splits in two.

**The single branch point in the entire run is the single missing edge.** The
algorithm does not so much search as walk directly to the fork. Compare §5: 51
calls, 28 memo entries, 24 masks generated, 22 discarded.

`X` never becomes non-empty here — with one missing edge there is no chance to
build a non-maximal clique. On denser graphs `X` does the real work.

#### Step 5 — keep the cliques containing the goal

```python
if goal_bit is not None:
    masks = [m for m in masks if (m >> goal_bit) & 1]
```

This is §5's `completable()` test, relocated. Previously it was a separate question
asked at every leaf: *can an ordering of this set still reach the goal afterwards?*
Now the goal is an ordinary vertex, so "can still reach the goal" **is** "is
comparable with the goal" — an edge like any other. A clique containing the goal is
a set that can be toured and escaped from.

Two subtleties make this exact rather than approximate:

- The goal absorbs — `adj` shows it reaching nothing — so wherever it appears in a
  clique it is necessarily **last**. There is no risk of an ordering that visits the
  goal and then continues.
- A maximal clique *without* the goal would be a set that can be toured but not
  escaped. Here there are none; on a board with a trap there would be, and they are
  dropped exactly as §5 dropped them.

When `goal_state` is not in `B` at all, `goal_bit` is `None` and every maximal
clique is returned — matching the old `completable()`, which returned `True`
unconditionally in that case.

#### Step 6 — `I`

```
I = [ [(1,2), (1,3), (2,1), (8,8)],     go east
      [(1,2), (2,1), (3,1), (8,8)] ]    go south
```

Identical to §5's answer, in a different row order — which nothing downstream reads
(see §9).

---

## 8. Complexity — the three side by side

### 8.1 Shared preprocessing

All three pay the same `O(n · |S| · |Act|)` for the reachability BFS and `O(n²)` to
compress it. Nothing below changes that; the comparison is about what happens
*after* `predecessors` exists.

### 8.2 Asymptotics

Write `N` for the number of MDP states and `m` for the size of the subset being
tested.

| | achievability queries | cost of ONE query | maximality | total time | space | cost driven by |
|---|---|---|---|---|---|---|
| **A** `main` | `2ⁿ` distinct | VI over `2^m · N` states | no-op (see §4) | **`Θ(3ⁿ · N · VI)`** | `O(2ⁿ · N)` peak | `3ⁿ`, **unconditionally** |
| **B** implementation | `O(A · n)` | `O(n)` bitmask ops, memoized | filter, `O(A·k)` | `O(A · n² + A·k)` | `O(A · n)` | `A`, the achievable subsets |
| **C** cliques | none | — | free (by construction) | `O(n² + output)`; `O(3^(n/3))` worst | `O(n²/w)` | `k`, **the answer size** |

Three observations, in increasing order of importance.

**A → B is an asymptotic win, and a larger one than it looks.** `3ⁿ` against `2ⁿ`
is already a separation, but the `N` factor is what dominates in practice: A's
achievability query builds an MDP over `2^m × N` states and runs value
iteration on it, where B's answers the same question with a handful of bitmask
operations against a memo. A 6×6 board was enough to have A killed by the OOM
killer during testing.

**Answering a reachability question with value iteration is what costs A the
exponent.** `Σ_{S ⊆ B} 2^|S| = 3ⁿ` is not the enumeration's fault — the enumeration
is `2ⁿ` in both A and B. The extra factor is entirely the augmented MDP inside
`CheckAchievability`, which re-encodes "which subgoals have I collected" as MDP
state and so pays a second powerset on top of the first. Had A called B's memoized
DP instead, the same enumeration would have cost `O(2ⁿ · n)`.

**B → C changes what the cost is proportional to.** This is the real difference, and
it is not visible in the worst-case column.

### 8.3 `A` against `k` — the gap that matters

`A` is the number of achievable subsets; `k = |I|` is how many of them are maximal.
The old algorithms pay for `A`. The answer is `k`.

Because the achievable family is downward-closed, `A` counts the entire downward
closure of the answer. The two can be exponentially far apart — and the worst case
is not exotic:

| board | \|B\| | live ℓ | `A` (achievable) | `2^ℓ` | `k = \|I\|` | `A/k` |
|---|---|---|---|---|---|---|
| open (no doors) | 15 | 10 | **1024** | 1024 | **1** | 1024 |
| open (no doors) | 7 | 6 | **64** | 64 | **1** | 64 |
| 3×3 rooms | 6 | 6 | 48 | 64 | 2 | 24 |
| 5×5 rooms | 5 | 5 | 32 | 32 | 1 | 32 |

Read the first row. On a **reversible** board — no one-way doors — every bottleneck
is mutually reachable with every other, the comparability graph is complete, and
`A = 2^ℓ` exactly: *every* subset is achievable. Meanwhile a complete graph has
exactly **one** maximal clique, so `k = 1`.

> **The old algorithm hits its absolute worst case precisely on the instance whose
> answer is trivial.** It enumerates 1024 subsets to report one hypothesis; the
> clique version sees a complete graph and returns in `ℓ` steps.

This is not a pathological corner — it is the open-board configuration
(`--rooms-per-side 1`), and it is exactly why `experiment.py` carries
`--min-hypotheses` to redraw past instances with `|I| = 1`. Those instances were
being *discarded* after being the most expensive ones to compute.

### 8.4 Honest limits of C

`O(3^(n/3))` is real, and it is tight: Moon and Moser showed a graph on `n`
vertices can have `3^(n/3)` maximal cliques, and the extremal example — the
complete multipartite graph `K₃,₃,…,₃` — **is** a comparability graph, so nothing
about this problem's structure rules it out.

Nor is `k` bounded in general here. On an `R × R` room grid, `|I|` is the number of
monotone routes, `C(2(R−1), R−1)`: 6 for 3×3, 20 for 4×4, 70 for 5×5, 924 for 7×7.
It grows exponentially in `R`.

But the bound to compare against is not `1` — it is the **output size**. No
algorithm can enumerate `k` hypotheses in less than `k` steps, so an
output-sensitive cost is the best available shape. C is within a polynomial factor
of it; A and B are not, because they pay `2ⁿ` (or `A`) *regardless* of `k`.

For this graph class the guarantee is stronger than the general bound suggests:
comparability graphs are **perfect**, and maximal-clique enumeration on them admits
polynomial delay — the time between two successive outputs is polynomial, so the
run never stalls, it only produces answers.

---

## 9. Validation

- **Exhaustive**: on 11 instances, the Held–Karp DP and the pairwise test agreed on
  **every one of ~36,000 subsets**.
- **End-to-end**: **38 instances** up to |B| = 28 returned identical `I`.
- **Full pipeline** at the default config, four grid games, 120 repetitions:
  `query_counts.csv` **byte-identical**, all size columns (`n_B`, `n_I`, `n_draws`,
  `n_skipped`, …) identical.

One difference is real but inert: the two implementations emit `I`'s rows in
different orders (B sorts by popcount out of the maximality filter, C emits in
recursion order). Nothing downstream depends on it — the terminal tests reduce with
`any()`, H1 and H4 sum over rows, H3 uses `all(0)`/`any(0)`, and tie-breaking is
over the *bit* order, which comes from `B`. The byte-identical CSV confirms it
rather than merely arguing it.

---

## 10. Where this stops being true

The reduction is a consequence of *this model*, not a general graph fact. It fails
the moment achievability stops meaning "some order exists, with free travel in
between":

- **restricted travel** — if a trajectory could not pass through bottlenecks it was
  not assigned, the relation is no longer plain reachability and is not transitive;
- **budgets** — "reach them within N steps" is not transitive: two cheap hops
  compose into an expensive one;
- **consumable edges** — single-use doors break composition of walks the same way.

Any of those puts you back in genuine Hamiltonian-path territory, and Held–Karp is
the right tool again. The DFS is kept as
`find_maximally_achievable_subsets_dfs` for exactly that reason, and as the oracle
the clique version is tested against.

A related point worth stating precisely, because it is easy to overclaim: **none of
this beats NP-completeness.** NP-completeness is a statement about the worst case
over *all* inputs. Restricting to a structured subclass routinely makes hard
problems easy, and that is a standard move, not a loophole — Hamiltonian path is
NP-complete in general but trivial in a transitive relation; graph colouring is
NP-complete in general but easy on interval graphs; SAT is the original NP-complete
problem but 2-SAT is linear. These instances were never general instances.

---

## 11. Reading (suggested by Claude)

**The DP being replaced**
- Held & Karp, *A Dynamic Programming Approach to Sequencing Problems*,
  J. SIAM 10(1):196–210, 1962. Accessible summary: Wikipedia, "Held–Karp algorithm".
  Bellman published the same idea independently that year.
- Jeff Erickson, *Algorithms* (free at `jeffe.cs.illinois.edu/teaching/algorithms/`),
  for the subset-DP idiom.

**Why the (⟸) direction is free**
- Rédei, *Ein kombinatorischer Satz*, Acta Sci. Math. (Szeged) 7:39–43, 1934. The
  paper proves the stronger statement that a tournament has an *odd* number of
  Hamiltonian paths; existence is the corollary used here. Readable statement:
  Wikipedia, "Tournament (graph theory)" § Hamiltonian path. A modern re-treatment:
  *The Tournament Theorem of Rédei revisited*, arXiv:2510.10659.
- Bang-Jensen & Gutin, *Digraphs: Theory, Algorithms and Applications*, 2nd ed.,
  Springer 2009 — Hamiltonicity of semicomplete digraphs.

**The clique enumeration**
- Bron & Kerbosch, *Algorithm 457: Finding all cliques of an undirected graph*,
  CACM 16(9):575–577, 1973. Wikipedia's page has the pivoting pseudocode used here.
- Tomita, Tanaka & Takahashi, *The worst-case time complexity for generating all
  maximal cliques*, Theoretical Computer Science 363(1):28–42, 2006 — the
  `O(3^(n/3))` bound.
- Moon & Moser, *On cliques in graphs*, Israel J. Math. 3:23–28, 1965 — that bound
  is tight.
- Eppstein, Löffler & Strash, *Listing All Maximal Cliques in Sparse Graphs in
  Near-Optimal Time*, arXiv:1006.5440 — degeneracy ordering, if more speed is needed.

**Order-theoretic framing**
- Golumbic, *Algorithmic Graph Theory and Perfect Graphs*, 2nd ed., Elsevier 2004 —
  comparability graphs are perfect, the structural reason clique problems are easy
  here.
- Johnson, Yannakakis & Papadimitriou, *On generating all maximal independent sets*,
  Information Processing Letters 27(3):119–123, 1988 — polynomial-delay enumeration.
- Wikipedia: "Comparability graph", "Dilworth's theorem".

**Complexity background**
- Karp, *Reducibility Among Combinatorial Problems*, 1972 — Hamiltonian path among
  the original 21 NP-complete problems.
- Wikipedia: "NP-completeness", "P versus NP problem".

**Where `B` comes from**
- Lengauer & Tarjan, *A fast algorithm for finding dominators in a flowgraph*,
  ACM TOPLAS 1(1):121–141, 1979.
