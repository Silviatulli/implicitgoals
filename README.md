# About this branch

I'm trying to write a unified version of the codebase — redundant files and dead code deleted,
only what the experiments actually use remains.

Two-stage split:
1. **Game modules** implement physics only, and output determinized
   transition matrices (`T_R`, `T_H_list`, `start_state`, `goal_state`). No
   notion of bottlenecks or queries.
2. **`experiment.py`** turns those matrices into a Query MDP (bottlenecks,
   Algorithm 1, solve) using the game-agnostic `bottlenecks.py`. 
   (Not paralelized because some config demand a lot of ram)

Four selection rules are wired in — `solve_query_mdp_exact` (Strategic VI),
plus H1 Info Gain, H3 Goal Proximity and H4 Query Frequency — each run twice,
once alone and once wearing the H2 dominance mask. Nine columns with the
random-order control.

Next step: more game variants, to grow `|I|` (the hypothesis space) beyond the
small/easy sizes we get today.

## Architecture — how a request flows through the code

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ 1. GAME LAYER -- physics only.                                                │
│    Produces T_R, T_H_list, start_state, goal_state.                           │
│                                                                               │
│ gridworld_core.py   shared GridWorld base + determinizer                      │
│   |-- gridworld.py       plain grid / four-rooms                              │
│   |-- puddleworld.py     grid + reward puddles                                │
│   |-- rockworld.py       grid + valuable/dangerous rocks                      │
│   `-- taxiworld.py       grid + pickup/dropoff passenger                      │
│                                                                               │
│ minigridworld.py    MiniGrid Unlock / UnlockPickup (own state, no base class) │
│ overcooked_env.py   kitchen/recipe rules (own state, no base class)           │
└───────────────────────────────────────────────────────────────────────────────┘
                                        |
                                        | T_R, T_H_list, start_state, goal_state
                                        v
┌───────────────────────────────────────────────────────────────────────┐
│ 2. QUERY-MDP LAYER -- game-agnostic.                                  │
│    Turns bare matrices into a Query MDP and solves it.                │
│                                                                       │
│ bottlenecks.py                                                        │
│   compute_bottlenecks_per_matrix  -> one set per human; union -> B    │
│   remove_toboggan_redundancies    -> B_filter    real decision nodes  │
│   find_maximally_achievable_subsets -> I         (Algorithm 1)        │
│   subsets_to_array                -> I_array                          │
│   Oracle                          -> simulated human answers          │
│   solve_query_mdp_exact           -> ExactQNet   VI baseline          │
│   solve_query_mdp_info_gain       -> GreedyQNet  H1                   │
│   solve_query_mdp_proximity       -> GreedyQNet  H3 (needs V_R)       │
│   solve_query_mdp_frequency       -> GreedyQNet  H4                   │
│   build_dominance                 -> (n,n) mask  H2, not a policy     │
│   evaluate_policy_on_real_human   -> query counts; owns termination   │
└───────────────────────────────────────────────────────────────────────┘
                                    |
                                    | I, I_array, one policy per condition
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. EXPERIMENT LAYER                                                        │
│                                                                            │
│ experiment.py       sweeps every (game, size, num_humans) combo, runs      │
│                      the pipeline --num-simu times per combo, and writes:  │
│                        results/compute_times.csv, results/query_counts.csv │
│                        results/compute_times.png, results/query_counts.png │
└────────────────────────────────────────────────────────────────────────────┘
```

## File-by-file

### Game layer

Four of the six games are grids, and they all share the same core:
`gridworld_core.py` defines the basic `GridWorld` MDP and a function that
turns any stochastic MDP into its determinized version. `gridworld.py`,
`puddleworld.py`, `rockworld.py`, and `taxiworld.py` each build on this
core and just add their own twist — puddles, rocks, or a taxi that picks
up and drops off a passenger — then call the same determinizer to get
their transition matrices.

`minigridworld.py` adds two more environments, MiniGrid's Unlock and
UnlockPickup tasks. It's in the repo and works probably, but I didn't add it to `experiment.py`yet.

`overcooked_env.py` stands on its own, separate from the grid games.
Overcooked isn't a grid, and its MDP is already deterministic, so there's
no stochastic-to-deterministic step here — it builds `T_R` and `T_H_list`
directly.

### Query-MDP layer (game-agnostic)

`bottlenecks.py` takes `T_R`, the candidate `T_H_list`, a start state and a
goal state, and finds the bottlenecks (`B`), filters them down to real
decision points (`B_filter`), enumerates the achievable subsets (`I`), builds
one policy per selection rule, and evaluates each of them against a
deterministic human oracle. It is the only file in this layer — one module for
the whole game-agnostic half of the pipeline.

Termination lives in `evaluate_policy_on_real_human`, never in a policy: a
condition is only a scoring rule over `B_filter`, and every one of them stops on
the same `I_hat ⊆ I_k` test `solve_query_mdp_exact` builds its absorbing masks
from. That is what makes the nine columns comparable.

### Experiment / results layer

`experiment.py` runs the whole pipeline for each game, size, and human
count, and times every step. It writes `results/compute_times.csv` and
`results/query_counts.csv`, then plots them into `results/compute_times.png`
and `results/query_counts.png`.

## TODO / next steps

- ~~**Replace 'n_states' with the number of states reachabel from the start state.** (only modify layer 3)~~ — **DONE**

  ~~The current plot express thee ~40k states instead of ~120 for Overcooked, which is misleading.~~

  The first panel of `compute_times.png` now plots `n_reachable`, the side of
  the pruned matrix the value iteration actually runs on (161 for Overcooked
  against 38 417 on paper). `n_states` is kept as a CSV column.

- ~~**Add the other Query-MDP solving methods.** (only modify layer 3)~~ — **DONE**

  ~~`solve_query_mdp_exact` (Strategic VI) is the only one wired in right
  now. I will try to copy paste your other methods developed on the other
  branch and hook them into `experiment.py`.~~

  All four hypotheses are wired into `experiment.py`, giving nine columns:
  Random, plus each of VI / H1 Info Gain / H3 Goal Proximity / H4 Query
  Frequency run with and without the H2 dominance mask. H2 has no column of
  its own because it is not a selection rule — it is applied at inference via
  `evaluate_policy_on_real_human(dominance=...)`.

- **Add more variants of the games.** (only modify layer 1)

  The goal is to grow the cardinality of `I` (the maximally achievable
  bottleneck subsets) so the query policies are tested on harder, more
  interesting instances than the current small-`|I|` ones.

- **Stop the grid generator from reseeding the global numpy RNG.** (only modify layer 1)

  `GridWorld.place_random_obstacles` calls `np.random.seed()`, so building a grid
  resets the process-wide numpy generator — once per robot and per human. The
  `np.random.seed(seed)` in `generate_determinized_models` is therefore dead, and
  `experiment.py`'s human draw depends on the last human's obstacle seed instead
  of the repetition seed. Fix: give `GridWorld` its own `np.random.Generator`.
