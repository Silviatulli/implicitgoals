# About this branch

Two branches matter:

- **`yacine_overcooked`** — the clean-up. It merges everything that came before
  into one codebase: redundant files and dead code deleted, only what the
  experiments really use kept. It fixes almost nothing and adds almost nothing;
  it just makes the rest readable.

- **`yacine_game_variants`** (this one) — the work branch, and the one furthest
  ahead. It adds variants of the games so that `|I|` — the number of hypotheses,
  written Φ in the paper — can be greater than 1, and every change made since
  then lives here too. It stays **exactly ahead of** `yacine_overcooked`: same
  history, extra commits on top, no divergence. You can ignore the `big_Q_games`
  folder, it is just a scratchpad for testing new ideas, not clean code.

## How the code is split

1. **Game modules** — physics only. They output determinized transition matrices
   (`T_R`, `T_H_list`, `start_state`, `goal_state`) and know nothing about
   bottlenecks or queries.
2. **`experiment.py`** — turns those matrices into a Query MDP (bottlenecks,
   Algorithm 1, solve) with the game-agnostic `bottlenecks.py`. Not parallelised,
   because some configurations need a lot of RAM.

Four selection rules are wired in — `solve_query_mdp_exact` (Strategic VI), H1
Info Gain, H3 Goal Proximity and H4 Query Frequency — reported against the
random-order control, so five columns. `--h2` runs each rule a second time
wearing the H2 dominance mask, for nine. H2 is not deleted, but it looks
irrelevant, so it is off by default.

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
│   compute_bottlenecks_per_matrix  -> per human; union -> B_nofilter   │
│   remove_toboggan_redundancies    -> B           real decision nodes  │
│   find_maximally_achievable_subsets -> I         (Algorithm 1)        │
│   subsets_to_array                -> I_array                          │
│   Oracle                          -> simulated human answers          │
│   solve_query_mdp_exact           -> ExactQNet   VI baseline          │
│   solve_query_mdp_info_gain       -> GreedyQNet  H1                   │
│   solve_query_mdp_proximity       -> GreedyQNet  H3 (needs geometry)  │
│   solve_query_mdp_frequency       -> GreedyQNet  H4                   │
│   build_dominance                 -> (n,n) mask  H2 — off by default  │
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

Four of the five games are grids, and they all share the same core:
`gridworld_core.py` defines the basic `GridWorld` MDP and a function that
turns any stochastic MDP into its determinized version. `gridworld.py`,
`puddleworld.py`, `rockworld.py`, and `taxiworld.py` each build on this
core and just add their own twist — puddles, rocks, or a taxi that picks
up and drops off a passenger — then call the same determinizer to get
their transition matrices.

`overcooked_env.py` stands on its own, separate from the grid games.
Overcooked isn't a grid, and its MDP is already deterministic, so there's
no stochastic-to-deterministic step here — it builds `T_R` and `T_H_list`
directly.

### Query-MDP layer (game-agnostic)

`bottlenecks.py` takes `T_R`, the candidate `T_H_list`, a start state and a
goal state, and finds the bottlenecks (`B_nofilter`), filters them down to real
decision points (`B`), enumerates the achievable subsets (`I`), builds
one policy per selection rule, and evaluates each of them against a
deterministic human oracle. It is the only file in this layer — one module for
the whole game-agnostic half of the pipeline.

Termination lives in `evaluate_policy_on_real_human`, never in a policy: a
condition is only a scoring rule over `B`, and every one of them stops on
the same `I_hat ⊆ I_k` test `solve_query_mdp_exact` builds its absorbing masks
from. Sharing a stopping rule that none of them owns is what makes the columns
comparable.

### Experiment / results layer

`experiment.py` runs the whole pipeline for each game, size, and human
count, and times every step. It writes `results/compute_times.csv` and
`results/query_counts.csv`, then plots them into `results/compute_times.png`
and `results/query_counts.png`.

Not every random map is usable, so each repetition redraws until it gets one
that passes two conditions: `|B| <= --max-bottlenecks` (affordable — Algorithm 1
is a 2^|B| search) and `|I| >= --min-hypotheses` (interesting — with a single
hypothesis there is nothing to ask about and every rule ties at zero queries).
The `n_draws` column reports how many maps were built per repetition, so the
strength of that conditioning stays visible.
