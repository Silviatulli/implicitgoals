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
random-order control. The H2 half is now optional and off unless `--h2` is
passed — see the remark on H2(ii) below for why.

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
  `evaluate_policy_on_real_human(dominance=...)`. The four "+ H2" columns are
  now behind `--h2` and off by default — see the remark on H2(ii) below.

- **Add more variants of the games.** (only modify layer 1)

  The goal is to grow the cardinality of `I` (the maximally achievable
  bottleneck subsets) so the query policies are tested on harder, more
  interesting instances than the current small-`|I|` ones.

- ~~**Stop the grid generator from reseeding the global numpy RNG.** (only modify layer 1)~~ — **DONE**

  ~~`GridWorld.place_random_obstacles` calls `np.random.seed()`, so building a grid
  resets the process-wide numpy generator — once per robot and per human. The
  `np.random.seed(seed)` in `generate_determinized_models` is therefore dead, and
  `experiment.py`'s human draw depends on the last human's obstacle seed instead
  of the repetition seed. Fix: give `GridWorld` its own `np.random.Generator`.~~

  `GridWorld` now owns a `self.rng` built by `gridworld_core.seeded_rng()`, and
  every draw in the four grid games goes through it. It is a
  `np.random.RandomState`, not a `default_rng`: RandomState is the same MT19937
  stream `np.random.seed` drove, so every map is byte-identical to the ones the
  old code produced for the same `obstacle_seed` — the only thing that changed is
  who else can see the stream. The `np.random.seed(seed)` in each
  `generate_determinized_models` is live again, and `experiment.py`'s human draw
  now follows the repetition seed.

  One bug fell out of it. `place_random_obstacles` ran *inside* `__init__`'s
  retry loop and reseeded on every entry, so a layout that failed
  `check_for_path` was redrawn identically `max_tries` times before the
  empty-map fallback. A retry now actually retries.


## Remark — the H2(ii) dominance mask may not be sound

Flagged to Silvia, waiting on her answer. **`dominance` is off by default in
`main` until then.** Nothing is deleted; `build_dominance` and the
`evaluate_policy_on_real_human(dominance=...)` path still work, and the nine
columns still run if you switch it back on.

### What the mask does

`build_dominance` marks a pair `(b1, b2)` when every subset in `I` that
contains `b1` also contains `b2`. At inference, if the oracle answers *no* on
`b2`, we rule out `b1` too, without spending a query.

### Why that needs `I_G ∈ I`

The argument is: the rule holds for every `I_k` in `I`, so it holds for the
human's true subgoal set `I_G`, so we can contrapose. That last step only
works if `I_G` is itself one of the subsets in `I`.

But it isn't guaranteed. `I` comes from **the robot**: Algorithm 1 enumerates
what `T_R` can achieve, and keeps only the *maximal* ones. `I_G` comes from
**the human**: it's the bottleneck set of one of the `T_H_list` matrices.
Nothing in the pipeline makes these two agree. `I_G` may not be achievable by
the robot at all, and even when it is, there's no reason for it to be maximal —
so in general we only get `I_G ⊆ I_k` for some `I_k`, not `I_G ∈ I`.

### What goes wrong

```
B        = {1, 2, 3}
I        = {{1,2}, {2,3}}     # robot can't do 1 and 3 in one plan
I_G      = {1}                # achievable, but not maximal -> I_G not in I

supp(1) = {{1,2}} ⊆ supp(2) = both   =>  mask marks (1, 2)

query 2  -> oracle says "no"
mask     -> rules out 1
           but 1 IS in I_G, and it was never queried.
```

The robot then commits to a plan that skips the human's only subgoal, and
never asks the one question that would have caught it.

### It probably isn't only H2

The consistency filter has the same requirement: dropping every `I_k` that
contains a denied bottleneck is only justified if the `I_k` are candidates for
`I_G` *exactly*. On the example above, answering *no* on `2` drops both
subsets and leaves the consistent set empty — even though `{1,2}` was a
perfectly good plan the whole time.

Worth noting that our termination test is already the other way round:
`evaluate_policy_on_real_human` stops on `I_hat ⊆ I_k`, i.e. *covering* `I_G`,
not *identifying* it. Covering is the weaker, safer condition and doesn't need
`I_G ∈ I`. So the codebase currently mixes the two readings — inclusion at
termination, equality in the filter and in the mask.

### Cheap check

Compute `I`, then test whether each per-human bottleneck set from
`compute_bottlenecks_per_matrix` is in `I`. If they all are, the assumption
holds for our configs and should be written down as a precondition. If some
aren't, we have a real counterexample from the actual experiments. If `|I|` is
much smaller than the number of humans, the robot is capable enough to merge
them and the hypothesis space has collapsed.