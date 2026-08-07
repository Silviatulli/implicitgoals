"""
benchmark_creation.py
=====================
Cross-game benchmark: run the same bottleneck / Query-MDP pipeline on all five
games and write the comparison to CSV.

    python benchmark_creation.py --num-simu 200 --sizes 4 5 6 --humans 1 3 5

Each (game, size, humans) combination is repeated --num-simu times, and every
repetition rebuilds the instance from scratch — a fresh random map for the four
grid games — so every number reported is a mean over those repetitions.

Writes two files into results/ , one row per combination:
    compute_times.csv   mean wall-clock time of every pipeline stage
    query_counts.csv    mean query count, Strategic Exact vs Query All

and two plots rendered from those same CSVs:
    query_counts.png    Strategic Exact vs Query All, mean queries per combination
    compute_times.png   problem size (|B_filter|) and wall-clock time per combination

All defaults (num_simu, sizes, humans, ...) live in parse_args() below.
"""

import os
import csv
import time
import random
import argparse
import contextlib
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Benchmark domains ─────────────────────────────────────────────────────────
from gridworld import generate_determinized_models as generate_determinized_gridworlds
from puddleworld import generate_determinized_models as generate_determinized_puddleworlds
from rockworld import generate_determinized_models as generate_determinized_rockworlds
from taxiworld import generate_determinized_models as generate_determinized_taxiworlds

from overcooked_env import (
    # no-movement MDP: state = inv * NUM_POT + pot
    build_transition_matrix_nomove,
    serving_matrices_nomove,
    # encoding constants
    CLIENT_SERVED,
    # the robot's stochastic model, pruned to the reachable states — Hypothesis 3
    build_stochastic_matrix as overcooked_stochastic_matrix,
)
from gridworld_core import build_stochastic_matrix as grid_stochastic_matrix
# Unlike the four grid domains, overcooked_env hands back (T_R, T_H_list)
# directly instead of a stochastic MDP to determinize: there is no
# augment_mdp_to_deterministic step, the matrices are already
# T[state, action] → next_state.


# ── Shared, game-agnostic pipeline ───────────────────────────────────────────
from bottlenecks import (
    Oracle,
    remove_toboggan_redundancies,
    find_maximally_achievable_subsets,
    subsets_to_array,
    compute_bottlenecks_per_matrix,
    solve_query_mdp_exact,
    # the four hypothesis conditions
    solve_query_mdp_info_gain,
    solve_query_mdp_transition,
    solve_query_mdp_proximity,
    solve_query_mdp_frequency,
    value_iteration_goal_probability,
)
from query_mdp_nn import evaluate_policy_on_real_human


# The six conditions the paper compares.  "strategic_exact" is the value-iteration
# baseline every hypothesis is measured against; "query_all" is the random-order
# control.  Order matters: it fixes the CSV column order and the plot legend.
CONDITIONS = ("strategic_exact", "info_gain", "transition",
              "proximity", "frequency", "query_all")

CONDITION_LABELS = {
    "strategic_exact": "VI baseline",
    "info_gain":       "H1 Information Gain",
    "transition":      "H2 Structural Redundancy",
    "proximity":       "H3 Goal Proximity",
    "frequency":       "H4 Query Frequency",
    "query_all":       "Random",
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-game instance builders
# ─────────────────────────────────────────────────────────────────────────────
# Every builder returns the same 4-tuple the pipeline consumes, plus the time
# spent producing it:
#     (T_R, T_H_list, start_state, goal_state), build_time
# For the four grid domains "build" means generate the stochastic MDPs and
# determinize them; for Overcooked it means assembling the serving matrices.

GRID_GAMES = ("gridworld", "puddleworld", "rockworld", "taxiworld")
ALL_GAMES  = GRID_GAMES + ("overcooked",)


@contextlib.contextmanager
def _quiet():
    """Swallow stdout — several pipeline functions print unconditionally."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def build_grid_instance(game, size, num_humans, seed=None, obstacles_percent=0.1,
                        puddle_percent=0.2, rock_percent=0.3, divide_rooms=False):
    """Generate + determinize one grid-domain instance.

    The density arguments are not shared: each generator accepts obstacles plus
    at most one domain-specific extra, so they are dispatched per game rather
    than passed as one common kwargs dict.
    """
    kwargs = dict(size=size, num_humans=num_humans, seed=seed,
                  obstacles_percent=obstacles_percent,
                  verbose=False, visualize=False)
    t0 = time.perf_counter()
    with _quiet():
        if game == "gridworld":
            out = generate_determinized_gridworlds(divide_rooms=divide_rooms, **kwargs)
        elif game == "puddleworld":
            out = generate_determinized_puddleworlds(puddle_percent=puddle_percent, **kwargs)
        elif game == "rockworld":
            out = generate_determinized_rockworlds(rock_percent=rock_percent, **kwargs)
        elif game == "taxiworld":
            out = generate_determinized_taxiworlds(**kwargs)
        else:
            raise ValueError(f"unknown grid game {game!r}")
    build_time = time.perf_counter() - t0

    T_R, start_state, goal_state, _ = out["robot"]
    T_H_list = [h[0] for h in out["humans"]]
    # Deferred, not built here: Hypothesis 3 is timed with its own value
    # iteration included, so the stochastic model is built inside that timer.
    sto_builder = lambda: grid_stochastic_matrix(out["robot_mdp"])
    return (T_R, T_H_list, start_state, goal_state, sto_builder), build_time


def build_overcooked_instance(num_humans=None, allow_drop=False, seed=None):
    """Assemble the no-movement Overcooked instance.

    Only `num_humans` is relevant here — grid size is not a parameter of this
    game (the kitchen is fixed) and there is no determinization step.  A
    candidate human is a recipe, drawn uniformly **with replacement** from the
    ~10 cookable ones, so `num_humans` is not capped at that number: asking for
    30 humans gives 30 candidates, most of them duplicates.  Sampling is what
    makes the repetitions differ — this is Overcooked's counterpart to the grid
    games' fresh random map.

    Duplicates are deliberate and are kept in T_H_list: they weight the human
    ensemble towards the repeated recipes, exactly as sampling with replacement
    should.  T_R is unaffected by them, since adding the same serving edge twice
    is idempotent — the robot ends up serving the *distinct* sampled recipes.

    The base transition matrix is rebuilt on every call, deliberately: it costs
    ~26 ms against a ~60 ms pipeline, and memoizing it would make `t_build` mean
    "one real build divided by num_simu" instead of the cost of building one
    instance.  Do not add a cache back.
    """
    t0 = time.perf_counter()
    with _quiet():
        T_base, recipes, _ = build_transition_matrix_nomove(allow_drop=allow_drop,
                                                            verbose=False)
    # Seeded after the build so the recipe draw stays fixed even if
    # build_transition_matrix_nomove ever starts consuming the global RNG.
    if seed is not None:
        np.random.seed(seed)

    if num_humans is not None:
        recipes = [recipes[i] for i in np.random.randint(len(recipes), size=num_humans)]
    with _quiet():
        T_R, T_H_list = serving_matrices_nomove(T_base, recipes)
    build_time = time.perf_counter() - t0

    # See build_grid_instance: deferred so its cost lands inside t_solve_proximity.
    sto_builder = lambda: overcooked_stochastic_matrix(
        np.asarray(T_R, dtype=np.int64), 0)
    return (T_R, T_H_list, 0, CLIENT_SERVED, sto_builder), build_time


# ─────────────────────────────────────────────────────────────────────────────
# The experiment — one repetition: one instance in, one timing row out
# ─────────────────────────────────────────────────────────────────────────────

def run_instance(T_R, T_H_list, start_state, goal_state, sto_builder=None,
                 max_exact_n=17, filter_toboggans=False, max_bottlenecks=18):
    """Run the whole pipeline once on one instance and time every stage.

    The Query MDP has two distinct inputs, and they are not the same set:

      * the **hypothesis space** I — the maximally achievable subsets returned by
        Algorithm 1.  These are the candidate answers: "the human wants I_k".
      * the **query alphabet** B_filter — what the robot is allowed to ask about.

    The alphabet is passed to solve_query_mdp_exact explicitly rather than being
    inferred from the labels present in I.  The difference is real: a bottleneck
    in B_filter that appears in no I_k can still be queried, and answering YES to
    it proves the human is incompatible with every hypothesis (failure), while
    answering NO is required before any hypothesis can be certified (success).
    Inferring the alphabet from I would silently drop exactly those bottlenecks.

    The alphabet is B_filter, not the raw union B: the exact solver allocates
    3^|alphabet| knowledge states, and |B| reaches ~44 on Overcooked (3^44 ≈
    10^21) against |B_filter| ~14 (3^14 ≈ 4.8M).  The toboggan filter is what
    makes the exact solve possible at all.

    max_bottlenecks : int
        Abandon the repetition when the bottleneck set is larger than this.
        Algorithm 1 is an include/exclude DFS over 2^|B| subsets, so a model
        that produces many bottlenecks can stall the whole sweep.  The test is
        applied to B_filter — i.e. *after* the toboggan filter for Overcooked,
        which is the only game that runs it — so a set that the filter brings
        back under the cap is still processed.

    filter_toboggans : bool
        Run remove_toboggan_redundancies between B and Algorithm 1.  Only
        Overcooked uses it — its bottleneck set is full of forced chains
        (31 raw states collapse to 10 real decision nodes), whereas the grid
        domains have few enough bottlenecks that the filter mostly just costs
        time.  When False, Algorithm 1 runs on the raw B, `n_B_filter` equals
        `n_B` and `t_toboggan_filter` is NaN to mark the stage as not run.

    sto_builder : callable or None
        Zero-argument builder returning (T_R_sto, index) — the robot's stochastic
        model pruned to its reachable states, plus the map from state ID to row.
        Called *inside* the Proximity timer, because Hypothesis 3's cost includes
        the value iteration it depends on.  None means H3 is skipped (NaN).

    Returns (row, counts, success): `row` holds the problem sizes and per-stage
    times, `counts` maps each of CONDITIONS to the queries that condition needed
    on one episode, and `success` says whether the episode identified the human
    at all.  Every count is NaN when the Query MDP was skipped — every return
    path has this same arity, so main() can unpack it unconditionally.

    `success` is deliberately a single flag rather than one per policy: an
    episode succeeds iff the drawn human's bottleneck set (restricted to the
    alphabet) is contained in some subset of I, which is a property of the
    instance and is therefore the same for every condition.  The individual
    episodes' own flags are discarded for that reason.

    One call is one repetition — main() averages over num_simu of them.
    """
    row: dict = {"n_states": int(T_R.shape[0]), "n_actions": int(T_R.shape[1]),
                 "n_humans": len(T_H_list)}

    # B comes from the *candidate humans*, not from T_R — this is what the
    # reference implementation does, and it is what makes the problem non-empty.
    # Sourcing B from T_R makes every bottleneck a dominator of T_R, hence
    # trivially achievable in T_R, hence |I| == 1 for any reversible domain.
    # Achievability below is still tested against T_R: "which of the waypoints
    # some human might care about can the robot actually visit, and together?"
    #
    # One dominator pass, two products: the per-matrix sets (the ensemble the
    # Oracle is built from, and the pool the evaluated human is drawn from) and
    # their union B.  Running extract_bottlenecks_union as well would repeat the
    # identical pass over the identical matrices, so t_bottlenecks carries the
    # whole cost and t_oracle_sets is only the union that falls out of it.
    t0 = time.perf_counter()
    with _quiet():
        oracle_sets = compute_bottlenecks_per_matrix(T_H_list, start_state, goal_state)
    row["t_bottlenecks"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Differs from extract_bottlenecks_union only for a human that cannot reach
    # the goal at all: that one is skipped there, and contributes {goal_state}
    # here.  goal_state is a bottleneck of every solvable matrix anyway, so the
    # union is unchanged unless *every* human is unsolvable.
    B = sorted(set().union(*oracle_sets)) if oracle_sets else []
    row["t_oracle_sets"] = time.perf_counter() - t0

    if filter_toboggans:
        t0 = time.perf_counter()
        with _quiet():
            B_filter = remove_toboggan_redundancies(T_R, B, goal_state)
        row["t_toboggan_filter"] = time.perf_counter() - t0
    else:
        B_filter = B
        row["t_toboggan_filter"] = float("nan")

    # Safety valve: bail out before Algorithm 1 rather than after, since it is
    # the 2^|B| stage that would hang.
    if len(B_filter) > max_bottlenecks:
        tqdm.write(f"skipping repetition: {len(B_filter)} bottlenecks > {max_bottlenecks}")
        row.update({"n_B": len(B), "n_B_filter": len(B_filter),
                    "n_I": float("nan"), "n_columns": float("nan")})
        # t_bottlenecks and t_oracle_sets already ran and hold real values —
        # only the stages that never got a chance to run are NaN'd here.
        for stage in ["t_algorithm1"] + SOLVE_TIMES + SIM_TIMES:
            row[stage] = float("nan")
        row["skipped"] = f"{len(B_filter)} bottlenecks > {max_bottlenecks}"
        return row, {c: float("nan") for c in CONDITIONS}, float("nan")

    t0 = time.perf_counter()
    with _quiet():
        I = find_maximally_achievable_subsets(B_filter, T_R, start_state,
                                              goal_state, verbose=False)
    row["t_algorithm1"] = time.perf_counter() - t0

    # The query alphabet: everything the robot may ask about.  It is B_filter,
    # *not* the labels occurring in I — see the docstring.  The same ordering is
    # handed to solve_query_mdp_exact, so the exact policy's action indices and
    # I_array's columns refer to the same bottleneck; letting the solver infer
    # its own order from I would silently desynchronise the two.
    B_sorted  = sorted(int(b) for b in B_filter)
    I_array  = subsets_to_array(I, B_sorted)
    b_to_int = {b: j for j, b in enumerate(B_sorted)}

    row.update({"n_B": len(B), "n_B_filter": len(B_filter),
                "n_I": len(I), "n_columns": len(B_sorted)})

    # ── Query MDP ────────────────────────────────────────────────────────────
    n = len(B_sorted)
    if n == 0 or n > max_exact_n:
        for stage in SOLVE_TIMES + SIM_TIMES:
            row[stage] = float("nan")
        row["skipped"] = "no bottlenecks" if n == 0 else f"3^{n} too large"
        tqdm.write(f"skipping Q-MDP exact because n = {n} bottlenecks")
        return row, {c: float("nan") for c in CONDITIONS}, float("nan")

    # Empirical P(YES | b) = fraction of candidate humans owning b, taken from the
    # very ensemble the evaluated human is drawn from below.  Without it the
    # solver assumes a uniform 50/50 prior over every bottleneck.
    oracle = Oracle(oracle_sets, n_states=max(int(T_R.shape[0]), int(goal_state) + 1))

    # ── Build one policy per condition, timing each in isolation ─────────────
    # Transition solves its own copy of the query MDP rather than reusing the
    # baseline's: solve_query_mdp_transition attaches `.dominance` to the policy
    # it is handed, which would silently turn the baseline into H2 as well.
    policies: dict = {}
    for name, solver in (("strategic_exact", solve_query_mdp_exact),
                         ("info_gain",       solve_query_mdp_info_gain),
                         ("transition",      solve_query_mdp_transition),
                         ("frequency",       solve_query_mdp_frequency)):
        t0 = time.perf_counter()
        with _quiet():
            policies[name] = solver(I, B_sorted, oracle=oracle)
        row[f"t_solve_{name}"] = time.perf_counter() - t0

    # Proximity: the value iteration is part of H3's cost, so the stochastic
    # model is built and solved inside this timer rather than hoisted out of it.
    t0 = time.perf_counter()
    if sto_builder is None:
        policies["proximity"] = None
        row["t_solve_proximity"] = float("nan")
    else:
        with _quiet():
            T_R_sto, sto_index = sto_builder()
            V_R = value_iteration_goal_probability(T_R_sto, sto_index[int(goal_state)])
            policies["proximity"] = solve_query_mdp_proximity(
                I, B_sorted, oracle=oracle, V_R=V_R, state_index=sto_index)
        row["t_solve_proximity"] = time.perf_counter() - t0

    policies["query_all"] = None   # the random-order control needs no solve

    # The same human faces every condition within a repetition, so the counts are
    # paired: their differences are not polluted by which human was drawn.
    oracle_bottlenecks = list(oracle_sets[np.random.randint(len(oracle_sets))])

    counts, success = {}, float("nan")
    for name in CONDITIONS:
        if name == "proximity" and policies["proximity"] is None:
            counts[name] = float("nan")
            row["t_sim_proximity"] = float("nan")
            continue
        t0 = time.perf_counter()
        counts[name], flag = _query_episode(policies[name], oracle_bottlenecks,
                                            I_array, b_to_int)
        row[f"t_sim_{name}"] = time.perf_counter() - t0
        # Every condition reports the same flag — see the docstring — so keeping
        # the last one is keeping the instance's flag.
        success = flag

    row["skipped"] = ""
    return row, counts, success


def _query_episode(policy, oracle_bottlenecks, I_array, b_to_int):
    """Queries one episode needs against the human owning `oracle_bottlenecks`.

    `policy=None` is the Query All baseline (bottlenecks in random order);
    an ExactQNet is the Strategic Exact policy.  Mirrors viz.compute.
    """
    with _quiet():
        res = evaluate_policy_on_real_human(
            true_bottlenecks=oracle_bottlenecks, policy_network=policy,
            n_runs=1, I_array=I_array, b_to_int=b_to_int,
        )
    return int(res["n_queries"][0]), int(res["success"][0])


# ─────────────────────────────────────────────────────────────────────────────
# Main — every (game, size, humans) combination
# ─────────────────────────────────────────────────────────────────────────────

# Every t_* and n_* column is a mean over the num_simu repetitions.
SOLVE_TIMES = [f"t_solve_{c}" for c in CONDITIONS if c != "query_all"]
SIM_TIMES   = [f"t_sim_{c}"   for c in CONDITIONS]

STAGE_TIMES = (["t_build", "t_bottlenecks", "t_toboggan_filter", "t_algorithm1",
                "t_oracle_sets"] + SOLVE_TIMES + SIM_TIMES + ["t_total"])
STAGE_SIZES = ["n_states", "n_actions", "n_humans",
               "n_B", "n_B_filter", "n_I", "n_columns"]

TIME_FIELDS = (["game", "size", "num_humans", "num_simu"] + STAGE_SIZES
               + STAGE_TIMES + ["t_total_std", "n_skipped", "skipped"])

# n_success + n_failure == n_episodes.  An episode "fails" when the drawn human
# is not representable in I, in which case its query count measures queries until
# the contradiction was proved, not queries until the human was identified — the
# two are not commensurable, hence the split means alongside the pooled ones.
QUERY_FIELDS = (["game", "size", "num_humans", "num_simu", "n_episodes",
                 "n_success", "n_failure"]
                + [f"{c}_{stat}" for c in CONDITIONS
                   for stat in ("mean", "std", "mean_success", "mean_failure")]
                + ["saved_queries", "skipped"])


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare the bottleneck / Query-MDP pipeline across the five games.")
    p.add_argument("--num-simu", type=int, default=50,
                   help="repetitions of the whole experiment per combination; every "
                        "reported time and query count is a mean over them "
                        "(default: 50)")
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 10],
                   help="grid side lengths to sweep (default: 8 10)")
    p.add_argument("--humans", type=int, nargs="+", default=[10, 20],
                   help="numbers of candidate human models to sweep; for Overcooked "
                        "these are drawn with replacement from the ~10 recipes, so "
                        "values above 10 are allowed (default: 10 20)")
    p.add_argument("--games", nargs="+", default=list(ALL_GAMES), choices=list(ALL_GAMES),
                   help="games to run (default: all five)")
    p.add_argument("--out-dir", default="results",
                   help="output folder, created if missing (default: results)")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; each combination is offset from it (default: 0)")
    # The two caps guard different stages and are deliberately different:
    # Algorithm 1 is a 2^n DFS (2^18 = 262k subsets, cheap), the exact Query MDP
    # allocates 3^n knowledge states.  A repetition may therefore clear
    # Algorithm 1 and still skip the exact solve.
    #
    # Both are sized so that Overcooked never skips at the default MDP: measured
    # over 160 instances (40 seeds x 5/10/20/30 humans) the toboggan filter puts
    # |B_filter| in 4..17, so 18 and 17 are the observed ceilings.  The price of
    # the 17 is steep and superlinear — one n=17 solve costs ~29 s and ~4.7 GB
    # peak, against ~0.08 s at n=13 — so a 20-human sweep spends most of its wall
    # clock here.  Lower --max-exact-n to trade completed repetitions for speed.
    #
    # Do not raise --max-exact-n to 18 without testing it alone first: 3^18 is
    # 3x the states of 3^17, extrapolating to ~14 GB.  And with
    # --overcooked-allow-drop the filtered set roughly doubles (~36), which no
    # cap value can bring back into reach of an exact solve.
    p.add_argument("--max-bottlenecks", type=int, default=18,
                   help="skip a repetition whose bottleneck set exceeds this, "
                        "measured after the toboggan filter; Algorithm 1 is a "
                        "2^|B| search (default: 18)")
    p.add_argument("--max-exact-n", type=int, default=17,
                   help="skip the exact Query MDP above this many bottlenecks, "
                        "since it allocates 3^n arrays; n=17 costs ~29 s and "
                        "~4.7 GB per repetition (default: 17)")
    p.add_argument("--obstacles-percent", type=float, default=0.1,
                   help="obstacle density of the four grid games (default: 0.1)")
    p.add_argument("--puddle-percent", type=float, default=0.2,
                   help="puddle density, puddleworld only (default: 0.2, "
                        "matches PuddleWorld's own default)")
    p.add_argument("--rock-percent", type=float, default=0.3,
                   help="rock density, rockworld only (default: 0.3)")
    p.add_argument("--divide-rooms", action="store_true",
                   help="use the four-rooms layout, gridworld only")
    p.add_argument("--overcooked-allow-drop", action="store_true",
                   help="use the allow_drop variant of the Overcooked MDP")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    # Overcooked has no grid size, so it gets one job per human count; the four
    # grid domains get the full (size x humans) cross product.
    jobs = [(g, s, h) for g in args.games if g in GRID_GAMES
            for s in args.sizes for h in args.humans]
    if "overcooked" in args.games:
        jobs += [("overcooked", None, h) for h in args.humans]

    time_rows, query_rows = [], []
    # One tqdm tick per repetition, so the bar reflects the real work: a
    # repetition rebuilds the instance (a fresh random map for the grid games)
    # and re-runs the whole pipeline on it.
    bar = tqdm(total=len(jobs) * args.num_simu, desc="benchmark", unit="rep")
    for job_idx, (game, size, num_humans) in enumerate(jobs):
        label = game if size is None else f"{game} {size}x{size}"
        bar.set_postfix_str(f"{label}, {num_humans} humans")

        reps, counts_per_rep, successes = [], [], []
        for rep in range(args.num_simu):
            # A distinct seed per repetition — that is what makes the average
            # meaningful for the grid games: each repetition is a new map.
            seed = args.seed + job_idx * args.num_simu + rep
            random.seed(seed)
            np.random.seed(seed)

            t_start = time.perf_counter()
            if game == "overcooked":
                instance, t_build = build_overcooked_instance(
                    num_humans=num_humans, allow_drop=args.overcooked_allow_drop, seed=seed)
            else:
                instance, t_build = build_grid_instance(
                    game, size, num_humans, seed=seed,
                    obstacles_percent=args.obstacles_percent,
                    puddle_percent=args.puddle_percent,
                    rock_percent=args.rock_percent,
                    divide_rooms=args.divide_rooms)

            row, counts, success = run_instance(
                *instance, max_exact_n=args.max_exact_n,
                filter_toboggans=(game == "overcooked"),
                max_bottlenecks=args.max_bottlenecks)
            row["t_build"] = t_build
            row["t_total"] = time.perf_counter() - t_start

            reps.append(row)
            counts_per_rep.append(counts)
            successes.append(success)
            bar.update(1)

        key = {"game": game, "size": "" if size is None else size,
               "num_humans": num_humans, "num_simu": args.num_simu}
        time_rows.append(_aggregate_times(key, reps))
        query_rows.append(_aggregate_queries(key, reps, counts_per_rep, successes))
    bar.close()

    times_path   = os.path.join(args.out_dir, "compute_times.csv")
    queries_path = os.path.join(args.out_dir, "query_counts.csv")
    _write_csv(times_path, TIME_FIELDS, time_rows)
    _write_csv(queries_path, QUERY_FIELDS, query_rows)
    query_plot_path, times_plot_path = _make_plots(times_path, queries_path, args.out_dir)

    print(f"\n{len(time_rows)} configurations x {args.num_simu} repetitions")
    print(f"  mean stage times   -> {times_path}")
    print(f"  mean query counts  -> {queries_path}")
    print(f"  query counts plot  -> {query_plot_path}")
    print(f"  compute times plot -> {times_plot_path}")
    _print_summary(time_rows, query_rows)


def _nanmean(values):
    """Mean ignoring NaNs, and NaN (not a warning) when everything is NaN."""
    arr = np.asarray([v for v in values], dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _nanstd(values):
    arr = np.asarray([v for v in values], dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.std()) if arr.size else float("nan")


def _aggregate_times(key, reps):
    """Average every stage time and problem size over the repetitions."""
    row = dict(key)
    for field in STAGE_SIZES + STAGE_TIMES:
        row[field] = _nanmean([r.get(field, float("nan")) for r in reps])
    row["t_total_std"] = _nanstd([r["t_total"] for r in reps])
    skips = [r["skipped"] for r in reps if r.get("skipped")]
    row["n_skipped"] = len(skips)
    row["skipped"] = skips[0] if skips else ""
    return row


def _mean_where(values, successes, want):
    """Mean of `values` over the repetitions whose success flag equals `want`.

    Skipped repetitions carry NaN in both lists and belong to neither bucket.
    """
    sel = [v for v, s in zip(values, successes)
           if not np.isnan(s) and int(s) == want and not np.isnan(v)]
    return float(np.mean(sel)) if sel else float("nan")


def _aggregate_queries(key, reps, counts_per_rep, successes):
    """Average every condition's query counts over the repetitions.

    Reported three ways: pooled over every completed episode, and split by
    whether the episode succeeded.  The split matters because a failed episode's
    query count measures queries-until-contradiction rather than
    queries-until-identification, so pooling the two averages incommensurable
    quantities — and would credit a policy for detecting a contradiction fast.
    `successes` holds one flag per repetition, shared by every condition.

    `counts_per_rep` is a list of {condition: count} dicts, one per repetition.
    """
    row = dict(key)
    by_cond = {c: [d.get(c, float("nan")) for d in counts_per_rep] for c in CONDITIONS}

    exact_counts = by_cond["strategic_exact"]
    row["n_episodes"] = int(sum(1 for c in exact_counts if not np.isnan(c)))
    row["n_success"] = int(sum(1 for s in successes if not np.isnan(s) and int(s) == 1))
    row["n_failure"] = int(sum(1 for s in successes if not np.isnan(s) and int(s) == 0))

    for cond, vals in by_cond.items():
        row[f"{cond}_mean"]         = _nanmean(vals)
        row[f"{cond}_std"]          = _nanstd(vals)
        row[f"{cond}_mean_success"] = _mean_where(vals, successes, 1)
        row[f"{cond}_mean_failure"] = _mean_where(vals, successes, 0)

    # Mean of the paired per-repetition differences: queries Strategic Exact
    # saves over Query All against the same human.
    paired = [q - e for e, q in zip(exact_counts, by_cond["query_all"])
              if not (np.isnan(e) or np.isnan(q))]
    row["saved_queries"] = float(np.mean(paired)) if paired else float("nan")
    skips = [r["skipped"] for r in reps if r.get("skipped")]
    row["skipped"] = skips[0] if skips else ""
    return row


def _write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _combo_label(row):
    """'gridworld 4x4 (10h)' for a grid game, 'overcooked (10h)' for Overcooked
    (which has no grid size)."""
    size = row["size"]
    base = row["game"] if size == "" or pd.isna(size) else f"{row['game']} {int(size)}x{int(size)}"
    return f"{base} ({int(row['num_humans'])}h)"


def _rotate_xticks(ax, labels, x):
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")


def _make_plots(times_path, queries_path, out_dir):
    """Read compute_times.csv / query_counts.csv back and render two PNGs
    into out_dir: one bar per (game, size, num_humans) combination in both.

    query_counts.png   Strategic Exact vs Query All, mean queries per episode
    compute_times.png  2x2 grid: state-space size (n_states), hypothesis-space
                        cardinality (n_I), problem size (|B_filter|), and mean
                        wall-clock time — so the cost of a combination can be
                        read against where it started (n_states) and what it
                        was solving for (n_I).

    Returns (query_plot_path, times_plot_path).
    """
    df_q = pd.read_csv(queries_path)
    df_t = pd.read_csv(times_path)

    # ── query_counts.png ────────────────────────────────────────────────────
    labels = df_q.apply(_combo_label, axis=1)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5.5))
    width = 0.8 / len(CONDITIONS)
    for i, cond in enumerate(CONDITIONS):
        offset = (i - (len(CONDITIONS) - 1) / 2) * width
        ax.bar(x + offset, df_q[f"{cond}_mean"], width,
               yerr=df_q[f"{cond}_std"], capsize=2,
               label=CONDITION_LABELS[cond])
    ax.set_ylabel("mean queries per episode")
    ax.set_title("Query conditions — the four hypotheses against the VI baseline")
    _rotate_xticks(ax, labels, x)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    query_plot_path = os.path.join(out_dir, "query_counts.png")
    fig.savefig(query_plot_path, dpi=150)
    plt.close(fig)

    # ── compute_times.png ───────────────────────────────────────────────────
    labels_t = df_t.apply(_combo_label, axis=1)
    x_t = np.arange(len(labels_t))

    fig, axes = plt.subplots(2, 2, figsize=(max(11, len(labels_t) * 1.1), 10))

    axes[0, 0].bar(x_t, df_t["n_states"], color="darkorange")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("n_states (log scale)")
    axes[0, 0].set_title("State-space size (start of the pipeline)")
    _rotate_xticks(axes[0, 0], labels_t, x_t)

    axes[0, 1].bar(x_t, df_t["n_I"], color="mediumseagreen")
    axes[0, 1].set_ylabel("|I|")
    axes[0, 1].set_title("Hypothesis-space cardinality (Algorithm 1 output)")
    _rotate_xticks(axes[0, 1], labels_t, x_t)

    axes[1, 0].bar(x_t, df_t["n_B_filter"], color="steelblue")
    axes[1, 0].set_ylabel("|B_filter|")
    axes[1, 0].set_title("Problem size (bottlenecks after toboggan filter)")
    _rotate_xticks(axes[1, 0], labels_t, x_t)

    axes[1, 1].bar(x_t, df_t["t_total"], color="indianred")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("mean t_total (s, log scale)")
    axes[1, 1].set_title("Wall-clock time per repetition")
    _rotate_xticks(axes[1, 1], labels_t, x_t)

    fig.tight_layout()
    times_plot_path = os.path.join(out_dir, "compute_times.png")
    fig.savefig(times_plot_path, dpi=150)
    plt.close(fig)

    return query_plot_path, times_plot_path


def _print_summary(time_rows, query_rows):
    """One line per configuration — everything shown is a mean over num_simu."""
    short = {"strategic_exact": "VI", "info_gain": "H1", "transition": "H2",
             "proximity": "H3", "frequency": "H4", "query_all": "rand"}
    header = "".join(f"{short[c]:>8}" for c in CONDITIONS)
    print(f"\n{'game':<12}{'size':>5}{'hum':>5}{'|B_f|':>7}{header}{'time(s)':>10}")
    for row, q in zip(time_rows, query_rows):
        if q["n_episodes"]:
            queries = "".join(f"{q[f'{c}_mean']:8.2f}" for c in CONDITIONS)
        else:
            queries = f"{row['skipped']:>{8 * len(CONDITIONS)}}"
        print(f"{row['game']:<12}{str(row['size']):>5}{row['num_humans']:>5}"
              f"{row['n_B_filter']:>7.1f}{queries}{row['t_total']:>10.2f}")


if __name__ == "__main__":
    main()


