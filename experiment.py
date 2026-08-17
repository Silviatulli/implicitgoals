"""
experiment.py
=============
Cross-game benchmark: run the same bottleneck / Query-MDP pipeline on all five
games and write the comparison to CSV.

    python experiment.py --num-simu 200 --room-sides 2 3 4 --humans 1 3 5

Each (game, room side, humans) combination is repeated --num-simu times, and
every repetition rebuilds the instance from scratch — a fresh random map for the
four grid games — so every number reported is a mean over those repetitions.

Board geometry (the four grid games).  Two numbers describe it:
--rooms-per-side R     rooms along each side of the board
--room-sides C         cells along each side of one room
so every room is a C x C square and the room grid is R x R.  The walls are thin —
they run *between* cells rather than occupying any — so the board is exactly
R * C cells per side, nothing is spent on the walls, and a door is an *edge*: you
stand west of a door, step right, and you are in the next room, never standing in
the door itself.  The default R=3, C=3 is a 9x9 board of nine 3x3 rooms, and
--rooms-per-side 1 is the degenerate case: one room, no walls, an open C x C board.
The defaults are R=5, C=5 — a 25x25 board of twenty-five 5x5 rooms, with 40
one-way doors.

Every wall between two neighbouring rooms carries exactly one door, and it is
**one-way** — east or south only — sitting at the middle of that wall.  Doors are
shared by the robot and every human, like the start, the goal and the taxi's
passenger; only the obstacles vary between models.  Drawing a door per model
instead put each human's mandatory waypoints on different cells, so B_nofilter —
the union over the humans — grew linearly with the human count and ran past
--max-bottlenecks, which Algorithm 1's 2^|B| cost cannot absorb.

The orientation is the point.  On an open grid a single trajectory can tour every
bottleneck and come back, so Algorithm 1 always finds exactly one maximally
achievable subset — |I| = 1 whatever the obstacle layout.  One-way doors make the
room grid a DAG: stepping through a door commits, the doors of the routes not
taken become unreachable, and |I| grows to the number of monotone routes through
the room grid: C(2(R-1), R-1), so 6 for 3x3, 20 for 4x4 and 70 for the default
5x5.

--humans is bounded by the same 2^|B| cost, since B_nofilter is the union over
the humans.  An instance that busts --max-bottlenecks is no longer skipped: it is
thrown away and *redrawn* from a fresh seed until one fits, so every
configuration reports the full --num-simu repetitions instead of however many
happened to survive.  The price is that the maps are now sampled conditioned on
|B| <= --max-bottlenecks, which favours the ones with fewer mandatory waypoints;
`n_draws` is the mean number of maps drawn per repetition and is what makes that
conditioning visible (1.0 means the cap never bound).  Raising --humans therefore
costs draws rather than repetitions.  Overcooked is unaffected — its bottlenecks
come from recipes, not maps.

Five conditions per repetition: the four selection rules (VI, H1 Info Gain,
H3 Goal Proximity, H4 Query Frequency), plus the random-order "query all"
control.  --h2 adds a second run of each rule wearing the H2 dominance mask, for
nine conditions; without the flag the mask is never built.  H2 is unsound and was
dropped from the paper — see bottlenecks.build_dominance — so the flag exists to
reproduce the effect, not to report numbers.

Writes two files into results/ , one row per combination:
    compute_times.csv   mean wall-clock time of every pipeline stage
    query_counts.csv    mean query count, one column per condition

and two plots rendered from those same CSVs:
    query_counts.png    one subplot per configuration, one bar per condition
    compute_times.png   n_reachable, |I|, |B| and wall-clock time

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
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from tqdm import tqdm

# ── Benchmark domains ─────────────────────────────────────────────────────────
from gridworld import generate_determinized_models as generate_determinized_gridworlds
from puddleworld import generate_determinized_models as generate_determinized_puddleworlds
from rockworld import (
    generate_determinized_models as generate_determinized_rockworlds,
    MAX_VALUABLE_ROCKS,
)
from taxiworld import generate_determinized_models as generate_determinized_taxiworlds

from overcooked_env import (
    # no-movement MDP: state = inv * NUM_POT + pot
    build_transition_matrix_nomove,
    serving_matrices_nomove,
    # encoding constants
    CLIENT_SERVED,
)
from gridworld_core import (board_side, status_line, set_status_sink,
                            RETRY_REPORT_EVERY)
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
    bottleneck_index,
    compute_bottlenecks_per_matrix,
    solve_query_mdp_exact,
    # the selection rules, and H2's inference-time mask
    solve_query_mdp_info_gain,
    solve_query_mdp_proximity,
    solve_query_mdp_frequency,
    build_dominance,
    get_reachable_states,
    evaluate_policy_on_real_human,
)


# Four selection rules plus the random-order control: five columns, or nine with
# --h2, which runs each rule a second time wearing the dominance mask.
#
# H2 gets no column of its own because it is not a selection rule: it never
# chooses a query, it only widens K_not after a NO.  It is applied at inference
# (evaluate_policy_on_real_human(dominance=...)), so the paired columns share one
# policy object and differ only in whether the mask is passed.  With --h2 off the
# mask is never built, and the CSV fields, plots and summary all follow the
# condition list the run actually used.
#
# H2 is unsound and no longer in the paper.  The "+ H2" columns are lower than
# their twins because the mask answers questions the oracle was never asked, not
# because it asks better ones — do not read the gap as a saving.
BASES = ("strategic_exact", "info_gain", "proximity", "frequency")


def conditions_for(use_h2):
    """Column order for one run: Random, then each rule, each immediately
    followed by its "+ H2" twin when the dominance layer is switched on."""
    return ("query_all",) + tuple(
        c for b in BASES for c in ((b, f"{b}_h2") if use_h2 else (b,)))


CONDITIONS = conditions_for(True)    # every column the pipeline can produce

BASE_LABELS = {
    "strategic_exact": "VI baseline",
    "info_gain":       "H1 Info Gain",
    "proximity":       "H3 Goal Proximity",
    "frequency":       "H4 Query Frequency",
}
CONDITION_LABELS = {"query_all": "Random"}
for _b, _lab in BASE_LABELS.items():
    CONDITION_LABELS[_b] = _lab
    CONDITION_LABELS[f"{_b}_h2"] = f"{_lab} + H2"

# One hue per selection rule, two shades of it: hue answers "which rule?", shade
# answers "with H2 or not?", so the gap within a pair is what the dominance layer
# bought.  tab20 is built for this — ten (dark, light) pairs of one hue.  Random
# is grey: a control, not a rule.
_TAB20 = plt.get_cmap("tab20").colors
CONDITION_COLORS = {"query_all": "#9e9e9e"}
for _i, _b in enumerate(BASES):
    CONDITION_COLORS[_b]         = _TAB20[2 * _i + 1]   # light — rule alone
    CONDITION_COLORS[f"{_b}_h2"] = _TAB20[2 * _i]       # dark  — rule + H2


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


def build_grid_instance(game, room_side, num_humans, seed=None, obstacle_density=0.1,
                        puddle_density=0.2, rock_density=0.3, rooms_per_side=3,
                        slip_prob=0.0):
    """Generate + determinize one grid-domain instance.

    ``room_side`` is the side of one **room**, which is what --room-sides sweeps;
    the board the generators build is ``board_side(rooms_per_side, room_side)``.

    The density arguments are not shared: each generator accepts obstacles plus
    at most one domain-specific extra, so they are dispatched per game rather
    than passed as one common kwargs dict.
    """
    kwargs = dict(num_humans=num_humans, seed=seed,
                  obstacle_density=obstacle_density,
                  rooms_per_side=rooms_per_side, room_side=room_side,
                  slip_prob=slip_prob, verbose=False, visualize=False)
    t0 = time.perf_counter()
    with _quiet():
        if game == "gridworld":
            out = generate_determinized_gridworlds(**kwargs)
        elif game == "puddleworld":
            out = generate_determinized_puddleworlds(puddle_density=puddle_density, **kwargs)
        elif game == "rockworld":
            out = generate_determinized_rockworlds(rock_density=rock_density, **kwargs)
        elif game == "taxiworld":
            out = generate_determinized_taxiworlds(**kwargs)
        else:
            raise ValueError(f"unknown grid game {game!r}")
    build_time = time.perf_counter() - t0

    T_R, start_state, goal_state, _ = out["robot"]
    T_H_list = [h[0] for h in out["humans"]]
    # The MDP object travels with the instance: Hypothesis 3 reads its state
    # space to get each state's (row, col) cell.
    return (T_R, T_H_list, start_state, goal_state, out["robot_mdp"]), build_time


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
        T_base, recipes = build_transition_matrix_nomove(allow_drop=allow_drop,
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

    # A size-4 instance: no MDP object (Overcooked has none). run_instance builds
    # the stochastic matrix from T_R itself, inside H3's timer.
    return (T_R, T_H_list, 0, CLIENT_SERVED), build_time


# ─────────────────────────────────────────────────────────────────────────────
# Drawing an instance the pipeline can actually afford
# ─────────────────────────────────────────────────────────────────────────────

def compute_bottleneck_sets(T_R, T_H_list, start_state, goal_state,
                            filter_toboggans):
    """The per-human bottleneck sets, their union, the query set, and the timings.

    Split out of run_instance because the caller needs |B| *before* it commits to
    an instance: one whose query set busts --max-bottlenecks is thrown away and
    redrawn, and deciding that takes exactly these three stages and nothing after
    them.  Handing the timings back with the sets is what lets the accepted
    draw's stage columns hold real numbers instead of being recomputed and
    re-timed inside run_instance.

    Returns (oracle_sets, B_nofilter, B, stage_times)
      oracle_sets  one frozenset per candidate human — the Oracle's ensemble, and
                   the pool the evaluated human is drawn from
      B_nofilter   their union, before the toboggan filter
      B            the query set: what the robot may ask about, and the bit order
                   every policy and I_array agree on.  Equal to B_nofilter unless
                   `filter_toboggans`, which only Overcooked sets
      stage_times  {t_bottlenecks, t_oracle_sets, t_toboggan_filter}
    """
    # B_nofilter comes from the *candidate humans*, not from T_R.  Sourcing it
    # from T_R makes every bottleneck a dominator of T_R, hence trivially
    # achievable, hence |I| == 1 for any reversible domain.  Achievability is
    # still tested against T_R: "which of these waypoints can the robot visit,
    # and together?"
    #
    # One dominator pass, two products: the per-matrix sets and their union.
    # t_bottlenecks therefore carries the whole cost and t_oracle_sets only the
    # union that falls out of it.
    times = {}
    t0 = time.perf_counter()
    with _quiet():
        oracle_sets = compute_bottlenecks_per_matrix(T_H_list, start_state, goal_state)
    times["t_bottlenecks"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # A human that cannot reach the goal at all still contributes {goal_state}:
    # extract_bottlenecks always includes it.  goal_state is a bottleneck of
    # every solvable matrix anyway, so the union is unchanged unless *every*
    # human is unsolvable.
    B_nofilter = sorted(set().union(*oracle_sets)) if oracle_sets else []
    times["t_oracle_sets"] = time.perf_counter() - t0

    if filter_toboggans:
        t0 = time.perf_counter()
        with _quiet():
            B = remove_toboggan_redundancies(T_R, B_nofilter, goal_state)
        times["t_toboggan_filter"] = time.perf_counter() - t0
    else:
        B = B_nofilter
        times["t_toboggan_filter"] = float("nan")

    return oracle_sets, B_nofilter, B, times


# An instance whose query set busts --max-bottlenecks used to be skipped
# outright, which cost up to 31 of 50 repetitions on taxiworld and left each
# configuration averaging over a different, unstated number of episodes.  It is
# redrawn instead, and only a repetition that cannot find a usable map in this
# many draws is given up on.
MAX_INSTANCE_TRIES = 100_000

# Draw n reuses the repetition seed offset by n * this stride.  A redraw has to
# change the seed at all — every builder reseeds `random`/`numpy` from the seed it
# is handed, so redrawing with the same one would rebuild the identical map for
# ever and the loop would spin to MAX_INSTANCE_TRIES without ever varying.  The
# stride is larger than any (job_idx * num_simu + rep) a run can produce, so a
# redraw can never land on another repetition's seed and collapse two
# repetitions onto one map, which is the whole reason they are seeded apart.
RETRY_SEED_STRIDE = 1_000_003


def draw_instance_under_cap(game, room_side, num_humans, seed, args,
                            max_bottlenecks, max_tries=MAX_INSTANCE_TRIES):
    """Draw instances until one's query set fits under `max_bottlenecks`.

    Returns (instance, bundle, t_build, t_rejected, n_draws).  `instance` is None
    when all `max_tries` draws came back too big, which skips the repetition —
    the one case left where a repetition produces no episode.

    `t_build` times the accepted draw alone, so it still means "what one instance
    costs to build" and stays comparable with runs made before redrawing existed.
    Everything burned on the rejected draws — their builds *and* their bottleneck
    passes — is reported separately, as `t_rejected`.

    Sampling note: conditioning on |B| <= max_bottlenecks is not free.  The maps
    that survive are the ones with fewer mandatory waypoints, so every mean the
    run reports describes that conditioned population rather than the raw one.
    `n_draws` is what makes the strength of the conditioning readable.
    """
    filter_toboggans = (game == "overcooked")
    t_rejected = 0.0
    for attempt in range(max_tries):
        if attempt and attempt % RETRY_REPORT_EVERY == 0:
            status_line(f"draw {attempt:,}/{max_tries:,} before skipping the config")
        t0 = time.perf_counter()
        attempt_seed = seed + attempt * RETRY_SEED_STRIDE
        if game == "overcooked":
            instance, t_build = build_overcooked_instance(
                num_humans=num_humans, allow_drop=args.overcooked_allow_drop,
                seed=attempt_seed)
        else:
            instance, t_build = build_grid_instance(
                game, room_side, num_humans, seed=attempt_seed,
                obstacle_density=args.obstacle_density,
                puddle_density=args.puddle_density,
                rock_density=args.rock_density,
                rooms_per_side=args.rooms_per_side,
                slip_prob=args.slip_prob)
        # instance is a 5-tuple for the grid games and a 4-tuple for Overcooked,
        # but the first four slots are the same four things in both.
        bundle = compute_bottleneck_sets(instance[0], instance[1], instance[2],
                                         instance[3], filter_toboggans)
        if len(bundle[2]) <= max_bottlenecks:
            status_line(None)
            return instance, bundle, t_build, t_rejected, attempt + 1
        # Too big: this instance and its matrices go out of scope here and the
        # next iteration builds a fresh one over them.  Only the cost survives.
        t_rejected += time.perf_counter() - t0
    status_line(None)
    return None, None, float("nan"), t_rejected, max_tries


# ─────────────────────────────────────────────────────────────────────────────
# The experiment — one repetition: one instance in, one timing row out
# ─────────────────────────────────────────────────────────────────────────────

def run_instance(T_R, T_H_list, start_state, goal_state, mdp_R=None,
                 max_exact_n=17, filter_toboggans=False, max_bottlenecks=17,
                 use_h2=False, bottleneck_bundle=None):
    # mdp_R is what Hypothesis 3 reads coordinates out of:
    #   * a robot MDP object (grid games) → state[0] is the cell, so H3 ranks
    #     bottlenecks by Euclidean distance to the goal;
    #   * None (Overcooked, which has no MDP object and no geometry) → no H3.
    """Run the whole pipeline once on one instance and time every stage.

    The Query MDP has two distinct inputs, and they are not the same set:

      * the **hypothesis space** I — the maximally achievable subsets returned by
        Algorithm 1.  These are the candidate answers: "the human wants I_k".
      * the **query set** B — what the robot is allowed to ask about.

    B is passed to solve_query_mdp_exact explicitly rather than being inferred
    from the labels present in I.  The difference is real: a bottleneck in B that
    appears in no I_k can still be queried, and answering YES to it proves the
    human is incompatible with every hypothesis (failure), while answering NO is
    required before any hypothesis can be certified (success).  Inferring it from
    I would silently drop exactly those bottlenecks.

    It is B, not the unfiltered union B_nofilter: the exact solver allocates 3^|B|
    knowledge states, and |B_nofilter| reaches ~44 on Overcooked (3^44 ≈ 10^21)
    against |B| ~14 (3^14 ≈ 4.8M).  The toboggan filter is what makes the exact
    solve possible at all.

    bottleneck_bundle : tuple or None
        The (oracle_sets, B_nofilter, B, stage_times) compute_bottleneck_sets
        returns.  main() passes it in because it already had to compute |B| to
        decide whether to keep this instance at all; None recomputes it, which is
        the standalone path for running one instance with no redraw loop around
        it.

    max_bottlenecks : int
        Abandon the repetition when the query set is larger than this.
        Algorithm 1 is an include/exclude DFS over 2^|B| subsets, so a model that
        produces many bottlenecks can stall the whole sweep.  Under main() this
        is unreachable — draw_instance_under_cap redraws until the instance fits,
        so run_instance is only ever handed one that already passed — and it
        stays as the guard on the standalone path.

    filter_toboggans : bool
        Run remove_toboggan_redundancies between B_nofilter and Algorithm 1.
        Only Overcooked uses it — its bottleneck set is full of forced chains
        (31 raw states collapse to 10 real decision nodes), whereas the grid
        domains have few enough bottlenecks that the filter mostly just costs
        time.  When False, Algorithm 1 runs on the unfiltered union, `n_B` equals
        `n_B_nofilter` and `t_toboggan_filter` is NaN to mark the stage as not run.

    mdp_R : MDP object or None
        The robot's un-determinized model.  Hypothesis 3 reads its state space
        for each state's cell; None (Overcooked) means H3 is skipped and its
        columns are NaN, since Euclidean proximity needs a geometry the game
        does not have.

    use_h2 : bool
        Add the "+ H2" twin of every selection rule.  Off by default: the
        dominance mask is not even built, and neither `t_dominance` nor any
        `*_h2` column appears in the returned row or counts.  H2 is unsound and
        no longer in the paper — see bottlenecks.build_dominance.

    Returns (row, counts, success): `row` holds the problem sizes and per-stage
    times, `counts` maps each of the run's conditions to the queries that condition needed
    on one episode, and `success` says whether the episode identified the human
    at all.  Every count is NaN when the Query MDP was skipped — every return
    path has this same arity, so main() can unpack it unconditionally.

    `success` is deliberately a single flag rather than one per policy: an
    episode succeeds iff the drawn human's bottleneck set (restricted to B) is
    contained in some subset of I, which is a property of the
    instance and is therefore the same for every condition.  The individual
    episodes' own flags are discarded for that reason.

    One call is one repetition — main() averages over num_simu of them.
    """
    # The columns this repetition produces, and the stage timers an early return
    # has to NaN out — both narrower when H2 is off.
    conditions = conditions_for(use_h2)
    unrun_stages = (["t_dominance"] if use_h2 else []) + \
        [f"t_solve_{c}" for c in conditions if c != "query_all"] + \
        [f"t_sim_{c}" for c in conditions]

    # n_reachable: how much of the state space the robot can actually get to,
    # against n_states, which is how big the game is on paper (161 against 38 417
    # for Overcooked).  A BFS over the determinized matrix, so it costs nothing
    # and every early return below already carries the column.  It used to fall
    # out of the stochastic matrix H3 built; H3 no longer builds one.
    row: dict = {"n_states": int(T_R.shape[0]), "n_actions": int(T_R.shape[1]),
                 "n_humans": len(T_H_list),
                 "n_reachable": int(get_reachable_states(
                     np.asarray(T_R), int(start_state)).sum())}

    # Normally precomputed: main() has to know |B| *before* it accepts an
    # instance, so it runs these three stages itself and hands the result down
    # rather than paying for them twice.  Recomputing is the standalone path.
    if bottleneck_bundle is None:
        bottleneck_bundle = compute_bottleneck_sets(T_R, T_H_list, start_state,
                                                    goal_state, filter_toboggans)
    oracle_sets, B_nofilter, B, stage_times = bottleneck_bundle
    row.update(stage_times)

    # Safety valve: bail out before Algorithm 1 rather than after, since it is
    # the 2^|B| stage that would hang.  Unreachable under main(), which redraws
    # until the instance fits; this is the standalone path's guard.
    if len(B) > max_bottlenecks:
        tqdm.write(f"skipping repetition: {len(B)} bottlenecks > {max_bottlenecks}")
        row.update({"n_B_nofilter": len(B_nofilter), "n_B": len(B),
                    "n_I": float("nan"), "n_columns": float("nan")})
        # t_bottlenecks and t_oracle_sets already ran and hold real values —
        # only the stages that never got a chance to run are NaN'd here.
        for stage in ["t_algorithm1"] + unrun_stages:
            row[stage] = float("nan")
        row["skipped"] = f"{len(B)} bottlenecks > {max_bottlenecks}"
        return row, {c: float("nan") for c in conditions}, float("nan")

    t0 = time.perf_counter()
    with _quiet():
        I = find_maximally_achievable_subsets(B, T_R, start_state,
                                              goal_state, verbose=False)
    row["t_algorithm1"] = time.perf_counter() - t0

    # The query set: everything the robot may ask about.  It is B, *not* the
    # labels occurring in I — see the docstring.  The same ordering goes to
    # solve_query_mdp_exact, so action indices and I_array columns mean the same
    # bottleneck; letting the solver infer its own order would desynchronise them.
    I_array  = subsets_to_array(I, B)
    b_to_int = bottleneck_index(B)

    row.update({"n_B_nofilter": len(B_nofilter), "n_B": len(B),
                "n_I": len(I), "n_columns": len(B)})

    # ── Query MDP ────────────────────────────────────────────────────────────
    n = len(B)
    if n == 0:
        for stage in unrun_stages:
            row[stage] = float("nan")
        row["skipped"] = "no bottlenecks"
        return row, {c: float("nan") for c in conditions}, float("nan")

    # Empirical P(YES | b) = fraction of candidate humans owning b, taken from the
    # very ensemble the evaluated human is drawn from below.  Without it the
    # solver assumes a uniform 50/50 prior over every bottleneck.
    oracle = Oracle(oracle_sets, n_states=max(int(T_R.shape[0]), int(goal_state) + 1))

    # ── Build one policy per selection rule ─────────────────────────────────
    # Each t_solve_* below is what that condition would cost *run on its own*, so
    # shared work is added to every condition needing it rather than amortised:
    # the dominance mask is built once but charged to all four "+ H2" columns.
    #
    # max_exact_n gates the VI baseline *alone*, because solve_query_mdp_exact
    # allocates 3^n knowledge states while the greedy rules cost microseconds at
    # any n — skipping them alongside it would hide the regime they are for.
    # (max_bottlenecks is the other kind of cap: I is needed by every condition,
    # so exceeding it skips the repetition whole.)
    policies, base_solve = {}, {}
    if n > max_exact_n:
        policies["strategic_exact"]   = None
        base_solve["strategic_exact"] = float("nan")
        row["skipped"] = f"VI skipped: 3^{n} too large"
        tqdm.write(f"skipping the VI baseline because n = {n} bottlenecks "
                   f"(the greedy rules still run)")
    else:
        t0 = time.perf_counter()
        with _quiet():
            policies["strategic_exact"] = solve_query_mdp_exact(I, B, oracle=oracle)
        base_solve["strategic_exact"] = time.perf_counter() - t0

    for name, solver in (("info_gain", solve_query_mdp_info_gain),
                         ("frequency", solve_query_mdp_frequency)):
        t0 = time.perf_counter()
        with _quiet():
            policies[name] = solver(I, B, oracle=oracle)
        base_solve[name] = time.perf_counter() - t0

    # Proximity is a coordinate lookup and a norm — H3 ranks bottlenecks by
    # Euclidean distance to the goal, so there is no model to build and no value
    # iteration to time.  What used to sit in this timer was an O(|S|^2 |A|)
    # stochastic-matrix build plus a value iteration, and it was the dominant
    # cost of the whole H3 stage.
    t0 = time.perf_counter()
    with _quiet():
        if mdp_R is not None:
            # state[0] is the (row, col) cell in every grid world, and
            # augment_mdp_to_deterministic indexes states in get_state_space()
            # order, so the raw bottleneck IDs address this list directly.
            positions = {i: tuple(s[0])
                         for i, s in enumerate(mdp_R.get_state_space())}
            policies["proximity"] = solve_query_mdp_proximity(
                I, B, oracle=oracle, positions=positions,
                goal_state=int(goal_state))
        else:
            # Overcooked has no geometry — a state is a bit-packed (inventory,
            # pot) pair, not a place — so Euclidean proximity is undefined and
            # the game simply has no H3 column.  None flows through the same
            # path as a skipped VI baseline below, giving NaN.
            policies["proximity"] = None
    base_solve["proximity"] = (float("nan") if policies["proximity"] is None
                               else time.perf_counter() - t0)

    # H2's mask — computed once, charged to each column that wears it.  Not built
    # at all when no column wears it.
    dominance, t_dominance = None, float("nan")
    if use_h2:
        t0 = time.perf_counter()
        with _quiet():
            dominance = build_dominance(I, B)
        t_dominance = time.perf_counter() - t0
        row["t_dominance"] = t_dominance

    for base in BASES:
        row[f"t_solve_{base}"] = base_solve[base]
        if use_h2:
            row[f"t_solve_{base}_h2"] = base_solve[base] + t_dominance

    # The same human faces every condition within a repetition, so the counts are
    # paired: their differences are not polluted by which human was drawn.
    oracle_bottlenecks = list(oracle_sets[np.random.randint(len(oracle_sets))])

    counts, success = {}, float("nan")
    for name in conditions:
        base = name[:-3] if name.endswith("_h2") else name
        if base != "query_all" and policies[base] is None:
            counts[name] = float("nan")
            row[f"t_sim_{name}"] = float("nan")
            continue
        t0 = time.perf_counter()
        counts[name], flag = _query_episode(
            policies.get(base), oracle_bottlenecks, I_array, b_to_int,
            dominance=dominance if name.endswith("_h2") else None)
        row[f"t_sim_{name}"] = time.perf_counter() - t0
        # Read the flag off a condition that does not wear H2.  Every condition
        # reports the same one: success means the drawn human is representable in
        # I, a property of the instance rather than of the rule.
        #
        # Sourcing it from a plain condition is *not* a check on H2 — the success
        # flag cannot detect H2's unsoundness at all, since the mask only ever
        # adds K_not bits, which shrinks I_hat and makes the success test easier
        # and the failure test harder.  It is only to keep this column meaning
        # the instance's property rather than the last condition's.
        if not name.endswith("_h2"):
            success = flag

    row.setdefault("skipped", "")   # may already hold the VI-skipped note
    return row, counts, success


def _query_episode(policy, oracle_bottlenecks, I_array, b_to_int, dominance=None):
    """Queries one episode needs against the human owning `oracle_bottlenecks`.

    `policy=None` is the Query All baseline (bottlenecks in random order);
    otherwise it is one of the four selection rules.  `dominance` switches H2 on
    for this episode only — the policy object is untouched, which is what lets
    the same one serve both the plain and the "+ H2" column.
    """
    with _quiet():
        res = evaluate_policy_on_real_human(
            true_bottlenecks=oracle_bottlenecks, policy_network=policy,
            n_runs=1, I_array=I_array, b_to_int=b_to_int, dominance=dominance,
        )
    return int(res["n_queries"][0]), int(res["success"][0])


# ─────────────────────────────────────────────────────────────────────────────
# Main — every (game, size, humans) combination
# ─────────────────────────────────────────────────────────────────────────────

# Every t_* and n_* column is a mean over the num_simu repetitions.
# The t_solve_* columns are per-condition standalone costs, so shared work is
# added into each of them.  t_dominance is that shared piece reported once on its
# own, so it is *not* a term to add to a total — the t_solve_* columns already
# count it four times, deliberately.  It appears only when H2 runs.
#
# n_draws and t_rejected are the redraw loop's accounting: how many instances had
# to be built to get one under --max-bottlenecks, and what the discarded ones
# cost.  t_build stays the accepted draw alone, so it means the same thing it
# always did and t_build + t_rejected is the true cost of arriving at an instance.
STAGE_SIZES = ["n_states", "n_reachable", "n_actions", "n_humans", "n_draws",
               "n_B_nofilter", "n_B", "n_I", "n_columns"]


def column_layout(use_h2):
    """The CSV header of a run, derived from the conditions it will produce.

    Returns (conditions, stage_times, time_fields, query_fields).  Keeping the
    four in one place is what keeps the H2-off run from writing empty `*_h2`
    columns nothing filled in.

    n_success + n_failure == n_episodes.  An episode "fails" when the drawn human
    is not representable in I, in which case its query count measures queries
    until the contradiction was proved, not queries until the human was
    identified — the two are not commensurable, hence the split means alongside
    the pooled ones.
    """
    conditions  = conditions_for(use_h2)
    solve_times = [f"t_solve_{c}" for c in conditions if c != "query_all"]
    sim_times   = [f"t_sim_{c}"   for c in conditions]

    stage_times = (["t_build", "t_rejected", "t_bottlenecks", "t_toboggan_filter",
                    "t_algorithm1", "t_oracle_sets"]
                   + (["t_dominance"] if use_h2 else [])
                   + solve_times + sim_times + ["t_total"])

    # All three geometry columns are written: "room_side" is what --room-sides
    # asked for, "board_side" is what actually drives the state count and the run
    # time, and neither can be recovered from the other without "rooms_per_side".
    time_fields = (["game", "room_side", "rooms_per_side", "board_side",
                    "num_humans", "num_simu"] + STAGE_SIZES
                   + stage_times + ["t_total_std", "n_skipped", "skipped"])

    query_fields = (["game", "room_side", "rooms_per_side", "board_side",
                     "num_humans", "num_simu", "n_episodes",
                     "n_success", "n_failure"]
                    + [f"{c}_{stat}" for c in conditions
                       for stat in ("mean", "std", "mean_success", "mean_failure")]
                    + ["saved_queries", "skipped"])
    return conditions, stage_times, time_fields, query_fields


def parse_args(argv=None):
    """The command line, grouped by what each flag decides.

    argparse prints --help in declaration order, so this order is the help screen
    a reader gets: what to sweep, then the board that sweep is run on, then which
    conditions are reported, then the cost ceilings, then the odds and ends.
    Flags that refer to each other are kept adjacent — --room-sides is the side of
    a room and --rooms-per-side the number of them, and neither means anything
    without the other.
    """
    p = argparse.ArgumentParser(
        description="Compare the bottleneck / Query-MDP pipeline across the five games.")

    # ── What to sweep ────────────────────────────────────────────────────────
    p.add_argument("--games", nargs="+", default=list(ALL_GAMES), choices=list(ALL_GAMES),
                   help="games to run (default: all five)")
    p.add_argument("--humans", type=int, nargs="+", default=[12, 16, 20],
                   help="numbers of candidate human models to sweep; B is their "
                        "union, so more humans means a bigger query set and more "
                        "instances redrawn for exceeding --max-bottlenecks "
                        "(default: 12 16 20)")
    p.add_argument("--num-simu", type=int, default=50,
                   help="repetitions of the whole experiment per combination; every "
                        "reported time and query count is a mean over them "
                        "(default: 50)")

    # ── The board the four grid games are built on ───────────────────────────
    p.add_argument("--rooms-per-side", type=int, default=5,
                   help="rooms per side of the board, joined by one-way doors; "
                        "1 = open board, and |I| = 1 with it (default: 5)")
    p.add_argument("--room-sides", type=int, nargs="+", default=[5],
                   help="cells per side of one ROOM, swept; the board is "
                        "rooms-per-side x this (default: 5, so a 25x25 board)")
    p.add_argument("--obstacle-density", type=float, default=0.05,
                   help="obstacle density of the four grid games (default: 0.05)")
    p.add_argument("--puddle-density", type=float, default=0.2,
                   help="puddle density, puddleworld only (default: 0.2, "
                        "matches PuddleWorld's own default)")
    p.add_argument("--rock-density", type=float, default=0.3,
                   help="rock density, rockworld only (default: 0.3)")
    p.add_argument("--slip-prob", type=float, default=0.0,
                   help="probability of slipping into each unintended neighbour "
                        "on a move, for the four grid games; 0 makes every move "
                        "deterministic, so determinization is a no-op and the "
                        "augmented action set stays the 4 real moves instead of "
                        "growing to ~20 (default: 0.0)")

    # ── Which conditions to report ───────────────────────────────────────────
    p.add_argument("--h2", action="store_true",
                   help="also run every rule wearing the H2 dominance mask, "
                        "doubling the four rule columns to eight.  H2 is UNSOUND "
                        "and was dropped from the paper — it lowers query counts "
                        "by granting answers the oracle never gave, so its "
                        "columns are not comparable with the others (default: off)")

    # ── Cost ceilings ────────────────────────────────────────────────────────
    # Maintainer's note, deliberately not in the help text below: it is about the
    # two default *values*, so its reader is whoever edits them.
    #
    # The caps guard different stages: Algorithm 1 is a 2^n DFS (2^17 = 131k
    # subsets, cheap), the exact Query MDP allocates 3^n knowledge states.  They
    # are now the same number, so an instance accepted at all can also be solved
    # exactly; while they differed (18 against 17) a repetition could clear
    # Algorithm 1 and then skip the VI baseline, leaving that column averaged
    # over a different subset of repetitions than the greedy ones.
    #
    # Both are sized so Overcooked never redraws at the default MDP: over 160
    # instances (40 seeds x 5/10/20/30 humans) the toboggan filter put |B| in
    # 4..17, so 17 is the observed ceiling.  It is expensive — one n=17 solve
    # costs ~29 s and ~4.7 GB peak against ~0.08 s at n=13 — so lower
    # --max-bottlenecks to trade draws for speed.
    #
    # Do not raise --max-exact-n to 18 without testing it alone first: 3^18
    # extrapolates to ~14 GB.  And --overcooked-allow-drop roughly doubles the
    # filtered set (~36), which no cap value brings back into reach — there the
    # redraw loop cannot help either, since every draw busts the cap.
    p.add_argument("--max-bottlenecks", type=int, default=17,
                   help="redraw an instance whose query set exceeds this, "
                        "measured after the toboggan filter; Algorithm 1 is a "
                        "2^|B| search (default: 17)")
    p.add_argument("--max-exact-n", type=int, default=17,
                   help="skip the exact Query MDP above this many bottlenecks, "
                        "since it allocates 3^n arrays; n=17 costs ~29 s and "
                        "~4.7 GB per repetition (default: 17)")

    # ── Variant, reproducibility, output ─────────────────────────────────────
    p.add_argument("--overcooked-allow-drop", action="store_true",
                   help="use the allow_drop variant of the Overcooked MDP")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; each combination is offset from it (default: 0)")
    p.add_argument("--out-dir", default="results",
                   help="output folder, created if missing (default: results)")
    return p.parse_args(argv)


# A RockWorld board above this many states is flagged as slow before the sweep
# starts.  An 8x8 board is 505 states at the default densities and already costs
# ~3.4 s per repetition against ~0.08 s for gridworld, so the cut sits just below
# it and leaves 6x6 (281) unflagged.
ROCKWORLD_SLOW_STATES = 400


def _rockworld_state_estimate(board, rock_density, valuable_rock_ratio=0.4):
    """States of one RockWorld board: board^2 positions x 2^k collection sets.

    ``board`` is the side of the whole grid, *not* the room side that
    --room-sides sweeps: the rocks are scattered over the entire board, so it is
    board_side() that drives the state count and therefore the cost.

    Mirrors RockWorld.place_rocks and create_state_space — k valuable rocks,
    capped at MAX_VALUABLE_ROCKS, contribute one bit each, and the goal collapses
    to a single sink whatever has been collected (hence the -1 / +1).
    `valuable_rock_ratio` repeats RockWorld's own default; the experiment never
    overrides it, so it is not exposed on the command line.
    """
    total_rocks = int(board * board * rock_density)
    k = min(int(total_rocks * valuable_rock_ratio), MAX_VALUABLE_ROCKS)
    return (board * board - 1) * 2 ** k + 1


def _warn_slow_rockworld(jobs, num_simu, rock_density, rooms_per_side=3):
    """Announce the RockWorld jobs that will crawl, and where on the bar.

    Nothing here is at risk of diverging or being skipped: the 2^k collection
    bits simply multiply the state space that both the determinized build and
    the determinized build walks, so those jobs run an order of magnitude
    slower per repetition than the plain grids.  Printing the slice of the
    progress bar they occupy is the point — an hour of near-frozen bar in that
    range is expected, not a hang.
    """
    slow = [i for i, (game, room_side, _) in enumerate(jobs)
            if game == "rockworld"
            and _rockworld_state_estimate(board_side(rooms_per_side, room_side),
                                          rock_density) >= ROCKWORLD_SLOW_STATES]
    if not slow:
        return
    # The flagged jobs are contiguous (one game, sizes in order), so first and
    # last bound the whole stretch of the bar they own.
    total = len(jobs) * num_simu
    lo = 100.0 * slow[0] * num_simu / total
    hi = 100.0 * (slow[-1] + 1) * num_simu / total
    # Report the board, and the room side that produced it: --room-sides named the
    # second, but it is the first that sets the state count being warned about.
    room_sides = sorted({jobs[i][1] for i in slow})
    sizes = ", ".join(
        f"{board_side(rooms_per_side, c)}x{board_side(rooms_per_side, c)} boards "
        f"({rooms_per_side}x{rooms_per_side} rooms of {c}x{c}, "
        f"~{_rockworld_state_estimate(board_side(rooms_per_side, c), rock_density)} states)"
        for c in room_sides)
    print(f"warning: rockworld {sizes} should converge, but each repetition "
          f"determinizes every model over board^2 x 2^{MAX_VALUABLE_ROCKS} "
          f"states, so they are much slower "
          f"than the other games — they are {hi - lo:.0f}% of the progress bar, "
          f"from {lo:.0f}% to {hi:.0f}%.")


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    conditions, stage_times, time_fields, query_fields = column_layout(args.h2)

    # Overcooked has no grid size, so it gets one job per human count; the four
    # grid domains get the full (room side x humans) cross product.
    jobs = [(g, c, h) for g in args.games if g in GRID_GAMES
            for c in args.room_sides for h in args.humans]
    if "overcooked" in args.games:
        jobs += [("overcooked", None, h) for h in args.humans]

    # State the geometry once, up front: --room-sides names the room, and every
    # board size printed from here on is the derived one.
    if any(g in GRID_GAMES for g in args.games):
        r = args.rooms_per_side
        boards = ", ".join(f"{board_side(r, c)}x{board_side(r, c)}" for c in args.room_sides)
        print(f"[geometry] {r}x{r} rooms of {args.room_sides} cells a side "
              f"-> boards {boards}"
              + ("  (one room, no walls: the open board)" if r == 1
                 else f"  ({2 * r * (r - 1)} one-way doors per board)"))

    _warn_slow_rockworld(jobs, args.num_simu, args.rock_density, args.rooms_per_side)

    time_rows, query_rows = [], []
    # One tqdm tick per repetition, so the bar reflects the real work: a
    # repetition rebuilds the instance (a fresh random map for the grid games)
    # and re-runs the whole pipeline on it.
    bar = tqdm(total=len(jobs) * args.num_simu, desc="benchmark", unit="rep")
    for job_idx, (game, room_side, num_humans) in enumerate(jobs):
        # The board, not the room side: it is what the reader of a progress bar
        # wants, and what every other size in the output refers to.
        board = None if room_side is None else board_side(args.rooms_per_side, room_side)
        label = game if board is None else f"{game} {board}x{board}"
        # Retry status lands in the bar's own postfix, which already redraws in
        # place: a 100 000-draw loop then costs no scrollback and no second writer
        # competes with the bar for the line.  The job label stays as the prefix,
        # so the bar never stops saying which configuration it is running.
        postfix = f"{label}, {num_humans} humans"
        bar.set_postfix_str(postfix)
        set_status_sink(lambda text, p=postfix:
                        bar.set_postfix_str(p if text is None else f"{p} | {text}"))

        reps, counts_per_rep, successes = [], [], []
        for rep in range(args.num_simu):
            # A distinct seed per repetition — that is what makes the average
            # meaningful for the grid games: each repetition is a new map.
            seed = args.seed + job_idx * args.num_simu + rep
            random.seed(seed)
            np.random.seed(seed)

            t_start = time.perf_counter()
            instance, bundle, t_build, t_rejected, n_draws = draw_instance_under_cap(
                game, room_side, num_humans, seed, args, args.max_bottlenecks)

            if instance is None:
                note = (f"no instance under {args.max_bottlenecks} bottlenecks "
                        f"in {MAX_INSTANCE_TRIES:,} draws")
                tqdm.write(f"skipping repetition: {note}")
                row = {f: float("nan") for f in STAGE_SIZES + stage_times}
                row["n_humans"] = num_humans
                row["skipped"] = note
                counts  = {c: float("nan") for c in conditions}
                success = float("nan")
            else:
                # The evaluated human is drawn inside run_instance off the global
                # numpy stream.  Reseeding with the *repetition* seed here keeps
                # that draw tied to the repetition rather than to whichever draw
                # happened to be accepted, so which human is evaluated does not
                # silently depend on how many maps were rejected first.
                random.seed(seed)
                np.random.seed(seed)
                row, counts, success = run_instance(
                    *instance, max_exact_n=args.max_exact_n,
                    filter_toboggans=(game == "overcooked"),
                    max_bottlenecks=args.max_bottlenecks, use_h2=args.h2,
                    bottleneck_bundle=bundle)
            row["t_build"]    = t_build
            row["t_rejected"] = t_rejected
            row["n_draws"]    = n_draws
            row["t_total"]    = time.perf_counter() - t_start

            reps.append(row)
            counts_per_rep.append(counts)
            successes.append(success)
            bar.update(1)
        set_status_sink(None)

        # room_side = what --room-sides asked for; board_side = what it built.
        # Overcooked has neither, so both stay blank for it.
        key = {"game": game, "room_side": "" if room_side is None else room_side,
               "rooms_per_side": "" if room_side is None else args.rooms_per_side,
               "board_side": "" if board is None else board,
               "num_humans": num_humans, "num_simu": args.num_simu}
        time_rows.append(_aggregate_times(key, reps, stage_times))
        query_rows.append(_aggregate_queries(key, reps, counts_per_rep, successes,
                                             conditions))
    bar.close()

    times_path   = os.path.join(args.out_dir, "compute_times.csv")
    queries_path = os.path.join(args.out_dir, "query_counts.csv")
    _write_csv(times_path, time_fields, time_rows)
    _write_csv(queries_path, query_fields, query_rows)
    query_plot_path, times_plot_path = _make_plots(times_path, queries_path,
                                                   args.out_dir, conditions)

    print(f"\n{len(time_rows)} configurations x {args.num_simu} repetitions")
    print(f"  mean stage times   -> {times_path}")
    print(f"  mean query counts  -> {queries_path}")
    print(f"  query counts plot  -> {query_plot_path}")
    print(f"  compute times plot -> {times_plot_path}")
    _print_summary(time_rows, query_rows, conditions)


def _nanmean(values):
    """Mean ignoring NaNs, and NaN (not a warning) when everything is NaN."""
    arr = np.asarray([v for v in values], dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _nanstd(values):
    arr = np.asarray([v for v in values], dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.std()) if arr.size else float("nan")


def _aggregate_times(key, reps, stage_times):
    """Average every stage time and problem size over the repetitions."""
    row = dict(key)
    for field in STAGE_SIZES + stage_times:
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


def _aggregate_queries(key, reps, counts_per_rep, successes, conditions):
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
    by_cond = {c: [d.get(c, float("nan")) for d in counts_per_rep] for c in conditions}

    exact_counts = by_cond["strategic_exact"]
    # Counted on query_all, not on the VI baseline: --max-exact-n can skip VI on
    # a repetition the greedy rules still completed, and those episodes are real.
    row["n_episodes"] = int(sum(1 for c in by_cond["query_all"] if not np.isnan(c)))
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
    side = row["board_side"]
    base = row["game"] if side == "" or pd.isna(side) else f"{row['game']} {int(side)}x{int(side)}"
    return f"{base} ({int(row['num_humans'])}h)"


def _rotate_xticks(ax, labels, x):
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")


def _make_plots(times_path, queries_path, out_dir, conditions=CONDITIONS):
    """Read compute_times.csv / query_counts.csv back and render two PNGs
    into out_dir: one bar per (game, size, num_humans) combination in both.

    query_counts.png   one subplot per configuration — a (size, humans) pair for
                        the grid games, and one for Overcooked, which has no grid
                        size.  One bar per condition: the four selection rules,
                        each doubled when `conditions` carries its "+ H2" twin,
                        plus Random.  Per-config subplots rather than
                        one shared axis because the configurations differ by an
                        order of magnitude in |B|, and a shared y-axis
                        flattens the small ones into indistinguishable stubs.
    compute_times.png  2x2 grid: the robot-reachable state count
                        (n_reachable), hypothesis-space cardinality (n_I),
                        problem size (|B|), and mean wall-clock time — so
                        the cost of a combination can be read against where it
                        started and what it was solving for.

    Returns (query_plot_path, times_plot_path).
    """
    df_q = pd.read_csv(queries_path)
    df_t = pd.read_csv(times_path)
    with_h2 = any(c.endswith("_h2") for c in conditions)

    # ── query_counts.png — one subplot per configuration ────────────────────
    # A configuration is a (board side, num_humans) pair; Overcooked has no board
    # and groups on num_humans alone.  Grid games sharing a pair share a subplot,
    # one bar group per game.
    configs, seen = [], set()
    for _, r in df_q.iterrows():
        key = ("overcooked", r["num_humans"]) if r["game"] == "overcooked" \
              else ("grid", r["board_side"], r["num_humans"])
        if key not in seen:
            seen.add(key)
            configs.append(key)

    # Three across at most, except four, which goes 2x2 rather than 3+1 with two
    # blanks.  Four is the count the default sweep produces, and the only one
    # under nine where "three across" is the wrong answer: 3, 6 and 9 fill their
    # rows exactly, and 5, 7 and 8 leave blanks whatever the width.
    ncols = 2 if len(configs) == 4 else min(3, len(configs))
    nrows = int(np.ceil(len(configs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(6.5 * ncols, 4.6 * nrows))
    flat = [a for rowaxes in axes for a in rowaxes]

    for ax, key in zip(flat, configs):
        if key[0] == "overcooked":
            sub = df_q[df_q["game"] == "overcooked"]
            sub = sub[sub["num_humans"] == key[1]]
            title = f"overcooked — {key[1]} humans"
        else:
            sub = df_q[(df_q["game"] != "overcooked")
                       & (df_q["board_side"] == key[1])
                       & (df_q["num_humans"] == key[2])]
            # board_side arrives as float: Overcooked leaves the column empty,
            # which makes pandas read the whole column as float.
            side  = int(key[1])
            title = f"{side}x{side} grid — {int(key[2])} humans"

        games = list(sub["game"])
        x = np.arange(len(games))
        width = 0.85 / len(conditions)
        for i, cond in enumerate(conditions):
            offset = (i - (len(conditions) - 1) / 2) * width
            ax.bar(x + offset, sub[f"{cond}_mean"], width,
                   yerr=sub[f"{cond}_std"], capsize=2,
                   color=CONDITION_COLORS[cond],
                   label=CONDITION_LABELS[cond])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("mean queries per episode")
        _rotate_xticks(ax, games, x)

    for ax in flat[len(configs):]:      # unused cells in the last row
        ax.axis("off")

    # One shared legend — repeating it per subplot would eat the axes.  With H2
    # on, each rule gets ONE entry whose swatch is its light|dark pair, so the
    # legend has five entries instead of nine and the shade convention is shown
    # rather than spelled out.  Nine flat entries also laid out badly: a legend
    # fills column-major, so "VI + H2" ended up stacked above
    # "H1".  With H2 off the same five entries carry a single swatch each.
    pair_handles = [Patch(facecolor=CONDITION_COLORS["query_all"])]
    pair_labels  = [CONDITION_LABELS["query_all"]]
    for b in BASES:
        shades = (Patch(facecolor=CONDITION_COLORS[b]),)
        if with_h2:
            shades += (Patch(facecolor=CONDITION_COLORS[f"{b}_h2"]),)
        pair_handles.append(shades)
        pair_labels.append(BASE_LABELS[b])
    fig.legend(pair_handles, pair_labels, loc="lower center",
               ncol=len(pair_labels), fontsize=10, frameon=False,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
               handlelength=3.0, handletextpad=0.6, columnspacing=2.4,
               title=("left bar = rule alone   ·   right bar = rule + H2"
                      if with_h2 else "one bar per selection rule"),
               title_fontsize=9)
    fig.suptitle("Query conditions — four selection rules"
                 + (", with and without H2" if with_h2 else ""),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.09, 1, 0.97])
    query_plot_path = os.path.join(out_dir, "query_counts.png")
    fig.savefig(query_plot_path, dpi=150)
    plt.close(fig)

    # ── compute_times.png ───────────────────────────────────────────────────
    labels_t = df_t.apply(_combo_label, axis=1)
    x_t = np.arange(len(labels_t))

    fig, axes = plt.subplots(2, 2, figsize=(max(11, len(labels_t) * 1.1), 10))

    # n_reachable, not n_states: how much of the state space the robot can
    # actually get to.  On Overcooked the two differ by more than two orders of
    # magnitude — ~38k states on paper, a few hundred reachable.
    axes[0, 0].bar(x_t, df_t["n_reachable"], color="darkorange")
    axes[0, 0].set_ylabel("n_reachable")
    axes[0, 0].set_title("State space reachable from the start")
    _rotate_xticks(axes[0, 0], labels_t, x_t)

    axes[0, 1].bar(x_t, df_t["n_I"], color="mediumseagreen")
    axes[0, 1].set_ylabel("|I|")
    axes[0, 1].set_title("Hypothesis-space cardinality (Algorithm 1 output)")
    _rotate_xticks(axes[0, 1], labels_t, x_t)

    axes[1, 0].bar(x_t, df_t["n_B"], color="steelblue")
    axes[1, 0].set_ylabel("|B|")
    axes[1, 0].set_title("Problem size")
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


def _print_summary(time_rows, query_rows, conditions=CONDITIONS):
    """One line per configuration — everything shown is a mean over num_simu."""
    short = {"query_all": "rand", "strategic_exact": "VI", "info_gain": "H1",
             "proximity": "H3", "frequency": "H4"}
    short.update({f"{b}_h2": f"{short[b]}+2" for b in BASES})
    header = "".join(f"{short[c]:>8}" for c in conditions)
    print(f"\n{'game':<12}{'board':>6}{'hum':>5}{'|B|':>7}{header}{'time(s)':>10}")
    for row, q in zip(time_rows, query_rows):
        if q["n_episodes"]:
            queries = "".join(f"{q[f'{c}_mean']:8.2f}" for c in conditions)
        else:
            queries = f"{row['skipped']:>{8 * len(conditions)}}"
        print(f"{row['game']:<12}{str(row['board_side']):>6}{row['num_humans']:>5}"
              f"{row['n_B']:>7.1f}{queries}{row['t_total']:>10.2f}")


if __name__ == "__main__":
    main()


