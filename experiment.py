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
the door itself.

The defaults are R=5, C=5: a 25x25 board of twenty-five 5x5 rooms, with 40
one-way doors.  --rooms-per-side 1 is the degenerate case — one room, no walls,
an open C x C board — and |I| = 1 on it, for the reason given just below.

Every wall between two neighbouring rooms carries exactly one door, and it is
**one-way** — east or south only — sitting at the middle of that wall.

Which parts of an instance are shared, and which vary, is a deliberate design:
the start, the goal, the doors, the taxi's passenger, RockWorld's rocks and
PuddleWorld's puddles are identical across the robot and every candidate human,
and **only the obstacles differ**.  That is what keeps the query set small.  B is
the *union* of the humans' bottleneck sets, so anything drawn per model puts each
human's mandatory waypoints on different cells and makes |B| grow linearly with
--humans — which Algorithm 1's 2^|B| cost cannot absorb.  Sharing them instead
makes the humans disagree about which *route* is forced, which is the interesting
disagreement.

The orientation is the point.  On an open grid a single trajectory can tour every
bottleneck and come back, so Algorithm 1 always finds exactly one maximally
achievable subset — |I| = 1 whatever the obstacle layout.  One-way doors make the
room grid a DAG: stepping through a door commits, the doors of the routes not
taken become unreachable, and |I| grows to the number of monotone routes through
the room grid: C(2(R-1), R-1), so 6 for 3x3, 20 for 4x4 and 70 for the default
5x5.

--humans is bounded by the same 2^|B| cost, since B is the union over the humans.

Not every random map is usable, and a repetition *redraws* until it gets one that
is — a fresh seed, a fresh map, the previous one discarded.  Two conditions have
to hold (see draw_instance_under_cap):

    |B| <= --max-bottlenecks   affordable: Algorithm 1 is a 2^|B| search and the
                               exact solver allocates 3^|B| knowledge states.
    |I| >= --min-hypotheses    interesting: with a single hypothesis there is
                               nothing to ask about, every rule stops at once and
                               all five conditions tie at zero queries.

So every configuration reports the full --num-simu repetitions.  The price is
that the maps are sampled *conditioned* on both, which is not a neutral sample:
it keeps the maps with fewer mandatory waypoints and more surviving routes.
`n_draws` is the mean number of maps drawn per repetition and is what makes the
strength of that conditioning readable — 1.0 means neither condition ever bound,
and 48.0 means 47 maps were built and thrown away for every one reported.
Raising --humans therefore costs draws rather than repetitions.  Overcooked is
barely affected — its bottlenecks come from recipes, not maps.

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
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from tqdm import tqdm

# ── Benchmark domains ─────────────────────────────────────────────────────────
from gridworld import (
    generate_determinized_models as generate_determinized_gridworlds,
    plan_determinized_models as plan_gridworlds,
)
from puddleworld import (
    generate_determinized_models as generate_determinized_puddleworlds,
    plan_determinized_models as plan_puddleworlds,
)
from rockworld import (
    generate_determinized_models as generate_determinized_rockworlds,
    plan_determinized_models as plan_rockworlds,
)
from taxiworld import (
    generate_determinized_models as generate_determinized_taxiworlds,
    plan_determinized_models as plan_taxiworlds,
)

from overcooked_env import (
    # no-movement MDP: state = inv * NUM_POT + pot
    build_transition_matrix_nomove,
    serving_matrices_nomove,
    # encoding constants
    CLIENT_SERVED,
)
from gridworld_core import (board_side, status_line, set_status_sink,
                            RETRY_REPORT_EVERY, UnsolvableLayout)
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
    extract_bottlenecks,
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
# H2 is unsound and is not in the paper.  The "+ H2" columns are lower than their
# twins because the mask answers questions the oracle was never asked, not because
# it asks better ones — do not read the gap as a saving.
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
                        puddle_density=0.1, rock_density=0.1, rooms_per_side=3,
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
    "one real build divided by num_simu" rather than the cost of building one
    instance.  That is why there is no cache here — please do not add one.
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


def compute_hypothesis_space(T_R, B, start_state, goal_state):
    """Algorithm 1 over the query set, timed.  Returns (I, t_algorithm1).

    Split out for the same reason compute_bottleneck_sets was: the draw loop has
    to know |I| before it accepts an instance, and running it there means it must
    not run again in run_instance.
    """
    t0 = time.perf_counter()
    with _quiet():
        I = find_maximally_achievable_subsets(B, T_R, start_state,
                                              goal_state, verbose=False)
    return I, time.perf_counter() - t0


GRID_PLANNERS = {
    "gridworld":   plan_gridworlds,
    "puddleworld": plan_puddleworlds,
    "rockworld":   plan_rockworlds,
    "taxiworld":   plan_taxiworlds,
}


def draw_grid_instance_lazily(game, room_side, num_humans, seed, args,
                              max_bottlenecks):
    """Build a grid instance one model at a time, abandoning it as soon as it
    cannot fit under ``max_bottlenecks``.

    B is the *union* of the humans' bottleneck sets, and a union only grows.  So
    once the running union exceeds the cap, no remaining human can bring it back
    down and none of them need building.  On taxiworld that verdict arrives after
    about 5 humans of 20, and roughly 90% of draws are rejected this way — which
    is where this run spends most of its time.

    The robot is built last, and only if the cap is cleared.  Nothing before
    Algorithm 1 reads T_R: compute_bottleneck_sets touches it solely for the
    toboggan filter, which no grid game runs.

    Identical to the eager path, because plan_determinized_models draws every
    random choice up front and building consumes no randomness — so a model gets
    the same map whether or not its siblings were skipped, and a draw that is
    accepted here is the same instance build_grid_instance would have produced.

    Returns ``(instance, oracle_sets, B_nofilter, times, t_build)`` with
    ``instance`` None when the cap was busted.  ``t_build`` counts only what was
    actually built.
    """
    extra = {}
    if game == "puddleworld":
        extra["puddle_density"] = args.puddle_density
    elif game == "rockworld":
        extra["rock_density"] = args.rock_density

    times = {}
    t_build = 0.0
    t_bottlenecks = 0.0

    t0 = time.perf_counter()
    with _quiet():
        build, n_models = GRID_PLANNERS[game](
            num_humans=num_humans, obstacle_density=args.obstacle_density,
            seed=seed, rooms_per_side=args.rooms_per_side,
            room_side=room_side, slip_prob=args.slip_prob, **extra)
    t_build += time.perf_counter() - t0

    # Models 1..num_humans are the humans; model 0 is the robot.
    oracle_sets, union = [], set()
    start_state = goal_state = None
    T_H_list = []
    for i in range(1, n_models):
        t0 = time.perf_counter()
        with _quiet():
            T_H, s0, g, _det, _mdp = build(i)
        t_build += time.perf_counter() - t0
        T_H_list.append(T_H)
        # Every model of an instance shares the board, so its state space — and
        # therefore these two indices — are the same for all of them.
        start_state, goal_state = int(s0), int(g)

        t0 = time.perf_counter()
        with _quiet():
            bset = frozenset(extract_bottlenecks(T_H, start_state, goal_state))
        t_bottlenecks += time.perf_counter() - t0
        oracle_sets.append(bset)
        union |= bset
        if len(union) > max_bottlenecks:
            # Hopeless: the union cannot shrink.  Skip the rest of the humans and
            # the robot entirely.
            times["t_bottlenecks"] = t_bottlenecks
            times["t_oracle_sets"] = 0.0
            times["t_toboggan_filter"] = float("nan")
            return None, oracle_sets, sorted(union), times, t_build

    t0 = time.perf_counter()
    with _quiet():
        T_R, _s0, _g, _det, mdp_R = build(0)
    t_build += time.perf_counter() - t0

    times["t_bottlenecks"] = t_bottlenecks
    times["t_oracle_sets"] = 0.0        # the union was accumulated above
    times["t_toboggan_filter"] = float("nan")
    return ((T_R, T_H_list, start_state, goal_state, mdp_R),
            oracle_sets, sorted(union), times, t_build)


# How many maps a single repetition may draw before it gives up and reports
# nothing.  Rejecting an instance is cheap and common (see draw_instance_under_cap
# for the two conditions it has to pass), so the budget is generous: giving up is
# meant to mean "this configuration is impossible", never "this one was unlucky".
# A repetition that exhausts it is the only way a configuration ends up averaging
# over fewer than --num-simu episodes.
MAX_INSTANCE_TRIES = 100_000

# Draw n reuses the repetition seed offset by n * this stride.  A redraw has to
# change the seed at all — every builder reseeds `random`/`numpy` from the seed it
# is handed, so redrawing with the same one would rebuild the identical map for
# ever and the loop would spin to MAX_INSTANCE_TRIES without ever varying.  The
# stride is larger than any (job_idx * num_simu + rep) a run can produce, so a
# redraw can never land on another repetition's seed and collapse two
# repetitions onto one map, which is the whole reason they are seeded apart.
# It is also small enough that the last draw's seed still fits in the 32 bits
# numpy allows, which is what the assertion below pins.
RETRY_SEED_STRIDE = 40_009
assert RETRY_SEED_STRIDE * MAX_INSTANCE_TRIES < 2 ** 32, \
    "stride x tries must fit in 32 bits: numpy seeds are bounded"


def draw_instance_under_cap(game, room_side, num_humans, seed, args,
                            max_bottlenecks, min_hypotheses=1,
                            max_tries=MAX_INSTANCE_TRIES):
    """Draw instances until one is both affordable and interesting.

    Two constraints, tested in this order because the second is far more
    expensive than the first:

      1. ``len(B) <= max_bottlenecks`` — affordability.  Algorithm 1 is a 2^|B|
         DFS and the exact solver allocates 3^|B| knowledge states, so an
         oversized query set has to be rejected *before* either runs.
      2. ``len(I) >= min_hypotheses`` — interest.  A repetition with one
         hypothesis poses no question at all: I_hat is already inside the only
         I_k, every rule terminates at once, and all five conditions tie at zero
         queries.  Those instances say nothing about which rule is better, so
         they are drawn past rather than reported.

    Returns (instance, bundle, t_build, t_rejected, n_draws, t_accepted).
    `instance` is None
    when all `max_tries` draws failed one of the two, which skips the repetition
    — the one case left where a repetition produces no episode.  The bundle
    carries I, so Algorithm 1 is not run a second time downstream.

    `t_build` times the accepted draw alone, so it means "what one instance costs
    to build" and nothing else.  Everything burned on the rejected draws — their
    builds, their bottleneck passes *and* their Algorithm 1 runs — is reported
    separately, as `t_rejected`.
    `t_accepted` is how long the accepted attempt took, from fixing its seed to
    having I in hand.  The caller adds what the solves and simulations then cost
    to get `t_total`, so the reported figure is one instance's pipeline and
    excludes the search for a usable map.

    A duration rather than a start timestamp because the two halves need not run
    in the same process: with --jobs above 1 the draw happens in a worker and the
    solve later in the parent, and a wall-clock span across them would be mostly
    queueing.  Summing the two measured intervals is what one instance costs
    either way.

    Sampling note: neither constraint is free.  Conditioning on |B| keeps the
    maps with fewer mandatory waypoints; conditioning on |I| keeps the ones whose
    routes survive the obstacles, which is a much stronger filter — the raw |I|
    distribution is dominated by 1.  Every mean the run reports therefore
    describes that doubly-conditioned population, and `n_draws` is what makes the
    strength of it readable.
    """
    filter_toboggans = (game == "overcooked")
    t_rejected = 0.0
    for attempt in range(max_tries):
        if attempt and attempt % RETRY_REPORT_EVERY == 0:
            status_line(f"draw {attempt:,}/{max_tries:,} before skipping the config")
        t0 = time.perf_counter()
        attempt_seed = seed + attempt * RETRY_SEED_STRIDE
        if game == "overcooked":
            # Eager: the toboggan filter can *shrink* B below the cap, so an
            # oversized raw union proves nothing here and there is no sound point
            # to stop early at.  Overcooked never redraws anyway (n_draws = 1.0).
            instance, t_build = build_overcooked_instance(
                num_humans=num_humans, allow_drop=args.overcooked_allow_drop,
                seed=attempt_seed)
            T_R, T_H_list, start_state, goal_state = instance[:4]
            oracle_sets, B_nofilter, B, times = compute_bottleneck_sets(
                T_R, T_H_list, start_state, goal_state, filter_toboggans)
        else:
            # Lazy: stop building the moment the running union busts the cap.
            instance, oracle_sets, B_nofilter, times, t_build = \
                draw_grid_instance_lazily(game, room_side, num_humans,
                                          attempt_seed, args, max_bottlenecks)
            B = B_nofilter          # no grid game runs the toboggan filter
            if instance is None:
                t_rejected += time.perf_counter() - t0
                continue
            _, _, start_state, goal_state = instance[:4]
        if len(B) <= max_bottlenecks:
            # Only now is Algorithm 1 affordable to run, and it has to run here
            # rather than downstream because |I| is the second constraint.
            I, times["t_algorithm1"] = compute_hypothesis_space(
                instance[0], B, start_state, goal_state)
            if len(I) >= min_hypotheses:
                status_line(None)
                return (instance, (oracle_sets, B_nofilter, B, I, times),
                        t_build, t_rejected, attempt + 1,
                        time.perf_counter() - t0)
        # Rejected on one constraint or the other: this instance and its matrices
        # go out of scope here and the next iteration builds a fresh one over
        # them.  Only the cost survives.
        t_rejected += time.perf_counter() - t0
    status_line(None)
    return None, None, float("nan"), t_rejected, max_tries, float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# The experiment — one repetition: one instance in, one timing row out
# ─────────────────────────────────────────────────────────────────────────────

def run_instance(T_R, T_H_list, start_state, goal_state, mdp_R=None,
                 max_exact_n=17, filter_toboggans=False, max_bottlenecks=17,
                 use_h2=False, bottleneck_bundle=None):
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
        (oracle_sets, B_nofilter, B, I, stage_times), as draw_instance_under_cap
        returns it.  main() passes it in because it already had to compute both
        |B| and |I| to decide whether to keep this instance at all, and neither
        stage should run twice.  None recomputes both, which is the standalone
        path for running one instance with no redraw loop around it.

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
        is not in the paper — see bottlenecks.build_dominance.

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
    # for Overcooked).  A BFS over the determinized matrix, so it costs nothing —
    # which is why it is computed up front and every early return below can carry
    # the column.
    row: dict = {"n_states": int(T_R.shape[0]), "n_actions": int(T_R.shape[1]),
                 "n_humans": len(T_H_list),
                 "n_reachable": int(get_reachable_states(
                     np.asarray(T_R), int(start_state)).sum())}

    # Normally precomputed: main() has to know |B| *before* it accepts an
    # instance, so it runs these three stages itself and hands the result down
    # rather than paying for them twice.  Recomputing is the standalone path.
    if bottleneck_bundle is None:
        # Standalone path: no redraw loop above, so Algorithm 1 has not run yet
        # and I is filled in after the cap check below.
        oracle_sets, B_nofilter, B, stage_times = compute_bottleneck_sets(
            T_R, T_H_list, start_state, goal_state, filter_toboggans)
        I = None
    else:
        oracle_sets, B_nofilter, B, I, stage_times = bottleneck_bundle
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

    if I is None:
        I, row["t_algorithm1"] = compute_hypothesis_space(
            T_R, B, start_state, goal_state)

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

    # H3 ranks bottlenecks by straight-line distance to the goal, so building it
    # is a coordinate lookup and a norm — no model, no value function.  This timer
    # is therefore microseconds, unlike the VI baseline's.
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


def _draw_one(payload):
    """Phase 1 for a single repetition: draw until an instance is accepted.

    Runs in a worker process when --jobs > 1, and inline when it is 1.  Returns
    ``(instance, bundle, t_build, t_rejected, n_draws, t_accepted, note)``, where
    `note` is None on success and the skip reason otherwise.

    Safe to run out of order, in another process, or not at all, because **a
    repetition is a pure function of its seed**.  main() reseeds both RNGs from a
    seed it computes arithmetically, draw_instance_under_cap reseeds again per
    attempt from ``seed + attempt * RETRY_SEED_STRIDE``, and
    plan_determinized_models makes every remaining draw at a known point.  No
    state carries from one repetition to the next, so scheduling cannot change
    what any of them produces.
    """
    game, room_side, num_humans, seed, args, max_bottlenecks, min_hyp = payload
    # A worker has no progress bar to write to, and several of them sharing
    # stdout would shred the parent's.  Swallow the retry status instead.
    set_status_sink(lambda text: None)
    random.seed(seed)
    np.random.seed(seed)
    try:
        instance, bundle, t_build, t_rejected, n_draws, t_accepted = \
            draw_instance_under_cap(game, room_side, num_humans, seed, args,
                                    max_bottlenecks, min_hypotheses=min_hyp)
        note = None if instance is not None else (
            f"no instance with |B| <= {max_bottlenecks} and "
            f"|I| >= {min_hyp} in {MAX_INSTANCE_TRIES:,} draws")
    except UnsolvableLayout as exc:
        # The generator could not lay out a solvable board at these densities.
        # It raises instead of substituting an obstacle-free one, so the
        # repetition is reported as NaN rather than quietly averaging in a board
        # the run never asked for.  Caught per repetition so one impossible
        # configuration cannot take the whole sweep down with it.
        instance = bundle = None
        t_build = t_rejected = n_draws = t_accepted = float("nan")
        note = str(exc)
    return instance, bundle, t_build, t_rejected, n_draws, t_accepted, note


def resolve_jobs(requested):
    """How many worker processes phase 1 should use.

    ``0`` means auto: one per core bar one, capped at 8.

    Two reasons for the cap, neither of them cores.  Memory: a worker imports
    this module, torch included, and sits at ~275 MB RSS whether or not it is
    drawing — 8 of them is ~2.2 GB held for the whole run, on top of the ~4.7 GB
    phase 2 peaks at when |B| = 17.  And diminishing returns: phase 2 is ~14% of
    the run and stays sequential, so Amdahl caps the whole sweep near 7x however
    many workers phase 1 gets.

    Auto returns 1 on a single- or dual-core machine, so the default degrades to
    the sequential path on small hardware with nothing to configure.  ``--jobs 1``
    forces it everywhere, and a larger value than the cap can be asked for
    explicitly.
    """
    if requested and requested > 0:
        return int(requested)
    return max(1, min((os.cpu_count() or 1) - 1, 8))


def draw_phase(game, room_side, num_humans, seeds, args, pool, bar, postfix):
    """Phase 1 for every repetition of one configuration, in seed order.

    ``pool`` is a ProcessPoolExecutor, or None to draw inline.  Only the *draw*
    is parallelised: it is ~86% of the run and costs about a megabyte per
    instance, whereas phase 2 allocates 3^|B| knowledge states — ~4.7 GB at
    |B| = 17 — and would exhaust memory long before it exhausted cores.  So the
    expensive-in-time half is spread out and the expensive-in-memory half is
    left alone.

    The pool is owned by main() and outlives every configuration, because a
    worker pays for importing this module — torch and matplotlib included — the
    moment it starts.  Spawning one pool per configuration would pay that
    fifteen times over for a sweep that runs each of them once.

    Results are indexed by repetition, never by completion order, so the rows are
    assembled exactly as a sequential run assembles them.
    """
    payloads = [(game, room_side, num_humans, s, args,
                 args.max_bottlenecks, args.min_hypotheses) for s in seeds]
    n = len(payloads)
    if pool is None:
        out = []
        for i, p in enumerate(payloads):
            bar.set_postfix_str(f"{postfix} | drawing {i + 1}/{n}")
            out.append(_draw_one(p))
        bar.set_postfix_str(postfix)
        return out

    results = [None] * n
    futures = {pool.submit(_draw_one, p): i for i, p in enumerate(payloads)}
    for done, fut in enumerate(as_completed(futures), start=1):
        results[futures[fut]] = fut.result()
        bar.set_postfix_str(f"{postfix} | drawing {done}/{n}")
    bar.set_postfix_str(postfix)
    return results


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
# t_total is the accepted instance alone — its clock starts at that draw's seed —
# so t_total + t_rejected, not t_total, is what a repetition took on the wall.
# The split is deliberate: how many maps had to be thrown away is a property of
# the generator and the two caps, not of the pipeline being measured.
STAGE_SIZES = ["n_states", "n_reachable", "n_actions", "n_humans", "n_draws",
               "n_B_nofilter", "n_B", "n_I", "n_columns"]

# What identifies a configuration.  Every aggregate is a mean over the
# repetitions sharing these, so they are also the group key when the per-episode
# file is read back.
CONFIG_KEYS = ["game", "room_side", "rooms_per_side", "board_side",
               "num_humans", "num_simu"]


def column_layout(use_h2):
    """The CSV header of a run, derived from the conditions it will produce.

    Returns (conditions, stage_times, episode_fields, time_fields, query_fields).
    Keeping them in one place is what keeps the H2-off run from writing empty
    `*_h2` columns nothing filled in.

    `episode_fields` is the header of the one file a run actually writes: one row
    per repetition, carrying every size, every stage time and every condition's
    query count.  `time_fields` and `query_fields` describe the two *aggregate*
    views, which are means over the repetitions and are derived from that file
    rather than accumulated alongside it — see aggregate_episodes().

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

    # One row per repetition: the raw record everything else is derived from.
    # `rep` and `seed` identify it; `success` and the per-condition query counts
    # are what the aggregate query view averages.
    episode_fields = (CONFIG_KEYS + ["rep", "seed"] + STAGE_SIZES + stage_times
                      + ["success"] + list(conditions) + ["skipped"])
    return conditions, stage_times, episode_fields, time_fields, query_fields


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
    p.add_argument("--puddle-density", type=float, default=0.1,
                   help="puddle density, puddleworld only (default: 0.1, "
                        "matches --rock-density so the three grid variants "
                        "differ in kind and not in how much they scatter)")
    p.add_argument("--rock-density", type=float, default=0.1,
                   help="rock density, rockworld only (default: 0.1)")
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
    # The two caps guard different stages: --max-bottlenecks guards Algorithm 1,
    # a 2^n DFS (2^17 = 131k subsets, cheap); --max-exact-n guards the exact Query
    # MDP, which allocates 3^n knowledge states.  Keep them equal.  An instance
    # that clears the first can then always be solved exactly, so every column is
    # averaged over the same repetitions.  Were the first the larger of the two, a
    # repetition could pass Algorithm 1 and then have its VI baseline skipped, and
    # that column alone would average over a subset of the others.
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
    p.add_argument("--min-hypotheses", type=int, default=3,
                   help="redraw an instance whose hypothesis space is smaller "
                        "than this.  |I| = 1 poses no question — every rule "
                        "terminates immediately and all five conditions tie — so "
                        "such repetitions measure nothing; raising this costs "
                        "draws, and the raw |I| distribution is dominated by 1 "
                        "(default: 3)")
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
    p.add_argument("--jobs", type=int, default=0,
                   help="worker processes for the draw phase, which is ~86%% of "
                        "the run; 0 = auto (cores - 1, capped at 8), 1 = "
                        "sequential.  Only the draw is parallel: the exact Query "
                        "MDP allocates 3^|B| states (~4.7 GB at |B|=17) and stays "
                        "sequential, which also caps the whole sweep at ~7x. "
                        "Results do not depend on this: a repetition is a pure "
                        "function of its seed (default: 0)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    (conditions, stage_times, episode_fields,
     time_fields, query_fields) = column_layout(args.h2)

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

    n_jobs = resolve_jobs(args.jobs)
    if n_jobs > 1:
        print(f"[jobs] drawing on {n_jobs} worker processes; the solve stays "
              f"sequential (3^|B| states).  Results are unaffected — a "
              f"repetition is a pure function of its seed.")
    # One pool for the whole sweep: a worker imports this module on startup, and
    # that cost should be paid once, not once per configuration.
    pool = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None

    # The run's only output: one row per repetition, written as each finishes.
    # Everything reported later is a mean over these rows, computed on read.
    # Writing during the sweep rather than at the end is deliberate — a run that
    # dies in its fifth hour still leaves every repetition it completed.
    episodes_path = os.path.join(args.out_dir, "episodes.csv")
    episodes_file = open(episodes_path, "w", newline="")
    episodes = csv.DictWriter(episodes_file, fieldnames=episode_fields,
                              extrasaction="ignore")
    episodes.writeheader()

    # One tqdm tick per repetition, so the bar reflects the real work: a
    # repetition rebuilds the instance (a fresh random map for the grid games)
    # and re-runs the whole pipeline on it.
    bar = tqdm(total=len(jobs) * args.num_simu, desc="benchmark", unit="rep")

    # Everything the run has to say while the bar is up goes through one of two
    # channels, and neither writes to the terminal itself:
    #   * transient retry counts -> the bar's postfix, via this sink.  The
    #     postfix already redraws in place, so a 100 000-draw loop leaves no
    #     scrollback and no second writer competes for the line;
    #   * anything worth keeping (a skipped repetition) -> tqdm.write, which
    #     scrolls it above the bar.
    # `job` is rebound by the loop below and read here at call time, so the label
    # of the running configuration is never lost behind a status message.
    job = ""

    def show_status(text=None):
        bar.set_postfix_str(job if text is None else f"{job} | {text}")

    set_status_sink(show_status)

    for job_idx, (game, room_side, num_humans) in enumerate(jobs):
        # The board, not the room side: it is what the reader of a progress bar
        # wants, and what every other size in the output refers to.
        board = None if room_side is None else board_side(args.rooms_per_side, room_side)
        label = game if board is None else f"{game} {board}x{board}"
        job = f"{label}, {num_humans} humans"
        show_status()

        # ── phase 1: draw every repetition of this configuration ────────────
        # A distinct seed per repetition — that is what makes the average
        # meaningful for the grid games: each repetition is a new map.  The seeds
        # are arithmetic, so phase 1 needs no state from phase 2 and the draws
        # can run in any order, in any process.
        seeds = [args.seed + job_idx * args.num_simu + rep
                 for rep in range(args.num_simu)]
        drawn = draw_phase(game, room_side, num_humans, seeds, args, pool,
                           bar, job)

        # ── phase 2: solve and simulate, one repetition at a time ───────────
        # Sequential on purpose: the exact Query MDP allocates 3^|B| knowledge
        # states, so this is the memory-bound half and running it wide would
        # exhaust RAM long before it exhausted cores.
        for rep, seed in enumerate(seeds):
            (instance, bundle, t_build, t_rejected,
             n_draws, t_accepted, note) = drawn[rep]
            t_phase2 = time.perf_counter()

            if note is not None:
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
            # What one usable instance costs end to end: the accepted draw (its
            # build, determinization included, plus Algorithm 1) and then the
            # solves and simulations — never the redraws that preceded it.  The
            # two halves are added rather than spanned by one clock, because with
            # --jobs > 1 they happen in different processes and a wall-clock span
            # across them would be mostly queueing.  NaN when every draw failed:
            # there is no accepted instance to time.
            row["t_total"]    = (float("nan") if instance is None
                                 else t_accepted + time.perf_counter() - t_phase2)

            episodes.writerow({**key_of(game, room_side, board, num_humans,
                                          args),
                               "rep": rep, "seed": seed, "success": success,
                               **row, **counts})
            bar.update(1)

        episodes_file.flush()      # one configuration's worth is now safe on disk
    bar.close()
    set_status_sink(None)          # the bar is gone; status goes back to stdout
    if pool is not None:
        pool.shutdown()
    episodes_file.close()

    # Everything below is a *view* of episodes.csv, recomputed from it rather
    # than carried alongside it.
    time_rows, query_rows = aggregate_episodes(episodes_path, conditions,
                                               stage_times, time_fields,
                                               query_fields)
    query_plot_path, times_plot_path = _make_plots(time_rows, query_rows,
                                                   args.out_dir, conditions)
    tables_path = write_tables(episodes_path, query_rows, time_rows,
                               args.out_dir, conditions)

    print(f"\n{len(time_rows)} configurations x {args.num_simu} repetitions")
    print(f"  per-episode results -> {episodes_path}")
    print(f"  summary tables      -> {tables_path}")
    print(f"  query counts plot   -> {query_plot_path}")
    print(f"  compute times plot  -> {times_plot_path}")
    _print_summary(time_rows, query_rows, conditions)


def key_of(game, room_side, board, num_humans, args):
    """The configuration a repetition belongs to, as CSV columns.

    room_side is what --room-sides asked for, board_side what it built, and
    neither can be recovered from the other without rooms_per_side, so all three
    are written.  Overcooked has no geometry at all and leaves them blank.
    """
    return {"game": game,
            "room_side": "" if room_side is None else room_side,
            "rooms_per_side": "" if room_side is None else args.rooms_per_side,
            "board_side": "" if board is None else board,
            "num_humans": num_humans, "num_simu": args.num_simu}


def aggregate_episodes(episodes_path, conditions, stage_times,
                       time_fields, query_fields):
    """Rebuild the two aggregate views from the per-episode file.

    The run writes exactly one file — one row per repetition — and everything
    reported is a mean over the repetitions of a configuration, computed here.
    Nothing is accumulated during the sweep, so the file on disk is the only
    source of truth and a run that dies half way still leaves usable data.

    Returns (time_rows, query_rows): lists of dicts with the columns
    `time_fields` and `query_fields` describe.
    """
    df = pd.read_csv(episodes_path)
    # Overcooked leaves the geometry columns empty, which pandas reads as NaN;
    # a NaN key would silently drop those groups.
    for k in CONFIG_KEYS:
        if df[k].isna().any():
            df[k] = df[k].fillna("")
    df["skipped"] = df["skipped"].fillna("")

    time_rows, query_rows = [], []
    # sort=False keeps the configurations in the order the sweep ran them.
    for key_vals, sub in df.groupby(CONFIG_KEYS, sort=False, dropna=False):
        # The geometry columns come back as floats, because Overcooked leaves
        # them blank and that makes pandas type the whole column float.  Put the
        # integers back so the aggregate reads exactly as it did when it was
        # accumulated in memory: "9", not "9.0".
        key = {k: (int(v) if isinstance(v, float) and v == int(v) else v)
               for k, v in zip(CONFIG_KEYS, key_vals)}
        reps = sub.to_dict("records")
        counts_per_rep = [{c: r[c] for c in conditions} for r in reps]
        successes = [r["success"] for r in reps]
        time_rows.append(_aggregate_times(key, reps, stage_times))
        query_rows.append(_aggregate_queries(key, reps, counts_per_rep,
                                             successes, conditions))
    return time_rows, query_rows


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


def _make_plots(time_rows, query_rows, out_dir, conditions=CONDITIONS):
    """Render two PNGs into out_dir from the aggregated views of episodes.csv:
    one bar per (game, size, num_humans) combination in both.

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
                        started and what it was solving for.  The time shown is
                        t_total: the accepted instance from its seed onwards,
                        with the rejected draws (t_rejected) left out.

    Returns (query_plot_path, times_plot_path).
    """
    df_q = pd.DataFrame(query_rows)
    df_t = pd.DataFrame(time_rows)
    # The geometry columns are blank for Overcooked; make them numeric-with-NaN
    # so the grid/Overcooked split below reads the same as it did from CSV.
    for d in (df_q, df_t):
        for c in ("room_side", "rooms_per_side", "board_side"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
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
    axes[1, 1].set_ylabel("mean t_total (s)")
    axes[1, 1].set_title("Wall-clock time per instance")
    _rotate_xticks(axes[1, 1], labels_t, x_t)

    fig.tight_layout()
    times_plot_path = os.path.join(out_dir, "compute_times.png")
    fig.savefig(times_plot_path, dpi=150)
    plt.close(fig)

    return query_plot_path, times_plot_path


# ─────────────────────────────────────────────────────────────────────────────
# The two summary tables, written next to the plots
# ─────────────────────────────────────────────────────────────────────────────

def _levene_rows(episodes, a="strategic_exact", b="info_gain"):
    """Levene's test on the raw per-episode query counts, one row per domain.

    Pools the human counts, so each sample is num_simu x |--humans| episodes.
    The test needs the individual episodes, which is why it lives here and not
    in the aggregate view: means and standard deviations are not enough to run
    it.

    Levene with the median (Brown-Forsythe) rather than an F-test, because query
    counts are discrete and far from normal, which is exactly the case Levene is
    robust to.

    scipy is imported here rather than at module scope so a machine without it
    still completes a sweep; the table then reports the variance ratio and
    leaves the p-value blank.
    """
    try:
        from scipy.stats import levene
    except ImportError:
        levene = None

    rows = []
    for game, sub in episodes.groupby("game", sort=False):
        x = sub[a].dropna().to_numpy()
        y = sub[b].dropna().to_numpy()
        if len(x) < 3 or len(y) < 3:
            continue
        var_x, var_y = x.var(ddof=1), y.var(ddof=1)
        p = "" if levene is None else f"{levene(x, y, center='median')[1]:.3e}"
        rows.append([game, int(sub.n_states.iloc[0]), len(x),
                     round(var_x, 3), round(var_y, 3),
                     round(var_y / var_x, 2) if var_x else "", p])
    return sorted(rows, key=lambda r: r[1])


def write_tables(episodes_path, query_rows, time_rows, out_dir, conditions):
    """Write results/tables.csv: the two summary tables, one above the other.

    Plain CSV so it opens in a spreadsheet and reads in a terminal, with a `#`
    line naming each table.  Both are *views* of episodes.csv and are rebuilt
    from it every time, so they cannot drift from the data they describe.

    Table 1 — mean queries per condition.  Grid domains as a range across the
    four, Overcooked on its own, since the four grids agree closely and the
    range is what the paper reports.  |B| and |Phi| are included because both
    are observed rather than set: instances are redrawn until |B| <=
    --max-bottlenecks and |Phi| >= --min-hypotheses, so every mean is
    conditioned on that screen.

    Table 2 — variance of the query count, Str.VI against Info Gain, with
    Levene's p-value.
    """
    episodes = pd.read_csv(episodes_path)
    q = pd.DataFrame(query_rows)
    t = pd.DataFrame(time_rows)
    m = q.merge(t[["game", "num_humans", "n_B", "n_I"]], on=["game", "num_humans"])
    grid = m[m.game != "overcooked"]
    over = m[m.game == "overcooked"]
    humans = sorted(m.num_humans.unique())
    named = [c for c in conditions]

    def rng(sub, cond):
        return f"{sub[cond + '_mean'].min():.2f}-{sub[cond + '_mean'].max():.2f}"

    path = os.path.join(out_dir, "tables.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        n_simu = int(m.num_simu.iloc[0])
        # No commas inside these header lines: one cell holding commas is
        # re-split by anything reading the file as CSV, including `column -s,`.
        w.writerow([f"# TABLE 1 - mean queries until the human's subgoal set is "
                    f"identified ({n_simu} repetitions per configuration). "
                    f"Grid rows give the range across the four grid domains. "
                    f"|B| and |Phi| are observed rather than set."])
        w.writerow(["Domain", "Humans", "|B|", "|Phi|"]
                   + [CONDITION_LABELS[c] for c in named])
        for h in humans:
            s = grid[grid.num_humans == h]
            if len(s):
                w.writerow(["Grid family", h,
                            f"{s.n_B.min():.1f}-{s.n_B.max():.1f}",
                            f"{s.n_I.min():.1f}-{s.n_I.max():.1f}"]
                           + [rng(s, c) for c in named])
        for h in humans:
            s = over[over.num_humans == h]
            if len(s):
                r = s.iloc[0]
                # "n/a" rather than an empty cell: H3 ranks by distance to the
                # goal and Overcooked has no geometry, which is a fact about the
                # game and not a missing measurement.
                w.writerow(["Overcooked", h, f"{r.n_B:.1f}", f"{r.n_I:.1f}"]
                           + [("n/a" if pd.isna(r[c + "_mean"]) else
                               f"{r[c + '_mean']:.2f} +/- {r[c + '_std']:.2f}")
                              for c in named])

        w.writerow([])
        w.writerow(["# TABLE 2 - variance of the query count (VI baseline vs "
                    "Info Gain) by Levene's median-centred test on the raw "
                    "per-episode counts. Ratio > 1 means Info Gain varies more."])
        w.writerow(["Domain", "States", "n episodes", "var (VI)",
                    "var (Info Gain)", "ratio", "Levene p"])
        for row in _levene_rows(episodes):
            w.writerow(row)
    return path


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


