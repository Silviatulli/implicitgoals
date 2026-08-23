"""Assemble one per-human query-count table for every policy in the study.

Per-human, never per-episode: the same 12 humans face every policy, so the
design is paired and every comparison is on per-human differences.

Writes results/benchmark_per_human.csv with columns policy,human,n_queries.
Classical policies are recomputed here (seconds); DQN rows are read from the
dqn_humans_*.csv files that dqn.py appends as each run finishes.
"""
import os, sys, csv, glob
from itertools import combinations
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "big_Q_games"))
from bottlenecks import (evaluate_policy_on_real_human, solve_query_mdp_info_gain,
                         solve_query_mdp_proximity, solve_query_mdp_frequency)
from dqn import build_instance, per_human
from rails import RailsOnlyNet, RailsEntropyNet, RailsStaticNet, RailsProbNet

RES = os.path.join(ROOT, "big_Q_games", "results")


def floor_queries(bits, I_array, B):
    r"""Fewest queries ANY policy could need against a human owning `bits`.

    Representable (S_i inside some I_k): only truthful NOs shrink I_hat, so the cost
    is |B \ I_k| for the roomiest I_k containing S_i; failure can never fire.
    Not representable: success can never fire, so it is the fewest YESes fitting in
    no hypothesis at all.
    """
    covering = [k for k in range(len(I_array)) if (I_array[k] | ~bits).all()]
    if covering:
        return min(int((~I_array[k]).sum()) for k in covering)
    owned = np.flatnonzero(bits)
    for size in range(1, len(owned) + 1):
        for Y in combinations(owned, size):
            m = np.zeros(len(B), bool); m[list(Y)] = True
            if not any((I_array[k] | ~m).all() for k in range(len(I_array))):
                return size
    return int(bits.sum())


def classical(ins):
    I, B, I_array, b2i = ins["I"], ins["B"], ins["I_array"], ins["b_to_int"]
    def q(pol, runs=1):
        return np.array([evaluate_policy_on_real_human(
            true_bottlenecks=list(s), policy_network=pol, n_runs=runs,
            I_array=I_array, b_to_int=b2i)["n_queries"].mean() for s in ins["oracle_sets"]])
    out = {}
    out["Query All"] = q(None, runs=50)          # random order; averaged over 50 episodes
    out["H1 Info Gain"] = q(solve_query_mdp_info_gain(I, B, oracle=ins["oracle"]))
    out["H3 Goal Proximity"] = q(solve_query_mdp_proximity(
        I, B, oracle=ins["oracle"], positions=ins["positions"], goal_state=ins["goal"]))
    out["H4 Query Frequency"] = q(solve_query_mdp_frequency(I, B, oracle=ins["oracle"]))
    # rails with no learning: averaged over 20 random tier-2 orderings
    out["Rails only (no NN)"] = np.mean(
        [per_human(RailsOnlyNet(I_array, seed=s), ins) for s in range(20)], axis=0)
    out["Rails + entropy"] = per_human(RailsEntropyNet(I_array), ins)
    out["Rails + P(YES) order"] = np.mean(
        [per_human(RailsProbNet(I_array, ins["probs"], seed=s), ins) for s in range(20)], axis=0)
    out["Rails static (H3-style)"] = np.mean(
        [per_human(RailsStaticNet(I_array, seed=s), ins) for s in range(20)], axis=0)
    out["floor"] = np.array([floor_queries(hs, I_array, B) for hs in ins["hs"]])
    out["|B|"] = np.full(len(ins["oracle_sets"]), len(B), dtype=float)
    return out


LABEL = {"joint": "DQN joint (leaky)", "marginal": "DQN marginal",
         "uniform": "DQN uniform 1/2", "uniform_avg": "DQN uniform avg"}


def dqn_rows():
    """Per-human means over whatever seeds each dqn_humans_*.csv currently holds."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RES, "dqn_humans_*.csv"))):
        tag = os.path.basename(path)[len("dqn_humans_"):-len(".csv")]
        railed = tag.endswith("_railed")
        base = tag[:-len("_railed")] if railed else tag
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        seeds = sorted({int(r["seed"]) for r in rows})
        v = np.mean([[float(r["n_queries"]) for r in rows if int(r["seed"]) == s]
                     for s in seeds], axis=0)
        out[LABEL.get(base, base) + (" + rails" if railed else "")] = (v, len(seeds))
    return out


if __name__ == "__main__":
    ins = build_instance(0)
    rows = [["policy", "human", "n_queries", "n_seeds"]]
    for name, v in classical(ins).items():
        for i, x in enumerate(v):
            rows.append([name, i, round(float(x), 4), ""])
    for name, (v, ns) in dqn_rows().items():
        for i, x in enumerate(v):
            rows.append([name, i, round(float(x), 4), ns])
    path = os.path.join(RES, "benchmark_per_human.csv")
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    import collections
    agg = collections.defaultdict(list)
    for r in rows[1:]:
        agg[r[0]].append(r[2])
    print(f"{'policy':<26} {'mean':>7} {'std':>7}  seeds")
    for k, v in sorted(agg.items(), key=lambda kv: np.mean(kv[1])):
        ns = next((r[3] for r in rows[1:] if r[0] == k), "")
        print(f"{k:<26} {np.mean(v):>7.2f} {np.std(v, ddof=1):>7.2f}  {ns}")
    print(f"\n-> {path}")
