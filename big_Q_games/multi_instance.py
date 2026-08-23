"""Do the rails help on maps other than seed 0?

Everything else in this study is one hand-picked instance (seed 0 has |I| = 24, the
maximum over 40 screened seeds).  The rails are analytic and need no training, so the
population question is answerable in minutes rather than GPU-hours.
"""
import sys, os, csv
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "big_Q_games"))
from bottlenecks import (evaluate_policy_on_real_human, solve_query_mdp_info_gain,
                         solve_query_mdp_proximity, solve_query_mdp_frequency)
from dqn import build_instance, per_human
from rails import RailsOnlyNet, RailsEntropyNet
from benchmark import floor_queries

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = os.path.join(ROOT, "big_Q_games", "results", "multi_instance.csv")
rows = [["seed", "n_B", "n_I", "policy", "mean_queries"]]
per_seed = {}

print(f"{'seed':>4} {'|B|':>4} {'|I|':>4} {'QueryAll':>9} {'H1':>7} {'H3':>7} {'H4':>7} "
      f"{'rails':>7} {'r+ent':>7} {'floor':>7}")
for sd in range(N):
    ins = build_instance(sd)
    I, B, I_array = ins["I"], ins["B"], ins["I_array"]
    def q(pol, runs=1):
        return np.array([evaluate_policy_on_real_human(
            true_bottlenecks=list(s), policy_network=pol, n_runs=runs,
            I_array=I_array, b_to_int=ins["b_to_int"])["n_queries"].mean()
            for s in ins["oracle_sets"]])
    res = {
        "Query All": q(None, runs=20),
        "H1": q(solve_query_mdp_info_gain(I, B, oracle=ins["oracle"])),
        "H3": q(solve_query_mdp_proximity(I, B, oracle=ins["oracle"],
                                          positions=ins["positions"], goal_state=ins["goal"])),
        "H4": q(solve_query_mdp_frequency(I, B, oracle=ins["oracle"])),
        "Rails only": np.mean([per_human(RailsOnlyNet(I_array, seed=s), ins)
                               for s in range(5)], axis=0),
        "Rails + entropy": per_human(RailsEntropyNet(I_array), ins),
        "floor": np.array([floor_queries(h, I_array, B) for h in ins["hs"]]),
    }
    per_seed[sd] = res
    for k, v in res.items():
        rows.append([sd, len(B), len(I), k, round(float(v.mean()), 3)])
    print(f"{sd:>4} {len(B):>4} {len(I):>4} " +
          " ".join(f"{res[k].mean():>7.2f}" for k in
                   ("Query All", "H1", "H3", "H4", "Rails only", "Rails + entropy", "floor")))
    with open(OUT + ".partial", "w", newline="") as f:
        csv.writer(f).writerows(rows)   # promoted to OUT only on completion

print(f"\n{'='*74}\nAcross {N} instances -- mean of per-instance means")
names = ["Query All", "H1", "H3", "H4", "Rails only", "Rails + entropy", "floor"]
M = {k: np.array([per_seed[s][k].mean() for s in per_seed]) for k in names}
for k in sorted(names, key=lambda k: M[k].mean()):
    print(f"  {k:<18} {M[k].mean():>7.2f} +/- {M[k].std(ddof=1):>5.2f}")
print(f"\nPaired across instances (n = {N}), rails-only vs each heuristic:")
for k in ("H1", "H3", "H4", "Query All"):
    d = M["Rails only"] - M[k]
    t = stats.ttest_rel(M["Rails only"], M[k])
    print(f"  vs {k:<10} diff {d.mean():+7.2f}   wins on {int((d<0).sum())}/{N} instances"
          f"   p = {t.pvalue:.5f}")
# A killed run leaves OUT untouched and its progress in OUT.partial.
os.replace(OUT + ".partial", OUT)
print(f"\n-> {OUT}")
