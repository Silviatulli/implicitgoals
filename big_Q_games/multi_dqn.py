"""Does the railed DQN's advantage survive off map seed 0?

Trains the best configuration (uniform_avg oracle + rails) on several instances and
scores it against the best training-free policy on the SAME instance, so the comparison
is paired at the instance level.

150k steps rather than 300k: the seed-0 curves are within 0.5 queries of their final
value by step 76,000 and flat from 100k, so the second half bought nothing.
"""
import os, sys, csv, time
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "big_Q_games"))
from dqn import build_instance, train, per_human
from rails import RailsProbNet, RailsOnlyNet
from benchmark import floor_queries
from bottlenecks import evaluate_policy_on_real_human, solve_query_mdp_proximity

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 150_000
OUT = os.path.join(ROOT, "big_Q_games", "results", "multi_dqn.csv")

rows = [["inst_seed", "n_B", "n_I", "policy", "mean_queries"]]
print(f"{'inst':>4} {'|B|':>4} {'|I|':>4} {'QueryAll':>9} {'H3':>7} {'rails+P':>8} "
      f"{'DQN+rails':>10} {'floor':>7} {'sec':>6}")
acc = {k: [] for k in ("Query All", "H3", "Rails + P(YES)", "DQN uniform_avg + rails", "floor")}
for sd in range(N):
    ins = build_instance(sd)
    I_array, B = ins["I_array"], ins["B"]
    qa = np.array([evaluate_policy_on_real_human(
        true_bottlenecks=list(s), policy_network=None, n_runs=20,
        I_array=I_array, b_to_int=ins["b_to_int"])["n_queries"].mean()
        for s in ins["oracle_sets"]])
    h3 = per_human(solve_query_mdp_proximity(ins["I"], B, oracle=ins["oracle"],
                   positions=ins["positions"], goal_state=ins["goal"]), ins)
    rp = np.mean([per_human(RailsProbNet(I_array, ins["probs"], seed=s), ins)
                  for s in range(5)], axis=0)
    fl = np.array([floor_queries(h, I_array, B) for h in ins["hs"]])
    t = time.time()
    net, _ = train(ins, seed=0, mode="uniform_avg", railed=True, n_steps=STEPS, verbose=False)
    dq = per_human(net, ins, railed=True)
    dt = time.time() - t
    res = {"Query All": qa, "H3": h3, "Rails + P(YES)": rp,
           "DQN uniform_avg + rails": dq, "floor": fl}
    for k, v in res.items():
        acc[k].append(v.mean())
        rows.append([sd, len(B), len(ins["I"]), k, round(float(v.mean()), 3)])
    print(f"{sd:>4} {len(B):>4} {len(ins['I']):>4} {qa.mean():>9.2f} {h3.mean():>7.2f} "
          f"{rp.mean():>8.2f} {dq.mean():>10.2f} {fl.mean():>7.2f} {dt:>6.0f}", flush=True)
    with open(OUT + ".partial", "w", newline="") as f:
        csv.writer(f).writerows(rows)   # promoted to OUT only on completion

A = {k: np.array(v) for k, v in acc.items()}
print(f"\n{'='*72}\nAcross {N} instances")
for k in sorted(A, key=lambda k: A[k].mean()):
    print(f"  {k:<26} {A[k].mean():>7.2f} +/- {A[k].std(ddof=1):>5.2f}")
print(f"\nPaired across instances (n = {N}):")
for a, b in (("DQN uniform_avg + rails", "Rails + P(YES)"),
             ("DQN uniform_avg + rails", "H3"),
             ("Rails + P(YES)", "H3")):
    d = A[a] - A[b]
    print(f"  {a:<26} vs {b:<16} {d.mean():+7.2f}  wins {int((d<0).sum())}/{N}"
          f"  p = {stats.ttest_rel(A[a], A[b]).pvalue:.5f}")
gap = lambda k: 100 * np.mean((A["Query All"] - A[k]) / (A["Query All"] - A["floor"]))
print(f"\ngap to floor closed:")
for k in ("H3", "Rails + P(YES)", "DQN uniform_avg + rails"):
    print(f"  {k:<26} {gap(k):>5.0f}%")
# A killed run leaves OUT untouched and its progress in OUT.partial.
os.replace(OUT + ".partial", OUT)
print(f"\n-> {OUT}")
