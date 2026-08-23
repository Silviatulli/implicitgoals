"""Does the density scalar have to come from the evaluation humans?

Retrains the best configuration with a leave-one-out constant -- the mean density of
OTHER instances -- instead of this instance's own.  If the score holds, the column
carries no test-set knowledge beyond a domain constant.
"""
import os, sys
import numpy as np
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "big_Q_games"))
from dqn import build_instance, train, per_human

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 150_000
POOL = [s for s in range(12) if s != TARGET]

dens = []
for sd in POOL:
    i2 = build_instance(sd)
    dens.append(float(i2["hs"].sum(1).mean()) / i2["I_array"].shape[1])
p_loo = float(np.mean(dens))

ins = build_instance(TARGET)
p_own = float(ins["hs"].sum(1).mean()) / ins["I_array"].shape[1]
print(f"instance {TARGET}: own density {p_own:.4f}, leave-one-out {p_loo:.4f} "
      f"({100*abs(p_loo-p_own)/p_own:.1f}% apart)\n")

out = {}
for label, p in (("own density (leaky)", p_own), ("leave-one-out (clean)", p_loo)):
    v = []
    for sd in (0, 1):
        net, _ = train(ins, seed=sd, mode="uniform_avg", railed=True, n_steps=STEPS,
                       p_const=p, verbose=False)
        v.append(per_human(net, ins, railed=True))
        print(f"  {label:<24} seed {sd}: {v[-1].mean():.2f}", flush=True)
    out[label] = np.mean(v, axis=0)
a, b = out["own density (leaky)"], out["leave-one-out (clean)"]
print(f"\nown {a.mean():.2f}   leave-one-out {b.mean():.2f}   diff {b.mean()-a.mean():+.2f}"
      f"   p = {stats.ttest_rel(b, a).pvalue:.4f}")
np.save(os.path.join(ROOT, "big_Q_games", "results", "loo_check.npy"), np.vstack([a, b]))
