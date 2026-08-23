"""How often, and where, the rails actually bite.

Walks real episodes and records the tier composition at every decision, which is
what says whether a rail can matter on this instance at all.
"""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottlenecks import _terminal
from rails import tiers, RailsOnlyNet
from dqn import build_instance


def walk(net, ins, seed=0):
    I_array = ins["I_array"]; n = I_array.shape[1]
    recs = []
    for h_i, s in enumerate(ins["oracle_sets"]):
        true_bits = {ins["b_to_int"][b] for b in s}
        K_I, K_not = np.zeros(n, bool), np.zeros(n, bool)
        done, _ = _terminal(K_I, K_not, I_array)
        step = 0
        while not done and step < n:
            t1, t3 = tiers(K_I, K_not, I_array)
            live = ~(K_I | K_not)
            obs = torch.as_tensor(np.concatenate([K_I, K_not])[None], dtype=torch.float32)
            with torch.no_grad():
                a = int(net(obs).masked_fill(~torch.as_tensor(live[None]), -1e9).argmax(1))
            recs.append(dict(human=h_i, step=step, n_live=int(live.sum()),
                             n_t1=int(t1.sum()), n_t3=int(t3.sum()),
                             n_t2=int(live.sum() - t1.sum() - t3.sum()),
                             forced=bool(t1[a])))
            if a in true_bits:
                K_I[a] = True
            else:
                K_not[a] = True
            done, _ = _terminal(K_I, K_not, I_array)
            step += 1
    return recs


if __name__ == "__main__":
    ins = build_instance(0)
    recs = walk(RailsOnlyNet(ins["I_array"], seed=0), ins)
    n_t1 = np.array([r["n_t1"] for r in recs]); n_t2 = np.array([r["n_t2"] for r in recs])
    n_t3 = np.array([r["n_t3"] for r in recs]); live = np.array([r["n_live"] for r in recs])
    forced = np.array([r["forced"] for r in recs])
    print(f"{len(recs)} decisions across {len(ins['oracle_sets'])} humans\n")
    print(f"{'':<34}{'mean':>8}{'% of live':>11}")
    for lab, arr in (("tier 1  (must ask first)", n_t1), ("tier 2  (policy decides)", n_t2),
                     ("tier 3  (must never ask)", n_t3)):
        print(f"{lab:<34}{arr.mean():>8.1f}{100*arr.sum()/live.sum():>10.0f}%")
    print(f"\nlive actions per decision: {live.mean():.1f}")
    print(f"decisions where tier 1 forced the choice: {forced.sum()}/{len(recs)} "
          f"({100*forced.mean():.0f}%)")
    print(f"decisions with a tier-3 mask available  : {(n_t3>0).sum()}/{len(recs)} "
          f"({100*(n_t3>0).mean():.0f}%)")
    print(f"\n-> the policy genuinely chooses on {100*(1-forced.mean()):.0f}% of decisions;"
          f" the rest are forced by rail 1")
    # how the tiers evolve within an episode
    print(f"\n{'step':>5}{'live':>7}{'t1':>6}{'t2':>6}{'t3':>6}")
    for st in (0, 2, 5, 10, 20, 30, 40):
        m = np.array([r["step"] == st for r in recs])
        if m.sum():
            print(f"{st:>5}{live[m].mean():>7.1f}{n_t1[m].mean():>6.1f}"
                  f"{n_t2[m].mean():>6.1f}{n_t3[m].mean():>6.1f}")
