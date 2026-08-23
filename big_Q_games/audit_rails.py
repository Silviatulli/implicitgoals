"""End-to-end audit: does the deployed railed policy actually obey its rails?

Replays real episodes against the real humans, recomputing the tiers at every
decision point and checking the chosen action against them.  Also counts how
often each tier fires, which is what says whether the rails can matter at all
on this instance.
"""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottlenecks import _terminal
from rails import tiers, RailedQNet
from dqn import build_instance, QNet


def audit_episode(net, I_array, true_bits, n):
    """One episode, mirroring bottlenecks._run_query_episode, with tier checks."""
    K_I, K_not = np.zeros(n, bool), np.zeros(n, bool)
    log = []
    done, success = _terminal(K_I, K_not, I_array)
    if done:
        return log, 0, success
    for n_q in range(1, n + 1):
        t1, t3 = tiers(K_I, K_not, I_array)
        obs = torch.as_tensor(np.concatenate([K_I, K_not])[None], dtype=torch.float32)
        vmask = torch.as_tensor((~(K_I | K_not))[None])
        with torch.no_grad():
            a = int(net(obs).masked_fill(~vmask, -1e9).argmax(1).item())
        log.append(dict(action=a, tier1_available=int(t1.sum()), tier3_available=int(t3.sum()),
                        chose_tier1=bool(t1[a]), chose_tier3=bool(t3[a]),
                        legal=bool(not (K_I | K_not)[a])))
        if a in true_bits:
            K_I[a] = True
        else:
            K_not[a] = True
        done, success = _terminal(K_I, K_not, I_array)
        if done:
            return log, n_q, success
    raise RuntimeError("episode did not absorb")


if __name__ == "__main__":
    ins = build_instance(0)
    I_array, n = ins["I_array"], ins["I_array"].shape[1]
    torch.manual_seed(0)
    net = RailedQNet(QNet(n, 256, 2), I_array)     # untrained: rails must hold anyway

    viol_t3 = viol_t1 = illegal = 0
    steps = 0
    t1_open = t3_open = 0
    lens = []
    for s in ins["oracle_sets"]:
        true_bits = {ins["b_to_int"][b] for b in s}
        log, nq, _ = audit_episode(net, I_array, true_bits, n)
        lens.append(nq)
        for e in log:
            steps += 1
            illegal += (not e["legal"])
            viol_t3 += e["chose_tier3"]
            if e["tier1_available"]:
                t1_open += 1
                viol_t1 += (not e["chose_tier1"])
            if e["tier3_available"]:
                t3_open += 1
    print(f"{len(lens)} episodes, {steps} decisions, mean length {np.mean(lens):.2f}")
    print(f"\nRAIL VIOLATIONS")
    print(f"  chose an already-answered bottleneck : {illegal}")
    print(f"  chose a tier-3 (provably useless)    : {viol_t3}")
    print(f"  tier-1 available but not chosen      : {viol_t1}")
    print(f"\nHOW OFTEN THE RAILS CAN BITE (untrained net, so this is the instance's own shape)")
    print(f"  decisions with >=1 tier-1 available  : {t1_open}/{steps} ({100*t1_open/steps:.0f}%)")
    print(f"  decisions with >=1 tier-3 available  : {t3_open}/{steps} ({100*t3_open/steps:.0f}%)")
    ok = (illegal == 0 and viol_t3 == 0 and viol_t1 == 0)
    print(f"\n{'AUDIT PASSED' if ok else 'AUDIT FAILED'}")
    sys.exit(0 if ok else 1)
