"""How often does each existing heuristic break the dynamic rails?

H3 already carries both rails, but computes them once from the full hypothesis space
and freezes them (bottlenecks.py: "The tiers come from I ... and are fixed for the
episode").  H1 and H4 do re-derive the consistent set, but score it in ways that
misrank the two railed tiers.  This measures the consequence.
"""
import sys, os
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "big_Q_games"))
from dqn import build_instance
from rails import tiers, RailsOnlyNet
from bottlenecks import (_terminal, solve_query_mdp_proximity, solve_query_mdp_info_gain,
                         solve_query_mdp_frequency)


def walk(net, ins):
    I_array = ins["I_array"]; n = I_array.shape[1]
    t3 = t1m = steps = 0
    for s in ins["oracle_sets"]:
        tb = {ins["b_to_int"][b] for b in s}
        K_I, K_not = np.zeros(n, bool), np.zeros(n, bool)
        while not _terminal(K_I, K_not, I_array)[0]:
            m1, m3 = tiers(K_I, K_not, I_array)
            live = ~(K_I | K_not)
            obs = torch.as_tensor(np.concatenate([K_I, K_not])[None], dtype=torch.float32)
            with torch.no_grad():
                a = int(net(obs).masked_fill(~torch.as_tensor(live[None]), -1e9).argmax(1))
            steps += 1
            t3 += int(m3[a])
            t1m += int(m1.any() and not m1[a])
            (K_I if a in tb else K_not)[a] = True
    return steps, t3, t1m


if __name__ == "__main__":
    ins = build_instance(0)
    pols = [
        ("H3 (static tiers)", solve_query_mdp_proximity(
            ins["I"], ins["B"], oracle=ins["oracle"],
            positions=ins["positions"], goal_state=ins["goal"])),
        ("H1 (entropy)", solve_query_mdp_info_gain(ins["I"], ins["B"], oracle=ins["oracle"])),
        ("H4 (marginal)", solve_query_mdp_frequency(ins["I"], ins["B"], oracle=ins["oracle"])),
        ("Rails only (dynamic)", RailsOnlyNet(ins["I_array"], seed=0)),
    ]
    print(f"{'policy':<24}{'queries':>9}{'tier-3 asked':>16}{'tier-1 skipped':>17}")
    for name, p in pols:
        s, t3, t1m = walk(p, ins)
        print(f"{name:<24}{s:>9}{t3:>10} ({100*t3/s:>3.0f}%){t1m:>11} ({100*t1m/s:>3.0f}%)")
