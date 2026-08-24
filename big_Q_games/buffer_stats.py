"""What does the replay buffer actually contain?

`push` is called unconditionally, so every primitive transition is stored -- including
the ones where a rail forced the action and the network's output was irrelevant.  This
measures the tier composition of the actions actually taken, separately for the
exploration branch and the greedy branch, with and without the two fixes.
"""
import os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "big_Q_games"))
from dqn import build_instance, QueryMDPVecEnv, QNet
from rails import tiers_batch, rail_bias


def compose(ins, eps, rail_explore, steps=300, n_envs=64, seed=0):
    I_array = ins["I_array"]; n = I_array.shape[1]
    p = float(ins["hs"].sum(1).mean()) / n
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    q = QNet(n, 256, 2); I_t = torch.as_tensor(I_array)
    env = QueryMDPVecEnv(I_array, -10.0, 1.0, 0.0, 0.99, n_envs, rng,
                         oracle_probs=np.full(n, p))
    c = dict(t1=0, t2=0, t3=0)
    for _ in range(steps):
        K_I, K_not = env.K_I.copy(), env.K_not.copy()
        m1, m3 = tiers_batch(K_I, K_not, I_array)
        live = ~(K_I | K_not)
        if rng.random() < eps:
            pool = live
            if rail_explore:
                pool = np.where(m1.any(1, keepdims=True), m1, live & ~m3)
                pool = np.where(pool.any(1, keepdims=True), pool, live)
            g = rng.random(pool.shape); g[~pool] = -1.0
            a = g.argmax(1).astype(np.int64)
        else:
            obs = torch.as_tensor(np.concatenate([K_I, K_not], 1).astype(np.float32))
            a = (q(obs) + rail_bias(obs, I_t)).masked_fill(
                ~torch.as_tensor(live), -1e9).argmax(1).numpy()
        i = np.arange(n_envs)
        c["t1"] += int(m1[i, a].sum()); c["t3"] += int(m3[i, a].sum())
        c["t2"] += int((live & ~m1 & ~m3)[i, a].sum())
        env.step(a)
    tot = sum(c.values())
    return {k: 100 * v / tot for k, v in c.items()}


if __name__ == "__main__":
    ins = build_instance(0)
    print("Tier of the action actually stored in the buffer\n")
    print(f"{'branch':<34}{'tier1':>8}{'tier2':>8}{'tier3':>8}")
    print(f"{'':<34}{'forced':>8}{'NN picks':>8}{'useless':>8}\n")
    for lab, eps, rx in (("exploration (eps=1), as shipped", 1.0, False),
                         ("exploration (eps=1), rails on", 1.0, True),
                         ("greedy (eps=0.05), as shipped", 0.05, False),
                         ("greedy (eps=0.05), rails on", 0.05, True)):
        c = compose(ins, eps, rx)
        print(f"{lab:<34}{c['t1']:>7.0f}%{c['t2']:>7.0f}%{c['t3']:>7.0f}%")
    print("\nOnly tier-2 rows are decisions the network can influence.")
