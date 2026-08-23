"""DQN on the Query MDP, logging the real metric -- mean queries against the
12 candidate humans -- alongside the training diagnostics, on one time axis.

The loop is here rather than in drafts/query_mdp_nn.train() because that one
logs only loss and maxQ(s0): it never calls bottlenecks.evaluate_policy_on_real_human
and has no hook where it could.  Env, network and replay buffer are imported
from it unchanged.
"""
import os, sys, time, copy, json, random
from argparse import Namespace
import numpy as np, torch, torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "big_Q_games", "drafts"))
from experiment import draw_instance_under_cap
from bottlenecks import (Oracle, subsets_to_array, bottleneck_index,
                         evaluate_policy_on_real_human, solve_query_mdp_proximity)
from query_mdp_nn import QueryMDPVecEnv, QNet, TensorBuffer, valid_from_obs_t, _NEG


def build_instance(seed=0):
    args = Namespace(obstacle_density=0.15, rooms_per_side=5, slip_prob=0.0,
                     puddle_density=0.1, rock_density=0.1, overcooked_allow_drop=False)
    random.seed(seed); np.random.seed(seed)
    inst, bundle, *_ = draw_instance_under_cap("gridworld", 5, 12, seed, args, 1e3,
                                               min_hypotheses=6, max_tries=100000)
    T_R, _, _, g, mdp = inst
    oracle_sets, _, B, I, _ = bundle
    oracle = Oracle(oracle_sets, n_states=max(int(T_R.shape[0]), int(g) + 1))
    return dict(B=B, I=I, I_array=subsets_to_array(I, B), oracle_sets=oracle_sets,
                hs=oracle.bottleneck_matrix(B), b_to_int=bottleneck_index(B),
                oracle=oracle, goal=int(g),
                positions={i: tuple(s[0]) for i, s in enumerate(mdp.get_state_space())})


def mean_queries(net, ins):
    """Mean over the 12 real humans.  The evaluator builds observations on cpu,
    so a net living on mps has to come back first."""
    was = next(net.parameters(), None)
    was = was.device if was is not None else None
    if was is not None:
        net.to("cpu")
    v = float(np.mean([evaluate_policy_on_real_human(
        true_bottlenecks=list(s), policy_network=net, n_runs=1,
        I_array=ins["I_array"], b_to_int=ins["b_to_int"])["n_queries"][0]
        for s in ins["oracle_sets"]]))
    if was is not None:
        net.to(was)
    return v


def run(ins, seed=0, n_steps=120_000, log_every=2_000, n_envs=64, batch_size=256,
        hidden=256, n_layers=2, lr=1e-3, target_sync=200, buffer_cap=30_000,
        warmup_steps=400, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.5,
        grad_clip=5.0, c_q=-10.0, p_i=1.0, p_f=0.0, gamma=0.99, device=None):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device) if device else torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu")

    I_array = ins["I_array"]; n = I_array.shape[1]
    env = QueryMDPVecEnv(I_array, c_q, p_i, gamma, n_envs, rng, p_f, None, ins["hs"])
    q = QNet(n, hidden, n_layers).to(device)
    q_target = copy.deepcopy(q).to(device)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = TensorBuffer(buffer_cap, 2 * n, device)
    decay = max(1, int(eps_decay_frac * n_steps))

    def push(obs_np, act_np):
        nobs_np, rew_np, done_np = env.step(act_np)
        buf.push(torch.as_tensor(obs_np, device=device),
                 torch.as_tensor(act_np, dtype=torch.long, device=device),
                 torch.as_tensor(rew_np, device=device),
                 torch.as_tensor(nobs_np, device=device),
                 torch.as_tensor(done_np.astype(np.float32), device=device))

    def rand_actions(valid_np):
        g = rng.random(valid_np.shape); g[~valid_np] = -1.0
        return g.argmax(1).astype(np.int64)

    for _ in range(warmup_steps):
        push(env.obs(), rand_actions(env.valid()))

    hist = {k: [] for k in ("step", "queries", "loss", "maxq0", "mean_q", "eps", "sec")}
    lbuf, t0 = [], time.time()
    for step in range(n_steps):
        obs_np, valid_np = env.obs(), env.valid()
        eps = eps_start + min(1.0, step / decay) * (eps_end - eps_start)
        if rng.random() < eps:
            act_np = rand_actions(valid_np)
        else:
            with torch.no_grad():
                qv = q(torch.as_tensor(obs_np, device=device))
                qv = qv.masked_fill(~torch.as_tensor(valid_np, device=device), _NEG)
                act_np = qv.argmax(1).cpu().numpy().astype(np.int64)
        push(obs_np, act_np)

        if buf.size >= batch_size:
            s, a, r, ns, d = buf.sample(batch_size)
            with torch.no_grad():
                nvalid = valid_from_obs_t(ns)
                a_star = q(ns).masked_fill(~nvalid, _NEG).argmax(1, keepdim=True)
                target = r + gamma * (1.0 - d) * q_target(ns).gather(1, a_star).squeeze(1)
            loss = nn.functional.smooth_l1_loss(
                q(s).gather(1, a.unsqueeze(1)).squeeze(1), target)
            lbuf.append(loss.item())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), grad_clip); opt.step()
            if step % target_sync == 0:
                q_target.load_state_dict(q.state_dict())

        if step % log_every == 0 and lbuf:
            with torch.no_grad():
                q0 = q(torch.zeros(1, 2 * n, device=device))
            hist["step"].append(step)
            hist["queries"].append(mean_queries(q, ins))
            hist["loss"].append(float(np.mean(lbuf)))
            hist["maxq0"].append(q0.max().item())
            hist["mean_q"].append(q0.mean().item())
            hist["eps"].append(eps)
            hist["sec"].append(round(time.time() - t0, 1))
            lbuf.clear()
            print(f"  step {step:>7,}  q {hist['queries'][-1]:6.2f}  "
                  f"loss {hist['loss'][-1]:8.4f}  maxq0 {hist['maxq0'][-1]:7.3f}  "
                  f"eps {eps:4.2f}", flush=True)
    hist["final_queries"] = mean_queries(q, ins)
    hist["wall_sec"] = round(time.time() - t0, 1)
    hist["device"] = str(device)
    return hist, q


if __name__ == "__main__":
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 120_000
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    ins = build_instance(0)
    h3 = mean_queries(solve_query_mdp_proximity(
        ins["I"], ins["B"], oracle=ins["oracle"], positions=ins["positions"],
        goal_state=ins["goal"]), ins)
    print(f"|B| = {len(ins['B'])}, |I| = {len(ins['I'])}, "
          f"{len(ins['oracle_sets'])} humans | H3 {h3:.2f}\n", flush=True)
    out = {"h3": h3, "n_steps": n_steps, "runs": {}}
    OUT = os.path.join(ROOT, "big_Q_games", "results", "dqn_curves.json")
    for sd in seeds:
        print(f"seed {sd}", flush=True)
        hist, _ = run(ins, seed=sd, n_steps=n_steps)
        out["runs"][sd] = hist
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"  -> final {hist['final_queries']:.2f}  ({hist['wall_sec']}s)\n", flush=True)
    print(f"-> {OUT}")
