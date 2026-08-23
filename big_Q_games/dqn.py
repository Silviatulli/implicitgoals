"""DQN on the Query MDP.

Self-contained: the environment, the network and the replay buffer are here, so
the only imports are the tracked modules (bottlenecks.py, experiment.py).

Metrics go to CSV as each log point is produced, not at the end -- same reason
experiment.py writes episodes.csv during the sweep: a run killed at minute 40
still leaves everything it measured.  The logged query count comes from
bottlenecks.evaluate_policy_on_real_human, so it is the same measurement the
heuristics are scored with.
"""
import os, sys, csv, time, copy, random
from argparse import Namespace
import numpy as np, torch, torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiment import draw_instance_under_cap
from bottlenecks import (Oracle, subsets_to_array, bottleneck_index,
                         evaluate_policy_on_real_human)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rails import RailedQNet, rail_bias

_NEG = -1e9          # finite stand-in for -inf, so a fully-masked row still argmaxes


class QueryMDPVecEnv:
    """n_envs knowledge states (K_I, K_not) stepped in parallel; auto-reset on absorption.

    Two oracle modes, exactly one of which must be given:

    bottleneck_sets (n_sets, n) bool -- episodic.  Each env draws one row per
        episode and answers every query from it, so answers within an episode are
        consistent with a single human.  Knowing these rows is knowing the joint.
    oracle_probs (n,) float -- marginal.  Each query is answered by an independent
        coin at probs[b].  This is the information solve_query_mdp_exact has, and
        it is strictly weaker: it cannot represent the correlations between
        bottlenecks that belonging to one human induces.

    Termination is bottlenecks._terminal vectorised over envs -- failure when K_I
    fits no hypothesis, success when I_hat = ~K_not fits one.  Identical in both
    modes: only the answers differ.
    """
    def __init__(self, I_array, c_q, p_i, p_f, gamma, n_envs, rng,
                 bottleneck_sets=None, oracle_probs=None):
        self.T, self.n = I_array, I_array.shape[1]
        self.c_q, self.p_i, self.p_f, self.gamma = c_q, p_i, p_f, gamma
        self.n_envs, self.rng = n_envs, rng
        self.K_I = np.zeros((n_envs, self.n), dtype=bool)
        self.K_not = np.zeros((n_envs, self.n), dtype=bool)
        if (bottleneck_sets is None) == (oracle_probs is None):
            raise ValueError("give exactly one of bottleneck_sets / oracle_probs")
        self._bsets = None if bottleneck_sets is None else np.asarray(bottleneck_sets, dtype=bool)
        self.probs = None if oracle_probs is None else np.asarray(oracle_probs, dtype=np.float32)
        self.active = (None if self._bsets is None
                       else rng.integers(0, len(self._bsets), size=n_envs))

    def obs(self):
        return np.concatenate([self.K_I, self.K_not], axis=1).astype(np.float32)

    def valid(self):
        return ~(self.K_I | self.K_not)

    def step(self, actions):
        idx = np.arange(self.n_envs)
        yes = (self._bsets[self.active, actions] if self._bsets is not None
               else self.rng.random(self.n_envs) < self.probs[actions])
        self.K_I[idx[yes], actions[yes]] = True
        self.K_not[idx[~yes], actions[~yes]] = True

        KI, Tb = self.K_I[:, None, :], self.T[None, :, :]
        failure = ~(~((KI & ~Tb).any(2))).any(1)
        success = (~(((~self.K_not)[:, None, :] & ~Tb).any(2))).any(1) & ~failure
        done = failure | success
        reward = np.full(self.n_envs, self.c_q, dtype=np.float32)
        reward[success] += self.gamma * self.p_i
        reward[failure] += self.gamma * self.p_f

        next_obs = self.obs()
        self.K_I[done] = False
        self.K_not[done] = False
        if self._bsets is not None and done.any():
            self.active[done] = self.rng.integers(0, len(self._bsets), size=done.sum())
        return next_obs, reward, done


class QNet(nn.Module):
    """Observation (2n,) -> Q-values (n,).  The contract evaluate_policy_on_real_human wants."""
    def __init__(self, n, hidden=256, n_layers=2):
        super().__init__()
        layers = [nn.Linear(2 * n, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, n))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TensorBuffer:
    """On-device ring buffer taking a whole rollout batch per push."""
    def __init__(self, cap, obs_dim, device):
        self.cap, self.device, self.ptr, self.size = cap, device, 0, 0
        self.obs = torch.zeros((cap, obs_dim), device=device)
        self.nobs = torch.zeros((cap, obs_dim), device=device)
        self.act = torch.zeros((cap,), dtype=torch.long, device=device)
        self.rew = torch.zeros((cap,), device=device)
        self.done = torch.zeros((cap,), device=device)

    def push(self, obs, act, rew, nobs, done):
        b = obs.shape[0]
        i = (self.ptr + torch.arange(b, device=self.device)) % self.cap
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.nobs[i], self.done[i] = nobs, done
        self.ptr = int((self.ptr + b) % self.cap)
        self.size = min(self.size + b, self.cap)

    def sample(self, batch):
        i = torch.randint(0, self.size, (batch,), device=self.device)
        return self.obs[i], self.act[i], self.rew[i], self.nobs[i], self.done[i]


def valid_from_obs(obs):
    n = obs.shape[1] // 2
    return (obs[:, :n] + obs[:, n:]) < 0.5


def build_instance(seed=0, num_humans=12, room_side=5, min_hypotheses=6,
                   max_bottlenecks=1e3, obstacle_density=0.15):
    args = Namespace(obstacle_density=obstacle_density, rooms_per_side=5, slip_prob=0.0,
                     puddle_density=0.1, rock_density=0.1, overcooked_allow_drop=False)
    random.seed(seed); np.random.seed(seed)
    inst, bundle, *_ = draw_instance_under_cap("gridworld", room_side, num_humans, seed,
                                               args, max_bottlenecks,
                                               min_hypotheses=min_hypotheses,
                                               max_tries=100000)
    T_R, _, _, g, mdp = inst
    oracle_sets, _, B, I, _ = bundle
    oracle = Oracle(oracle_sets, n_states=max(int(T_R.shape[0]), int(g) + 1))
    return dict(B=B, I=I, I_array=subsets_to_array(I, B), oracle_sets=oracle_sets,
                hs=oracle.bottleneck_matrix(B), probs=oracle.probs_for_raw_ids(B),
                b_to_int=bottleneck_index(B),
                oracle=oracle, goal=int(g),
                positions={i: tuple(s[0]) for i, s in enumerate(mdp.get_state_space())})


def per_human(net, ins, railed=False):
    """One query count per candidate human -- the vector a paired test needs.

    The evaluator builds its observations on cpu, so a net on mps comes back first.
    """
    if railed:
        net = RailedQNet(net, ins["I_array"])
    p = next(net.parameters(), None)
    home = p.device if p is not None else None
    if home is not None:
        net.to("cpu")
    v = np.array([evaluate_policy_on_real_human(
        true_bottlenecks=list(s), policy_network=net, n_runs=1,
        I_array=ins["I_array"], b_to_int=ins["b_to_int"])["n_queries"][0]
        for s in ins["oracle_sets"]], dtype=float)
    if home is not None:
        net.to(home)
    return v


def train(ins, seed=0, mode="joint", railed=False, n_steps=120_000, log_every=2_000, writer=None, csv_file=None,
          n_envs=64, batch_size=256, hidden=256, n_layers=2, lr=3e-4, target_sync=1_000,
          buffer_cap=500_000, warmup_steps=400, eps_start=1.0, eps_end=0.05,
          eps_decay_steps=60_000, grad_clip=5.0, c_q=-10.0, p_i=1.0, p_f=0.0,
          gamma=0.99, device=None, verbose=True, p_const=None):
    """eps_decay_steps is absolute, not a fraction of n_steps: budget and
    exploration schedule have to vary independently for a length sweep to mean
    anything."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device) if device else torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu")

    n = ins["I_array"].shape[1]
    n_b = ins["I_array"].shape[1]
    # uniform_avg: one constant p from |B| and the mean bottlenecks-per-human only,
    # i.e. aggregate counts without knowing which bottlenecks are the rare ones.
    # p_const overrides the instance's own density, so a leave-one-out constant can be
    # used instead of one measured on the evaluation humans.
    p_avg = p_const if p_const is not None else float(ins["hs"].sum(1).mean()) / n_b
    oracle_kw = ({"bottleneck_sets": ins["hs"]} if mode == "joint" else
                 {"oracle_probs": ins["probs"]} if mode == "marginal" else
                 {"oracle_probs": np.full(n_b, p_avg)} if mode == "uniform_avg" else
                 {"oracle_probs": np.full(n_b, 0.5)})
    env = QueryMDPVecEnv(ins["I_array"], c_q, p_i, p_f, gamma, n_envs, rng, **oracle_kw)
    q = QNet(n, hidden, n_layers).to(device)
    q_target = copy.deepcopy(q).to(device)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = TensorBuffer(buffer_cap, 2 * n, device)
    I_t = torch.as_tensor(np.asarray(ins["I_array"], dtype=bool), device=device)

    def railed_q(net_out, obs_t):
        """Rails applied identically at rollout and in the target, so the learned
        Q is the value of the railed policy rather than of an unrailed one."""
        return net_out + rail_bias(obs_t, I_t) if railed else net_out

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

    losses, t0 = [], time.time()
    for step in range(n_steps):
        obs_np, valid_np = env.obs(), env.valid()
        eps = eps_start + min(1.0, step / eps_decay_steps) * (eps_end - eps_start)
        if rng.random() < eps:
            act_np = rand_actions(valid_np)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_np, device=device)
                qv = railed_q(q(obs_t), obs_t)
                qv = qv.masked_fill(~torch.as_tensor(valid_np, device=device), _NEG)
                act_np = qv.argmax(1).cpu().numpy().astype(np.int64)
        push(obs_np, act_np)

        if buf.size >= batch_size:
            s, a, r, ns, d = buf.sample(batch_size)
            with torch.no_grad():
                a_star = railed_q(q(ns), ns).masked_fill(
                    ~valid_from_obs(ns), _NEG).argmax(1, keepdim=True)
                target = r + gamma * (1.0 - d) * q_target(ns).gather(1, a_star).squeeze(1)
            loss = nn.functional.smooth_l1_loss(
                q(s).gather(1, a.unsqueeze(1)).squeeze(1), target)
            losses.append(loss.item())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), grad_clip); opt.step()
            if step % target_sync == 0:
                q_target.load_state_dict(q.state_dict())

        if step % log_every == 0 and losses:
            with torch.no_grad():
                q0 = q(torch.zeros(1, 2 * n, device=device))
            row = {"seed": seed, "step": step,
                   "queries": round(float(per_human(q, ins, railed).mean()), 3),
                   "loss": round(float(np.mean(losses)), 5),
                   "maxq0": round(q0.max().item(), 3),
                   "mean_q": round(q0.mean().item(), 3),
                   "eps": round(eps, 4), "sec": round(time.time() - t0, 1)}
            losses.clear()
            if writer is not None:
                writer.writerow(row)
                csv_file.flush()     # this log point is now safe on disk
            if verbose:
                print(f"  step {row['step']:>7,}  q {row['queries']:6.2f}  "
                      f"loss {row['loss']:8.4f}  maxq0 {row['maxq0']:8.1f}  "
                      f"eps {row['eps']:4.2f}", flush=True)
    # Release the replay buffer before returning.  It is 500k x 2n floats (~530 MB at
    # n = 66) and the mps/cuda caching allocators hold freed blocks, so a caller that
    # trains in a loop accumulates one buffer per iteration unless the cache is emptied.
    elapsed = time.time() - t0
    del buf, q_target, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return q, elapsed


FIELDS = ["seed", "step", "queries", "loss", "maxq0", "mean_q", "eps", "sec"]

if __name__ == "__main__":
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 120_000
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    mode = sys.argv[3] if len(sys.argv) > 3 else "joint"
    railed = len(sys.argv) > 4 and sys.argv[4] == "railed"
    assert mode in ("joint", "marginal", "uniform", "uniform_avg")
    tag = mode + ("_railed" if railed else "")
    out_dir = os.path.join(ROOT, "big_Q_games", "results")
    os.makedirs(out_dir, exist_ok=True)

    ins = build_instance(0)
    print(f"|B| = {len(ins['B'])}, |I| = {len(ins['I'])}, {len(ins['oracle_sets'])} humans, "
          f"{n_steps:,} steps, oracle = {mode}, rails = {railed}\n", flush=True)

    mpath = os.path.join(out_dir, f"dqn_metrics_{tag}.csv")
    hpath = os.path.join(out_dir, f"dqn_humans_{tag}.csv")
    fresh = not os.path.exists(mpath)
    with open(mpath, "a", newline="") as mf, open(hpath, "a", newline="") as hf:
        mw = csv.DictWriter(mf, fieldnames=FIELDS)
        hw = csv.writer(hf)
        if fresh:
            mw.writeheader(); hw.writerow(["seed", "human", "n_queries"])
        finals = []
        for sd in seeds:
            print(f"seed {sd}", flush=True)
            net, sec = train(ins, seed=sd, mode=mode, railed=railed, n_steps=n_steps,
                             writer=mw, csv_file=mf)
            v = per_human(net, ins, railed)
            for i, x in enumerate(v):
                hw.writerow([sd, i, x])
            hf.flush()
            torch.save(net.to("cpu").state_dict(), os.path.join(out_dir, f"dqn_{tag}_seed{sd}.pt"))
            finals.append(v.mean())
            print(f"  -> {v.mean():.2f}  ({sec:.0f}s)\n", flush=True)
    print(f"{tag}: {len(finals)} seed(s), mean {np.mean(finals):.2f}  "
          f"std {np.std(finals, ddof=1) if len(finals) > 1 else 0:.2f}  "
          f"{[round(x, 2) for x in finals]}")
    print(f"-> {mpath}\n-> {hpath}")
