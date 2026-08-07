"""
query_mdp_nn.py
===============
Neural network infrastructure for the Query MDP: DQN training, environment,
and trajectory simulation.

Game-agnostic: works on abstract bit indices 0..n-1 and a bool I_array matrix.
Imports from bottlenecks.py for the helpers it reuses.
"""

import copy
import json
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import time
from tqdm import tqdm

from bottlenecks import (
    shuffle_bottlenecks,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_NEG = -1e9   # finite "minus infinity" for masked Q-values (avoids NaN in argmax)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading / saving
# ─────────────────────────────────────────────────────────────────────────────

def load_real_I_array(filepath: str):
    """
    Load a Part-1 save file and return (I_array, n_bottleneck, b_to_int).

    Accepts two formats:
      - dict  {"I_array": [...], "bottlenecks": [...]}  — new format
      - list  [...]                                      — legacy (I_array only)

    The bottleneck list gives the canonical column order for I_array.
    Legacy files derive the order from sorted(union of all I_array).

    Returns
    -------
    I_array : bool ndarray, shape (len_I_array, n_bottleneck)
    n_bottleneck   : int
    b_to_int       : dict  {bottleneck_tuple: column_index}
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        I_array_raw   = data["I_array"]
        bottlenecks   = [tuple(b) for b in data["bottlenecks"]]
    else:
        I_array_raw = data
        unique_b = set()
        for t in I_array_raw:
            for b in t:
                unique_b.add(tuple(b) if isinstance(b, list) else b)
        bottlenecks = sorted(unique_b)

    b_to_int       = {b: i for i, b in enumerate(bottlenecks)}
    n_bottleneck   = len(bottlenecks)
    I_array = np.zeros((len(I_array_raw), n_bottleneck), dtype=bool)
    for i, t in enumerate(I_array_raw):
        for b in t:
            I_array[i, b_to_int[tuple(b) if isinstance(b, list) else b]] = True

    return I_array, n_bottleneck, b_to_int


def save_training_results(out_dict,
                          json_path="saves/flat_dqn/metrics/training_results.json",
                          model_path="saves/flat_dqn/weights/q_network.pt"):
    """
    Persist a training output dict produced by train().

    The Q-network state dict is written to model_path; everything else
    (history, metrics, hyperparameters) is JSON-serialized to json_path.
    NumPy arrays are converted to Python lists automatically.
    """
    if "q" in out_dict and out_dict["q"] is not None:
        torch.save(out_dict["q"].state_dict(), model_path)

    clean_out = {}
    for key, value in out_dict.items():
        if key == "q":
            clean_out[key] = f"Model saved to {model_path}"
        elif isinstance(value, np.ndarray):
            clean_out[key] = value.tolist()
        else:
            clean_out[key] = value

    with open(json_path, "w") as f:
        json.dump(clean_out, f, indent=4)

    print(f"Saved: metrics → {json_path}  |  model → {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Hyperparameters for a single DQN training run on the Query MDP."""
    # ---- instance ----
    n_bottleneck: int      = 10
    len_I_array:    int      = 6
    c_q:          float    = -10.0
    p_i:          float    = 1.0
    p_f:          float    = 0.0          # failure terminal value (negative = penalty)
    gamma:        float    = 0.99
    oracle_probs:    np.ndarray = None  # shape (n_bottleneck,), marginal P(YES); ignored when bottleneck_sets is set
    bottleneck_sets: np.ndarray = None  # shape (n_sets, n_bottleneck) bool; one set sampled per episode
    seed:         int      = 0
    I_array:      np.ndarray = None    # if None, build_instance() is called

    # ---- DQN ----
    n_envs:          int   = 64
    buffer_cap:      int   = 30_000
    n_steps:         int   = 6_000
    batch_size:      int   = 256
    warmup_steps:    int   = 400
    lr:              float = 1e-3
    hidden:          int   = 128
    n_layers:        int   = 2         # number of hidden layers in QNet
    target_sync:     int   = 200       # steps between target-net syncs
    double_dqn:      bool  = False     # Double DQN target (curbs Q overestimation blow-up)
    eps_start:       float = 1.0
    eps_end:         float = 0.05
    eps_decay_frac:  float = 0.5       # fraction of n_steps over which ε decays
    grad_clip:       float = 5.0

    # ---- bookkeeping ----
    eval_every:   int  = 500
    device:       str  = None          # None → auto-detect (mps / cuda / cpu)
    verbose:      bool = True

    # ---- tuning / long-run control (optional; no effect when left at defaults) ----
    checkpoint_path: str   = None      # if set, dump partial history JSON here every checkpoint_every steps
    checkpoint_every: int  = None      # None → falls back to 10 * eval_every
    max_wall_sec:    float = None      # if set, break the main loop once training wall time exceeds this


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def build_instance(n, len_I_array, seed):
    """
    Random I_array matrix where every bottleneck appears in at least one row
    (mirrors unique_B drawn from I_decoded — no unreachable bottlenecks).
    """
    rng = np.random.default_rng(seed)
    while True:
        T = rng.random((len_I_array, n)) < 0.5
        T[T.sum(1) == 0, 0] = True
        if T.any(0).all():
            return T


def compute_exact_value(I_array, c_q, p_i, gamma, p_f=0.0):
    """
    Compact recursive backward induction for small n (≤ ~16).

    Unlike solve_query_mdp, this accepts a bool I_array matrix
    (shape len_I_array × n_bottleneck) rather than I_decoded, and returns
    only the scalar V*(0, 0) — used to validate DQN convergence.
    """
    n    = I_array.shape[1]
    T    = [int(''.join('1' if b else '0' for b in row[::-1]), 2) for row in I_array]
    FULL = (1 << n) - 1
    memo = {}

    def failure(KI):
        return all((KI & t) != KI for t in T)

    def success(KN):
        Ih = FULL & ~KN
        return any((Ih & t) == Ih for t in T)

    def V(KI, KN):
        if failure(KI):          return p_f
        if success(KN):          return p_i
        if (KI | KN) == FULL:    return p_f   # all queried, no success → failure
        k = (KI, KN)
        if k in memo: return memo[k]
        asked = KI | KN
        best  = -1e18
        for b in range(n):
            if (asked >> b) & 1: continue
            bit  = 1 << b
            q    = c_q + gamma * 0.5 * (V(KI | bit, KN) + V(KI, KN | bit))
            best = max(best, q)
        memo[k] = best
        return best

    return V(0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized environment
# ─────────────────────────────────────────────────────────────────────────────

class QueryMDPVecEnv:
    """
    Vectorized Query MDP environment.

    Maintains n_envs independent (K_I, K_not) knowledge states in parallel.
    On each step, an action selects which bottleneck to query.

    Oracle modes
    ------------
    bottleneck_sets : bool ndarray (n_sets, n_bottlenecks)
        Episodic mode — at the start of each trajectory a random row is drawn
        once; all queries in that episode are answered deterministically from
        that row (YES iff the bottleneck belongs to the chosen set).
    oracle_probs : float32 ndarray (n_bottlenecks,)
        Marginal mode — each query independently draws YES with probability
        probs[action].  Used as a fallback when bottleneck_sets is None.
    Neither → uniform Bernoulli(0.5) per query.

    Episodes reset automatically upon reaching an absorbing state.
    """
    def __init__(self, I_array, c_q, p_i, gamma, n_envs, rng, p_f=0.0,
                 oracle_probs=None, bottleneck_sets=None, dominance=None):
        # dominance: (n, n) bool or None — Hypothesis 2(ii).  dominance[b2, b1]
        # means an oracle NO on b2 entails NO on b1, so b1 is ruled out without
        # spending a query.  None (default) leaves stepping exactly as it was.
        self.dominance = None if dominance is None else np.asarray(dominance, dtype=bool)
        self.T         = I_array
        self.len_I_array, self.n = I_array.shape
        self.c_q, self.p_i, self.p_f, self.gamma = c_q, p_i, p_f, gamma
        self.n_envs    = n_envs
        self.rng       = rng
        self.K_I   = np.zeros((n_envs, self.n), dtype=bool)
        self.K_not = np.zeros((n_envs, self.n), dtype=bool)

        if bottleneck_sets is not None:
            # Episodic oracle: each env gets one fixed recipe per trajectory.
            self._bsets = np.asarray(bottleneck_sets, dtype=bool)  # (n_sets, n)
            self._n_sets = len(self._bsets)
            # Draw initial recipes for all envs.
            self.active_set = self.rng.integers(0, self._n_sets, size=n_envs)
            self.probs = None
        else:
            self._bsets = None
            # Marginal oracle: i.i.d. Bernoulli per query.
            self.probs = (np.asarray(oracle_probs, dtype=np.float32)
                          if oracle_probs is not None
                          else np.full(self.n, 0.5, dtype=np.float32))

    def obs(self):
        """Observation: [K_I || K_not], shape (n_envs, 2n), float32."""
        return np.concatenate([self.K_I, self.K_not], axis=1).astype(np.float32)

    def valid(self):
        """Boolean mask of unqueried (valid) actions, shape (n_envs, n)."""
        return ~(self.K_I | self.K_not)

    def _terminal_and_reward(self):
        KI        = self.K_I[:, None, :]          # (n_envs, 1, n)
        Tb        = self.T[None, :, :]             # (1, len_I_array, n)
        ki_subset = ~((KI & ~Tb).any(2))           # K_I ⊆ T_t?  (n_envs, len_I_array)
        failure   = ~ki_subset.any(1)              # K_I ⊄ any row of I_array
        Ih          = (~self.K_not)[:, None, :]     # possible-YES set = K_I ∪ unqueried
        ih_sub_tb   = ~((Ih & ~Tb).any(2))         # Ih ⊆ T_t?  (no bit of Ih outside T_t)
        success     = ih_sub_tb.any(1) & ~failure
        done      = failure | success
        reward    = np.full(self.n_envs, self.c_q, dtype=np.float32)
        reward[success] += self.gamma * self.p_i
        reward[failure] += self.gamma * self.p_f
        return done, reward

    def apply_dominance(self):
        """Hypothesis 2(ii): grant the NOs that the current K_not already entails.

        b1 ⪯ b2 and b2 ∉ I_G ⇒ b1 ∉ I_G, so those bits are ruled out without
        spending a query.  `dominance` is transitively closed, so one pass
        reaches the fixpoint.  Bits the oracle answered YES are left alone —
        contradicting one means no hypothesis survives, which the terminal test
        already reports as a failure.

        No-op when dominance is None, which is every condition except H2.
        """
        if self.dominance is None:
            return
        entailed    = (self.K_not[:, :, None] & self.dominance[None, :, :]).any(1)
        self.K_not |= entailed & ~self.K_I

    def step(self, actions):
        idx = np.arange(self.n_envs)
        if self._bsets is not None:
            # Episodic oracle: answer deterministically from this episode's recipe.
            yes = self._bsets[self.active_set, actions]
        else:
            yes = self.rng.random(self.n_envs) < self.probs[actions]
        self.K_I  [idx[ yes], actions[ yes]] = True
        self.K_not[idx[~yes], actions[~yes]] = True
        self.apply_dominance()
        done, reward = self._terminal_and_reward()
        next_obs     = self.obs()
        # Auto-reset: clear knowledge state and sample new recipe for done envs.
        self.K_I  [done] = False
        self.K_not[done] = False
        if self._bsets is not None and done.any():
            self.active_set[done] = self.rng.integers(0, self._n_sets, size=done.sum())
        return next_obs, reward, done


def valid_from_obs_t(obs):
    """Valid-action mask from an observation tensor [B, 2n] → bool [B, n]."""
    n = obs.shape[1] // 2
    return (obs[:, :n] + obs[:, n:]) < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Neural network models
# ─────────────────────────────────────────────────────────────────────────────

class QNet(nn.Module):
    """MLP: observation (2n,) → Q-values (n,). n_layers controls hidden depth (default 2)."""
    def __init__(self, n, hidden, n_layers=2):
        super().__init__()
        layers = [nn.Linear(2 * n, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, n))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TensorBuffer:
    """On-device ring buffer that accepts whole rollout batches at once."""
    def __init__(self, cap, obs_dim, device):
        self.cap, self.device = cap, device
        self.obs  = torch.zeros((cap, obs_dim), device=device)
        self.act  = torch.zeros((cap,),         dtype=torch.long, device=device)
        self.rew  = torch.zeros((cap,),                           device=device)
        self.nobs = torch.zeros((cap, obs_dim), device=device)
        self.done = torch.zeros((cap,),                           device=device)
        self.ptr  = 0
        self.size = 0

    def push(self, obs, act, rew, nobs, done):
        b   = obs.shape[0]
        idx = (self.ptr + torch.arange(b, device=self.device)) % self.cap
        self.obs[idx],  self.act[idx],  self.rew[idx]  = obs, act, rew
        self.nobs[idx], self.done[idx]                 = nobs, done
        self.ptr  = int((self.ptr + b) % self.cap)
        self.size = min(self.size + b, self.cap)

    def sample(self, batch_size):
        i = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[i], self.act[i], self.rew[i], self.nobs[i], self.done[i]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config) -> dict:
    """
    Train a masked DQN on the Query MDP defined by cfg.

    The action at each step is a single masked choice over n bottlenecks.
    Q(s, b) alone drives the policy (argmax over valid actions); the
    conditional "which query next depends on previous answers" structure
    is recovered by sampling the ½-½ oracle in the environment.

    Returns
    -------
    dict with keys:
        history     — step / loss / maxq0 / gap / eps / mean_q lists
        v_star      — None (exact solver disabled)
        wall_sec    — training wall time in seconds
        device      — device string
        final_maxq0 — last logged max Q(s0)
        final_gap   — |final_maxq0 − v_star| (nan if v_star is None)
        q           — trained QNet
        I_array     — bool I_array matrix used
        n           — number of bottlenecks
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng    = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device) if cfg.device else torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else "cpu")

    I_array = cfg.I_array if cfg.I_array is not None else \
        build_instance(cfg.n_bottleneck, cfg.len_I_array, cfg.seed)
    n       = I_array.shape[1]
    obs_dim = 2 * n

    v_star = None

    env      = QueryMDPVecEnv(I_array, cfg.c_q, cfg.p_i, cfg.gamma, cfg.n_envs, rng,
                              cfg.p_f, cfg.oracle_probs, cfg.bottleneck_sets)
    q        = QNet(n, cfg.hidden, cfg.n_layers).to(device)
    q_target = copy.deepcopy(q).to(device)
    opt      = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    buf      = TensorBuffer(cfg.buffer_cap, obs_dim, device)

    eps_decay_steps = max(1, int(cfg.eps_decay_frac * cfg.n_steps))

    def epsilon(step):
        f = min(1.0, step / eps_decay_steps)
        return cfg.eps_start + f * (cfg.eps_end - cfg.eps_start)

    def push_rollout(obs_np, actions_np):
        nobs_np, rew_np, done_np = env.step(actions_np)
        buf.push(
            torch.as_tensor(obs_np,                      device=device),
            torch.as_tensor(actions_np, dtype=torch.long, device=device),
            torch.as_tensor(rew_np,                      device=device),
            torch.as_tensor(nobs_np,                     device=device),
            torch.as_tensor(done_np.astype(np.float32),  device=device),
        )

    def random_valid_actions(valid_np):
        g = rng.random(valid_np.shape)
        g[~valid_np] = -1.0
        return g.argmax(1).astype(np.int64)

    # ── warmup: fill buffer with random transitions ───────────────────────────
    for _ in range(cfg.warmup_steps):
        obs_np, valid_np = env.obs(), env.valid()
        push_rollout(obs_np, random_valid_actions(valid_np))

    hist = {"step": [], "loss": [], "loss_std": [], "maxq0": [], "q0_std": [], "gap": [], "eps": [], "mean_q": []}
    _loss_buf: list = []
    _ckpt_every = cfg.checkpoint_every if cfg.checkpoint_every else 10 * cfg.eval_every
    stopped_early = False
    t0   = time.time()

    def _dump_checkpoint(cur_step):
        # Partial-progress snapshot so a long run is inspectable before it finishes.
        try:
            with open(cfg.checkpoint_path, "w") as f:
                json.dump({"history": hist, "step": cur_step, "n_steps": cfg.n_steps,
                           "wall_sec": time.time() - t0, "device": str(device)}, f)
        except OSError:
            pass  # never let a checkpoint I/O hiccup kill a training run

    for step in tqdm(range(cfg.n_steps)):
        # ── act: ε-greedy with action masking ────────────────────────────────
        obs_np, valid_np = env.obs(), env.valid()
        eps = epsilon(step)
        if rng.random() < eps:
            actions_np = random_valid_actions(valid_np)
        else:
            with torch.no_grad():
                qv = q(torch.as_tensor(obs_np, device=device))
                qv = qv.masked_fill(~torch.as_tensor(valid_np, device=device), _NEG)
                actions_np = qv.argmax(1).cpu().numpy().astype(np.int64)
        push_rollout(obs_np, actions_np)

        # ── learn: Bellman update with frozen target net ──────────────────────
        if buf.size >= cfg.batch_size:
            s, a, r, ns, d = buf.sample(cfg.batch_size)
            with torch.no_grad():
                nvalid = valid_from_obs_t(ns)
                if cfg.double_dqn:
                    # Double DQN: online net picks the next action, target net evaluates it.
                    # Decouples selection from evaluation → curbs the max-overestimation
                    # spiral that blows up Q-values (seen on the marginal oracle).
                    a_star = q(ns).masked_fill(~nvalid, _NEG).argmax(1, keepdim=True)
                    next_q = q_target(ns).gather(1, a_star).squeeze(1)
                else:
                    next_q = q_target(ns).masked_fill(~nvalid, _NEG).max(1)[0]
                target = r + cfg.gamma * (1.0 - d) * next_q
            cur  = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(cur, target)
            _loss_buf.append(loss.item())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)
            opt.step()
            if step % cfg.target_sync == 0:
                q_target.load_state_dict(q.state_dict())

            # ── log ──────────────────────────────────────────────────────────
            if step % cfg.eval_every == 0:
                with torch.no_grad():
                    s0     = torch.zeros(1, obs_dim, device=device)
                    q0     = q(s0)
                    maxq0  = q0.max().item()
                    mean_q = q0.mean().item()
                    q0_std = q0.std().item()
                gap       = abs(maxq0 - v_star) if v_star is not None else float("nan")
                loss_mean = float(np.mean(_loss_buf))
                loss_std  = float(np.std(_loss_buf))
                _loss_buf.clear()
                hist["step"].append(step);      hist["loss"].append(loss_mean)
                hist["loss_std"].append(loss_std)
                hist["maxq0"].append(maxq0);    hist["q0_std"].append(q0_std)
                hist["gap"].append(gap)
                hist["eps"].append(eps);        hist["mean_q"].append(mean_q)
                if cfg.verbose:
                    extra = f" | V*={v_star:.3f} gap={gap:.3f}" if v_star else ""
                    tqdm.write(f"step {step:6d}  loss {loss.item():7.4f}  "
                               f"maxQ(s0) {maxq0:7.3f}  eps {eps:4.2f}{extra}")

                # ── partial checkpoint (optional) ─────────────────────────────
                if cfg.checkpoint_path and step % _ckpt_every == 0:
                    _dump_checkpoint(step)

        # ── wall-clock cap (optional): stop cleanly, keep what we have ─────────
        if cfg.max_wall_sec is not None and (time.time() - t0) > cfg.max_wall_sec:
            stopped_early = True
            break

    wall = time.time() - t0
    if cfg.checkpoint_path:
        _dump_checkpoint(hist["step"][-1] if hist["step"] else 0)
    out  = {
        "history":     hist,
        "v_star":      v_star,
        "wall_sec":    wall,
        "device":      str(device),
        "final_maxq0": hist["maxq0"][-1] if hist["maxq0"] else float("nan"),
        "final_gap":   hist["gap"][-1]   if hist["gap"]   else float("nan"),
        "q":           q,
        "I_array":     I_array,
        "n":           n,
        "stopped_early": stopped_early,
    }
    if cfg.verbose:
        tqdm.write(f"done in {wall:.1f}s on {device}"
                   + (f"  | final gap = {out['final_gap']:.4f}" if v_star else ""))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory simulation under NN policy
# ─────────────────────────────────────────────────────────────────────────────

def simulate_nn_trajectories(q_net, env_class, I_decoded, num_tries, device="cpu"):
    unique_b = set()
    for recipe in I_decoded:
        for b in recipe:
            unique_b.add(tuple(b) if isinstance(b, list) else b)

    b_to_int = {b: idx for idx, b in enumerate(sorted(unique_b))}
    n_bottleneck = len(unique_b)
    len_I_array = len(I_decoded)

    I_array = np.zeros((len_I_array, n_bottleneck), dtype=bool)
    for i, recipe in enumerate(I_decoded):
        for b in recipe:
            b_tuple = tuple(b) if isinstance(b, list) else b
            I_array[i, b_to_int[b_tuple]] = True

    rng = np.random.default_rng()
    env = env_class(I_array, 0.0, 0.0, 1.0, n_envs=num_tries, rng=rng)

    q_net.eval()
    obs_np = env.obs()
    valid_np = env.valid()

    dones = np.zeros(num_tries, dtype=bool)
    steps = np.zeros(num_tries, dtype=int)

    while not dones.all():
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_np, device=device)
            qv = q_net(obs_t)
            vmask = torch.as_tensor(valid_np, device=device)
            qv = qv.masked_fill(~vmask, -1e9)
            actions = qv.argmax(1).cpu().numpy().astype(np.int64)

        obs_np, _, done_step = env.step(actions)
        valid_np = env.valid()
        steps[~dones] += 1
        dones = dones | done_step

    return steps


# ─────────────────────────────────────────────────────────────────────────────
# Real-human evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _run_query_episode(env, true_canonical, choose_action, max_queries):
    """Run one query episode; returns (n_queries, terminal_reward).

    choose_action(env) → action index

    The empty knowledge state is tested for absorption *before* the loop.  It can
    already be terminal: I_hat = ~K_not starts as the whole alphabet, so if that
    alphabet is contained in some I_k the instance is solved with zero questions.
    Entering the loop regardless would bill one query for a problem that never
    posed a question — and would pick that query from an all-zero
    best_action_mask, since a policy has no meaningful action at an absorbing
    state.
    """
    n_q             = 0
    terminal_reward = env.c_q  # default: failure
    done_arr, reward_arr = env._terminal_and_reward()
    if done_arr[0]:
        return 0, float(reward_arr[0])
    for _ in range(max_queries):
        action = choose_action(env)
        if action in true_canonical:
            env.K_I[0, action] = True
        else:
            env.K_not[0, action] = True
        n_q += 1
        env.apply_dominance()   # H2 free NOs; no-op for every other condition
        done_arr, reward_arr = env._terminal_and_reward()
        if done_arr[0]:
            terminal_reward = float(reward_arr[0])
            break
    return n_q, terminal_reward


def evaluate_policy_on_real_human(
    true_bottlenecks,
    policy_network,       # QNet / ExactQNet, or None for the random-order ("query-all") baseline
    n_runs,
    I_array,
    b_to_int,
    c_q=-10.0,
    p_i=1.0,
    p_f=0.0,
    gamma=0.99,
    device="cpu",
):
    """
    Evaluate a query policy against a real human with known implicit subgoals.

    The oracle answers deterministically: YES iff the queried bottleneck is in
    true_bottlenecks.

    If policy_network is None ("query-all" baseline), bottlenecks are queried
    in a fresh random order each run via shuffle_bottlenecks — no network needed.
    If policy_network is a QNet, the network selects the next query at each step.

    Parameters
    ----------
    true_bottlenecks : list of decoded bottleneck identifiers (tuples or lists)
        The bottlenecks the human would answer YES to.
    policy_network   : QNet, ExactQNet, or None
        None → random-order baseline; QNet/ExactQNet → guided policy.
    n_runs           : int  number of independent episodes to run
    I_array          : bool ndarray, shape (len_I_array, n_bottleneck)
    b_to_int         : dict  {bottleneck_tuple: column_index}
    c_q, p_i, gamma  : MDP parameters — must match those used during training
    device           : torch device string (ignored when policy_network is None)

    Returns
    -------
    dict with:
        'n_queries'    : int ndarray, length n_runs
        'total_reward' : float ndarray  (n_q * c_q + gamma * p_i if success)
        'success'      : bool ndarray
    """
    n_bottleneck = I_array.shape[1]

    # Canonical column indices that correspond to true_bottlenecks
    true_canonical = frozenset(
        b_to_int[tuple(b) if isinstance(b, (list, np.ndarray)) else b]
        for b in true_bottlenecks
        if (tuple(b) if isinstance(b, (list, np.ndarray)) else b) in b_to_int
    )

    n_queries_arr    = np.zeros(n_runs, dtype=int)
    total_reward_arr = np.zeros(n_runs, dtype=float)
    success_arr      = np.zeros(n_runs, dtype=bool)
    rng              = np.random.default_rng()

    if policy_network is None:
        def choose_action(env):
            return next(env._random_iter)
    else:
        policy_network.eval()
        def choose_action(env):
            obs_t = torch.as_tensor(env.obs(), dtype=torch.float32, device=device)
            vmask = torch.as_tensor(env.valid(), dtype=torch.bool,  device=device)
            with torch.no_grad():
                qv = policy_network(obs_t).masked_fill(~vmask, -1e9)
            return int(qv.argmax(1).cpu().item())

    for run in range(n_runs):
        env = QueryMDPVecEnv(I_array, c_q, p_i, gamma, n_envs=1, rng=rng, p_f=p_f,
                             dominance=getattr(policy_network, "dominance", None))
        if policy_network is None:
            env._random_iter = iter(shuffle_bottlenecks(list(range(n_bottleneck))))
        n_q, terminal_reward = _run_query_episode(env, true_canonical, choose_action,
                                                  max_queries=n_bottleneck)
        n_queries_arr[run]    = n_q
        total_reward_arr[run] = (n_q - 1) * c_q + terminal_reward
        success_arr[run]      = terminal_reward > c_q

    return {
        "n_queries":    n_queries_arr,
        "total_reward": total_reward_arr,
        "success":      success_arr,
    }
