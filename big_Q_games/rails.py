"""Provable guardrails for a Query-MDP policy.

Given the knowledge state (K_I, K_not) and the hypothesis matrix I_array, every
unqueried bottleneck falls into one of three tiers.  Two of them have a
provably correct action, so a learned policy should never be consulted about
them; the rails enforce that.

Compatible hypotheses:  C = {k : K_I subset of I_k}.
Only compatible k matter, because success needs I_hat subset of I_k, and
I_hat contains K_I, so any k witnessing success is compatible already.

  tier 1  b in NO I_k for k in C      -> query first
          YES: K_I + b sits outside every compatible hypothesis, so no
               hypothesis survives -> failure fires, episode over.
          NO : b leaves I_hat.  b blocked I_hat subset of I_k for EVERY k,
               so one query removes a blocker from all of them at once.

  tier 3  b in EVERY I_k for k in C   -> never query
          YES: every compatible k already contains b, so C is unchanged and
               failure cannot fire.
          NO : b leaves I_hat, but b was in I_k, so b was never the reason
               I_hat failed to fit I_k.  Nothing moves.
          Neither answer can change (done, success) -> the query is wasted.

  tier 2  everything else             -> the learned policy decides
"""
import numpy as np


def compatible(K_I, I_array):
    """(len_I,) bool -- hypotheses that still contain every YES answer."""
    return ~(K_I & ~I_array).any(1)


def tiers(K_I, K_not, I_array):
    """(tier1, tier3) bool masks over bottlenecks, for one knowledge state.

    Both are restricted to unqueried bottlenecks: an answered one is not a
    legal action and its tier is meaningless.
    """
    C = compatible(K_I, I_array)
    unqueried = ~(K_I | K_not)
    if not C.any():                      # already failed; no live actions
        return np.zeros_like(unqueried), np.zeros_like(unqueried)
    live = I_array[C]                    # (|C|, n)
    tier1 = unqueried & ~live.any(0)     # in no compatible hypothesis
    tier3 = unqueried & live.all(0)      # in every compatible hypothesis
    return tier1, tier3


def tiers_batch(K_I, K_not, I_array):
    """Vectorised over a batch of knowledge states.  K_I, K_not: (batch, n)."""
    C = ~((K_I[:, None, :] & ~I_array[None]).any(2))          # (batch, len_I)
    unqueried = ~(K_I | K_not)
    inc = C[:, :, None] & I_array[None]                        # (batch, len_I, n)
    any_c = inc.any(1)                                         # in some compatible
    all_c = (~C[:, :, None] | I_array[None]).all(1)            # in every compatible
    alive = C.any(1)[:, None]
    return (unqueried & ~any_c & alive), (unqueried & all_c & alive)


# ── torch mirror, so training needs no CPU round-trip ────────────────────────

import torch
import torch.nn as nn

TIER1_BONUS = 1e6      # finite: an answered bit is masked at -1e9 by the
TIER3_PENALTY = 1e6    # evaluator, and must stay the worst option of all


def tiers_torch(obs, I_t):
    """(tier1, tier3) bool tensors from an observation batch.

    obs : (batch, 2n) float -- [K_I || K_not]
    I_t : (len_I, n) bool
    """
    n = obs.shape[1] // 2
    K_I = obs[:, :n] > 0.5
    K_not = obs[:, n:] > 0.5
    C = ~((K_I[:, None, :] & ~I_t[None]).any(2))               # (batch, len_I)
    unqueried = ~(K_I | K_not)
    any_c = (C[:, :, None] & I_t[None]).any(1)
    all_c = (~C[:, :, None] | I_t[None]).all(1)
    alive = C.any(1)[:, None]
    return (unqueried & ~any_c & alive), (unqueried & all_c & alive)


def rail_bias(obs, I_t):
    """Additive score bias implementing both rails."""
    t1, t3 = tiers_torch(obs, I_t)
    return t1.float() * TIER1_BONUS - t3.float() * TIER3_PENALTY


class RailedQNet(nn.Module):
    """A QNet with the two provable rails wrapped around it.

    Drop-in for evaluate_policy_on_real_human: same (batch, 2n) -> (batch, n)
    contract.  Tier 1 is forced to the front of the ordering, tier 3 to the
    back, and the wrapped network only ever decides among tier 2.
    """
    def __init__(self, net, I_array):
        super().__init__()
        self.net = net
        self.register_buffer("I_t", torch.as_tensor(np.asarray(I_array, dtype=bool)))

    def forward(self, x):
        return self.net(x) + rail_bias(x, self.I_t)


class RailsOnlyNet(nn.Module):
    """The guardrails with no learning at all: tier 1 first, tier 3 never,
    uniformly random among tier 2.  Isolates what the rails alone are worth."""
    def __init__(self, I_array, seed=0):
        super().__init__()
        self.register_buffer("I_t", torch.as_tensor(np.asarray(I_array, dtype=bool)))
        self.g = torch.Generator().manual_seed(seed)

    def forward(self, x):
        r = torch.rand(x.shape[0], self.I_t.shape[1], generator=self.g)
        return r + rail_bias(x, self.I_t)


class RailsEntropyNet(nn.Module):
    """Rails, then one-step entropy over the *compatible* hypotheses inside tier 2.

    H1 uses the same entropy rule without the rails, and that is precisely its
    failure mode: a tier-1 bottleneck has f = 0 and a tier-3 has f = 1, so both
    score zero entropy and H1 avoids them -- including the tier-1 queries whose
    YES ends the episode outright.
    """
    def __init__(self, I_array):
        super().__init__()
        self.register_buffer("I_t", torch.as_tensor(np.asarray(I_array, dtype=bool)))

    def forward(self, x):
        n = x.shape[1] // 2
        K_I = x[:, :n] > 0.5
        C = ~((K_I[:, None, :] & ~self.I_t[None]).any(2))          # (batch, len_I)
        cnt = C.sum(1, keepdim=True).clamp(min=1).float()
        f = (C[:, :, None] & self.I_t[None]).sum(1).float() / cnt  # P(YES | compatible)
        p = f.clamp(1e-6, 1 - 1e-6)
        H = -(p * p.log2() + (1 - p) * (1 - p).log2())             # binary entropy
        return H + rail_bias(x, self.I_t)


class RailsStaticNet(nn.Module):
    """The rails computed ONCE at the empty knowledge state and then frozen.

    This is H3's tier structure with H3's geometry removed -- the control that
    isolates dynamic recomputation as the single changed variable.  By the
    absorbing property both tiers only grow, so a frozen mask is never wrong,
    merely incomplete.
    """
    def __init__(self, I_array, seed=0):
        super().__init__()
        I_t = torch.as_tensor(np.asarray(I_array, dtype=bool))
        self.register_buffer("I_t", I_t)
        n = I_t.shape[1]
        z = torch.zeros(1, 2 * n)
        t1, t3 = tiers_torch(z, I_t)                 # tiers at the empty state
        self.register_buffer("t1", t1[0])
        self.register_buffer("t3", t3[0])
        self.g = torch.Generator().manual_seed(seed)

    def forward(self, x):
        r = torch.rand(x.shape[0], self.I_t.shape[1], generator=self.g)
        return (r + self.t1.float() * TIER1_BONUS - self.t3.float() * TIER3_PENALTY)


class RailsProbNet(nn.Module):
    """Rails, with tier 1 ordered by descending marginal P(YES).

    Tier-1 queries are mandatory (see test_rails.py section 3), so their order only
    changes how soon a YES can end the episode.  A tier-1 YES is possible only for a
    non-representable human: if b were in the human's set S and S were inside some
    I_k, that I_k would be compatible and b would not be tier 1.  So asking the
    likeliest YES first shortens exactly the episodes that end in failure, and costs
    nothing on the others, where every tier-1 answer is a NO regardless of order.

    Uses only the per-bottleneck marginals -- the information solve_query_mdp_exact has.
    """
    def __init__(self, I_array, probs, seed=0):
        super().__init__()
        self.register_buffer("I_t", torch.as_tensor(np.asarray(I_array, dtype=bool)))
        self.register_buffer("p", torch.as_tensor(np.asarray(probs, dtype=np.float32)))
        self.g = torch.Generator().manual_seed(seed)

    def forward(self, x):
        t1, t3 = tiers_torch(x, self.I_t)
        r = torch.rand(x.shape[0], self.I_t.shape[1], generator=self.g)
        # inside tier 1 the marginal breaks the tie; elsewhere it is ignored
        return (r + t1.float() * (TIER1_BONUS + self.p[None] * 10.0)
                  - t3.float() * TIER3_PENALTY)
