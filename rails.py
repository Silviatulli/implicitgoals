"""Provable guardrails for a Query-MDP policy — the "+ rails" columns of
experiment.py.

Given the knowledge state (K_I, K_not), only the hypotheses still containing
every YES answer can witness success: C = {k : K_I ⊆ I_k}.  Every unqueried
bottleneck b then falls into one of three tiers:

  tier 1  b in NO I_k of C     -> ask first.  A YES leaves no compatible
                                  hypothesis (failure); a NO is needed before any
                                  I_k of C can be certified.  Never wasted.
  tier 3  b in EVERY I_k of C  -> never ask.  Neither answer can move the
                                  failure or the success test.
  tier 2  everything else      -> the wrapped policy decides.

Both wrappers keep the (batch, 2n) -> (batch, n) contract of ExactQNet and
GreedyQNet, so evaluate_policy_on_real_human takes them unchanged.
"""
import numpy as np
import torch
import torch.nn as nn

# Finite on purpose: evaluate_policy_on_real_human masks answered bottlenecks to
# -1e9, and a tier-3 action must still rank above those.
TIER_BIAS = 1e6


def rail_bias(obs, I_t):
    """(batch, n) additive bias: +TIER_BIAS on tier 1, -TIER_BIAS on tier 3.

    obs : (batch, 2n) float, [K_I || K_not]
    I_t : (len(I), n) bool, row k is I_k in B's bit order
    """
    n = obs.shape[1] // 2
    K_I, K_not = obs[:, :n] > 0.5, obs[:, n:] > 0.5
    C = ~(K_I[:, None, :] & ~I_t[None]).any(2)          # (batch, len(I))
    unqueried = ~(K_I | K_not)
    alive = C.any(1, keepdim=True)                      # no C: already failed
    in_some = (C[:, :, None] & I_t[None]).any(1)
    in_all = (~C[:, :, None] | I_t[None]).all(1)
    tier1 = unqueried & alive & ~in_some
    tier3 = unqueried & alive & in_all
    return (tier1.float() - tier3.float()) * TIER_BIAS


class RailedQNet(nn.Module):
    """`net` with the rails around it: tier 1 first, tier 3 last, `net` ranks
    tier 2.  The tiers are recomputed from (K_I, K_not) on every forward."""

    def __init__(self, net, I_array):
        super().__init__()
        self.net = net
        self.register_buffer("I_t", torch.as_tensor(np.asarray(I_array, dtype=bool)))

    def forward(self, x):
        return self.net(x) + rail_bias(x, self.I_t)


class RailsOnlyNet(nn.Module):
    """The rails with no rule inside: uniformly random among tier 2.  The railed
    counterpart of the Random column."""

    def __init__(self, I_array, seed=0):
        super().__init__()
        self.register_buffer("I_t", torch.as_tensor(np.asarray(I_array, dtype=bool)))
        self.g = torch.Generator().manual_seed(seed)

    def forward(self, x):
        r = torch.rand(x.shape[0], self.I_t.shape[1], generator=self.g)
        return r + rail_bias(x, self.I_t)
