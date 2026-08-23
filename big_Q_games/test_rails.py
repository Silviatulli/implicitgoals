"""Exhaustive tests for rails.py.

Small instances only, so every knowledge state can be enumerated and every
claim checked against the real termination rule rather than against itself.
"""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bottlenecks import _terminal
from rails import tiers, tiers_batch, compatible

FAIL = []
def check(cond, msg):
    if not cond:
        FAIL.append(msg)


def all_states(n):
    """Every (K_I, K_not) with disjoint supports -- the 3^n knowledge states."""
    for code in itertools.product((0, 1, 2), repeat=n):
        a = np.array(code)
        yield (a == 1), (a == 2)


def exhaustive(I_array, label):
    n = I_array.shape[1]
    n_t1 = n_t3 = n_states = 0
    for K_I, K_not in all_states(n):
        done, _ = _terminal(K_I, K_not, I_array)
        t1, t3 = tiers(K_I, K_not, I_array)

        # batch path must agree with the scalar path everywhere
        b1, b3 = tiers_batch(K_I[None], K_not[None], I_array)
        check((b1[0] == t1).all() and (b3[0] == t3).all(), f"{label}: batch != scalar")

        # a tier is only ever claimed for a legal (unqueried) action
        check(not (t1 & (K_I | K_not)).any(), f"{label}: tier1 on answered bit")
        check(not (t3 & (K_I | K_not)).any(), f"{label}: tier3 on answered bit")
        # the two tiers are disjoint unless no hypothesis is compatible
        check(not (t1 & t3).any(), f"{label}: tier1 and tier3 overlap")

        if done:
            continue
        n_states += 1
        for b in np.flatnonzero(t1):
            n_t1 += 1
            yK = K_I.copy(); yK[b] = True
            d_yes, s_yes = _terminal(yK, K_not, I_array)
            check(d_yes and not s_yes, f"{label}: tier1 YES did not fail-terminate")
        for b in np.flatnonzero(t3):
            n_t3 += 1
            yK = K_I.copy(); yK[b] = True
            nN = K_not.copy(); nN[b] = True
            check(_terminal(yK, K_not, I_array) == (False, False),
                  f"{label}: tier3 YES changed termination")
            check(_terminal(K_I, nN, I_array) == (False, False),
                  f"{label}: tier3 NO changed termination")
            # and YES must leave the compatible set untouched
            check((compatible(yK, I_array) == compatible(K_I, I_array)).all(),
                  f"{label}: tier3 YES changed the compatible set")
    return n_states, n_t1, n_t3


print("=" * 68)
print("1. hand-built instance, checked by hand")
print("=" * 68)
#  b:        0  1  2  3
#  I_0 = {0,1}, I_1 = {0,2}, I_2 = {1,2,3}
I = np.array([[1,1,0,0],[1,0,1,0],[0,1,1,1]], dtype=bool)
K_I = np.zeros(4, bool); K_not = np.zeros(4, bool)
t1, t3 = tiers(K_I, K_not, I)
print(f"  empty state       tier1={np.flatnonzero(t1)}  tier3={np.flatnonzero(t3)}")
check(not t1.any() and not t3.any(), "empty state should have no tier1/tier3 here")

# after YES on 3: only I_2 stays compatible, so 1 and 2 become 'in every compatible'
K_I = np.array([0,0,0,1], bool)
t1, t3 = tiers(K_I, K_not, I)
print(f"  after YES on b3   compatible={np.flatnonzero(compatible(K_I, I))}  "
      f"tier1={np.flatnonzero(t1)}  tier3={np.flatnonzero(t3)}")
check(list(np.flatnonzero(t1)) == [0], "b0 should be tier1 after YES on b3")
check(list(np.flatnonzero(t3)) == [1, 2], "b1,b2 should be tier3 after YES on b3")
print("  -> tiers migrate as the compatible set shrinks: confirmed")

print()
print("=" * 68)
print("2. exhaustive over every knowledge state, random instances")
print("=" * 68)
rng = np.random.default_rng(0)
tot = [0, 0, 0]
for n, len_I, reps in ((4, 3, 8), (5, 4, 8), (6, 5, 6), (7, 4, 3)):
    for r in range(reps):
        while True:
            A = rng.random((len_I, n)) < 0.5
            if A.any(1).all() and A.any(0).all():
                break
        s, a, b = exhaustive(A, f"n={n} r={r}")
        tot[0] += s; tot[1] += a; tot[2] += b
print(f"  {tot[0]:,} non-terminal states checked")
print(f"  {tot[1]:,} tier-1 claims verified (YES must fail-terminate)")
print(f"  {tot[2]:,} tier-3 claims verified (neither answer may move termination)")

print()
print("=" * 68)
print(f"{'ALL TESTS PASSED' if not FAIL else 'FAILURES:'}")
for f in FAIL[:20]:
    print("  " + f)
print("=" * 68)


# ── structural properties, appended after the first audit round ──────────────
print()
print("=" * 68)
print("3. absorbing tiers, and tier-1 mandatory")
print("=" * 68)
rng2 = np.random.default_rng(0)
lost1 = lost3 = early_success = n_ck = 0
for n, len_I in ((4, 3), (5, 4), (6, 4)):
    for _ in range(10):
        while True:
            A = rng2.random((len_I, n)) < 0.5
            if A.any(1).all() and A.any(0).all():
                break
        for K_I, K_not in all_states(n):
            if _terminal(K_I, K_not, A)[0]:
                continue
            t1, t3 = tiers(K_I, K_not, A)
            n_ck += 1
            for b in range(n):
                if (K_I | K_not)[b]:
                    continue
                for ans in (True, False):
                    nI, nN = K_I.copy(), K_not.copy()
                    (nI if ans else nN)[b] = True
                    if _terminal(nI, nN, A)[0]:
                        continue
                    u1, u3 = tiers(nI, nN, A)
                    live = ~(nI | nN)
                    lost1 += int((t1 & live & ~u1).any())
                    lost3 += int((t3 & live & ~u3).any())
            if t1.any():
                ih = ~K_not          # a tier-1 bit is still inside I_hat here
                early_success += int(any((A[k] | ~ih).all() for k in range(len_I)))
print(f"  {n_ck:,} non-terminal states")
print(f"  tier-1 membership lost after a query : {lost1}")
print(f"  tier-3 membership lost after a query : {lost3}")
print(f"  success reached with a tier-1 bit unasked : {early_success}")
check(lost1 == 0, "tier 1 is not absorbing")
check(lost3 == 0, "tier 3 is not absorbing")
check(early_success == 0, "success reached without asking a tier-1 bit")
print(f"\n  {'CONFIRMED' if not FAIL else 'FAILED'}: both tiers absorbing, tier-1 mandatory")
sys.exit(1 if FAIL else 0)
