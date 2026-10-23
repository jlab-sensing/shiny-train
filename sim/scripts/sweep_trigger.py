"""Sweep a variance-based replanning trigger over (risk level x depth x energy budget).

Trigger rule, from Cantelli's (one-sided Chebyshev) inequality:

    P(E_D >= budget) <= var_D / (var_D + (budget - mean_D)^2),   if mean_D < budget
                     <= 1,                                         otherwise

Replan at the first depth D where this bound reaches p_target. Equivalently,
replan when

    var_D >= p_target / (1 - p_target) * (budget - mean_D)^2,

which is a variance threshold that is justified by a brownout-risk level and
scales with the energy actually stored, rather than a fixed constant in J^2.
"""
from collections import defaultdict

import numpy as np

from sim.leaf_distribution import moments, sweep

CONST_VOLTAGE = 3.3
SCALE = 10.0  # measured power; 1.0 reproduces the earlier under-scaled values

# State order: measure, analyze, tx, rx. Analyze is set to twice the measure power
# (unmeasured; breaks the measure/analyze symmetry). Baseline was 11.68e-4.
power = np.array([11.68e-4, 23.36e-4, 86.52e-4, 20.03e-4]) * CONST_VOLTAGE * SCALE
duration = np.array([0.511, 0.511, 0.285, 0.937])
e = power * duration  # joules per task

# Same parameterization as your __main__:
#   a = P(analyze -> measure), b = P(rx -> tx)
a, b = 0.95, 0.02
P = [[0, 1, 0, 0], [a, 0, 1 - a, 0], [0, 0, 0, 1], [1 - b, 0, b, 0]]


def usable_energy(farads, v, v_min=0.1):
    """Energy above v_min, matching E_allowed in callback.build_model."""
    return farads * (v**2 - v_min**2) / 2


def cantelli_bound(mean, var, budget):
    gap = budget - mean
    return 1.0 if gap <= 0 else var / (var + gap * gap)


def cantelli_lower(mean, var, budget, floor=None):
    """LOWER bound on P(E_D > budget) from the lower-tail Cantelli inequality.

    floor: energy of the cheapest trajectory at this horizon, E_min(D). No
    trajectory uses less, so the risk is exactly 1 for budget < floor. Above the
    floor, Cantelli remains the sharpest bound available from mean and variance:
    its extremal two-point distribution puts mass at the budget itself, which
    respects the floor."""
    if floor is not None and budget < floor:
        return 1.0
    gap = mean - budget
    return 0.0 if gap <= 0 else gap * gap / (var + gap * gap)


def min_trajectory_energy(P, e, start, max_depth):
    """E_min(D) for D = 0..max_depth: energy of the cheapest trajectory, by a
    min-plus recursion. Uses only which transitions are possible, not their
    probabilities, so it is a structural fact of the chain."""
    best, out = {start: e[start]}, [e[start]]
    for _ in range(max_depth):
        nxt = {}
        for s, E in best.items():
            for j, pij in enumerate(P[s]):
                if pij > 0:
                    nxt[j] = min(nxt.get(j, float("inf")), E + e[j])
        best = nxt
        out.append(min(best.values()))
    return out


def failure_budget(mean, var, p, floor=None):
    """Failure budget B-_p: below it, brownout risk is guaranteed to be >= p.
    Cantelli gives mean - std * sqrt(p / (1 - p)); with the cheapest-trajectory
    floor, every budget below E_min(D) has risk 1, so B-_p >= E_min(D)."""
    b = mean - np.sqrt(var) * np.sqrt(p / (1 - p))
    return b if floor is None else max(b, floor)


def expected_completed(profile, budget, risk=None, e_min=None):
    """Expected number of tasks completed before brownout,
    E[N(B)] = sum_D P(E_D <= B). Pass risk=f(mean, var, marginal, B) to
    substitute a bound for the exact tail.

    With e_min (the smallest per-task energy), depths with (D+1)*e_min > B
    are skipped: no path can be that cheap, so P(E_D <= B) = 0 there. This
    cutoff is required for the Cantelli-based UPPER bound, whose terms decay
    only like 1/D and would otherwise sum to infinity."""
    rows = [r for r in profile if e_min is None or (r[0] + 1) * e_min <= budget]
    if risk is None:
        return sum(1 - exact_tail(mg, budget) for (_, _, _, mg) in rows)
    return sum(1 - risk(m, v, mg, budget) for (_, m, v, mg) in rows)


def progress_steps(profile):
    """Exact E[N(B)] as a step function: breakpoints at every atom energy of
    every depth, with jump P(E_D = x) summed over D. Returns (x, y)."""
    jumps = defaultdict(float)
    for _, _, _, mg in profile:
        for E, p in mg.items():
            jumps[E] += p
    x = np.array(sorted(jumps))
    return x, np.cumsum([jumps[v] for v in x])


def exact_tail(marginal, budget):
    """Computed brownout risk r_D(B) = P(E_D > B) from the leaf distribution."""
    return sum(p for E, p in marginal.items() if E > budget)


def required_budget(mean, var, p):
    """Cantelli safe budget B+_p: smallest budget for which the upper bound on
    risk stays at or below p, B = mean + std * sqrt((1 - p) / p)."""
    return mean + np.sqrt(var) * np.sqrt((1 - p) / p)


def exact_required_budget(marginal, p):
    """Quantile budget B_p(D): smallest atom B with P(E_D > B) <= p, computed
    from the leaf distribution (the (1-p)-quantile of E_D)."""
    tail = 1.0
    for E in sorted(marginal):
        tail -= marginal[E]
        if tail <= p + 1e-12:
            return E
    return max(marginal)


def capacitance_for(budget, v=3.2, v_min=0.1):
    return 2 * budget / (v**2 - v_min**2)


def depth_profile(P, e, start, max_depth):
    """One pass over depth: per-D mean, variance, and the exact marginal."""
    rows = []
    for D, marg in sweep(P, e, start, max_depth):
        mean, var = moments(marg)
        rows.append((D, mean, var, marg))
    return rows


def trigger_depth(profile, budget, p_target):
    """First depth whose Cantelli bound reaches p_target, or None."""
    for D, mean, var, _ in profile:
        if cantelli_bound(mean, var, budget) >= p_target:
            return D
    return None


if __name__ == "__main__":
    max_depth = 20
    profile = depth_profile(P, e, start=0, max_depth=max_depth)

    one_cap = usable_energy(6e-3, 3.2)
    two_caps = 2 * one_cap

    print(f"SCALE={SCALE}  alpha(A->B)={1-a:.2f}  beta(B->A)={1-b:.2f}  "
          f"lambda={a+b-1:+.2f}")
    print(f"budgets: one 6 mF cap {one_cap*1e3:.1f} mJ, two {two_caps*1e3:.1f} mJ\n")

    print(" D   mean(mJ)  std(mJ)   computed P(E>30.7mJ)  Cantelli bound")
    for D, mean, var, marg in profile:
        print(f"{D:2d}  {mean*1e3:8.2f}  {np.sqrt(var)*1e3:7.2f}   "
              f"{exact_tail(marg, one_cap):20.4f}  {cantelli_bound(mean, var, one_cap):14.4f}")

    print("\nTrigger depth D* (Cantelli) and computed brownout risk at D*:")
    for budget, label in [(one_cap, "one cap"), (two_caps, "two caps")]:
        for p in [0.01, 0.05, 0.10, 0.25]:
            D = trigger_depth(profile, budget, p)
            risk = exact_tail(profile[D][3], budget) if D is not None else float("nan")
            print(f"  {label:8s} p_target={p:4.2f}  D*={D}  computed risk at D*={risk:.4f}")

    print("\nSizing: budget and capacitance (at 3.2 V) needed for a D-task horizon at risk p")
    print(" D    p    B+_Cantelli(mJ)  B_quantile(mJ)  C+_Cantelli(mF)  C_quantile(mF)")
    for D, mean, var, marg in profile[1:9]:
        for p in [0.01, 0.05]:
            bc, be = required_budget(mean, var, p), exact_required_budget(marg, p)
            print(f"{D:2d}  {p:4.2f}  {bc*1e3:15.1f}  {be*1e3:14.1f}  "
                  f"{capacitance_for(bc)*1e3:15.1f}  {capacitance_for(be)*1e3:14.1f}")

    print("\nSmallest single capacitor (3.2 V) that can run each task alone:")
    for name, E in zip(["measure", "analyze", "tx", "rx"], e):
        print(f"  {name:8s} {E*1e3:6.1f} mJ -> {capacitance_for(E)*1e3:5.1f} mF")

    print("\nExpected tasks completed before brownout, E[N(B)]:")
    long_profile = depth_profile(P, e, start=0, max_depth=40)
    for B, label in [(one_cap, "one cap"), (two_caps, "two caps")]:
        print(f"  {label:8s} B={B*1e3:5.1f} mJ  E[N]={expected_completed(long_profile, B):.3f}")

    print("\nFigures: run figures.py")
