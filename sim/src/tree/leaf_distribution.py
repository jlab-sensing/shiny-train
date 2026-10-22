"""Exact leaf-energy distributions for a finite Markov task chain."""
from collections import defaultdict


def leaf_distribution(P, e, start=None, steps=1, dist=None, ndigits=12):
    """Advance the joint (state, cumulative energy) distribution by `steps` generations.

    Parameters
    ----------
    P : sequence of sequences of float, shape (n, n)
        Row-stochastic transition matrix. P[i][j] is the probability of
        moving from state i to state j. State order must match `e`.
    e : sequence of float, length n
        Energy consumed by each state's task, in joules (power * duration).
    start : int, optional
        Index of the root state. Required when `dist` is None; ignored otherwise.
    steps : int, default 1
        Number of generations to advance. steps=0 with dist=None returns
        the depth-0 distribution (the root alone).
    dist : dict[(int, float), float], optional
        Joint distribution returned by a previous call, mapping
        (current state, cumulative energy) -> probability. If None, a new
        distribution is created at depth 0 containing only the root.
        Never mutated, so earlier distributions stay valid.
    ndigits : int, default 12
        Decimal places to which cumulative energies are rounded, so paths
        with equal energy merge into one key despite float error.

    Returns
    -------
    dist : dict[(int, float), float]
        Joint distribution after advancing. Pass it back in to continue.
        It does not record its own depth; the caller tracks that.
    marginal : dict[float, float]
        Distribution of cumulative energy over leaves, summed over
        terminal states.
    """
    if dist is None:
        if start is None:
            raise ValueError("start is required when dist is None")
        dist = {(start, round(e[start], ndigits)): 1.0}
    for _ in range(steps):
        nxt = defaultdict(float)
        for (s, E), p in dist.items():
            for j, pij in enumerate(P[s]):
                if pij > 0:
                    nxt[(j, round(E + e[j], ndigits))] += p * pij
        dist = dict(nxt)
    return dist, marginalize(dist)


def marginalize(dist):
    """Sum a joint (state, energy) distribution over states -> {energy: probability}."""
    out = defaultdict(float)
    for (_, E), p in dist.items():
        out[E] += p
    return dict(out)


def moments(marginal):
    """Mean and variance of a {energy: probability} distribution.
    Variance is computed in centered form to avoid cancellation error."""
    mean = sum(E * p for E, p in marginal.items())
    var = sum((E - mean) ** 2 * p for E, p in marginal.items())
    return mean, var


def sweep(P, e, start, max_depth, ndigits=12):
    """Yield (D, marginal) for D = 0, 1, ..., max_depth, advancing one
    generation at a time so no depth is ever recomputed from the root."""
    dist, marginal = leaf_distribution(P, e, start=start, steps=0, ndigits=ndigits)
    yield 0, marginal
    for D in range(1, max_depth + 1):
        dist, marginal = leaf_distribution(P, e, steps=1, dist=dist, ndigits=ndigits)
        yield D, marginal
