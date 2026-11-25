"""Paired comparisons of two systems on the same queries.

Both functions take one value per query for system A and for system B, in
the same query order, and look at the differences A - B.

GermanDPR has many questions about the same passage (1025 questions, 192
distinct positive passages). Questions about the same passage are not
independent: a model that misreads a passage tends to miss all its
questions. Both functions therefore accept a cluster label per query and
then resample or flip whole clusters, not single queries. Without labels
every query is its own cluster.
"""

import itertools
import math
from collections.abc import Sequence

import numpy as np


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a share `successes / trials`; with z = 1.96 the 95 % interval.

    Unlike "share plus or minus" it stays between 0 and 1 and is still
    sensible for shares of 0 or 1, which happen with small groups.
    """
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError(f"cannot compute an interval for {successes} out of {trials}")
    share = successes / trials
    denominator = 1 + z * z / trials
    centre = (share + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(share * (1 - share) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _cluster_sums(a: Sequence[float], b: Sequence[float], clusters: Sequence[str] | None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("a and b must be two lists of the same, non-zero length")
    differences = a - b
    if clusters is None:
        return differences, np.ones_like(differences)
    if len(clusters) != a.size:
        raise ValueError("there must be one cluster label per query")
    _, index = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.bincount(index, weights=differences)
    counts = np.bincount(index).astype(float)
    return sums, counts


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[str] | None = None,
    samples: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
) -> dict:
    """Mean difference A - B and its percentile bootstrap interval.

    One bootstrap sample draws as many clusters as there are, with
    replacement, and takes the mean difference over all queries in them.
    If the interval contains 0, the data cannot tell which system is better.
    """
    sums, counts = _cluster_sums(a, b, clusters)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sums.size, size=(samples, sums.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return {
        "mean_difference": float(sums.sum() / counts.sum()),
        "low": float(low),
        "high": float(high),
        "level": level,
        "clusters": int(sums.size),
        "samples": samples,
    }


def paired_randomization_test(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[str] | None = None,
    samples: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided p-value of the paired sign-flip test for "A and B are equally good".

    If both systems were equally good, the sign of each cluster's difference
    would be a coin toss. The p-value is the share of sign patterns whose
    mean difference is at least as far from 0 as the observed one. With up
    to 16 clusters all 2**n patterns are counted exactly; with more, `samples`
    random patterns are drawn and the observed one is counted as one of them.
    """
    sums, counts = _cluster_sums(a, b, clusters)
    total = counts.sum()
    observed = abs(sums.sum()) / total
    tolerance = 1e-12  # differences of floats that are equal on paper
    if sums.size <= 16:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=sums.size)))
        flipped = np.abs(signs @ sums) / total
        return float(np.mean(flipped >= observed - tolerance))
    rng = np.random.default_rng(seed)
    signs = rng.choice((1.0, -1.0), size=(samples, sums.size))
    flipped = np.abs(signs @ sums) / total
    return float((1 + np.sum(flipped >= observed - tolerance)) / (samples + 1))
