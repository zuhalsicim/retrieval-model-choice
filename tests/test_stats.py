import numpy as np
import pytest

from ragbench.stats import paired_bootstrap, paired_randomization_test, wilson_interval


def test_constant_difference_has_a_zero_width_interval():
    a = [0.8, 0.6, 0.9, 0.5]
    b = [0.7, 0.5, 0.8, 0.4]
    result = paired_bootstrap(a, b, samples=2000)
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["low"] == pytest.approx(0.1)
    assert result["high"] == pytest.approx(0.1)


def test_identical_systems():
    a = [0.1, 0.5, 1.0, 0.0, 0.3]
    result = paired_bootstrap(a, a)
    assert result["mean_difference"] == 0.0
    assert result["low"] == result["high"] == 0.0
    assert paired_randomization_test(a, a) == 1.0


def test_interval_contains_zero_for_symmetric_noise():
    # Differences +1 and -1 in equal numbers: the mean is 0 and the interval must cover it.
    a = [1.0, 0.0] * 50
    b = [0.0, 1.0] * 50
    result = paired_bootstrap(a, b)
    assert result["mean_difference"] == 0.0
    assert result["low"] < 0 < result["high"]


def test_bootstrap_interval_matches_the_normal_approximation():
    # 400 differences of +1 or 0 with share 1/2: mean 0.5, standard error 0.025.
    a = [1.0, 0.0] * 200
    b = [0.0] * 400
    result = paired_bootstrap(a, b, samples=20_000, seed=1)
    assert result["low"] == pytest.approx(0.5 - 1.96 * 0.025, abs=0.006)
    assert result["high"] == pytest.approx(0.5 + 1.96 * 0.025, abs=0.006)


def test_bootstrap_is_reproducible_with_the_seed():
    rng = np.random.default_rng(0)
    a, b = rng.random(50), rng.random(50)
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


def test_clusters_widen_the_interval_when_queries_move_together():
    # 20 clusters of 10 queries; within a cluster all differences are the same.
    rng = np.random.default_rng(3)
    cluster_effect = np.repeat(rng.normal(0.05, 0.2, 20), 10)
    clusters = np.repeat([f"c{i}" for i in range(20)], 10)
    a, b = cluster_effect, np.zeros(200)
    per_query = paired_bootstrap(a, b, seed=1)
    clustered = paired_bootstrap(a, b, clusters, seed=1)
    assert clustered["mean_difference"] == pytest.approx(per_query["mean_difference"])
    assert clustered["clusters"] == 20
    assert (clustered["high"] - clustered["low"]) > 2 * (per_query["high"] - per_query["low"])


def test_cluster_mean_weights_every_query_once():
    # Cluster x has three queries with difference 1, cluster y one query with difference 0.
    a = [1.0, 1.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0, 0.0]
    result = paired_bootstrap(a, b, ["x", "x", "x", "y"])
    assert result["mean_difference"] == 0.75


def test_randomization_test_exact_by_hand():
    # Five clusters, all differences +1: only "all +" and "all -" are as extreme, 2 of 32 patterns.
    a = [1.0] * 5
    b = [0.0] * 5
    assert paired_randomization_test(a, b) == pytest.approx(2 / 32)


def test_randomization_test_exact_with_one_reversed_cluster():
    # Differences +1, +1, +1, +1, -1: |sum| = 3. Patterns with |sum| >= 3: 2 * (1 + 5) = 12 of 32.
    a = [1.0, 1.0, 1.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0, 0.0, 1.0]
    assert paired_randomization_test(a, b) == pytest.approx(12 / 32)


def test_randomization_test_sampled_for_many_clusters():
    a = [1.0] * 30
    b = [0.0] * 30
    # The exact p-value is 2 / 2**30; with 10,000 samples the smallest possible value is 1 / 10,001.
    assert paired_randomization_test(a, b, samples=10_000) == pytest.approx(1 / 10_001)
    noise = np.random.default_rng(5).choice([-1.0, 1.0], 30)
    assert paired_randomization_test(noise, np.zeros(30)) > 0.05


def test_randomization_test_counts_clusters_not_queries():
    # Ten queries, all +1, but in only two clusters: 2 of the 4 sign patterns are as extreme.
    a = [1.0] * 10
    b = [0.0] * 10
    clusters = ["x"] * 5 + ["y"] * 5
    assert paired_randomization_test(a, b, clusters) == pytest.approx(0.5)


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0, 2.0], ["only one label"])


def test_wilson_half_of_one_hundred():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=0.0001)
    assert high == pytest.approx(0.5962, abs=0.0001)


def test_wilson_edges():
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
