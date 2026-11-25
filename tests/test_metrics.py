import math

import pytest

from ragbench.metrics import dcg_at_k, means, mrr_at_k, ndcg_at_k, per_query, recall_at_k


def test_hit_at_rank_one_is_perfect():
    assert ndcg_at_k(["a", "b", "c"], {"a"}) == 1.0
    assert mrr_at_k(["a", "b", "c"], {"a"}) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a"}, 1) == 1.0


def test_single_hit_at_rank_three_by_hand():
    # One relevant passage at rank 3: DCG = 1 / log2(4) = 0.5, ideal DCG = 1 / log2(2) = 1.
    assert dcg_at_k(["x", "y", "a"], {"a"}, 10) == pytest.approx(0.5)
    assert ndcg_at_k(["x", "y", "a"], {"a"}) == pytest.approx(0.5)
    assert mrr_at_k(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_two_relevant_passages_by_hand():
    # Relevant at ranks 2 and 4: DCG = 1/log2(3) + 1/log2(5); ideal = 1/log2(2) + 1/log2(3).
    ranking = ["x", "a", "y", "b"]
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranking, {"a", "b"}) == pytest.approx(expected)
    assert recall_at_k(ranking, {"a", "b"}, 2) == 0.5
    assert recall_at_k(ranking, {"a", "b"}, 4) == 1.0
    assert mrr_at_k(ranking, {"a", "b"}) == 0.5


def test_hit_below_the_cut_off_counts_nothing():
    ranking = [f"x{i}" for i in range(10)] + ["a"]  # "a" at rank 11
    assert ndcg_at_k(ranking, {"a"}, 10) == 0.0
    assert mrr_at_k(ranking, {"a"}, 10) == 0.0
    assert recall_at_k(ranking, {"a"}, 10) == 0.0
    assert recall_at_k(ranking, {"a"}, 100) == 1.0


def test_ndcg_falls_with_the_rank():
    values = [ndcg_at_k(["x"] * (r - 1) + ["a"], {"a"}) for r in range(1, 11)]
    assert values == sorted(values, reverse=True)
    assert values[-1] == pytest.approx(1 / math.log2(11))


def test_more_relevant_than_k_still_reaches_one():
    # The ideal ranking is cut at k as well, so a full top-k of hits is perfect.
    relevant = {f"r{i}" for i in range(15)}
    assert ndcg_at_k(sorted(relevant), relevant, 10) == pytest.approx(1.0)


def test_query_without_relevant_passages_is_an_error():
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set())
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 10)


def test_per_query_and_means():
    rankings = {"q1": ["a", "b"], "q2": ["b", "a"]}
    qrels = {"q1": {"a"}, "q2": {"a"}}
    values = per_query(rankings, qrels)
    assert values["mrr@10"] == {"q1": 1.0, "q2": 0.5}
    assert means(values)["mrr@10"] == 0.75
    assert means(values)["recall@10"] == 1.0
