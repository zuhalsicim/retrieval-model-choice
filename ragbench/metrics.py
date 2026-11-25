"""Ranking metrics for one query. Relevance is a set of passage ids; every relevant passage counts 1.

GermanDPR has exactly one relevant passage per question, but the functions
work for any number, so that the tests can check them on hand-made cases.
"""

import math
from collections.abc import Sequence


def dcg_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Discounted cumulative gain: a hit at rank r (1-based) is worth 1 / log2(r + 1)."""
    return sum(1.0 / math.log2(rank + 1) for rank, pid in enumerate(ranking[:k], start=1) if pid in relevant)


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """DCG divided by the DCG of a perfect ranking, which puts all relevant passages first."""
    if not relevant:
        raise ValueError("a query without relevant passages has no nDCG")
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg_at_k(ranking, relevant, k) / ideal


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the relevant passages that are in the first k places."""
    if not relevant:
        raise ValueError("a query without relevant passages has no recall")
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def mrr_at_k(ranking: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """1 / rank of the first relevant passage, or 0 if none is in the first k places."""
    for rank, pid in enumerate(ranking[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


METRICS = {
    "ndcg@10": lambda ranking, relevant: ndcg_at_k(ranking, relevant, 10),
    "recall@10": lambda ranking, relevant: recall_at_k(ranking, relevant, 10),
    "recall@20": lambda ranking, relevant: recall_at_k(ranking, relevant, 20),
    "recall@100": lambda ranking, relevant: recall_at_k(ranking, relevant, 100),
    "mrr@10": lambda ranking, relevant: mrr_at_k(ranking, relevant, 10),
}


def per_query(rankings: dict[str, Sequence[str]], qrels: dict[str, set[str]]) -> dict[str, dict[str, float]]:
    """Every metric for every query: {metric: {query_id: value}}."""
    return {name: {qid: function(rankings[qid], qrels[qid]) for qid in rankings} for name, function in METRICS.items()}


def means(values: dict[str, dict[str, float]]) -> dict[str, float]:
    return {name: sum(per.values()) / len(per) for name, per in values.items()}
