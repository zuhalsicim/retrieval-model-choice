"""Exact search over a small corpus with numpy: float32, int8 and binary vectors.

The corpus has 2876 passages, so a flat index (compare the query with every
passage) is exact and fast enough. A production system with millions of
passages would use an approximate index; this benchmark measures the
models, not the index.
"""

import numpy as np


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first. Ties go to the lower index."""
    scores = np.atleast_2d(scores)
    k = min(k, scores.shape[1])
    # argsort of the negated scores with a stable sort keeps ties in index order.
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def dot_scores(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Dot products. For normalized vectors this is the cosine similarity."""
    return np.atleast_2d(queries).astype(np.float32) @ corpus.astype(np.float32).T


def int8_scores(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Dot products of int8 vectors, computed in int32 so that nothing overflows."""
    return np.atleast_2d(queries).astype(np.int32) @ corpus.astype(np.int32).T


def int8_ranges(corpus: np.ndarray) -> np.ndarray:
    """Per dimension the smallest and largest value over the corpus: shape (2, dims)."""
    return np.stack([corpus.min(axis=0), corpus.max(axis=0)]).astype(np.float32)


def dequantize_int8(values: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Turn int8 codes back into floats: the middle of the step each code stands for.

    quantize_embeddings maps x to floor((x - start) / step) - 128 with
    step = (max - min) / 255, so the code c stands for start + (c + 128.5) * step.
    """
    start = ranges[0]
    step = (ranges[1] - start) / 255
    step = np.where(step == 0, 1, step)
    return (start + (values.astype(np.float32) + 128.5) * step).astype(np.float32)


def hamming_distances(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Number of differing bits between packed binary vectors (uint8, 8 dimensions per byte)."""
    queries = np.atleast_2d(queries)
    out = np.empty((queries.shape[0], corpus.shape[0]), dtype=np.int32)
    for row, query in enumerate(queries):
        out[row] = np.bitwise_count(np.bitwise_xor(corpus, query)).sum(axis=1)
    return out


def binary_then_rescore(
    query_float: np.ndarray,
    query_bits: np.ndarray,
    corpus_bits: np.ndarray,
    corpus_rescore: np.ndarray,
    depth: int,
    k: int,
) -> np.ndarray:
    """Binary search for `depth` candidates, then reorder them with the float query against `corpus_rescore`.

    This is the usual pattern: the small binary index sits in memory, the
    larger vectors for rescoring (here dequantized int8) are read only for
    the few candidates. Positions after `depth` keep the binary order.
    """
    order = top_k(-hamming_distances(query_bits, corpus_bits).astype(np.float64), max(depth, k))[0]
    head, tail = order[:depth], order[depth:]
    rescored = query_float.astype(np.float32) @ corpus_rescore[head].astype(np.float32).T
    head = head[np.argsort(-rescored, kind="stable")]
    return np.concatenate([head, tail])[:k]
