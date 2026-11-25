"""Quantize the stored embeddings of one model and measure what it costs in retrieval quality.

python -m ragbench.run_quant e5-base [--out results/quantization]

No model is loaded: the float32 vectors of corpus and queries come from the
cache that run_embed wrote. Five ways to store and search the corpus:

- float32: as the model returns them, 4 bytes per dimension.
- int8 x int8: every dimension mapped to 256 steps between its minimum and
  maximum over the corpus (sentence-transformers' quantize_embeddings, which
  clips), 1 byte. The query is quantized the same way and the search takes
  the dot product of the int8 numbers.
- int8, float query: the same 1-byte passages, turned back into floats
  (dequantized) and compared with the unquantized query.
- binary: only the sign of every dimension, 1 bit, compared by Hamming distance.
- binary + int8 rescoring: binary search for BINARY_RESCORE_DEPTH candidates,
  then the float query is compared with the dequantized int8 vectors of those
  candidates. Positions after BINARY_RESCORE_DEPTH keep the binary order.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ragbench import data, metrics, settings
from ragbench.machine import Timer, latency_summary
from ragbench.search import (
    binary_then_rescore,
    dequantize_int8,
    dot_scores,
    hamming_distances,
    int8_ranges,
    int8_scores,
    top_k,
)

VARIANTS = ["float32", "int8 x int8", "int8, float query", "binary", "binary + int8 rescoring"]


def quantize(corpus: np.ndarray, queries: np.ndarray) -> dict[str, np.ndarray]:
    from sentence_transformers.util.quantization import quantize_embeddings

    ranges = int8_ranges(corpus)
    corpus_int8 = quantize_embeddings(corpus, "int8", ranges=ranges)
    return {
        "ranges": ranges,
        "corpus_int8": corpus_int8,
        "corpus_dequantized": dequantize_int8(corpus_int8, ranges),
        "queries_int8": quantize_embeddings(queries, "int8", ranges=ranges),
        "corpus_bits": quantize_embeddings(corpus, "ubinary"),
        "queries_bits": quantize_embeddings(queries, "ubinary"),
    }


def run(key: str, out: Path, cache: Path, smoke: bool) -> dict:
    tag = "smoke-" if smoke else ""
    corpus = np.load(cache / f"{tag}{key}.corpus.npy")
    queries = np.load(cache / f"{tag}{key}.queries.npy")
    ids = json.loads((cache / f"{tag}{key}.ids.json").read_text())
    _, _, qrels, _ = data.load()
    pids, qids = ids["passages"], ids["queries"]
    q = quantize(corpus, queries)
    depth = settings.RETRIEVE_DEPTH

    def search(name: str, rows: np.ndarray) -> np.ndarray:
        """Rankings (passage indices, best first) for the queries with these row numbers."""
        if name == "float32":
            return top_k(dot_scores(queries[rows], corpus), depth)
        if name == "int8 x int8":
            return top_k(int8_scores(q["queries_int8"][rows], q["corpus_int8"]).astype(np.float64), depth)
        if name == "int8, float query":
            return top_k(dot_scores(queries[rows], q["corpus_dequantized"]), depth)
        if name == "binary":
            return top_k(-hamming_distances(q["queries_bits"][rows], q["corpus_bits"]).astype(np.float64), depth)
        return np.stack(
            [
                binary_then_rescore(
                    queries[i],
                    q["queries_bits"][i],
                    q["corpus_bits"],
                    q["corpus_dequantized"],
                    settings.BINARY_RESCORE_DEPTH,
                    depth,
                )
                for i in rows
            ]
        )

    index_bytes = {
        "float32": corpus.nbytes,
        "int8 x int8": q["corpus_int8"].nbytes,
        "int8, float query": q["corpus_int8"].nbytes,
        "binary": q["corpus_bits"].nbytes,
        "binary + int8 rescoring": q["corpus_bits"].nbytes,  # in memory; the int8 vectors can stay on disk
    }
    all_rows = np.arange(len(qids))
    variants = {}
    for name in VARIANTS:
        order = search(name, all_rows)
        rankings = {qid: [pids[i] for i in row] for qid, row in zip(qids, order, strict=True)}
        per_query = metrics.per_query(rankings, {qid: {qrels[qid]} for qid in qids})
        # Search time for one query, on this small corpus, for completeness.
        times = []
        for i in range(min(len(qids), settings.LATENCY_QUERIES)):
            with Timer() as t:
                search(name, np.array([i]))
            times.append(t.seconds)
        variants[name] = {
            "index_bytes": int(index_bytes[name]),
            "index_bytes_per_passage": index_bytes[name] / len(pids),
            "search_one_query": latency_summary(times),
            "means": metrics.means(per_query),
            "per_query": per_query,
        }

    result = {
        "model": key,
        "passages": len(pids),
        "dimensions": int(corpus.shape[1]),
        "binary_rescore_depth": settings.BINARY_RESCORE_DEPTH,
        "variants": variants,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{key}.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key", choices=[e.key for e in settings.EMBEDDERS])
    parser.add_argument("--out", default="results/quantization")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.key, Path(args.out), Path(args.cache), args.smoke)
    for name, v in result["variants"].items():
        print(f"{args.key} {name}: nDCG@10 {v['means']['ndcg@10']:.3f}, {v['index_bytes'] / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
