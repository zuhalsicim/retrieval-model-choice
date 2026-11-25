"""Run one embedding model: encode the corpus, search, score, and time it. One model per process.

python -m ragbench.run_embed e5-base [--out results] [--smoke]

Every model runs in a process of its own, so that the peak memory belongs
to that model alone.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ragbench import data, metrics, settings
from ragbench.fetch import fetch_model
from ragbench.machine import Timer, describe, latency_summary, peak_memory_bytes
from ragbench.search import dot_scores, top_k


def smoke_subset(corpus, queries, qrels):
    """A few queries, their positive passages and the first 200 passages of the corpus as distractors."""
    chosen = queries[: settings.SMOKE_QUERIES]
    ids = [q["id"] for q in chosen]
    keep = {qrels[q] for q in ids} | set(list(corpus)[:200])
    small = {pid: text for pid, text in corpus.items() if pid in keep}
    return small, chosen, {q: qrels[q] for q in ids}, ids


def weight_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.suffix in {".safetensors", ".bin"})


def run(key: str, out: Path, cache: Path, smoke: bool) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    model_settings = settings.embedder(key)
    corpus, queries, qrels, subset = data.load()
    if smoke:
        corpus, queries, qrels, subset = smoke_subset(corpus, queries, qrels)

    folder = fetch_model(model_settings.repo, model_settings.revision)
    with Timer() as load:
        model = SentenceTransformer(model_settings.repo, revision=model_settings.revision, device="cpu")
        model.max_seq_length = model_settings.max_seq_length

    pids = list(corpus)
    passages = [model_settings.passage_prefix + corpus[pid] for pid in pids]
    texts = [model_settings.query_prefix + q["text"] for q in queries]
    with torch.inference_mode():
        with Timer() as encode_corpus:
            corpus_vectors = model.encode(
                passages, batch_size=settings.ENCODE_BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True
            )
        with Timer() as encode_queries:
            query_vectors = model.encode(
                texts, batch_size=settings.ENCODE_BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True
            )

        order = top_k(dot_scores(query_vectors, corpus_vectors), settings.RETRIEVE_DEPTH)
        rankings = {q["id"]: [pids[i] for i in row] for q, row in zip(queries, order, strict=True)}
        per_query = metrics.per_query(rankings, {q: {p} for q, p in qrels.items()})

        # Latency as a live system sees it: one query at a time, encode plus search.
        timed = queries[: settings.LATENCY_QUERIES]
        for q in timed[:5]:  # warm-up, not counted
            model.encode([model_settings.query_prefix + q["text"]], normalize_embeddings=True)
        encode_times, search_times, total_times = [], [], []
        for q in timed:
            with Timer() as total:
                with Timer() as encode_one:
                    vector = model.encode(
                        [model_settings.query_prefix + q["text"]], batch_size=1, normalize_embeddings=True
                    )
                with Timer() as search_one:
                    top_k(dot_scores(vector, corpus_vectors), settings.RETRIEVE_DEPTH)
            encode_times.append(encode_one.seconds)
            search_times.append(search_one.seconds)
            total_times.append(total.seconds)

    tag = "smoke-" if smoke else ""
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / f"{tag}{key}.corpus.npy", corpus_vectors)
    np.save(cache / f"{tag}{key}.queries.npy", query_vectors)
    (cache / f"{tag}{key}.ids.json").write_text(json.dumps({"passages": pids, "queries": [q["id"] for q in queries]}))
    (cache / f"{tag}{key}.rankings.json").write_text(json.dumps(rankings))

    dimensions = int(corpus_vectors.shape[1])
    result = {
        "model": {
            "key": key,
            "repo": model_settings.repo,
            "revision": model_settings.revision,
            "params_millions": model_settings.params_millions,
            "weight_file_bytes": weight_bytes(folder),
            "query_prefix": model_settings.query_prefix,
            "passage_prefix": model_settings.passage_prefix,
            "max_seq_length": model_settings.max_seq_length,
        },
        "data": {"passages": len(pids), "queries": len(queries), "smoke": smoke},
        "dimensions": dimensions,
        "index_bytes_float32": len(pids) * dimensions * 4,
        "load_seconds": load.seconds,
        "corpus_encode_seconds": encode_corpus.seconds,
        "passages_per_second": len(pids) / encode_corpus.seconds,
        "query_batch_encode_seconds": encode_queries.seconds,
        "latency_one_query": {
            "encode": latency_summary(encode_times),
            "search": latency_summary(search_times),
            "total": latency_summary(total_times),
        },
        "peak_memory_bytes": peak_memory_bytes(),
        "means": metrics.means(per_query),
        "per_query": per_query,
        "machine": describe(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{key}.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key", choices=[e.key for e in settings.EMBEDDERS])
    parser.add_argument("--out", default="results/retrieval")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.key, Path(args.out), Path(args.cache), args.smoke)
    means = ", ".join(f"{k} {v:.3f}" for k, v in result["means"].items())
    print(f"{args.key}: {means}; corpus {result['corpus_encode_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
