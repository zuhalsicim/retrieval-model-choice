"""Rerank the first stage's top passages with one cross-encoder. One reranker per process.

python -m ragbench.run_rerank bge-reranker-v2-m3 --first-stage granite-278m [--out results] [--smoke]

The first stage is an embedding model whose rankings are in the cache
(written by run_embed). The reranker reorders its top RERANK_DEPTH passages
for the fixed subset of RERANK_QUERIES queries; the rest of the list stays.
"""

import argparse
import json
from pathlib import Path

from ragbench import data, metrics, settings
from ragbench.fetch import fetch_model
from ragbench.machine import Timer, describe, latency_summary, peak_memory_bytes
from ragbench.run_embed import weight_bytes


def rerank_one(scores, candidates: list[str], rest: list[str]) -> list[str]:
    """Candidates sorted by score, best first (ties keep the first-stage order), then the rest unchanged."""
    order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
    return [candidates[i] for i in order] + rest


def run(key: str, first_stage: str, out: Path, cache: Path, smoke: bool) -> dict:
    import torch
    from sentence_transformers import CrossEncoder

    model_settings = settings.reranker(key)
    corpus, queries, qrels, subset = data.load()
    tag = "smoke-" if smoke else ""
    rankings = json.loads((cache / f"{tag}{first_stage}.rankings.json").read_text())
    if smoke:
        subset = sorted(rankings)
    elif model_settings.queries < len(subset):
        subset = data.rerank_subset(subset, model_settings.queries, settings.SEED)
    texts = {q["id"]: q["text"] for q in queries}

    folder = fetch_model(model_settings.repo, model_settings.revision)
    with Timer() as load:
        model = CrossEncoder(
            model_settings.repo, revision=model_settings.revision, device="cpu", max_length=model_settings.max_length
        )

    depth = settings.RERANK_DEPTH
    reranked, times = {}, []
    with torch.inference_mode():
        for qid in subset[:3]:  # warm-up, not counted
            model.predict(
                [(texts[qid], corpus[p]) for p in rankings[qid][:depth]], batch_size=settings.RERANK_BATCH_SIZE
            )
        for qid in subset:
            candidates = rankings[qid][:depth]
            with Timer() as one:
                scores = model.predict(
                    [(texts[qid], corpus[p]) for p in candidates], batch_size=settings.RERANK_BATCH_SIZE
                )
            times.append(one.seconds)
            reranked[qid] = rerank_one(scores, candidates, rankings[qid][depth:])

    relevant = {q: {qrels[q]} for q in subset}
    after = metrics.per_query(reranked, relevant)
    before = metrics.per_query({q: rankings[q] for q in subset}, relevant)
    result = {
        "model": {
            "key": key,
            "repo": model_settings.repo,
            "revision": model_settings.revision,
            "params_millions": model_settings.params_millions,
            "weight_file_bytes": weight_bytes(folder),
            "max_length": model_settings.max_length,
        },
        "first_stage": first_stage,
        "depth": depth,
        "data": {"queries": len(subset), "smoke": smoke},
        "load_seconds": load.seconds,
        "latency_per_query": latency_summary(times),
        "pairs_per_second": len(subset) * depth / sum(times),
        "peak_memory_bytes": peak_memory_bytes(),
        "means_before": metrics.means(before),
        "means_after": metrics.means(after),
        "per_query_before": before,
        "per_query_after": after,
        "machine": describe(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{key}.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key", choices=[r.key for r in settings.RERANKERS])
    parser.add_argument("--first-stage", required=True, choices=[e.key for e in settings.EMBEDDERS])
    parser.add_argument("--out", default="results/rerank")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.key, args.first_stage, Path(args.out), Path(args.cache), args.smoke)
    print(
        f"{args.key} on {args.first_stage}: nDCG@10 {result['means_before']['ndcg@10']:.3f} -> "
        f"{result['means_after']['ndcg@10']:.3f}, p95 {result['latency_per_query']['p95_ms']:.0f} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
