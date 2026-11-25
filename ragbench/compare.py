"""Paired comparisons between the result files, written to results/comparisons.json.

Every comparison uses the same queries for both sides and clusters the
queries by their positive passage (see ragbench/stats.py).
"""

import json
from pathlib import Path

from ragbench import data, settings, stats
from ragbench.licences import ENTRIES


def licence_verdict(repo: str) -> str:
    return next(e["commercial"] for e in ENTRIES if e["repo"] == repo)


def load_folder(folder: Path) -> dict[str, dict]:
    if not folder.exists():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}


def first_stage(results: Path = Path("results")) -> str:
    """The best embedding model by nDCG@10 among those whose licence verdict is "yes"."""
    runs = load_folder(results / "retrieval")
    clear = {k: r for k, r in runs.items() if licence_verdict(r["model"]["repo"]) == "yes"}
    if not clear:
        raise ValueError("no embedding model with a clear licence has results yet")
    return max(clear, key=lambda k: clear[k]["means"]["ndcg@10"])


def compare(name: str, metric: str, a: dict[str, float], b: dict[str, float], clusters: dict[str, str]) -> dict:
    ids = sorted(set(a) & set(b))
    x, y, c = [a[i] for i in ids], [b[i] for i in ids], [clusters[i] for i in ids]
    clustered = stats.paired_bootstrap(x, y, c, settings.BOOTSTRAP_SAMPLES, settings.SEED)
    per_query = stats.paired_bootstrap(x, y, None, settings.BOOTSTRAP_SAMPLES, settings.SEED)
    p = stats.paired_randomization_test(x, y, c, settings.PERMUTATIONS, settings.SEED)
    return {
        "comparison": name,
        "metric": metric,
        "queries": len(ids),
        "clusters": clustered["clusters"],
        "mean_a": sum(x) / len(x),
        "mean_b": sum(y) / len(y),
        "difference": clustered["mean_difference"],
        "ci95_clustered": [clustered["low"], clustered["high"]],
        "ci95_per_query": [per_query["low"], per_query["high"]],
        "p_clustered": p,
        # Resolved only if both say so: the clustered interval excludes 0 and the test is below 0.05.
        "resolved": (clustered["low"] > 0 or clustered["high"] < 0) and p < 0.05,
    }


def run(results: Path = Path("results")) -> dict:
    _, _, qrels, _ = data.load()
    clusters = dict(qrels)  # query id -> id of its positive passage
    out = {"embedding": [], "rerank": [], "quantization": []}

    embedding = load_folder(results / "retrieval")
    if embedding:
        ranked = sorted(embedding, key=lambda k: -embedding[k]["means"]["ndcg@10"])
        best = ranked[0]
        for other in ranked[1:]:
            for metric in ("ndcg@10", "recall@100"):
                out["embedding"].append(
                    compare(
                        f"{best} - {other}",
                        metric,
                        embedding[best]["per_query"][metric],
                        embedding[other]["per_query"][metric],
                        clusters,
                    )
                )

    rerank = load_folder(results / "rerank")
    for key, r in rerank.items():
        for metric in ("ndcg@10", "mrr@10", "recall@10"):
            out["rerank"].append(
                compare(
                    f"{r['first_stage']} + {key} - {r['first_stage']} alone",
                    metric,
                    r["per_query_after"][metric],
                    r["per_query_before"][metric],
                    clusters,
                )
            )
    keys = sorted(rerank)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            out["rerank"].append(
                compare(
                    f"{a} - {b} (shared queries)",
                    "ndcg@10",
                    rerank[a]["per_query_after"]["ndcg@10"],
                    rerank[b]["per_query_after"]["ndcg@10"],
                    clusters,
                )
            )

    for key, q in load_folder(results / "quantization").items():
        base = q["variants"]["float32"]["per_query"]
        for name, variant in q["variants"].items():
            if name == "float32":
                continue
            for metric in ("ndcg@10", "recall@100"):
                out["quantization"].append(
                    compare(f"{key} {name} - float32", metric, variant["per_query"][metric], base[metric], clusters)
                )

    (results / "comparisons.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out
