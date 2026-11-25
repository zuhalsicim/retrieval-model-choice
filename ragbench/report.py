"""Print the tables of the README from the result files. Nothing in the README is typed in by hand.

python -m ragbench report [results]
"""

import json
from pathlib import Path

from ragbench.compare import licence_verdict, load_folder
from ragbench.licences import ENTRIES


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def interval(low: float, high: float) -> str:
    return f"{100 * low:+.1f} to {100 * high:+.1f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def retrieval_table(results: Path) -> str:
    runs = load_folder(results / "retrieval")
    rows = []
    for key, r in sorted(runs.items(), key=lambda kv: -kv[1]["means"]["ndcg@10"]):
        m, lat = r["means"], r["latency_one_query"]["total"]
        rows.append(
            [
                key,
                str(r["model"]["params_millions"]),
                licence_verdict(r["model"]["repo"]),
                pct(m["ndcg@10"]),
                pct(m["recall@10"]),
                pct(m["recall@100"]),
                pct(m["mrr@10"]),
                f"{r['corpus_encode_seconds']:.0f}",
                f"{r['passages_per_second']:.1f}",
                f"{lat['p50_ms']:.0f} / {lat['p95_ms']:.0f}",
                f"{r['peak_memory_bytes'] / 1e9:.2f}",
                str(r["dimensions"]),
                f"{r['index_bytes_float32'] / 1e6:.1f}",
                f"{r['model']['weight_file_bytes'] / 1e9:.2f}",
            ]
        )
    header = [
        "Model",
        "Params (M)",
        "Licence for a product",
        "nDCG@10",
        "Recall@10",
        "Recall@100",
        "MRR@10",
        "Corpus encode (s)",
        "Passages/s",
        "One query p50 / p95 (ms)",
        "Peak RAM (GB)",
        "Dims",
        "Index float32 (MB)",
        "Weights (GB)",
    ]
    return table(header, rows)


def comparison_table(items: list[dict]) -> str:
    rows = [
        [
            c["comparison"],
            c["metric"],
            str(c["queries"]),
            f"{100 * c['difference']:+.1f}",
            interval(*c["ci95_clustered"]),
            interval(*c["ci95_per_query"]),
            f"{c['p_clustered']:.4f}",
            "yes" if c["resolved"] else "no",
        ]
        for c in items
    ]
    header = [
        "Comparison (A - B)",
        "Metric",
        "Queries",
        "Difference (points)",
        "95 % interval, clustered",
        "95 % interval, per query",
        "p (clustered)",
        "Resolved?",
    ]
    return table(header, rows)


def rerank_table(results: Path) -> str:
    rows = []
    for key, r in sorted(load_folder(results / "rerank").items()):
        b, a, lat = r["means_before"], r["means_after"], r["latency_per_query"]
        rows.append(
            [
                f"{r['first_stage']} + {key}",
                licence_verdict(r["model"]["repo"]),
                str(r["data"]["queries"]),
                f"{pct(b['ndcg@10'])} -> {pct(a['ndcg@10'])}",
                f"{pct(b['mrr@10'])} -> {pct(a['mrr@10'])}",
                f"{pct(b['recall@10'])} -> {pct(a['recall@10'])}",
                pct(b["recall@20"]),
                f"{lat['p50_ms']:.0f} / {lat['p95_ms']:.0f}",
                f"{r['pairs_per_second']:.1f}",
                f"{r['peak_memory_bytes'] / 1e9:.2f}",
            ]
        )
    header = [
        "First stage + reranker",
        "Licence for a product",
        "Queries",
        "nDCG@10 before -> after",
        "MRR@10 before -> after",
        "Recall@10 before -> after",
        "Recall@20 (ceiling)",
        "Rerank 20 passages p50 / p95 (ms)",
        "Pairs/s",
        "Peak RAM (GB)",
    ]
    return table(header, rows)


def quantization_table(results: Path) -> str:
    rows = []
    for key, q in sorted(load_folder(results / "quantization").items()):
        base = q["variants"]["float32"]["index_bytes"]
        for name, v in q["variants"].items():
            m = v["means"]
            rows.append(
                [
                    f"{key} {name}",
                    f"{v['index_bytes_per_passage']:.0f}",
                    f"{v['index_bytes'] / 1e6:.2f}",
                    f"{base / v['index_bytes']:.0f}x" if name != "float32" else "1x",
                    pct(m["ndcg@10"]),
                    pct(m["recall@10"]),
                    pct(m["recall@100"]),
                ]
            )
    header = ["Variant", "Bytes per passage", "Index (MB)", "Smaller by", "nDCG@10", "Recall@10", "Recall@100"]
    return table(header, rows)


def guard_table(results: Path) -> str:
    rows = []
    for key, g in sorted(load_folder(results / "guard").items()):
        for name, c in g["groups"].items():
            if not c["rows"]:
                continue

            def show(share):
                if share is None:
                    return "-"
                return f"{pct(share['value'])} ({pct(share['low'])} to {pct(share['high'])})"

            rows.append(
                [
                    f"{key}: {name}",
                    str(c["rows"]),
                    f"{c['tp']} / {c['fp']} / {c['tn']} / {c['fn']}",
                    show(c["precision"]),
                    show(c["recall"]),
                    show(c["false_positive_rate"]),
                ]
            )
    header = [
        "Group",
        "Rows",
        "TP / FP / TN / FN",
        "Precision % (95 %)",
        "Recall % (95 %)",
        "False positives on harmless % (95 %)",
    ]
    return table(header, rows)


def guard_latency(results: Path) -> str:
    lines = []
    for key, g in sorted(load_folder(results / "guard").items()):
        lat = g["latency_one_text"]
        lines.append(
            f"{key}: one text p50 {lat['p50_ms']:.0f} ms, p95 {lat['p95_ms']:.0f} ms, "
            f"peak RAM {g['peak_memory_bytes'] / 1e9:.2f} GB, labels seen {g['labels_seen']}"
        )
    return "\n".join(lines)


def licence_table() -> str:
    rows = [
        [
            f"`{e['repo']}`",
            f"`{e['revision'][:10]}`",
            e["card_licence"],
            e["commercial"],
            e["measured"],
            e["note"],
        ]
        for e in ENTRIES
    ]
    return table(["Repository", "Revision", "Licence on the card", "Product use", "Measured", "Note"], rows)


def main(results: Path = Path("results")) -> None:
    comparisons = json.loads((results / "comparisons.json").read_text(encoding="utf-8"))
    print("## Retrieval\n")
    print(retrieval_table(results))
    print("\n## Embedding comparisons\n")
    print(comparison_table(comparisons["embedding"]))
    print("\n## Reranking\n")
    print(rerank_table(results))
    print()
    print(comparison_table(comparisons["rerank"]))
    print("\n## Quantization\n")
    print(quantization_table(results))
    print()
    print(comparison_table(comparisons["quantization"]))
    print("\n## Prompt-injection guard\n")
    print(guard_table(results))
    print()
    print(guard_latency(results))
    print("\n## Licences\n")
    print(licence_table())
