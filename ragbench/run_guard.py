"""Evaluate one prompt-injection classifier on deepset/prompt-injections, one text at a time.

python -m ragbench.run_guard protectai-v2 [--out results/guard] [--smoke]

Label 1 in the data is an injection, 0 a harmless input. Every text is
classified alone (batch size 1), as a guard in front of a live system would
see it, and timed. The model's own top label is the decision; no threshold
is tuned, because tuning it on the same data would flatter the result.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ragbench import settings
from ragbench.fetch import fetch_dataset_file, fetch_model, file_sha256
from ragbench.language import guess_language
from ragbench.machine import Timer, describe, latency_summary, peak_memory_bytes
from ragbench.run_embed import weight_bytes
from ragbench.stats import wilson_interval


def load_rows() -> tuple[list[dict], dict]:
    rows, hashes = [], {}
    for split, filename in settings.INJECTION_FILES.items():
        path = fetch_dataset_file(settings.INJECTION_REPO, settings.INJECTION_REVISION, filename)
        hashes[filename] = file_sha256(path)
        for number, row in enumerate(pd.read_parquet(path).itertuples(index=False)):
            rows.append({"id": f"{split}-{number:03d}", "split": split, "text": row.text, "label": int(row.label)})
    return rows, hashes


def confusion(rows: list[dict]) -> dict:
    """Counts and shares for the rows given; `predicted` and `label` are 1 for injection."""
    tp = sum(r["predicted"] == 1 and r["label"] == 1 for r in rows)
    fp = sum(r["predicted"] == 1 and r["label"] == 0 for r in rows)
    tn = sum(r["predicted"] == 0 and r["label"] == 0 for r in rows)
    fn = sum(r["predicted"] == 0 and r["label"] == 1 for r in rows)

    def share(hits, total):
        if total == 0:
            return None
        low, high = wilson_interval(hits, total)
        return {"value": hits / total, "low": low, "high": high, "of": total}

    return {
        "rows": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": share(tp, tp + fp),  # of the inputs it blocks, how many were injections
        "recall": share(tp, tp + fn),  # of the injections, how many it blocks
        "false_positive_rate": share(fp, fp + tn),  # of the harmless inputs, how many it blocks
    }


def run(key: str, out: Path, smoke: bool) -> dict:
    import torch
    from transformers import pipeline

    guard = next(g for g in settings.GUARDS if g.key == key)
    rows, hashes = load_rows()
    if smoke:
        rows = rows[:4] + rows[-4:]
    for r in rows:
        r["language"] = guess_language(r["text"])

    folder = fetch_model(guard.repo, guard.revision)
    with Timer() as load:
        classify = pipeline(
            "text-classification",
            model=guard.repo,
            revision=guard.revision,
            device="cpu",
            truncation=True,
            max_length=512,
        )
    times = []
    with torch.inference_mode():
        for r in rows[:5]:  # warm-up, not counted
            classify(r["text"])
        for r in rows:
            with Timer() as one:
                output = classify(r["text"])[0]
            times.append(one.seconds)
            r["model_label"] = output["label"]
            r["score"] = float(output["score"])
            r["predicted"] = int(output["label"] == guard.injection_label)

    groups = {
        "all": rows,
        "test split": [r for r in rows if r["split"] == "test"],
        "train split": [r for r in rows if r["split"] == "train"],
        "German": [r for r in rows if r["language"] == "de"],
        "English": [r for r in rows if r["language"] == "en"],
        "language unknown": [r for r in rows if r["language"] == "unknown"],
    }
    result = {
        "model": {
            "key": key,
            "repo": guard.repo,
            "revision": guard.revision,
            "params_millions": guard.params_millions,
            "weight_file_bytes": weight_bytes(folder),
        },
        "data": {
            "repo": settings.INJECTION_REPO,
            "revision": settings.INJECTION_REVISION,
            "sha256": hashes,
            "smoke": smoke,
        },
        "labels_seen": sorted({r["model_label"] for r in rows}),
        "load_seconds": load.seconds,
        "latency_one_text": latency_summary(times),
        "peak_memory_bytes": peak_memory_bytes(),
        "groups": {name: confusion(group) for name, group in groups.items()},
        # Text is left out on purpose: the result file holds ids, labels and decisions only.
        "per_row": [{k: r[k] for k in ("id", "split", "language", "label", "predicted", "score")} for r in rows],
        "machine": describe(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{key}.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key", choices=[g.key for g in settings.GUARDS])
    parser.add_argument("--out", default="results/guard")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.key, Path(args.out), args.smoke)
    for name, c in result["groups"].items():
        if c["rows"]:
            p = c["precision"]["value"] if c["precision"] else float("nan")
            r = c["recall"]["value"] if c["recall"] else float("nan")
            print(f"{args.key} {name}: {c['rows']} rows, precision {p:.3f}, recall {r:.3f}", flush=True)


if __name__ == "__main__":
    main()
