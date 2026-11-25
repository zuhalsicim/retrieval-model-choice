"""Build the retrieval test data from GermanDPR and keep a SHA-256 manifest of every derived file.

The corpus is every distinct passage of the test split: the positive passage
of each question and its three hard negatives. That gives 2876 passages for
1025 questions, the same corpus that MTEB's GermanDPR task uses.
"""

import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from ragbench import settings
from ragbench.fetch import fetch_dataset_file, file_sha256

DERIVED_FILES = ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "rerank_queries.txt"]


def passage_id(text: str) -> str:
    """A stable id from the text itself, so that the same passage always gets the same id."""
    return "p" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def build(frame: pd.DataFrame) -> tuple[dict[str, str], list[dict], dict[str, str]]:
    """Return corpus (id -> text), queries and qrels (query id -> id of its one positive passage)."""
    corpus: dict[str, str] = {}
    queries: list[dict] = []
    qrels: dict[str, str] = {}
    for number, row in enumerate(frame.itertuples(index=False)):
        positives = list(row.positive_ctxs["text"])
        if len(positives) != 1:
            raise ValueError(f"row {number} has {len(positives)} positive passages, expected 1")
        query_id = f"q{number:04d}"
        for text in positives + list(row.hard_negative_ctxs["text"]):
            corpus[passage_id(text)] = text
        queries.append({"id": query_id, "text": row.question, "answers": list(row.answers)})
        qrels[query_id] = passage_id(positives[0])
    return dict(sorted(corpus.items())), queries, qrels


def rerank_subset(query_ids: list[str], size: int, seed: int) -> list[str]:
    """A fixed random subset of the queries, sorted, for the slower reranking step."""
    return sorted(random.Random(seed).sample(sorted(query_ids), size))


def write(folder: Path, corpus: dict[str, str], queries: list[dict], qrels: dict[str, str], subset: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "corpus.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for pid, text in corpus.items():
            handle.write(json.dumps({"id": pid, "text": text}, ensure_ascii=False) + "\n")
    with open(folder / "queries.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            handle.write(json.dumps(query, ensure_ascii=False) + "\n")
    with open(folder / "qrels.tsv", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("query_id\tpassage_id\trelevance\n")
        for query_id, pid in qrels.items():
            handle.write(f"{query_id}\t{pid}\t1\n")
    (folder / "rerank_queries.txt").write_text("\n".join(subset) + "\n", encoding="utf-8", newline="\n")


def write_manifest(folder: Path, source: Path) -> dict:
    """Record where the data came from and the SHA-256 of every file derived from it."""
    manifest = {
        "source": {
            "repository": f"https://huggingface.co/datasets/{settings.GERMANDPR_REPO}",
            "card_revision": settings.GERMANDPR_CARD_REVISION,
            "parquet_revision": settings.GERMANDPR_PARQUET_REVISION,
            "file": settings.GERMANDPR_TEST_FILE,
            "sha256": file_sha256(source),
            "licence": "CC BY 4.0",
        },
        "build": {"seed": settings.SEED, "rerank_queries": settings.RERANK_QUERIES},
        "files": {name: file_sha256(folder / name) for name in DERIVED_FILES},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def verify(folder: Path) -> None:
    """Stop if a derived file differs from the manifest. Every run calls this first."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        actual = file_sha256(folder / name)
        if actual != expected:
            raise ValueError(f"{folder / name} has SHA-256 {actual}, the manifest says {expected}")


def prepare(folder: Path = Path(settings.DATA_DIR)) -> dict:
    source = fetch_dataset_file(
        settings.GERMANDPR_REPO,
        settings.GERMANDPR_PARQUET_REVISION,
        settings.GERMANDPR_TEST_FILE,
        settings.GERMANDPR_TEST_SHA256,
    )
    corpus, queries, qrels = build(pd.read_parquet(source))
    subset = rerank_subset([q["id"] for q in queries], settings.RERANK_QUERIES, settings.SEED)
    write(folder, corpus, queries, qrels, subset)
    return write_manifest(folder, source)


def load(folder: Path = Path(settings.DATA_DIR)) -> tuple[dict[str, str], list[dict], dict[str, str], list[str]]:
    verify(folder)
    with open(folder / "corpus.jsonl", encoding="utf-8") as handle:
        corpus = {row["id"]: row["text"] for row in map(json.loads, handle)}
    with open(folder / "queries.jsonl", encoding="utf-8") as handle:
        queries = [json.loads(line) for line in handle]
    lines = (folder / "qrels.tsv").read_text(encoding="utf-8").splitlines()[1:]
    qrels = {q: p for q, p, _ in (line.split("\t") for line in lines)}
    subset = (folder / "rerank_queries.txt").read_text(encoding="utf-8").split()
    return corpus, queries, qrels, subset
