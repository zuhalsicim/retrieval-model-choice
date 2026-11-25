"""One command line for every step.

python -m ragbench prepare      build data/germandpr-test from GermanDPR and write its manifest
python -m ragbench embed [keys] run the embedding models, each in its own process
python -m ragbench rerank       rerank the best embedding model with a clear licence
python -m ragbench quantize     float32, int8 and binary versions of the stored vectors
python -m ragbench guard        the prompt-injection classifier
python -m ragbench compare      paired statistics -> results/comparisons.json
python -m ragbench report       print the README tables
python -m ragbench licences     check the licence fields of all cards again
python -m ragbench all          embed, rerank, quantize, guard, compare, report
python -m ragbench smoke        a tiny run of every step into output-smoke/ (for CI)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ragbench import settings


def step(module: str, *args: str) -> None:
    """Run one step in a fresh Python process, so that its peak memory is its own."""
    command = [sys.executable, "-m", f"ragbench.{module}", *args]
    print("+", " ".join(command[2:]), flush=True)
    subprocess.run(command, check=True)


def embed(keys: list[str], out: str = "results", smoke: bool = False) -> None:
    for key in keys or [e.key for e in settings.EMBEDDERS]:
        step("run_embed", key, "--out", f"{out}/retrieval", *(["--smoke"] if smoke else []))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ragbench")
    parser.add_argument("command")
    parser.add_argument("keys", nargs="*")
    args = parser.parse_args()

    from ragbench import compare, data, licences, report

    if args.command == "prepare":
        print(json.dumps(data.prepare(), indent=2))
    elif args.command == "embed":
        embed(args.keys)
    elif args.command == "rerank":
        first = compare.first_stage()
        for key in args.keys or [r.key for r in settings.RERANKERS]:
            step("run_rerank", key, "--first-stage", first)
    elif args.command == "quantize":
        # Every embedding model with stored vectors; it costs seconds, no model is loaded.
        stored = [e.key for e in settings.EMBEDDERS if (Path("cache") / f"{e.key}.corpus.npy").exists()]
        for key in args.keys or stored:
            step("run_quant", key)
    elif args.command == "guard":
        for key in args.keys or [g.key for g in settings.GUARDS]:
            step("run_guard", key)
    elif args.command == "compare":
        compare.run()
    elif args.command == "report":
        report.main()
    elif args.command == "licences":
        for row in licences.check():
            print(row)
    elif args.command == "all":
        embed([])
        main_steps = ["rerank", "quantize", "guard", "compare", "report"]
        for command in main_steps:
            subprocess.run([sys.executable, "-m", "ragbench", command], check=True)
    elif args.command == "smoke":
        out = "output-smoke"
        data.verify(Path(settings.DATA_DIR))
        embed([settings.SMOKE_EMBEDDER], out, smoke=True)
        step(
            "run_rerank",
            settings.SMOKE_RERANKER,
            "--first-stage",
            settings.SMOKE_EMBEDDER,
            "--out",
            f"{out}/rerank",
            "--smoke",
        )
        step("run_quant", settings.SMOKE_EMBEDDER, "--out", f"{out}/quantization", "--smoke")
        for guard in settings.GUARDS:
            step("run_guard", guard.key, "--out", f"{out}/guard", "--smoke")
        check_smoke(Path(out))
    else:
        parser.error(f"unknown command {args.command}")


def check_smoke(out: Path) -> None:
    """Every metric of the smoke run must be a share between 0 and 1."""
    files = list(out.rglob("*.json"))
    if not files:
        raise SystemExit("the smoke run wrote no result files")
    for path in files:
        result = json.loads(path.read_text(encoding="utf-8"))
        for name in ("means", "means_after"):
            for metric, value in result.get(name, {}).items():
                if not 0 <= value <= 1:
                    raise SystemExit(f"{path}: {metric} = {value}")
    print(f"smoke run: {len(files)} result files, all metrics between 0 and 1")


if __name__ == "__main__":
    main()
