"""Reproducible workflow commands; no hidden agent or paid API calls."""

import argparse
import json
from pathlib import Path
import random
from .data import download, read_rows
from .aspects import extract_rules, validate_findings


def main():
    parser = argparse.ArgumentParser(
        description="WorkBuddy: review intelligence and fit modeling"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "train", "run", "annotations"):
        command = sub.add_parser(name)
        command.add_argument(
            "--data", type=Path, default=Path("data/raw/renttherunway.json.gz")
        )
        if name in ("train", "run"):
            command.add_argument("--output", type=Path, default=Path("reports"))
            command.add_argument("--limit", type=int)
            command.add_argument("--max-iter", type=int, default=35)
        if name == "annotations":
            command.add_argument(
                "--output", type=Path, default=Path("data/annotations/reviews.jsonl")
            )
            command.add_argument("--count", type=int, default=200)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--root", type=Path, default=Path.cwd())
    evaluate = sub.add_parser("evaluate-aspects")
    evaluate.add_argument("annotations", type=Path)
    args = parser.parse_args()
    if args.command == "serve":
        from .web import create_app

        create_app(args.root).run(host="127.0.0.1", port=args.port, debug=False)
    elif args.command == "download":
        print(download(args.data))
    elif args.command in ("train", "run"):
        if args.command == "run" and not args.data.exists():
            download(args.data)
        from .train import run_experiment

        run_experiment(args.data, args.output, args.limit, args.max_iter)
    elif args.command == "annotations":
        rows, _ = read_rows(args.data)
        rows = [r for r in rows if r["review"].strip()]
        selected = random.Random(42).sample(rows, min(args.count, len(rows)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise SystemExit(
                "Annotation file exists; choose another --output to avoid overwriting labels."
            )
        with args.output.open("w") as f:
            for i, row in enumerate(selected):
                f.write(
                    json.dumps({"id": i, "review": row["review"], "findings": None})
                    + "\n"
                )
        print(
            "Exported",
            len(selected),
            "unlabeled reviews. Complete findings manually; do not publish raw dataset reviews.",
        )
    elif args.command == "evaluate-aspects":
        tp = fp = fn = total = 0
        for line in args.annotations.read_text().splitlines():
            row = json.loads(line)
            gold = validate_findings({"findings": row.get("findings")}, row["review"])
            gold = {(f["aspect"], f["sentiment"]) for f in gold}
            prediction = {
                (f["aspect"], f["sentiment"])
                for f in extract_rules(row["review"])["findings"]
            }
            tp += len(gold & prediction)
            fp += len(prediction - gold)
            fn += len(gold - prediction)
            total += 1
        print(
            json.dumps(
                {
                    "annotated_reviews": total,
                    "engine": "rules",
                    "metric": "micro aspect-sentiment pair F1",
                    "precision": tp / max(1, tp + fp),
                    "recall": tp / max(1, tp + fn),
                    "f1": 2 * tp / max(1, 2 * tp + fp + fn),
                },
                indent=2,
            )
        )
