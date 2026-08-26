import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from .benchmark import dataset_digest
from .web_benchmark import (
    DEFAULT_DOMAINS,
    SOURCE_FILENAME,
    SOURCE_LICENSE,
    SOURCE_REPO,
    SOURCE_REVISION,
    build_web_records,
    load_csv_rows,
    summarize_web_records,
)


DEFAULT_OUTPUT = Path(__file__).parent / "data" / "web_benchmark.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an independent 800-text MAGE benchmark for realistic "
            "out-of-dataset evaluation."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--records-per-domain",
        type=int,
        default=100,
        help="Even number, split equally between human and AI texts.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=80,
        help="Match the backend's minimum extracted article length.",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    print("Downloading independent MAGE test split (about 72 MB)", flush=True)
    source_path = hf_hub_download(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
        revision=SOURCE_REVISION,
        filename=SOURCE_FILENAME,
        cache_dir=args.cache_dir,
    )
    records = build_web_records(
        load_csv_rows(source_path),
        domains=DEFAULT_DOMAINS,
        records_per_domain=args.records_per_domain,
        min_words=args.min_words,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)

    summary = summarize_web_records(records)
    metadata = {
        "dataset": "MAGE independent natural-text benchmark",
        "purpose": "evaluation_only",
        "used_for_calibration": False,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_filename": SOURCE_FILENAME,
        "source_license": SOURCE_LICENSE,
        "domains": list(DEFAULT_DOMAINS),
        "records_per_domain": args.records_per_domain,
        "minimum_words": args.min_words,
        "sha256": dataset_digest(records),
        **summary,
    }
    write_json(args.output.with_suffix(".meta.json"), metadata)
    print(f"Wrote {len(records)} labelled texts to {args.output}")
    print(json.dumps({
        "labels": summary["labels"],
        "domains": summary["domains"],
        "ai_generators": len(summary["ai_generators"]),
        "word_counts": summary["word_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
