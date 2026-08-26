import argparse
import json
import os
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from .benchmark import (
    DEFAULT_DOMAINS,
    SOURCE_LICENSE,
    SOURCE_REPO,
    SOURCE_REVISION,
    build_records,
    dataset_digest,
    load_jsonl,
)


DEFAULT_OUTPUT = Path(__file__).parent / "data" / "hc3_benchmark.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic HC3 calibration and test benchmark."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pairs-per-domain",
        type=int,
        default=50,
        help=(
            "Target average per domain. Shortfalls are redistributed so the "
            "default output still contains 200 paired questions."
        ),
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=40,
        help="Minimum length of each source answer before composition.",
    )
    parser.add_argument(
        "--min-document-words",
        type=int,
        default=660,
        help="Minimum composed document length; 660 words gives at least 3 samples.",
    )
    parser.add_argument("--max-document-words", type=int, default=1100)
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
    rows_by_domain: dict[str, list[dict]] = {}
    for domain in DEFAULT_DOMAINS:
        print(f"Downloading HC3 domain: {domain}", flush=True)
        source_path = hf_hub_download(
            repo_id=SOURCE_REPO,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            filename=f"{domain}.jsonl",
            cache_dir=args.cache_dir,
        )
        rows_by_domain[domain] = load_jsonl(source_path)

    records = build_records(
        rows_by_domain,
        pairs_per_domain=args.pairs_per_domain,
        min_words=args.min_words,
        min_document_words=args.min_document_words,
        max_document_words=args.max_document_words,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)

    counts = Counter((item["split"], item["label"]) for item in records)
    domain_pairs = Counter(
        item["domain"] for item in records if item["label"] == "human"
    )
    metadata = {
        "dataset": "HC3 English deterministic balanced subset",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "domains": list(DEFAULT_DOMAINS),
        "target_pairs_per_domain": args.pairs_per_domain,
        "selected_pairs_by_domain": dict(sorted(domain_pairs.items())),
        "min_source_answer_words": args.min_words,
        "min_document_words": args.min_document_words,
        "max_document_words": args.max_document_words,
        "records": len(records),
        "sha256": dataset_digest(records),
        "counts": {
            f"{split}_{label}": count
            for (split, label), count in sorted(counts.items())
        },
    }
    write_json(args.output.with_suffix(".meta.json"), metadata)
    print(f"Wrote {len(records)} labelled texts to {args.output}")
    print(json.dumps(metadata["selected_pairs_by_domain"], indent=2))
    print(json.dumps(metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
