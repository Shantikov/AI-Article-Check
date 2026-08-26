import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from app.calibration import CalibrationProfile
from app.config import get_settings
from app.onnx_detector import LocalOnnxDetector

from .benchmark import dataset_digest, load_jsonl
from .build_web_benchmark import DEFAULT_OUTPUT
from .core import evaluate, summarize_probabilities, summarize_segments
from .score_cache import (
    score_records,
    scores_are_compatible,
    write_json,
    write_jsonl,
)


DEFAULT_CALIBRATION = Path(__file__).parents[1] / "calibration.json"
DEFAULT_REPORT = Path(__file__).parent / "reports" / "web_latest.json"
DEFAULT_SCORES = Path(__file__).parent / "data" / "web_scores.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the active calibration on the independent MAGE benchmark "
            "without fitting or changing any detector setting."
        )
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Ignore compatible cached raw scores and run the ONNX model again.",
    )
    return parser.parse_args()


def _metrics_for(items: list[dict], profile: CalibrationProfile) -> dict:
    return evaluate(
        [item["probability"] for item in items],
        [item["label"] for item in items],
        [item["segments"] for item in items],
        human_max=profile.human_max,
        ai_min=profile.ai_min,
        min_ai_segments=profile.min_ai_segments,
        profile=profile,
    )


def _ai_metrics_for(items: list[dict], profile: CalibrationProfile) -> dict:
    metrics = _metrics_for(items, profile)
    return {
        "samples": metrics["samples"],
        "detected_ai": metrics["true_positive"],
        "classified_human": metrics["false_negative"],
        "uncertain": metrics["uncertain"],
        "ai_recall": metrics["ai_recall"],
        "false_negative_rate": metrics["false_negative_rate"],
        "uncertain_rate": metrics["uncertain_rate"],
        "mean_ai_probability": fmean(
            item["probability"] for item in items
        ) if items else 0.0,
        "mean_segments": fmean(
            item["segments"] for item in items
        ) if items else 0.0,
    }


def build_report(
    predictions: list[dict],
    profile: CalibrationProfile,
    *,
    benchmark_sha256: str,
    model_id: str,
    model_filename: str,
    reused_scores: bool,
) -> dict:
    by_domain: dict[str, list[dict]] = defaultdict(list)
    ai_by_generator: dict[str, list[dict]] = defaultdict(list)
    ai_by_mode: dict[str, list[dict]] = defaultdict(list)
    by_length: dict[str, list[dict]] = defaultdict(list)
    for item in predictions:
        item["prediction"] = profile.classify(
            item["probability"], item["segments"]
        )
        by_domain[item["domain"]].append(item)
        if item["label"] == 1:
            ai_by_generator[item["generator"]].append(item)
            ai_by_mode[item["generation_mode"]].append(item)
        if item["segments"] == 1:
            length_bucket = "1_segment"
        elif item["segments"] == 2:
            length_bucket = "2_segments"
        else:
            length_bucket = "3_or_more_segments"
        by_length[length_bucket].append(item)

    probabilities = [item["probability"] for item in predictions]
    human_probabilities = [
        item["probability"] for item in predictions if item["label"] == 0
    ]
    ai_probabilities = [
        item["probability"] for item in predictions if item["label"] == 1
    ]

    def compact_example(item: dict) -> dict:
        return {
            "id": item["id"],
            "domain": item["domain"],
            "generator": item["generator"],
            "generation_mode": item["generation_mode"],
            "segments": item["segments"],
            "ai_probability": item["probability"],
            "prediction": item["prediction"],
        }

    false_positives = sorted(
        (
            item
            for item in predictions
            if item["label"] == 0 and item["prediction"] == "ai_likely"
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    missed_ai = sorted(
        (
            item
            for item in predictions
            if item["label"] == 1 and item["prediction"] == "human_likely"
        ),
        key=lambda item: item["probability"],
    )
    uncertain = sorted(
        (item for item in predictions if item["prediction"] == "uncertain"),
        key=lambda item: abs(item["probability"] - 0.5),
        reverse=True,
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "MAGE independent natural-text benchmark",
        "dataset_sha256": benchmark_sha256,
        "evaluation_only": True,
        "changed_calibration": False,
        "model_id": model_id,
        "model_filename": model_filename,
        "calibration_dataset": profile.dataset,
        "reused_cached_raw_scores": reused_scores,
        "overall": _metrics_for(predictions, profile),
        "segment_distribution": summarize_segments(
            [item["segments"] for item in predictions],
            profile.min_ai_segments,
        ),
        "probability_distribution": {
            "all": summarize_probabilities(probabilities),
            "human": summarize_probabilities(human_probabilities),
            "ai": summarize_probabilities(ai_probabilities),
        },
        "by_domain": {
            key: _metrics_for(items, profile)
            for key, items in sorted(by_domain.items())
        },
        "ai_by_generator": {
            key: _ai_metrics_for(items, profile)
            for key, items in sorted(ai_by_generator.items())
        },
        "ai_by_generation_mode": {
            key: _ai_metrics_for(items, profile)
            for key, items in sorted(ai_by_mode.items())
        },
        "by_length": {
            key: _metrics_for(items, profile)
            for key, items in sorted(by_length.items())
        },
        "error_examples": {
            "highest_scoring_false_positives": [
                compact_example(item) for item in false_positives[:20]
            ],
            "lowest_scoring_missed_ai": [
                compact_example(item) for item in missed_ai[:20]
            ],
            "most_extreme_uncertain": [
                compact_example(item) for item in uncertain[:20]
            ],
        },
        "thresholds": (
            {
                key: {
                    "human_max": band.human_max,
                    "ai_min": band.ai_min,
                }
                for key, band in sorted(profile.bands.items())
            }
            if profile.bands
            else {
                "human_max": profile.human_max,
                "ai_min": profile.ai_min,
                "min_ai_segments": profile.min_ai_segments,
            }
        ),
        "note": (
            "This dataset was used only for external evaluation. No threshold, "
            "calibration parameter, or runtime detector setting was fitted on it."
        ),
    }


def main() -> None:
    args = parse_args()
    if not args.benchmark.is_file():
        raise SystemExit(
            f"Web benchmark not found: {args.benchmark}\n"
            "Run: python -m evaluation.build_web_benchmark"
        )
    if not args.calibration.is_file():
        raise SystemExit(
            f"Calibration profile not found: {args.calibration}\n"
            "Run: python -m evaluation.calibrate_length_aware"
        )
    try:
        profile = CalibrationProfile.from_file(args.calibration)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Calibration profile is invalid: {exc}") from exc

    records = load_jsonl(args.benchmark)
    if not records:
        raise SystemExit("Web benchmark is empty.")
    if any(record.get("split") != "external_test" for record in records):
        raise SystemExit(
            "The independent report accepts only external_test records."
        )
    benchmark_sha256 = dataset_digest(records)
    settings = get_settings()

    scores: list[dict] = []
    reused_scores = False
    if args.scores.is_file() and not args.rescore:
        candidate = load_jsonl(args.scores)
        if scores_are_compatible(
            candidate,
            records,
            benchmark_sha256=benchmark_sha256,
            model_id=settings.local_model_id,
            model_filename=settings.local_model_filename,
            model_revision=settings.local_model_revision,
        ):
            scores = candidate
            reused_scores = True
            print(f"Reusing {len(scores)} compatible cached raw scores", flush=True)

    if not scores:
        detector = LocalOnnxDetector(
            settings.local_model_id,
            settings.local_model_filename,
            settings.local_model_cache_dir,
            calibration_path=None,
            revision=settings.local_model_revision,
        )
        scores = score_records(
            records,
            detector,
            benchmark_sha256=benchmark_sha256,
            model_id=settings.local_model_id,
            model_filename=settings.local_model_filename,
            model_revision=settings.local_model_revision,
        )
        write_jsonl(args.scores, scores)
        print(f"Raw scores cache: {args.scores}")

    score_by_id = {item["id"]: item for item in scores}
    predictions: list[dict] = []
    for record in records:
        score = score_by_id[record["id"]]
        predictions.append({
            "id": record["id"],
            "label": 1 if record["label"] == "ai" else 0,
            "domain": record["domain"],
            "generator": record["generator"],
            "generation_mode": record["generation_mode"],
            "segments": int(score["segments"]),
            "raw_score": float(score["raw_score"]),
            "probability": profile.predict(
                float(score["raw_score"]), int(score["segments"])
            ),
        })

    report = build_report(
        predictions,
        profile,
        benchmark_sha256=benchmark_sha256,
        model_id=settings.local_model_id,
        model_filename=settings.local_model_filename,
        reused_scores=reused_scores,
    )
    write_json(args.report, report)
    print(f"Independent web report: {args.report}")
    print(json.dumps(report["overall"], indent=2))
    print(json.dumps(report["segment_distribution"], indent=2))


if __name__ == "__main__":
    main()
