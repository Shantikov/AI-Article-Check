import argparse
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.calibration import CalibrationProfile
from app.config import get_settings
from app.onnx_detector import LocalOnnxDetector

from .benchmark import dataset_digest, load_jsonl
from .build_benchmark import DEFAULT_OUTPUT
from .core import (
    aggregate_scores,
    calibration_failures,
    choose_thresholds,
    evaluate,
    fit_platt,
    profile_from_fit,
    summarize_probabilities,
    summarize_segments,
)


DEFAULT_PROFILE = Path(__file__).parents[1] / "calibration.json"
DEFAULT_REPORT = Path(__file__).parent / "reports" / "latest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit calibration on the calibration split and report held-out metrics."
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--target-fnr", type=float, default=0.05)
    parser.add_argument("--min-ai-segments", type=int, default=3)
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
    if not args.benchmark.is_file():
        raise SystemExit(
            f"Benchmark not found: {args.benchmark}\n"
            "Run: python -m evaluation.build_benchmark"
        )
    records = load_jsonl(args.benchmark)
    if len(records) < 200:
        raise SystemExit("At least 200 labelled texts are required for calibration.")

    settings = get_settings()
    detector = LocalOnnxDetector(
        settings.local_model_id,
        settings.local_model_filename,
        settings.local_model_cache_dir,
        calibration_path=None,
        revision=settings.local_model_revision,
    )
    predictions: list[dict] = []
    for index, record in enumerate(records, start=1):
        chunk_scores = detector.score_text(record["text"])
        predictions.append({
            "id": record["id"],
            "split": record["split"],
            "label": 1 if record["label"] == "ai" else 0,
            "domain": record["domain"],
            "raw_score": aggregate_scores(chunk_scores),
            "segments": len(chunk_scores),
        })
        if index % 20 == 0 or index == len(records):
            print(f"Scored {index}/{len(records)} texts", flush=True)

    calibration = [item for item in predictions if item["split"] == "calibration"]
    test = [item for item in predictions if item["split"] == "test"]
    if not calibration or not test:
        raise SystemExit("Benchmark must contain both calibration and test splits.")

    fit = fit_platt(
        [item["raw_score"] for item in calibration],
        [item["label"] for item in calibration],
    )
    fitted_profile = profile_from_fit(fit)
    calibration_probabilities = [
        fitted_profile.predict(item["raw_score"])
        for item in calibration
    ]
    human_max, ai_min = choose_thresholds(
        calibration_probabilities,
        [item["label"] for item in calibration],
        target_fpr=args.target_fpr,
        target_fnr=args.target_fnr,
    )
    profile = {
        "schema_version": 3,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "HC3 English paired article-sized benchmark",
        "dataset_sha256": dataset_digest(records),
        "model_id": settings.local_model_id,
        "model_filename": settings.local_model_filename,
        "method": "article mean score plus regularized Platt calibration",
        "calibrator": {
            "kind": "regularized_platt",
            "feature": "standardized logit of article mean score",
            "slope": fit.slope,
            "intercept": fit.intercept,
            "feature_mean": fit.feature_mean,
            "feature_scale": fit.feature_scale,
            "probability_floor": fit.probability_floor,
            "regularization": 2.0,
            "finite_sample_targets": True,
        },
        "thresholds": {
            "human_max": human_max,
            "ai_min": ai_min,
            "min_ai_segments": max(1, args.min_ai_segments),
            "target_false_positive_rate": args.target_fpr,
            "target_false_negative_rate": args.target_fnr,
        },
    }
    runtime_profile = CalibrationProfile.from_dict(profile)

    def metrics_for(items: list[dict]) -> dict:
        probabilities = [runtime_profile.predict(item["raw_score"]) for item in items]
        return evaluate(
            probabilities,
            [item["label"] for item in items],
            [item["segments"] for item in items],
            human_max=human_max,
            ai_min=ai_min,
            min_ai_segments=runtime_profile.min_ai_segments,
        )

    domain_groups: dict[str, list[dict]] = defaultdict(list)
    for item in test:
        domain_groups[item["domain"]].append(item)
    test_metrics = metrics_for(test)
    test_probabilities = [
        runtime_profile.predict(item["raw_score"])
        for item in test
    ]
    test_probability_summary = summarize_probabilities(test_probabilities)
    test_segments = summarize_segments(
        [item["segments"] for item in test],
        runtime_profile.min_ai_segments,
    )
    failures = calibration_failures(
        test_metrics,
        test_segments,
        target_fpr=args.target_fpr,
        probability_summary=test_probability_summary,
    )
    report = {
        "dataset_sha256": profile["dataset_sha256"],
        "accepted": not failures,
        "rejection_reasons": failures,
        "calibration_split": metrics_for(calibration),
        "held_out_test": test_metrics,
        "held_out_segment_distribution": test_segments,
        "held_out_probability_distribution": test_probability_summary,
        "held_out_by_domain": {
            domain: metrics_for(items)
            for domain, items in sorted(domain_groups.items())
        },
        "thresholds": profile["thresholds"],
        "note": (
            "Thresholds and calibration parameters were fitted only on the calibration "
            "split. Held-out test records were used only for this report."
        ),
    }
    write_json(args.report, report)
    print(f"Held-out report: {args.report}")
    print(json.dumps(test_metrics, indent=2))
    print(json.dumps(test_segments, indent=2))
    print(json.dumps(test_probability_summary, indent=2))
    if failures:
        rejected_path = args.output.with_name(
            f"{args.output.stem}.rejected{args.output.suffix}"
        )
        write_json(rejected_path, profile)
        print(f"Rejected calibration candidate: {rejected_path}")
        raise SystemExit("Calibration rejected: " + " ".join(failures))

    write_json(args.output, profile)
    print(f"Calibration profile: {args.output}")


if __name__ == "__main__":
    main()
