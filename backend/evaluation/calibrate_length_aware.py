import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.calibration import CalibrationProfile, segment_band
from app.config import get_settings
from app.onnx_detector import LocalOnnxDetector

from .benchmark import dataset_digest, load_jsonl
from .build_validation_benchmark import DEFAULT_OUTPUT
from .core import (
    choose_thresholds,
    evaluate,
    fit_platt,
    profile_from_fit,
    summarize_probabilities,
    summarize_segments,
)
from .score_cache import (
    score_records,
    scores_are_compatible,
    write_json,
    write_jsonl,
)


DEFAULT_PROFILE = Path(__file__).parents[1] / "calibration.json"
DEFAULT_REPORT = Path(__file__).parent / "reports" / "length_calibration.json"
DEFAULT_SCORES = Path(__file__).parent / "data" / "web_validation_scores.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit independent calibration curves and conservative thresholds "
            "for 1, 2, and 3+ detector samples using MAGE validation data."
        )
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--short-target-fpr", type=float, default=0.01)
    parser.add_argument("--long-target-fpr", type=float, default=0.03)
    parser.add_argument("--target-fnr", type=float, default=0.05)
    parser.add_argument("--rescore", action="store_true")
    return parser.parse_args()


def _calibrator_payload(fit) -> dict:
    return {
        "kind": "regularized_platt",
        "feature": "standardized logit of mean sample score",
        "slope": fit.slope,
        "intercept": fit.intercept,
        "feature_mean": fit.feature_mean,
        "feature_scale": fit.feature_scale,
        "probability_floor": fit.probability_floor,
        "regularization": 2.0,
        "finite_sample_targets": True,
    }


def fit_length_aware_profile(
    predictions: list[dict],
    *,
    dataset_sha256: str,
    model_id: str,
    model_filename: str,
    model_revision: str | None = None,
    short_target_fpr: float = 0.01,
    long_target_fpr: float = 0.03,
    target_fnr: float = 0.05,
) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in predictions:
        grouped[segment_band(int(item["segments"]))].append(item)

    required = ("1_segment", "2_segments", "3_or_more_segments")
    bands: dict[str, dict] = {}
    for band_name in required:
        items = grouped[band_name]
        label_counts = Counter(int(item["label"]) for item in items)
        if label_counts[0] < 30 or label_counts[1] < 30:
            raise ValueError(
                f"{band_name} needs at least 30 human and 30 AI records; "
                f"found {label_counts[0]} and {label_counts[1]}"
            )
        fit = fit_platt(
            [float(item["raw_score"]) for item in items],
            [int(item["label"]) for item in items],
        )
        temporary_profile = profile_from_fit(fit)
        probabilities = [
            temporary_profile.predict(float(item["raw_score"]))
            for item in items
        ]
        target_fpr = (
            long_target_fpr
            if band_name == "3_or_more_segments"
            else short_target_fpr
        )
        human_max, ai_min = choose_thresholds(
            probabilities,
            [int(item["label"]) for item in items],
            target_fpr=target_fpr,
            target_fnr=target_fnr,
        )
        bands[band_name] = {
            "samples": len(items),
            "human_samples": label_counts[0],
            "ai_samples": label_counts[1],
            "calibrator": _calibrator_payload(fit),
            "thresholds": {
                "human_max": human_max,
                "ai_min": ai_min,
                "target_false_positive_rate": target_fpr,
                "target_false_negative_rate": target_fnr,
            },
        }

    payload = {
        "schema_version": 4,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "MAGE validation length-aware natural-text calibration",
        "dataset_sha256": dataset_sha256,
        "model_id": model_id,
        "model_filename": model_filename,
        "method": (
            "separate regularized Platt curves and conservative thresholds "
            "for 1, 2, and 3+ samples"
        ),
        "bands": bands,
        "external_test_used": False,
    }
    if model_revision:
        payload["model_revision"] = model_revision
    return payload


def validation_report(
    predictions: list[dict],
    profile: CalibrationProfile,
    *,
    dataset_sha256: str,
) -> dict:
    enriched: list[dict] = []
    for item in predictions:
        segments = int(item["segments"])
        enriched.append({
            **item,
            "probability": profile.predict(float(item["raw_score"]), segments),
        })

    def metrics(items: list[dict]) -> dict:
        return evaluate(
            [float(item["probability"]) for item in items],
            [int(item["label"]) for item in items],
            [int(item["segments"]) for item in items],
            human_max=profile.human_max,
            ai_min=profile.ai_min,
            min_ai_segments=profile.min_ai_segments,
            profile=profile,
        )

    by_band: dict[str, list[dict]] = defaultdict(list)
    for item in enriched:
        by_band[segment_band(int(item["segments"]))].append(item)
    return {
        "dataset_sha256": dataset_sha256,
        "fit_dataset": "MAGE validation",
        "external_test_used": False,
        "overall": metrics(enriched),
        "by_length": {
            key: metrics(items) for key, items in sorted(by_band.items())
        },
        "segment_distribution": summarize_segments(
            [int(item["segments"]) for item in enriched], 1
        ),
        "probability_distribution": summarize_probabilities(
            [float(item["probability"]) for item in enriched]
        ),
        "note": (
            "These are fit-set diagnostics, not final accuracy. Run "
            "evaluation.evaluate_web for the untouched MAGE test result."
        ),
    }


def main() -> None:
    args = parse_args()
    for value, name in (
        (args.short_target_fpr, "short-target-fpr"),
        (args.long_target_fpr, "long-target-fpr"),
        (args.target_fnr, "target-fnr"),
    ):
        if not 0 <= value <= 0.25:
            raise SystemExit(f"--{name} must be between 0 and 0.25")
    if not args.benchmark.is_file():
        raise SystemExit(
            f"Validation benchmark not found: {args.benchmark}\n"
            "Run: python -m evaluation.build_validation_benchmark"
        )

    records = load_jsonl(args.benchmark)
    if len(records) < 600:
        raise SystemExit("At least 600 validation texts are required.")
    if any(record.get("split") != "calibration" for record in records):
        raise SystemExit(
            "Refusing to calibrate on non-validation records. Run "
            "evaluation.build_validation_benchmark and use web_validation.jsonl."
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
            print(f"Reusing {len(scores)} compatible validation scores", flush=True)

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
        print(f"Validation score cache: {args.scores}")

    score_by_id = {item["id"]: item for item in scores}
    predictions = [{
        "id": record["id"],
        "label": 1 if record["label"] == "ai" else 0,
        "domain": record["domain"],
        "raw_score": float(score_by_id[record["id"]]["raw_score"]),
        "segments": int(score_by_id[record["id"]]["segments"]),
    } for record in records]

    try:
        payload = fit_length_aware_profile(
            predictions,
            dataset_sha256=benchmark_sha256,
            model_id=settings.local_model_id,
            model_filename=settings.local_model_filename,
            model_revision=settings.local_model_revision,
            short_target_fpr=args.short_target_fpr,
            long_target_fpr=args.long_target_fpr,
            target_fnr=args.target_fnr,
        )
        profile = CalibrationProfile.from_dict(payload)
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Length-aware calibration failed: {exc}") from exc

    report = validation_report(
        predictions, profile, dataset_sha256=benchmark_sha256
    )
    report["reused_cached_raw_scores"] = reused_scores
    write_json(args.report, report)
    if args.output.is_file():
        backup_path = args.output.with_name("calibration.previous.json")
        shutil.copy2(args.output, backup_path)
        print(f"Previous calibration backup: {backup_path}")
    write_json(args.output, payload)
    print(f"Length-aware calibration profile: {args.output}")
    print(f"Fit diagnostics: {args.report}")
    print(json.dumps(report["overall"], indent=2))
    print(json.dumps(report["by_length"], indent=2))


if __name__ == "__main__":
    main()
