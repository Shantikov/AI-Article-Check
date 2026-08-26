import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from app.calibration import CalibrationProfile, segment_band
from app.onnx_detector import ModelUnavailableError, sample_text_chunks

from .benchmark import dataset_digest, load_jsonl
from .build_validation_benchmark import DEFAULT_OUTPUT as DEFAULT_VALIDATION
from .build_web_benchmark import DEFAULT_OUTPUT as DEFAULT_TEST
from .calibrate_length_aware import (
    DEFAULT_SCORES as LEGACY_VALIDATION_SCORES,
    fit_length_aware_profile,
)
from .core import evaluate, summarize_probabilities, summarize_segments
from .evaluate_web import (
    DEFAULT_SCORES as LEGACY_TEST_SCORES,
    build_report,
)
from .model_candidates import (
    DEFAULT_CANDIDATE_KEYS,
    ModelCandidate,
    resolve_candidates,
)
from .score_cache import (
    score_records,
    scores_are_compatible,
    write_json,
    write_jsonl,
)


DEFAULT_REPORT = Path(__file__).parent / "reports" / "model_comparison_latest.json"
DEFAULT_DETAIL_DIR = Path(__file__).parent / "reports" / "model_comparison"
DEFAULT_SCORE_DIR = Path(__file__).parent / "data" / "model_comparison"
SPLIT_SEED = "ai-content-signal-model-selection-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pinned ONNX AI-text detectors without choosing a model on "
            "the same records used for its final external report."
        )
    )
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--detail-dir", type=Path, default=DEFAULT_DETAIL_DIR)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=list(DEFAULT_CANDIDATE_KEYS),
        help="Candidate keys. The current tmr baseline is mandatory.",
    )
    parser.add_argument("--selection-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-balanced-improvement", type=float, default=0.03)
    parser.add_argument("--maximum-fpr", type=float, default=0.03)
    parser.add_argument("--minimum-decided-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-ai-precision", type=float, default=0.95)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Ignore compatible candidate and legacy raw-score caches.",
    )
    return parser.parse_args()


def _length_band_for_record(record: dict) -> str:
    return segment_band(len(sample_text_chunks(record["text"])))


def split_validation_records(
    records: list[dict], selection_fraction: float
) -> tuple[list[dict], list[dict]]:
    """Create a deterministic label/domain/length-stratified selection holdout."""
    if not 0.15 <= selection_fraction <= 0.40:
        raise ValueError("selection_fraction must be between 0.15 and 0.40")
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[
            (record["label"], record["domain"], _length_band_for_record(record))
        ].append(record)

    fit: list[dict] = []
    selection: list[dict] = []
    for key in sorted(grouped):
        items = sorted(
            grouped[key],
            key=lambda item: hashlib.sha256(
                f"{SPLIT_SEED}:{item['id']}".encode()
            ).hexdigest(),
        )
        selection_count = min(
            len(items) - 1,
            max(1, round(len(items) * selection_fraction)),
        )
        selection.extend(items[:selection_count])
        fit.extend(items[selection_count:])
    return sorted(fit, key=lambda item: item["id"]), sorted(
        selection, key=lambda item: item["id"]
    )


def _predictions(records: list[dict], scores: list[dict]) -> list[dict]:
    score_by_id = {item["id"]: item for item in scores}
    return [{
        "id": record["id"],
        "label": 1 if record["label"] == "ai" else 0,
        "domain": record["domain"],
        "generator": record["generator"],
        "generation_mode": record["generation_mode"],
        "raw_score": float(score_by_id[record["id"]]["raw_score"]),
        "segments": int(score_by_id[record["id"]]["segments"]),
    } for record in records]


def _roc_auc(probabilities: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.0
    ranked = sorted(zip(probabilities, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            label for _, label in ranked[index:end]
        )
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def _average_precision(probabilities: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    if not positives:
        return 0.0
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(
        sorted(zip(probabilities, labels), key=lambda item: item[0], reverse=True),
        start=1,
    ):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def _quality_report(predictions: list[dict], profile: CalibrationProfile) -> dict:
    enriched = [{
        **item,
        "probability": profile.predict(
            float(item["raw_score"]), int(item["segments"])
        ),
    } for item in predictions]

    def metrics(items: list[dict]) -> dict:
        probabilities = [float(item["probability"]) for item in items]
        labels = [int(item["label"]) for item in items]
        result = evaluate(
            probabilities,
            labels,
            [int(item["segments"]) for item in items],
            human_max=profile.human_max,
            ai_min=profile.ai_min,
            min_ai_segments=profile.min_ai_segments,
            profile=profile,
        )
        result["roc_auc"] = _roc_auc(probabilities, labels)
        result["average_precision"] = _average_precision(probabilities, labels)
        return result

    by_length: dict[str, list[dict]] = defaultdict(list)
    for item in enriched:
        by_length[segment_band(int(item["segments"]))].append(item)
    probabilities = [float(item["probability"]) for item in enriched]
    return {
        "overall": metrics(enriched),
        "by_length": {
            key: metrics(items) for key, items in sorted(by_length.items())
        },
        "segment_distribution": summarize_segments(
            [int(item["segments"]) for item in enriched], 1
        ),
        "probability_distribution": summarize_probabilities(probabilities),
        "mean_raw_score": fmean(
            float(item["raw_score"]) for item in enriched
        ) if enriched else 0.0,
    }


def _qualification_failures(
    metrics: dict,
    *,
    maximum_fpr: float,
    minimum_decided_accuracy: float,
    minimum_ai_precision: float,
) -> list[str]:
    failures: list[str] = []
    if float(metrics["false_positive_rate"]) > maximum_fpr:
        failures.append(f"false-positive rate exceeds {maximum_fpr:.1%}")
    if float(metrics["decided_accuracy"]) < minimum_decided_accuracy:
        failures.append(
            f"decided accuracy is below {minimum_decided_accuracy:.1%}"
        )
    if float(metrics["ai_precision"]) < minimum_ai_precision:
        failures.append(f"AI precision is below {minimum_ai_precision:.1%}")
    if float(metrics["coverage"]) < 0.25:
        failures.append("decision coverage is below 25%")
    return failures


def choose_selection_winner(
    reports: dict[str, dict],
    *,
    minimum_balanced_improvement: float,
    maximum_fpr: float,
    minimum_decided_accuracy: float,
    minimum_ai_precision: float,
) -> tuple[str, dict[str, list[str]]]:
    baseline_metrics = reports["tmr"]["selection"]["overall"]
    failures: dict[str, list[str]] = {}
    eligible: list[str] = []
    for key, report in reports.items():
        if report.get("status") != "completed":
            failures[key] = [str(report.get("error", "candidate failed"))]
            continue
        metrics = report["selection"]["overall"]
        candidate_failures = _qualification_failures(
            metrics,
            maximum_fpr=maximum_fpr,
            minimum_decided_accuracy=minimum_decided_accuracy,
            minimum_ai_precision=minimum_ai_precision,
        )
        if key != "tmr":
            improvement = (
                float(metrics["balanced_accuracy"])
                - float(baseline_metrics["balanced_accuracy"])
            )
            if improvement < minimum_balanced_improvement:
                candidate_failures.append(
                    "balanced accuracy does not improve on TMR by at least "
                    f"{minimum_balanced_improvement:.1%}"
                )
            if float(metrics["brier_score"]) > (
                float(baseline_metrics["brier_score"]) + 0.01
            ):
                candidate_failures.append(
                    "calibrated Brier score is materially worse than TMR"
                )
        failures[key] = candidate_failures
        if key != "tmr" and not candidate_failures:
            eligible.append(key)

    if not eligible:
        return "tmr", failures
    winner = max(
        eligible,
        key=lambda key: (
            float(reports[key]["selection"]["overall"]["balanced_accuracy"]),
            -float(reports[key]["selection"]["overall"]["brier_score"]),
            float(reports[key]["selection"]["overall"]["coverage"]),
            -int(reports[key]["candidate"]["approximate_model_bytes"]),
        ),
    )
    return winner, failures


def _score_cache_paths(score_dir: Path, candidate: ModelCandidate, split: str):
    score_path = score_dir / f"{candidate.key}_{split}_scores.jsonl"
    return score_path, score_path.with_suffix(".meta.json")


def _load_score_meta(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _candidate_scores(
    records: list[dict],
    candidate: ModelCandidate,
    *,
    split: str,
    benchmark_sha256: str,
    score_dir: Path,
    cache_dir: str | None,
    rescore: bool,
) -> tuple[list[dict], dict]:
    score_path, meta_path = _score_cache_paths(score_dir, candidate, split)
    if score_path.is_file() and not rescore:
        cached = load_jsonl(score_path)
        if scores_are_compatible(
            cached,
            records,
            benchmark_sha256=benchmark_sha256,
            model_id=candidate.repo_id,
            model_filename=candidate.filename,
            model_revision=candidate.revision,
        ):
            return cached, {**_load_score_meta(meta_path), "reused_cache": True}

    legacy_path = (
        LEGACY_VALIDATION_SCORES if split == "validation" else LEGACY_TEST_SCORES
    )
    if candidate.key == "tmr" and legacy_path.is_file() and not rescore:
        legacy = load_jsonl(legacy_path)
        if scores_are_compatible(
            legacy,
            records,
            benchmark_sha256=benchmark_sha256,
            model_id=candidate.repo_id,
            model_filename=candidate.filename,
            model_revision=candidate.revision,
            accept_missing_revision=True,
        ):
            normalized = [{**item, "model_revision": candidate.revision} for item in legacy]
            write_jsonl(score_path, normalized)
            meta = {
                "reused_cache": True,
                "legacy_cache_used": True,
                "legacy_revision_assumption": (
                    "The legacy cache did not record a revision; the pinned TMR "
                    "revision was assumed because repository and filename match."
                ),
                "approximate_model_bytes": candidate.approximate_model_bytes,
            }
            write_json(meta_path, meta)
            return normalized, meta

    detector = candidate.create_detector(cache_dir)
    model_path = detector.prepare()
    started = time.perf_counter()
    scores = score_records(
        records,
        detector,
        benchmark_sha256=benchmark_sha256,
        model_id=candidate.repo_id,
        model_filename=candidate.filename,
        model_revision=candidate.revision,
    )
    elapsed = time.perf_counter() - started
    write_jsonl(score_path, scores)
    meta = {
        "reused_cache": False,
        "scoring_seconds": elapsed,
        "seconds_per_document": elapsed / len(records) if records else 0.0,
        "documents_per_second": len(records) / elapsed if elapsed else 0.0,
        "model_bytes": model_path.stat().st_size if model_path.is_file() else None,
    }
    write_json(meta_path, meta)
    return scores, meta


def _profile_for(
    records: list[dict],
    scores: list[dict],
    candidate: ModelCandidate,
) -> tuple[dict, CalibrationProfile]:
    payload = fit_length_aware_profile(
        _predictions(records, scores),
        dataset_sha256=dataset_digest(records),
        model_id=candidate.repo_id,
        model_filename=candidate.filename,
    )
    payload["model_revision"] = candidate.revision
    return payload, CalibrationProfile.from_dict(payload)


def _external_candidate_passes(
    candidate_report: dict,
    baseline_report: dict,
    *,
    minimum_balanced_improvement: float,
    maximum_fpr: float,
    minimum_decided_accuracy: float,
    minimum_ai_precision: float,
) -> list[str]:
    metrics = candidate_report["overall"]
    baseline = baseline_report["overall"]
    failures = _qualification_failures(
        metrics,
        maximum_fpr=maximum_fpr,
        minimum_decided_accuracy=minimum_decided_accuracy,
        minimum_ai_precision=minimum_ai_precision,
    )
    improvement = (
        float(metrics["balanced_accuracy"])
        - float(baseline["balanced_accuracy"])
    )
    if improvement < minimum_balanced_improvement:
        failures.append(
            "external balanced accuracy does not improve on TMR by at least "
            f"{minimum_balanced_improvement:.1%}"
        )
    if float(metrics["brier_score"]) > float(baseline["brier_score"]) + 0.01:
        failures.append("external calibrated Brier score is materially worse than TMR")
    return failures


def _validate_input_records(validation: list[dict], test: list[dict]) -> None:
    if len(validation) < 600:
        raise ValueError("At least 600 MAGE validation records are required")
    if any(record.get("split") != "calibration" for record in validation):
        raise ValueError("The validation input must contain only calibration records")
    if not test:
        raise ValueError("The external MAGE test benchmark is empty")
    if any(record.get("split") != "external_test" for record in test):
        raise ValueError("The final test input must contain only external_test records")
    if {record["id"] for record in validation} & {record["id"] for record in test}:
        raise ValueError("Validation and external test records overlap")


def main() -> None:
    args = parse_args()
    try:
        candidates = resolve_candidates(args.candidates)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for value, name in (
        (args.minimum_balanced_improvement, "minimum-balanced-improvement"),
        (args.maximum_fpr, "maximum-fpr"),
        (args.minimum_decided_accuracy, "minimum-decided-accuracy"),
        (args.minimum_ai_precision, "minimum-ai-precision"),
    ):
        if not 0 <= value <= 1:
            raise SystemExit(f"--{name} must be between 0 and 1")
    if not args.validation.is_file():
        raise SystemExit(
            f"Validation benchmark not found: {args.validation}\n"
            "Run: python -m evaluation.build_validation_benchmark"
        )
    if not args.test.is_file():
        raise SystemExit(
            f"External test benchmark not found: {args.test}\n"
            "Run: python -m evaluation.build_web_benchmark"
        )

    validation = load_jsonl(args.validation)
    test = load_jsonl(args.test)
    try:
        _validate_input_records(validation, test)
        fit_records, selection_records = split_validation_records(
            validation, args.selection_fraction
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid comparison data: {exc}") from exc
    validation_digest = dataset_digest(validation)
    test_digest = dataset_digest(test)
    fit_ids = {record["id"] for record in fit_records}
    selection_ids = {record["id"] for record in selection_records}

    candidate_reports: dict[str, dict] = {}
    validation_scores: dict[str, list[dict]] = {}
    for candidate in candidates:
        print(f"\nComparing {candidate.display_name}", flush=True)
        report = {"candidate": candidate.public_payload(), "status": "failed"}
        try:
            scores, timing = _candidate_scores(
                validation,
                candidate,
                split="validation",
                benchmark_sha256=validation_digest,
                score_dir=args.score_dir,
                cache_dir=args.cache_dir,
                rescore=args.rescore,
            )
            validation_scores[candidate.key] = scores
            fit_scores = [item for item in scores if item["id"] in fit_ids]
            selection_scores = [
                item for item in scores if item["id"] in selection_ids
            ]
            fit_payload, fit_profile = _profile_for(
                fit_records, fit_scores, candidate
            )
            report.update({
                "status": "completed",
                "scoring": timing,
                "fit_profile": fit_payload,
                "selection": _quality_report(
                    _predictions(selection_records, selection_scores), fit_profile
                ),
            })
            metrics = report["selection"]["overall"]
            print(json.dumps({
                "balanced_accuracy": metrics["balanced_accuracy"],
                "ai_recall": metrics["ai_recall"],
                "human_recall": metrics["human_recall"],
                "false_positive_rate": metrics["false_positive_rate"],
                "coverage": metrics["coverage"],
                "decided_accuracy": metrics["decided_accuracy"],
                "brier_score": metrics["brier_score"],
            }, indent=2), flush=True)
        except (OSError, ValueError, KeyError, TypeError, ModelUnavailableError) as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            print(f"Candidate failed: {report['error']}", flush=True)
            if candidate.key == "tmr":
                raise SystemExit("The mandatory TMR baseline could not be evaluated") from exc
        candidate_reports[candidate.key] = report

    winner, selection_failures = choose_selection_winner(
        candidate_reports,
        minimum_balanced_improvement=args.minimum_balanced_improvement,
        maximum_fpr=args.maximum_fpr,
        minimum_decided_accuracy=args.minimum_decided_accuracy,
        minimum_ai_precision=args.minimum_ai_precision,
    )
    print(f"\nValidation-selection winner: {winner}", flush=True)

    external_keys = ["tmr"] if winner == "tmr" else ["tmr", winner]
    external_reports: dict[str, dict] = {}
    full_profiles: dict[str, dict] = {}
    for key in external_keys:
        candidate = next(item for item in candidates if item.key == key)
        full_payload, full_profile = _profile_for(
            validation, validation_scores[key], candidate
        )
        full_profiles[key] = full_payload
        test_scores, timing = _candidate_scores(
            test,
            candidate,
            split="test",
            benchmark_sha256=test_digest,
            score_dir=args.score_dir,
            cache_dir=args.cache_dir,
            rescore=args.rescore,
        )
        predictions = _predictions(test, test_scores)
        for item in predictions:
            item["probability"] = full_profile.predict(
                float(item["raw_score"]), int(item["segments"])
            )
        external_report = build_report(
            predictions,
            full_profile,
            benchmark_sha256=test_digest,
            model_id=candidate.repo_id,
            model_filename=candidate.filename,
            reused_scores=bool(timing.get("reused_cache")),
        )
        external_report["model_revision"] = candidate.revision
        external_report["scoring"] = timing
        report_path = args.detail_dir / f"{key}_external.json"
        write_json(report_path, external_report)
        external_reports[key] = external_report

    external_failures: list[str] = []
    if winner != "tmr":
        external_failures = _external_candidate_passes(
            external_reports[winner],
            external_reports["tmr"],
            minimum_balanced_improvement=args.minimum_balanced_improvement,
            maximum_fpr=args.maximum_fpr,
            minimum_decided_accuracy=args.minimum_decided_accuracy,
            minimum_ai_precision=args.minimum_ai_precision,
        )
    recommended = winner if winner != "tmr" and not external_failures else "tmr"
    recommendation = (
        f"replace TMR with {recommended}"
        if recommended != "tmr"
        else "keep the current TMR model"
    )
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "Candidates were calibrated on a stratified fit portion of MAGE "
            "validation and ranked on a disjoint validation holdout. Only the "
            "holdout winner and TMR were then refit on all validation records "
            "and scored on the independent MAGE test set."
        ),
        "selection_seed": SPLIT_SEED,
        "validation_dataset_sha256": validation_digest,
        "external_test_dataset_sha256": test_digest,
        "counts": {
            "validation_total": len(validation),
            "candidate_fit": len(fit_records),
            "candidate_selection": len(selection_records),
            "external_test": len(test),
        },
        "acceptance_rules": {
            "minimum_balanced_accuracy_improvement": (
                args.minimum_balanced_improvement
            ),
            "maximum_false_positive_rate": args.maximum_fpr,
            "minimum_decided_accuracy": args.minimum_decided_accuracy,
            "minimum_ai_precision": args.minimum_ai_precision,
            "maximum_brier_regression": 0.01,
        },
        "candidate_reports": candidate_reports,
        "selection_failures": selection_failures,
        "selection_winner": winner,
        "full_calibration_profiles": full_profiles,
        "external_reports": external_reports,
        "external_winner_failures": external_failures,
        "recommended_model": recommended,
        "recommendation": recommendation,
        "automatic_runtime_change": False,
        "note": (
            "This command never changes app/config.py or calibration.json. "
            "A replacement is activated only after reviewing this report."
        ),
    }
    write_json(args.report, payload)
    print(f"\nModel comparison report: {args.report}")
    print(json.dumps({
        "selection_winner": winner,
        "recommended_model": recommended,
        "recommendation": recommendation,
        "external_winner_failures": external_failures,
    }, indent=2))


if __name__ == "__main__":
    main()
