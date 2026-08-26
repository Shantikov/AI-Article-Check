from collections import Counter
from dataclasses import dataclass
import math
from statistics import fmean
from typing import Iterable

from app.calibration import CalibrationProfile, display_percentage


@dataclass(frozen=True)
class PlattParameters:
    slope: float
    intercept: float
    feature_mean: float
    feature_scale: float
    probability_floor: float = 0.01


def aggregate_scores(scores: Iterable[float]) -> float:
    values = [min(1.0, max(0.0, float(score))) for score in scores]
    if not values:
        raise ValueError("At least one score is required")
    return fmean(values)


def _score_feature(score: float) -> float:
    value = min(1.0 - 1e-6, max(1e-6, float(score)))
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def fit_platt(
    scores: list[float],
    labels: list[int],
    *,
    regularization: float = 2.0,
    probability_floor: float = 0.01,
    max_iterations: int = 100,
) -> PlattParameters:
    """Fit a smooth monotonic probability curve with finite-sample shrinkage."""
    if len(scores) != len(labels) or len(scores) < 2:
        raise ValueError("Scores and labels must contain at least two paired values")
    if regularization <= 0:
        raise ValueError("Regularization must be positive")
    if not 0 < probability_floor < 0.5:
        raise ValueError("Probability floor must be between 0 and 0.5")
    for label in labels:
        if label not in (0, 1):
            raise ValueError("Labels must be 0 or 1")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Both human and AI examples are required")

    features = [_score_feature(score) for score in scores]
    feature_mean = fmean(features)
    variance = fmean((value - feature_mean) ** 2 for value in features)
    feature_scale = math.sqrt(variance)
    if feature_scale < 1e-9:
        raise ValueError("Calibration scores contain no usable variation")
    normalized = [(value - feature_mean) / feature_scale for value in features]

    # Platt's finite-sample targets avoid treating a limited benchmark as
    # absolute proof. L2 shrinkage also keeps separable data from producing an
    # almost vertical 0%-to-100% step.
    positive_target = (positives + 1.0) / (positives + 2.0)
    negative_target = 1.0 / (negatives + 2.0)
    targets = [positive_target if label else negative_target for label in labels]
    target_mean = fmean(targets)
    slope = 1.0
    intercept = math.log(target_mean / (1.0 - target_mean))

    for _ in range(max_iterations):
        probabilities = [
            _sigmoid(slope * feature + intercept)
            for feature in normalized
        ]
        weights = [max(1e-9, value * (1.0 - value)) for value in probabilities]
        gradient_slope = sum(
            (probability - target) * feature
            for probability, target, feature in zip(probabilities, targets, normalized)
        ) + regularization * slope
        gradient_intercept = sum(
            probability - target
            for probability, target in zip(probabilities, targets)
        )
        hessian_slope = sum(
            weight * feature * feature
            for weight, feature in zip(weights, normalized)
        ) + regularization
        hessian_cross = sum(
            weight * feature
            for weight, feature in zip(weights, normalized)
        )
        hessian_intercept = sum(weights)
        determinant = (
            hessian_slope * hessian_intercept
            - hessian_cross * hessian_cross
        )
        if determinant <= 1e-12:
            raise ValueError("Calibration optimization became singular")
        step_slope = (
            gradient_slope * hessian_intercept
            - gradient_intercept * hessian_cross
        ) / determinant
        step_intercept = (
            gradient_intercept * hessian_slope
            - gradient_slope * hessian_cross
        ) / determinant
        slope -= step_slope
        intercept -= step_intercept
        if max(abs(step_slope), abs(step_intercept)) < 1e-8:
            break

    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0:
        raise ValueError("Calibration did not produce a positive monotonic curve")
    return PlattParameters(
        slope=slope,
        intercept=intercept,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        probability_floor=probability_floor,
    )


def profile_from_fit(
    fit: PlattParameters,
    *,
    human_max: float = 0.25,
    ai_min: float = 0.75,
    min_ai_segments: int = 3,
) -> CalibrationProfile:
    return CalibrationProfile(
        slope=fit.slope,
        intercept=fit.intercept,
        feature_mean=fit.feature_mean,
        feature_scale=fit.feature_scale,
        human_max=human_max,
        ai_min=ai_min,
        min_ai_segments=min_ai_segments,
        probability_floor=fit.probability_floor,
    )


def choose_thresholds(
    probabilities: list[float],
    labels: list[int],
    *,
    target_fpr: float = 0.05,
    target_fnr: float = 0.05,
) -> tuple[float, float]:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("Probabilities and labels must be non-empty and paired")
    humans = sum(label == 0 for label in labels)
    ai = sum(label == 1 for label in labels)
    if not humans or not ai:
        raise ValueError("Both human and AI examples are required")

    displayed = [display_percentage(value) for value in probabilities]
    candidates = list(range(1, 100))
    valid_ai = [
        threshold
        for threshold in candidates
        if sum(
            label == 0 and percentage >= threshold
            for percentage, label in zip(displayed, labels)
        ) / humans <= target_fpr
    ]
    ai_limit = min(valid_ai) if valid_ai else 99

    valid_human = [
        threshold
        for threshold in candidates
        if sum(
            label == 1 and percentage <= threshold
            for percentage, label in zip(displayed, labels)
        ) / ai <= target_fnr
    ]
    human_limit = max(valid_human) if valid_human else 1

    # Keep at least one visible percentage point between categorical results.
    # Moving the human boundary down or the AI boundary up is conservative and
    # cannot worsen the requested false-positive/false-negative constraints.
    if ai_limit <= 1:
        ai_limit = 2
    human_limit = max(1, min(human_limit, ai_limit - 1))
    return human_limit / 100, ai_limit / 100


def evaluate(
    probabilities: list[float],
    labels: list[int],
    segments: list[int],
    *,
    human_max: float,
    ai_min: float,
    min_ai_segments: int = 3,
    profile: CalibrationProfile | None = None,
) -> dict[str, float | int]:
    predictions: list[str] = []
    human_limit_percentage = max(1, math.floor(human_max * 100 + 1e-9))
    ai_limit_percentage = min(99, math.ceil(ai_min * 100 - 1e-9))
    for probability, segment_count in zip(probabilities, segments):
        if profile is not None:
            runtime_label = profile.classify(probability, segment_count)
            predictions.append({
                "human_likely": "human",
                "ai_likely": "ai",
                "uncertain": "uncertain",
            }[runtime_label])
            continue
        percentage = display_percentage(probability)
        if percentage <= human_limit_percentage:
            predictions.append("human")
        elif percentage >= ai_limit_percentage and segment_count >= min_ai_segments:
            predictions.append("ai")
        else:
            predictions.append("uncertain")

    tp = sum(prediction == "ai" and label == 1 for prediction, label in zip(predictions, labels))
    fp = sum(prediction == "ai" and label == 0 for prediction, label in zip(predictions, labels))
    tn = sum(prediction == "human" and label == 0 for prediction, label in zip(predictions, labels))
    fn = sum(prediction == "human" and label == 1 for prediction, label in zip(predictions, labels))
    uncertain = predictions.count("uncertain")
    decided = len(predictions) - uncertain
    humans = sum(label == 0 for label in labels)
    ai = sum(label == 1 for label in labels)
    return {
        "samples": len(predictions),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "uncertain": uncertain,
        "coverage": decided / len(predictions) if predictions else 0.0,
        "uncertain_rate": uncertain / len(predictions) if predictions else 0.0,
        "accuracy_with_uncertain_as_error": (tp + tn) / len(predictions) if predictions else 0.0,
        "decided_accuracy": (tp + tn) / decided if decided else 0.0,
        "ai_precision": tp / (tp + fp) if tp + fp else 0.0,
        "ai_recall": tp / ai if ai else 0.0,
        "human_recall": tn / humans if humans else 0.0,
        "balanced_accuracy": (
            ((tp / ai) + (tn / humans)) / 2
            if ai and humans
            else 0.0
        ),
        "false_positive_rate": fp / humans if humans else 0.0,
        "false_negative_rate": fn / ai if ai else 0.0,
        "brier_score": (
            fmean(
                (probability - label) ** 2
                for probability, label in zip(probabilities, labels)
            )
            if probabilities
            else 0.0
        ),
        "log_loss": (
            fmean(
                -(
                    label * math.log(min(1.0 - 1e-12, max(1e-12, probability)))
                    + (1 - label)
                    * math.log(
                        min(1.0 - 1e-12, max(1e-12, 1.0 - probability))
                    )
                )
                for probability, label in zip(probabilities, labels)
            )
            if probabilities
            else 0.0
        ),
    }


def summarize_probabilities(probabilities: list[float]) -> dict[str, float | int]:
    if not probabilities:
        return {
            "minimum": 0.0,
            "maximum": 0.0,
            "mean": 0.0,
            "distinct_display_percentages": 0,
            "endpoint_share": 0.0,
        }
    displayed = [display_percentage(value) for value in probabilities]
    return {
        "minimum": min(probabilities),
        "maximum": max(probabilities),
        "mean": fmean(probabilities),
        "distinct_display_percentages": len(set(displayed)),
        "endpoint_share": sum(value in (1, 99) for value in displayed) / len(displayed),
    }


def summarize_segments(segments: list[int], minimum: int) -> dict[str, float | int | dict]:
    if not segments:
        return {
            "minimum": 0,
            "maximum": 0,
            "mean": 0.0,
            "below_required": 0,
            "counts": {},
        }
    counts = Counter(segments)
    return {
        "minimum": min(segments),
        "maximum": max(segments),
        "mean": fmean(segments),
        "below_required": sum(value < minimum for value in segments),
        "counts": {str(key): value for key, value in sorted(counts.items())},
    }


def calibration_failures(
    metrics: dict[str, float | int],
    segments: dict[str, float | int | dict],
    *,
    target_fpr: float,
    probability_summary: dict[str, float | int] | None = None,
) -> list[str]:
    failures: list[str] = []
    if int(segments["below_required"]) > 0:
        failures.append("Some held-out documents have fewer than the required samples.")
    if float(metrics["ai_recall"]) <= 0:
        failures.append("Held-out AI recall is zero.")
    if float(metrics["coverage"]) < 0.25:
        failures.append("Held-out decision coverage is below 25%.")
    allowed_fpr = max(0.10, target_fpr * 3)
    if float(metrics["false_positive_rate"]) > allowed_fpr:
        failures.append(
            f"Held-out false-positive rate exceeds {allowed_fpr:.0%}."
        )
    if float(metrics["decided_accuracy"]) < 0.70:
        failures.append("Held-out accuracy on decided cases is below 70%.")
    if probability_summary is not None:
        if int(probability_summary["distinct_display_percentages"]) < 5:
            failures.append(
                "Calibrated probabilities collapse to fewer than five displayed values."
            )
        if float(probability_summary["endpoint_share"]) > 0.80:
            failures.append(
                "More than 80% of calibrated probabilities sit at display endpoints."
            )
    return failures
