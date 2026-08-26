import json

import numpy as np

from app.calibration import CalibrationProfile, load_calibration
from app.onnx_detector import LocalOnnxDetector
from evaluation.core import (
    calibration_failures,
    choose_thresholds,
    evaluate,
    fit_platt,
    profile_from_fit,
    summarize_probabilities,
    summarize_segments,
)
from evaluation.calibrate_length_aware import fit_length_aware_profile


def make_profile(**overrides) -> CalibrationProfile:
    values = {
        "slope": 1.0,
        "intercept": 0.0,
        "feature_mean": 0.0,
        "feature_scale": 1.0,
        "human_max": 0.2,
        "ai_min": 0.8,
    }
    values.update(overrides)
    return CalibrationProfile(**values)


def test_profile_is_smooth_monotonic_and_classifies() -> None:
    profile = make_profile()

    assert 0.29 < profile.predict(0.3) < 0.31
    assert 0.49 < profile.predict(0.5) < 0.51
    assert 0.69 < profile.predict(0.7) < 0.71
    assert profile.classify(0.1, 1) == "human_likely"
    assert profile.classify(0.9, 2) == "uncertain"
    assert profile.classify(0.9, 3) == "ai_likely"


def test_profile_does_not_claim_absolute_certainty() -> None:
    profile = make_profile()

    assert profile.predict(0.0) == 0.01
    assert profile.predict(1.0) == 0.99
    assert profile.display_percentage(0.0) == 1
    assert profile.display_percentage(1.0) == 99


def test_visible_percentage_and_decision_use_same_rounding() -> None:
    profile = make_profile(ai_min=0.99)

    assert profile.display_percentage(0.986) == 99
    assert profile.classify(0.986, 3) == "ai_likely"
    assert profile.classify(0.986, 2) == "uncertain"
    assert profile.needs_more_samples(0.986, 2) is True

    strict_profile = make_profile(ai_min=1.0)
    assert strict_profile.classify(0.986, 3) == "ai_likely"


def test_invalid_or_legacy_profile_is_not_loaded(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"schema_version": 2, "points": []}), encoding="utf-8")

    assert load_calibration(path) is None


def test_length_aware_profile_uses_a_separate_short_text_curve() -> None:
    payload = {
        "schema_version": 4,
        "dataset": "test length-aware",
        "bands": {},
    }
    for name, intercept, ai_min in (
        ("1_segment", 2.0, 0.8),
        ("2_segments", 0.5, 0.75),
        ("3_or_more_segments", -1.0, 0.7),
    ):
        payload["bands"][name] = {
            "calibrator": {
                "kind": "regularized_platt",
                "slope": 1.0,
                "intercept": intercept,
                "feature_mean": 0.0,
                "feature_scale": 1.0,
                "probability_floor": 0.01,
            },
            "thresholds": {"human_max": 0.2, "ai_min": ai_min},
        }

    profile = CalibrationProfile.from_dict(payload)
    short_probability = profile.predict(0.7, 1)
    long_probability = profile.predict(0.7, 3)

    assert short_probability > long_probability
    assert profile.classify(short_probability, 1) == "ai_likely"
    assert profile.needs_more_samples(short_probability, 1) is False


def test_length_aware_fit_produces_all_three_calibration_bands() -> None:
    predictions = []
    for segments in (1, 2, 3):
        for index in range(35):
            predictions.append({
                "id": f"h-{segments}-{index}",
                "label": 0,
                "segments": segments,
                "raw_score": 0.05 + index * 0.005,
            })
            predictions.append({
                "id": f"a-{segments}-{index}",
                "label": 1,
                "segments": segments,
                "raw_score": 0.75 + index * 0.005,
            })

    payload = fit_length_aware_profile(
        predictions,
        dataset_sha256="digest",
        model_id="model",
        model_filename="model.onnx",
    )
    profile = CalibrationProfile.from_dict(payload)

    assert payload["schema_version"] == 4
    assert set(payload["bands"]) == {
        "1_segment",
        "2_segments",
        "3_or_more_segments",
    }
    assert profile.classify(profile.predict(0.9, 1), 1) == "ai_likely"


def test_regularized_platt_fit_is_smooth_and_monotonic() -> None:
    scores = [0.05, 0.10, 0.18, 0.28, 0.42, 0.55, 0.68, 0.78, 0.88, 0.96]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    profile = profile_from_fit(fit_platt(scores, labels))
    probabilities = [profile.predict(score) for score in scores]

    assert all(
        probabilities[index] < probabilities[index + 1]
        for index in range(len(probabilities) - 1)
    )
    assert len({profile.display_percentage(value) for value in probabilities}) >= 5
    assert all(0.0 < value < 1.0 for value in probabilities)


def test_thresholds_respect_target_false_positive_rate() -> None:
    probabilities = [0.02, 0.08, 0.14, 0.9, 0.95, 0.98]
    labels = [0, 0, 0, 1, 1, 1]
    human_max, ai_min = choose_thresholds(
        probabilities,
        labels,
        target_fpr=0,
        target_fnr=0,
    )
    metrics = evaluate(
        probabilities,
        labels,
        [3] * 6,
        human_max=human_max,
        ai_min=ai_min,
    )
    assert metrics["false_positive_rate"] == 0
    assert metrics["false_negative_rate"] == 0


def test_detector_uses_calibrated_probability(tmp_path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({
        "schema_version": 3,
        "dataset": "test",
        "calibrator": {
            "kind": "regularized_platt",
            "slope": 1.0,
            "intercept": 0.0,
            "feature_mean": 0.0,
            "feature_scale": 1.0,
            "probability_floor": 0.01,
        },
        "thresholds": {"human_max": 0.2, "ai_min": 0.8, "min_ai_segments": 3},
    }), encoding="utf-8")

    class FakeTokenizer:
        def __call__(self, chunks, **_kwargs):
            return {
                "input_ids": np.ones((len(chunks), 4)),
                "attention_mask": np.ones((len(chunks), 4)),
            }

    class FakeInput:
        def __init__(self, name):
            self.name = name

    class FakeSession:
        def get_inputs(self):
            return [FakeInput("input_ids"), FakeInput("attention_mask")]

        def run(self, _outputs, inputs):
            size = inputs["input_ids"].shape[0]
            return [np.tile(np.array([[0.0, 3.0]]), (size, 1))]

    detector = LocalOnnxDetector(
        "test/repo",
        "model.onnx",
        calibration_path=str(calibration_path),
    )
    detector._tokenizer = FakeTokenizer()
    detector._session = FakeSession()
    result = detector.analyze("word " * 1_000)

    assert result.label == "ai_likely"
    assert result.ai_probability is not None
    assert 0.9 < result.ai_probability < 1.0


def test_zero_recall_profile_is_rejected() -> None:
    metrics = {
        "ai_recall": 0.0,
        "coverage": 0.5,
        "false_positive_rate": 0.0,
        "decided_accuracy": 0.95,
    }
    segments = summarize_segments([3, 4, 5], minimum=3)

    failures = calibration_failures(metrics, segments, target_fpr=0.05)

    assert "Held-out AI recall is zero." in failures


def test_collapsed_probability_scale_is_rejected() -> None:
    metrics = {
        "ai_recall": 0.9,
        "coverage": 0.9,
        "false_positive_rate": 0.0,
        "decided_accuracy": 0.95,
    }
    segments = summarize_segments([3] * 10, minimum=3)
    probability_summary = summarize_probabilities([0.01] * 5 + [0.99] * 5)

    failures = calibration_failures(
        metrics,
        segments,
        target_fpr=0.05,
        probability_summary=probability_summary,
    )

    assert any("fewer than five" in failure for failure in failures)
    assert any("display endpoints" in failure for failure in failures)


def test_segment_summary_detects_short_documents() -> None:
    summary = summarize_segments([1, 3, 3, 4], minimum=3)

    assert summary["below_required"] == 1
    assert summary["counts"] == {"1": 1, "3": 2, "4": 1}
