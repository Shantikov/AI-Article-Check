from app.calibration import CalibrationProfile
from evaluation.evaluate_web import build_report
from evaluation.score_cache import scores_are_compatible
from evaluation.web_benchmark import build_web_records, parse_source


DOMAINS = ("cmv", "xsum")


def source_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for domain in DOMAINS:
        for index in range(8):
            rows.append({
                "text": " ".join(
                    f"human-{domain}-{index}-{word}" for word in range(90)
                ),
                "label": "1",
                "src": f"{domain}_human",
            })
        for generator in ("gpt_j", "gpt-3.5-trubo", "13B"):
            for index in range(5):
                rows.append({
                    "text": " ".join(
                        f"ai-{domain}-{generator}-{index}-{word}"
                        for word in range(90)
                    ),
                    "label": "0",
                    "src": f"{domain}_machine_continuation_{generator}",
                })
    return rows


def profile() -> CalibrationProfile:
    return CalibrationProfile(
        slope=1.0,
        intercept=0.0,
        feature_mean=0.0,
        feature_scale=1.0,
        human_max=0.25,
        ai_min=0.75,
        min_ai_segments=3,
        dataset="HC3 test",
    )


def test_source_parser_normalizes_models_and_modes() -> None:
    assert parse_source("xsum_human", "human") == ("xsum", "human", "human")
    assert parse_source(
        "xsum_machine_topical_gpt-3.5-trubo", "ai"
    ) == ("xsum", "gpt-3.5-turbo", "topical")
    assert parse_source(
        "cmv_machine_continuation_13B", "ai"
    ) == ("cmv", "llama_13b", "continuation")


def test_web_benchmark_is_balanced_diverse_and_deterministic() -> None:
    first = build_web_records(
        source_rows(), domains=DOMAINS, records_per_domain=8, min_words=80
    )
    second = build_web_records(
        reversed(source_rows()), domains=DOMAINS, records_per_domain=8, min_words=80
    )

    assert first == second
    assert len(first) == 16
    assert len({item["text"] for item in first}) == 16
    for domain in DOMAINS:
        domain_records = [item for item in first if item["domain"] == domain]
        assert sum(item["label"] == "human" for item in domain_records) == 4
        assert sum(item["label"] == "ai" for item in domain_records) == 4
        assert len({
            item["generator"]
            for item in domain_records
            if item["label"] == "ai"
        }) == 3
    assert all(item["split"] == "external_test" for item in first)


def test_compatible_score_cache_requires_same_model_dataset_and_ids() -> None:
    records = [{"id": "a"}, {"id": "b"}]
    scores = [
        {
            "id": record["id"],
            "benchmark_sha256": "digest",
            "model_id": "model",
            "model_filename": "model.onnx",
            "raw_score": 0.5,
            "segments": 3,
        }
        for record in records
    ]

    assert scores_are_compatible(
        scores,
        records,
        benchmark_sha256="digest",
        model_id="model",
        model_filename="model.onnx",
    )
    scores[0]["model_id"] = "other"
    assert not scores_are_compatible(
        scores,
        records,
        benchmark_sha256="digest",
        model_id="model",
        model_filename="model.onnx",
    )


def test_external_report_is_grouped_and_never_changes_calibration() -> None:
    predictions = [
        {
            "id": "human",
            "label": 0,
            "domain": "xsum",
            "generator": "human",
            "generation_mode": "human",
            "segments": 3,
            "probability": 0.1,
        },
        {
            "id": "ai",
            "label": 1,
            "domain": "xsum",
            "generator": "gpt_j",
            "generation_mode": "continuation",
            "segments": 3,
            "probability": 0.9,
        },
    ]

    report = build_report(
        predictions,
        profile(),
        benchmark_sha256="digest",
        model_id="model",
        model_filename="model.onnx",
        reused_scores=False,
    )

    assert report["evaluation_only"] is True
    assert report["changed_calibration"] is False
    assert report["overall"]["balanced_accuracy"] == 1.0
    assert report["by_domain"]["xsum"]["samples"] == 2
    assert report["ai_by_generator"]["gpt_j"]["ai_recall"] == 1.0
    assert report["error_examples"]["highest_scoring_false_positives"] == []
