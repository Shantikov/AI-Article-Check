from evaluation.compare_models import (
    choose_selection_winner,
    split_validation_records,
)
from evaluation.model_candidates import resolve_candidates


def _records() -> list[dict]:
    records: list[dict] = []
    for domain in ("cmv", "xsum"):
        for label in ("human", "ai"):
            for index in range(60):
                word_count = 100 if index < 20 else 300 if index < 40 else 700
                records.append({
                    "id": f"{domain}-{label}-{index}",
                    "label": label,
                    "domain": domain,
                    "text": "word " * word_count,
                })
    return records


def _candidate_report(
    balanced: float,
    *,
    fpr: float = 0.01,
    decided: float = 0.96,
    precision: float = 0.98,
    brier: float = 0.12,
    coverage: float = 0.70,
) -> dict:
    return {
        "status": "completed",
        "candidate": {"approximate_model_bytes": 100},
        "selection": {"overall": {
            "balanced_accuracy": balanced,
            "false_positive_rate": fpr,
            "decided_accuracy": decided,
            "ai_precision": precision,
            "brier_score": brier,
            "coverage": coverage,
        }},
    }


def test_candidate_registry_is_pinned_and_requires_tmr() -> None:
    candidates = resolve_candidates(["tmr", "glyph"])
    assert [candidate.key for candidate in candidates] == ["tmr", "glyph"]
    assert all(len(candidate.revision) == 40 for candidate in candidates)

    try:
        resolve_candidates(["glyph"])
    except ValueError as exc:
        assert "TMR baseline" in str(exc)
    else:
        raise AssertionError("TMR must be mandatory")


def test_validation_selection_split_is_deterministic_stratified_and_disjoint() -> None:
    first_fit, first_selection = split_validation_records(_records(), 0.25)
    second_fit, second_selection = split_validation_records(
        list(reversed(_records())), 0.25
    )

    assert [item["id"] for item in first_fit] == [item["id"] for item in second_fit]
    assert [item["id"] for item in first_selection] == [
        item["id"] for item in second_selection
    ]
    fit_ids = {item["id"] for item in first_fit}
    selection_ids = {item["id"] for item in first_selection}
    assert not fit_ids & selection_ids
    assert len(fit_ids | selection_ids) == len(_records())
    assert 0.22 < len(selection_ids) / len(_records()) < 0.28


def test_winner_requires_meaningful_safe_improvement() -> None:
    reports = {
        "tmr": _candidate_report(0.55, brier=0.15),
        "glyph": _candidate_report(0.72, brier=0.10),
        "fakespot": _candidate_report(0.74, fpr=0.08, brier=0.09),
    }
    winner, failures = choose_selection_winner(
        reports,
        minimum_balanced_improvement=0.03,
        maximum_fpr=0.03,
        minimum_decided_accuracy=0.90,
        minimum_ai_precision=0.95,
    )

    assert winner == "glyph"
    assert not failures["glyph"]
    assert any("false-positive" in failure for failure in failures["fakespot"])


def test_winner_falls_back_to_tmr_for_tiny_gain() -> None:
    reports = {
        "tmr": _candidate_report(0.55),
        "glyph": _candidate_report(0.57),
    }
    winner, failures = choose_selection_winner(
        reports,
        minimum_balanced_improvement=0.03,
        maximum_fpr=0.03,
        minimum_decided_accuracy=0.90,
        minimum_ai_precision=0.95,
    )

    assert winner == "tmr"
    assert any("does not improve" in failure for failure in failures["glyph"])
