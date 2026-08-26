from app.detector import (
    analyze_heuristically,
    combine_with_external,
    combine_with_local_model,
)
from app.models import DetectorOutput, Evidence, ExtractedArticle


def article(text: str, **kwargs) -> ExtractedArticle:
    return ExtractedArticle(
        title="Test",
        text=text,
        word_count=len(text.split()),
        **kwargs,
    )


def test_explicit_ai_disclosure_is_strong_signal() -> None:
    result = analyze_heuristically(
        article(
            "This article was generated with artificial intelligence and reviewed "
            "by an editor. "
            + "Useful information for readers. " * 100
        )
    )
    assert result.label == "ai_likely"
    assert result.evidence[0].kind == "strong"


def test_russian_model_leak_is_strong_signal() -> None:
    result = analyze_heuristically(
        article(
            "Как языковая модель искусственного интеллекта, я не обладаю личным "
            "опытом. "
            + "Это продолжение текста. " * 100
        )
    )
    assert result.label == "ai_likely"


def test_weak_signals_remain_uncertain() -> None:
    text = "\n".join(
        f"Paragraph {index} contains a normal sentence with a few distinct words."
        for index in range(40)
    )
    result = analyze_heuristically(article(text))
    assert result.label == "uncertain"
    assert all(len(item.message) <= 55 for item in result.evidence)


def test_external_detector_cannot_create_ai_result_by_itself() -> None:
    local = analyze_heuristically(article("A varied human paragraph. " * 120))
    result = combine_with_external(local, 0.95, 0.9)
    assert result.label == "uncertain"


def test_unanimous_local_samples_are_primary_signal() -> None:
    heuristic = analyze_heuristically(article("A varied human paragraph. " * 120))
    model = DetectorOutput(
        label="ai_likely",
        evidence=[Evidence(kind="weak", message="5 of 5 text samples were AI-like.")],
        sampled_word_count=1_000,
        segments_checked=5,
        ai_segments=5,
        non_ai_segments=0,
    )
    result = combine_with_local_model(heuristic, model)
    assert result.label == "ai_likely"
    assert result.ai_segments == 5


def test_explicit_disclosure_overrides_local_model() -> None:
    heuristic = analyze_heuristically(
        article("This article was generated with AI. " + "More text. " * 100)
    )
    model = DetectorOutput(
        label="human_likely",
        evidence=[Evidence(kind="human", message="5 of 5 samples were not AI-like.")],
        sampled_word_count=1_000,
        segments_checked=5,
        ai_segments=0,
        non_ai_segments=5,
    )
    result = combine_with_local_model(heuristic, model)
    assert result.label == "ai_likely"


def test_weak_style_matches_do_not_override_non_ai_samples() -> None:
    heuristic = analyze_heuristically(
        article("It is important to note. In conclusion. " * 200)
    )
    model = DetectorOutput(
        label="human_likely",
        evidence=[Evidence(kind="human", message="5 of 5 samples were not AI-like.")],
        sampled_word_count=1_000,
        segments_checked=5,
        ai_segments=0,
        non_ai_segments=5,
    )
    result = combine_with_local_model(heuristic, model)
    assert result.label == "human_likely"
