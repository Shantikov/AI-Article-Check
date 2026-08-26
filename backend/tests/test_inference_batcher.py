import asyncio

import pytest

from app.inference_batcher import InferenceBatcher
from app.onnx_detector import LocalOnnxDetector


def detector_with_fake_scores(calls: list[list[str]]) -> LocalOnnxDetector:
    detector = LocalOnnxDetector("test/repo", "model.onnx")

    def fake_score_chunks(chunks: list[str]) -> list[float]:
        calls.append(chunks)
        return [0.9 if chunk.startswith("alpha") else 0.1 for chunk in chunks]

    detector.score_chunks = fake_score_chunks  # type: ignore[method-assign]
    return detector


@pytest.mark.asyncio
async def test_nearby_articles_share_one_onnx_micro_batch() -> None:
    calls: list[list[str]] = []
    batcher = InferenceBatcher(
        detector_with_fake_scores(calls),
        max_batch_chunks=14,
        wait_ms=5,
    )

    ai_result, human_result = await asyncio.gather(
        batcher.analyze("alpha " * 1_540),
        batcher.analyze("beta " * 1_540),
    )

    assert [len(call) for call in calls] == [14]
    assert ai_result.segments_checked == 7
    assert ai_result.ai_segments == 7
    assert ai_result.label == "ai_likely"
    assert human_result.segments_checked == 7
    assert human_result.non_ai_segments == 7
    assert human_result.label == "human_likely"


@pytest.mark.asyncio
async def test_micro_batch_limit_does_not_drop_article_samples() -> None:
    calls: list[list[str]] = []
    batcher = InferenceBatcher(
        detector_with_fake_scores(calls),
        max_batch_chunks=7,
        wait_ms=0,
    )

    first, second = await asyncio.gather(
        batcher.analyze("alpha " * 1_540),
        batcher.analyze("beta " * 1_540),
    )

    assert [len(call) for call in calls] == [7, 7]
    assert first.segments_checked + second.segments_checked == 14


@pytest.mark.parametrize(
    ("batch_size", "wait_ms"),
    [(0, 40), (14, -1)],
)
def test_invalid_batch_settings_are_rejected(batch_size: int, wait_ms: int) -> None:
    with pytest.raises(ValueError):
        InferenceBatcher(
            LocalOnnxDetector("test/repo", "model.onnx"),
            max_batch_chunks=batch_size,
            wait_ms=wait_ms,
        )
