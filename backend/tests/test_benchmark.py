from evaluation.benchmark import build_records, dataset_digest
from app.onnx_detector import sample_text_chunks


def long_answer(prefix: str) -> str:
    return " ".join(f"{prefix}-{index}" for index in range(90))


def test_benchmark_is_balanced_paired_and_deterministic() -> None:
    rows = {
        "domain_a": [
            {
                "id": str(index),
                "question": f"Question {index}",
                "human_answers": [long_answer(f"human-a-{index}")],
                "chatgpt_answers": [long_answer(f"ai-a-{index}")],
            }
            for index in range(4)
        ],
        "domain_b": [
            {
                "id": str(index),
                "question": f"Other {index}",
                "human_answers": [long_answer(f"human-b-{index}")],
                "chatgpt_answers": [long_answer(f"ai-b-{index}")],
            }
            for index in range(4)
        ],
    }
    first = build_records(
        rows,
        pairs_per_domain=2,
        min_words=80,
        min_document_words=80,
    )
    second = build_records(
        rows,
        pairs_per_domain=2,
        min_words=80,
        min_document_words=80,
    )

    assert first == second
    assert dataset_digest(first) == dataset_digest(second)
    assert len(first) == 8
    assert sum(item["label"] == "human" for item in first) == 4
    assert sum(item["label"] == "ai" for item in first) == 4
    for pair_id in {item["pair_id"] for item in first}:
        pair = [item for item in first if item["pair_id"] == pair_id]
        assert len({item["split"] for item in pair}) == 1


def test_short_domain_is_redistributed_without_shrinking_benchmark() -> None:
    rows = {
        "small": [
            {
                "id": "only",
                "question": "Only pair",
                "human_answers": [long_answer("small-human")],
                "chatgpt_answers": [long_answer("small-ai")],
            }
        ],
        "large": [
            {
                "id": str(index),
                "question": f"Large {index}",
                "human_answers": [long_answer(f"large-human-{index}")],
                "chatgpt_answers": [long_answer(f"large-ai-{index}")],
            }
            for index in range(5)
        ],
    }

    records = build_records(
        rows,
        pairs_per_domain=2,
        min_words=80,
        min_document_words=80,
    )

    assert len(records) == 8
    assert sum(
        item["domain"] == "small" and item["label"] == "human"
        for item in records
    ) == 1
    assert sum(
        item["domain"] == "large" and item["label"] == "human"
        for item in records
    ) == 3


def test_short_answers_are_composed_into_three_sample_documents() -> None:
    rows = {
        "domain": [
            {
                "id": str(index),
                "question": f"Question {index}",
                "human_answers": [long_answer(f"human-{index}")],
                "chatgpt_answers": [long_answer(f"ai-{index}")],
            }
            for index in range(16)
        ]
    }

    records = build_records(
        rows,
        pairs_per_domain=2,
        min_words=80,
        min_document_words=660,
        max_document_words=1_100,
    )

    assert len(records) == 4
    assert all(len(item["text"].split()) >= 660 for item in records)
    assert all(len(sample_text_chunks(item["text"])) >= 3 for item in records)
