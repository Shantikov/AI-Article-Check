import numpy as np

from app.onnx_detector import LocalOnnxDetector, is_likely_english, sample_text_chunks


def test_recognizes_ordinary_english_article() -> None:
    text = (
        "The report is based on interviews with people who live in the city. "
        "It explains how the project was designed and why the council approved it. "
        "The authors also describe the evidence that was available at the time. "
    ) * 20
    assert is_likely_english(text) is True


def test_rejects_non_english_article() -> None:
    text = (
        "Это обычная статья на русском языке, в которой автор описывает события "
        "и приводит подробные объяснения для читателей. "
    ) * 30
    assert is_likely_english(text) is False


def test_samples_start_middle_and_end_of_long_article() -> None:
    text = " ".join(f"word-{index}" for index in range(1_000))
    chunks = sample_text_chunks(text, words_per_chunk=100)
    assert len(chunks) == 7
    assert chunks[0].startswith("word-0 ")
    assert "word-450" in chunks[3]
    assert chunks[6].endswith("word-999")


def test_short_article_uses_one_chunk() -> None:
    assert sample_text_chunks("one two three", words_per_chunk=10) == ["one two three"]


def test_detector_counts_ai_votes_from_second_logit() -> None:
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

    detector = LocalOnnxDetector("test/repo", "model.onnx")
    detector._tokenizer = FakeTokenizer()
    detector._session = FakeSession()

    result = detector.analyze("word " * 1_000)
    assert result.label == "ai_likely"
    assert result.segments_checked == 5
    assert result.ai_segments == 5
    assert result.non_ai_segments == 0
    assert result.evidence[0].kind == "weak"
    assert result.evidence[0].message == "5 of 5 text samples were AI-like."
    assert result.ai_probability is None
    assert "confidence" not in result.model_dump()


def test_mixed_sample_votes_remain_uncertain() -> None:
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
            rows = [[0.0, 3.0] if index % 2 == 0 else [3.0, 0.0] for index in range(size)]
            return [np.asarray(rows)]

    detector = LocalOnnxDetector("test/repo", "model.onnx")
    detector._tokenizer = FakeTokenizer()
    detector._session = FakeSession()

    result = detector.analyze("word " * 1_000)
    assert result.label == "uncertain"
    assert result.ai_segments == 3
    assert result.non_ai_segments == 2


def test_one_ai_like_sample_is_not_enough_for_strong_result() -> None:
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

        def run(self, _outputs, _inputs):
            return [np.asarray([[0.0, 3.0]])]

    detector = LocalOnnxDetector("test/repo", "model.onnx")
    detector._tokenizer = FakeTokenizer()
    detector._session = FakeSession()

    result = detector.analyze("word " * 100)
    assert result.label == "uncertain"
    assert result.segments_checked == 1


def test_detector_supports_single_sigmoid_logit_models() -> None:
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
            return [np.full((inputs["input_ids"].shape[0], 1), 2.0)]

    detector = LocalOnnxDetector(
        "test/repo",
        "model.onnx",
        output_kind="sigmoid_logit",
    )
    detector._tokenizer = FakeTokenizer()
    detector._session = FakeSession()

    scores = detector.score_chunks(["first", "second"])
    assert len(scores) == 2
    assert all(0.88 < score < 0.89 for score in scores)
