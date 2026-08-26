import re
import threading
from math import ceil
from pathlib import Path
from statistics import fmean
from typing import Any

from .calibration import load_calibration
from .models import DetectorOutput, Evidence


_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
_COMMON_ENGLISH_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


class ModelUnavailableError(RuntimeError):
    """Raised when the local ONNX model cannot be loaded or executed."""


def is_likely_english(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if len(letters) < 120:
        return False

    latin_letters = sum(
        1 for character in letters if ("a" <= character.lower() <= "z")
    )
    if latin_letters / len(letters) < 0.9:
        return False

    words = [word.casefold() for word in _ENGLISH_WORD_RE.findall(text[:30_000])]
    if len(words) < 40:
        return False

    common_hits = sum(word in _COMMON_ENGLISH_WORDS for word in words)
    distinct_hits = len(set(words) & _COMMON_ENGLISH_WORDS)
    return distinct_hits >= 4 and common_hits >= max(5, len(words) // 80)


def sample_text_chunks(
    text: str,
    words_per_chunk: int = 220,
    max_chunks: int = 7,
) -> list[str]:
    """Return non-overlapping samples spread across the entire article."""
    words = text.split()
    if not words:
        return []
    if len(words) <= words_per_chunk or max_chunks <= 1:
        return [" ".join(words)]

    chunk_count = min(max_chunks, max(2, ceil(len(words) / words_per_chunk)))
    chunk_width = min(words_per_chunk, len(words) // chunk_count)
    available_gap = len(words) - (chunk_width * chunk_count)
    chunks: list[str] = []
    for index in range(chunk_count):
        gap_before = round(available_gap * index / max(1, chunk_count - 1))
        start = index * chunk_width + gap_before
        chunk = " ".join(words[start : start + chunk_width])
        if chunk:
            chunks.append(chunk)
    return chunks


def _sample_position(index: int, total: int) -> str:
    if total <= 1:
        return "the available text"
    relative_position = index / (total - 1)
    if relative_position < 1 / 3:
        return "the beginning"
    if relative_position > 2 / 3:
        return "the end"
    return "the middle"


def _evidence_excerpt(text: str, max_chars: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    shortened = compact[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{shortened}…"


class LocalOnnxDetector:
    """Lazy local inference for compatible English ONNX text classifiers."""

    def __init__(
        self,
        repo_id: str,
        filename: str,
        cache_dir: str | None = None,
        calibration_path: str | None = None,
        *,
        revision: str | None = None,
        tokenizer_repo_id: str | None = None,
        tokenizer_revision: str | None = None,
        tokenizer_use_fast: bool | None = None,
        ai_label_index: int = 1,
        output_kind: str = "softmax_logits",
    ) -> None:
        if ai_label_index < 0:
            raise ValueError("ai_label_index must be non-negative")
        if output_kind not in {"softmax_logits", "sigmoid_logit", "probability"}:
            raise ValueError("Unsupported ONNX detector output kind")
        self.repo_id = repo_id
        self.filename = filename
        self.cache_dir = cache_dir
        self.revision = revision
        self.tokenizer_repo_id = tokenizer_repo_id or repo_id
        self.tokenizer_revision = tokenizer_revision or revision
        self.tokenizer_use_fast = tokenizer_use_fast
        self.ai_label_index = ai_label_index
        self.output_kind = output_kind
        self.calibration = load_calibration(calibration_path)
        self._tokenizer: Any = None
        self._session: Any = None
        self._model_path: Path | None = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._tokenizer is not None and self._session is not None

    def prepare(self) -> Path:
        self._ensure_loaded()
        return self._model_path or Path(self.filename)

    def analyze(self, text: str) -> DetectorOutput:
        chunks = sample_text_chunks(text)
        scores = self.score_chunks(chunks)
        segments_checked = len(scores)
        ai_segments = sum(score > 0.5 for score in scores)
        non_ai_segments = segments_checked - ai_segments
        ai_probability: float | None = None
        if self.calibration:
            ai_probability = self.calibration.predict(
                fmean(scores), segments_checked
            )
            label = self.calibration.classify(ai_probability, segments_checked)
        elif segments_checked >= 3 and ai_segments == segments_checked:
            label = "ai_likely"
        elif non_ai_segments == segments_checked:
            label = "human_likely"
        else:
            label = "uncertain"

        evidence: list[Evidence] = []
        if ai_segments:
            evidence.append(Evidence(
                kind="weak",
                message=(
                    f"{ai_segments} of {segments_checked} text samples "
                    "were AI-like."
                ),
            ))
        if non_ai_segments:
            evidence.append(Evidence(
                kind="human",
                message=(
                    f"{non_ai_segments} of {segments_checked} text samples "
                    "were not AI-like."
                ),
            ))
        if ai_segments:
            strongest_ai_index = max(range(segments_checked), key=scores.__getitem__)
            strongest_ai_position = _sample_position(
                strongest_ai_index,
                segments_checked,
            )
            evidence.append(Evidence(
                kind="weak",
                message=f"Strongest AI-like sample: {strongest_ai_position}.",
                detail=(
                    "This passage received the highest raw AI-pattern score among "
                    f"the {segments_checked} independently checked samples. The raw "
                    "score is used for comparison here, not presented as proof of "
                    "authorship."
                ),
                excerpt=_evidence_excerpt(chunks[strongest_ai_index]),
            ))
        if non_ai_segments:
            strongest_human_index = min(range(segments_checked), key=scores.__getitem__)
            strongest_human_position = _sample_position(
                strongest_human_index,
                segments_checked,
            )
            evidence.append(Evidence(
                kind="human",
                message=(
                    "Strongest human-like sample: "
                    f"{strongest_human_position}."
                ),
                detail=(
                    "This passage received the lowest raw AI-pattern score among "
                    f"the {segments_checked} independently checked samples. It is "
                    "the clearest counter-signal found by the model, not proof of "
                    "human authorship."
                ),
                excerpt=_evidence_excerpt(chunks[strongest_human_index]),
            ))
        if self.calibration and label == "uncertain":
            if self.calibration.needs_more_samples(ai_probability, segments_checked):
                evidence.append(Evidence(
                    kind="info",
                    message=(
                        f"At least {self.calibration.min_ai_segments} independently "
                        "checked samples are required for a strong AI result."
                    ),
                ))
            else:
                evidence.append(Evidence(
                    kind="info",
                    message=(
                        "The calibrated score stayed between the decision thresholds "
                        "for this amount of text."
                    ),
                ))
        elif (
            (not self.calibration or not self.calibration.bands)
            and segments_checked < 3
            and ai_segments == segments_checked
        ):
            evidence.append(Evidence(
                kind="info",
                message="At least 3 agreeing samples are required for a strong AI signal.",
            ))
        elif ai_segments and non_ai_segments:
            evidence.append(Evidence(
                kind="info",
                message="The independently checked text samples disagreed.",
            ))

        return DetectorOutput(
            label=label,
            evidence=evidence,
            sampled_word_count=sum(len(chunk.split()) for chunk in chunks),
            segments_checked=segments_checked,
            ai_segments=ai_segments,
            non_ai_segments=non_ai_segments,
            ai_probability=ai_probability,
        )

    def score_text(self, text: str) -> list[float]:
        return self.score_chunks(sample_text_chunks(text))

    def score_chunks(self, chunks: list[str]) -> list[float]:
        self._ensure_loaded()
        if not chunks:
            raise ModelUnavailableError("No text was available for model inference.")
        try:
            import numpy as np

            encoded = self._tokenizer(
                chunks,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            input_names = {item.name for item in self._session.get_inputs()}
            inputs = {
                name: np.asarray(value, dtype=np.int64)
                for name, value in encoded.items()
                if name in input_names
            }
            output = np.asarray(self._session.run(None, inputs)[0], dtype=np.float64)
            if self.output_kind == "softmax_logits":
                if output.ndim != 2 or output.shape[1] <= self.ai_label_index:
                    raise ValueError("The ONNX model returned an unexpected logit shape")
                output -= output.max(axis=1, keepdims=True)
                probabilities = np.exp(output)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                ai_probabilities = probabilities[:, self.ai_label_index]
            else:
                values = output.reshape(-1)
                if len(values) != len(chunks):
                    raise ValueError("The ONNX model returned an unexpected scalar shape")
                if self.output_kind == "sigmoid_logit":
                    ai_probabilities = 1.0 / (
                        1.0 + np.exp(-np.clip(values, -60.0, 60.0))
                    )
                else:
                    ai_probabilities = np.clip(values, 0.0, 1.0)
            return [float(value) for value in ai_probabilities]
        except Exception as exc:
            raise ModelUnavailableError("Local ONNX inference failed.") from exc

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from transformers import AutoTokenizer

                model_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    cache_dir=self.cache_dir,
                    revision=self.revision,
                )
                tokenizer_options: dict[str, Any] = {
                    "cache_dir": self.cache_dir,
                    "revision": self.tokenizer_revision,
                }
                if self.tokenizer_use_fast is not None:
                    tokenizer_options["use_fast"] = self.tokenizer_use_fast
                tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_repo_id,
                    **tokenizer_options,
                )
                session = ort.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                raise ModelUnavailableError(
                    "The local detector model could not be loaded."
                ) from exc

            self._tokenizer = tokenizer
            self._session = session
            self._model_path = Path(model_path)
