from dataclasses import dataclass

from app.onnx_detector import LocalOnnxDetector


@dataclass(frozen=True)
class ModelCandidate:
    """A reproducible ONNX detector candidate used only by evaluation tools."""

    key: str
    display_name: str
    repo_id: str
    filename: str
    revision: str
    tokenizer_use_fast: bool
    ai_label_index: int
    output_kind: str
    license: str
    approximate_model_bytes: int
    model_card_url: str
    notes: str

    def create_detector(self, cache_dir: str | None = None) -> LocalOnnxDetector:
        return LocalOnnxDetector(
            self.repo_id,
            self.filename,
            cache_dir,
            calibration_path=None,
            revision=self.revision,
            tokenizer_use_fast=self.tokenizer_use_fast,
            ai_label_index=self.ai_label_index,
            output_kind=self.output_kind,
        )

    def public_payload(self) -> dict[str, str | int]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "revision": self.revision,
            "license": self.license,
            "approximate_model_bytes": self.approximate_model_bytes,
            "model_card_url": self.model_card_url,
            "notes": self.notes,
        }


CANDIDATES: dict[str, ModelCandidate] = {
    "tmr": ModelCandidate(
        key="tmr",
        display_name="TMR AI Text Detector",
        repo_id="onnx-community/tmr-ai-text-detector-ONNX",
        filename="onnx/model_int8.onnx",
        revision="b9aa251e5bcda7e429fcc936767d921435945b60",
        tokenizer_use_fast=True,
        ai_label_index=1,
        output_kind="softmax_logits",
        license="MIT",
        approximate_model_bytes=126_000_000,
        model_card_url=(
            "https://huggingface.co/onnx-community/"
            "tmr-ai-text-detector-ONNX"
        ),
        notes="Current production baseline; RoBERTa-base trained on RAID.",
    ),
    "glyph": ModelCandidate(
        key="glyph",
        display_name="GLYPH v1.1",
        repo_id="ogmatrixllm/glyph-v1.1",
        filename="onnx_model_quantized.onnx",
        revision="105c9ffbe10a96c47ecc7bf1ec1a00e4c8d91fdc",
        tokenizer_use_fast=False,
        ai_label_index=1,
        output_kind="softmax_logits",
        license="MIT",
        approximate_model_bytes=244_000_000,
        model_card_url="https://huggingface.co/ogmatrixllm/glyph-v1.1",
        notes=(
            "DeBERTa-v3-base candidate trained across multiple human domains "
            "and AI model families."
        ),
    ),
    "fakespot": ModelCandidate(
        key="fakespot",
        display_name="Fakespot RoBERTa v1",
        repo_id="Lynote/fakespot-ai-roberta-base-ai-text-detection-v1-browser",
        filename="onnx/model.onnx",
        revision="75d794c69eecf566e98aa7717327aed159f21f21",
        tokenizer_use_fast=True,
        ai_label_index=1,
        output_kind="softmax_logits",
        license="Apache-2.0",
        approximate_model_bytes=499_000_000,
        model_card_url=(
            "https://huggingface.co/fakespot-ai/"
            "roberta-base-ai-text-detection-v1"
        ),
        notes=(
            "Unmodified FP32 ONNX conversion of the Fakespot RoBERTa-base "
            "detector; larger than the quantized baseline."
        ),
    ),
}

DEFAULT_CANDIDATE_KEYS = ("tmr", "glyph", "fakespot")


def resolve_candidates(keys: list[str] | tuple[str, ...]) -> list[ModelCandidate]:
    if len(set(keys)) != len(keys):
        raise ValueError("Candidate names must not be repeated")
    unknown = [key for key in keys if key not in CANDIDATES]
    if unknown:
        available = ", ".join(sorted(CANDIDATES))
        raise ValueError(
            f"Unknown model candidate(s): {', '.join(unknown)}. "
            f"Available: {available}"
        )
    if "tmr" not in keys:
        raise ValueError("The current TMR baseline must be included")
    return [CANDIDATES[key] for key in keys]
