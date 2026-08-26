import json
import os
from pathlib import Path

from app.onnx_detector import LocalOnnxDetector

from .core import aggregate_scores


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def scores_are_compatible(
    scores: list[dict],
    records: list[dict],
    *,
    benchmark_sha256: str,
    model_id: str,
    model_filename: str,
    model_revision: str | None = None,
    accept_missing_revision: bool = False,
) -> bool:
    if len(scores) != len(records):
        return False
    expected_ids = {record["id"] for record in records}
    if {score.get("id") for score in scores} != expected_ids:
        return False
    for score in scores:
        revision_matches = (
            model_revision is None
            or score.get("model_revision") == model_revision
            or (accept_missing_revision and score.get("model_revision") is None)
        )
        if not (
            score.get("benchmark_sha256") == benchmark_sha256
            and score.get("model_id") == model_id
            and score.get("model_filename") == model_filename
            and revision_matches
            and isinstance(score.get("raw_score"), (int, float))
            and isinstance(score.get("segments"), int)
            and score["segments"] > 0
        ):
            return False
    return True


def score_records(
    records: list[dict],
    detector: LocalOnnxDetector,
    *,
    benchmark_sha256: str,
    model_id: str,
    model_filename: str,
    model_revision: str | None = None,
) -> list[dict]:
    scores: list[dict] = []
    for index, record in enumerate(records, start=1):
        chunk_scores = detector.score_text(record["text"])
        scores.append({
            "id": record["id"],
            "benchmark_sha256": benchmark_sha256,
            "model_id": model_id,
            "model_filename": model_filename,
            "model_revision": model_revision,
            "raw_score": aggregate_scores(chunk_scores),
            "segments": len(chunk_scores),
        })
        if index % 20 == 0 or index == len(records):
            print(f"Scored {index}/{len(records)} texts", flush=True)
    return scores
