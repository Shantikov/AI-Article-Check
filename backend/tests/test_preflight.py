import json
from pathlib import Path

import pytest

from app.preflight import validate_calibration


PROJECT_CALIBRATION = Path(__file__).resolve().parents[1] / "calibration.json"
MODEL_ID = "onnx-community/tmr-ai-text-detector-ONNX"
MODEL_FILENAME = "onnx/model_int8.onnx"
MODEL_REVISION = "b9aa251e5bcda7e429fcc936767d921435945b60"


def test_bundled_calibration_matches_runtime_model() -> None:
    profile, payload = validate_calibration(
        PROJECT_CALIBRATION,
        model_id=MODEL_ID,
        model_filename=MODEL_FILENAME,
        model_revision=MODEL_REVISION,
    )

    assert payload["schema_version"] == 4
    assert payload["external_test_used"] is False
    assert set(profile.bands or {}) == {
        "1_segment",
        "2_segments",
        "3_or_more_segments",
    }


def test_preflight_rejects_calibration_for_another_model(tmp_path) -> None:
    payload = json.loads(PROJECT_CALIBRATION.read_text(encoding="utf-8"))
    payload["model_id"] = "another/model"
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different model repository"):
        validate_calibration(
            path,
            model_id=MODEL_ID,
            model_filename=MODEL_FILENAME,
            model_revision=MODEL_REVISION,
        )
