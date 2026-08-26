from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .calibration import CalibrationProfile
from .config import get_settings
from .onnx_detector import LocalOnnxDetector, ModelUnavailableError


def validate_calibration(
    path: str | Path,
    *,
    model_id: str,
    model_filename: str,
    model_revision: str,
) -> tuple[CalibrationProfile, dict[str, Any]]:
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise ValueError(f"Calibration file is missing: {calibration_path}")

    try:
        with calibration_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        profile = CalibrationProfile.from_dict(payload)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Calibration file is invalid: {calibration_path}") from exc

    if payload.get("model_id") != model_id:
        raise ValueError("Calibration was fitted for a different model repository")
    if payload.get("model_filename") != model_filename:
        raise ValueError("Calibration was fitted for a different model artifact")
    if payload.get("model_revision") != model_revision:
        raise ValueError("Calibration was fitted for a different model revision")
    if profile.bands is None:
        raise ValueError("Length-aware calibration is required")
    return profile, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local detector setup.")
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download and validate the configured ONNX model as part of the check.",
    )
    args = parser.parse_args()
    settings = get_settings()

    if not settings.local_model_enabled:
        raise SystemExit("Local model is disabled in the current configuration.")
    if not settings.local_calibration_path:
        raise SystemExit("LOCAL_CALIBRATION_PATH is disabled.")

    try:
        profile, payload = validate_calibration(
            settings.local_calibration_path,
            model_id=settings.local_model_id,
            model_filename=settings.local_model_filename,
            model_revision=settings.local_model_revision,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    print(
        "Calibration is ready: "
        f"schema 4, {len(profile.bands or {})} length bands, "
        f"{payload.get('dataset', 'unknown dataset')}."
    )

    if args.download_model:
        detector = LocalOnnxDetector(
            settings.local_model_id,
            settings.local_model_filename,
            settings.local_model_cache_dir,
            settings.local_calibration_path,
            revision=settings.local_model_revision,
        )
        print(f"Preparing model {settings.local_model_id} ...", flush=True)
        try:
            model_path = detector.prepare()
        except ModelUnavailableError:
            raise SystemExit(
                "Model preparation failed. Check the internet connection and run "
                "setup again."
            ) from None
        print(f"Model is ready: {model_path}")

    print("Backend preflight passed.")


if __name__ == "__main__":
    main()
