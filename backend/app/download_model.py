from .config import get_settings
from .onnx_detector import LocalOnnxDetector, ModelUnavailableError


def main() -> None:
    settings = get_settings()
    if not settings.local_model_enabled:
        raise SystemExit("LOCAL_MODEL_ENABLED is disabled.")

    detector = LocalOnnxDetector(
        settings.local_model_id,
        settings.local_model_filename,
        settings.local_model_cache_dir,
        calibration_path=None,
        revision=settings.local_model_revision,
    )
    print(f"Downloading {settings.local_model_id} ...", flush=True)
    try:
        detector.prepare()
    except ModelUnavailableError:
        raise SystemExit(
            "Model download failed. Check the internet connection and run the "
            "command again."
        ) from None
    print("Model is ready.")


if __name__ == "__main__":
    main()
