from functools import lru_cache
import re
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_EXTENSION_ID_RE = re.compile(r"^[a-p]{32}$")


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    app_environment: Literal["development", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8787
    allowed_extension_ids: str = ""
    allowed_web_origins: str = ""
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    trust_proxy_headers: bool = False
    preload_model: bool = False
    cache_ttl_seconds: int = 43_200
    max_download_bytes: int = 5_000_000
    fetch_timeout_seconds: float = 8.0
    fetch_max_retries: int = 0
    fetch_concurrency: int = 6
    inference_batch_size: int = 14
    inference_batch_wait_ms: int = 40
    local_model_enabled: bool = True
    local_model_id: str = "onnx-community/tmr-ai-text-detector-ONNX"
    local_model_filename: str = "onnx/model_int8.onnx"
    local_model_revision: str = "b9aa251e5bcda7e429fcc936767d921435945b60"
    local_model_cache_dir: str | None = None
    local_calibration_path: str | None = "calibration.json"
    external_detector_url: str | None = None
    external_detector_api_key: str | None = None

    @field_validator("rate_limit_requests", "rate_limit_window_seconds")
    @classmethod
    def validate_non_negative_integer(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Rate-limit settings must not be negative")
        return value

    @field_validator("fetch_max_retries")
    @classmethod
    def validate_non_negative_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("FETCH_MAX_RETRIES must not be negative")
        return value

    @field_validator("fetch_concurrency", "inference_batch_size")
    @classmethod
    def validate_positive_concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Concurrency and batch settings must be at least 1")
        return value

    @field_validator("inference_batch_wait_ms")
    @classmethod
    def validate_non_negative_batch_wait(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Inference batch wait must not be negative")
        return value

    @field_validator("allowed_extension_ids")
    @classmethod
    def validate_extension_ids(cls, value: str) -> str:
        invalid = [item for item in _csv_values(value) if not _EXTENSION_ID_RE.fullmatch(item)]
        if invalid:
            raise ValueError(
                "ALLOWED_EXTENSION_IDS must contain comma-separated Chrome "
                "extension IDs"
            )
        return value

    @property
    def cors_origins(self) -> list[str]:
        origins = _csv_values(self.allowed_web_origins)
        origins.extend(
            f"chrome-extension://{extension_id}"
            for extension_id in _csv_values(self.allowed_extension_ids)
        )
        if self.app_environment == "development":
            origins.extend(
                [
                    "http://127.0.0.1:8787",
                    "http://localhost:8787",
                ]
            )
        return list(dict.fromkeys(origins))

    @property
    def cors_extension_regex(self) -> str | None:
        if self.app_environment == "production":
            return None
        return r"^chrome-extension://[a-p]{32}$"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
