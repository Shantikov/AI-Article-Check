import pytest

from app.config import Settings
from app.rate_limit import SlidingWindowLimiter


def test_development_cors_accepts_unpacked_chrome_extensions() -> None:
    settings = Settings()

    assert settings.cors_extension_regex == r"^chrome-extension://[a-p]{32}$"
    assert "http://127.0.0.1:8787" in settings.cors_origins


def test_production_cors_uses_only_explicit_origins() -> None:
    first_id = "a" * 32
    second_id = "b" * 32
    settings = Settings(
        app_environment="production",
        allowed_extension_ids=f"{first_id},{second_id}",
        allowed_web_origins="https://admin.example.com",
    )

    assert settings.cors_extension_regex is None
    assert settings.cors_origins == [
        "https://admin.example.com",
        f"chrome-extension://{first_id}",
        f"chrome-extension://{second_id}",
    ]


def test_invalid_extension_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        Settings(
            app_environment="production",
            allowed_extension_ids="not-an-extension-id",
        )


@pytest.mark.asyncio
async def test_sliding_window_rate_limiter_recovers_after_window() -> None:
    limiter = SlidingWindowLimiter(limit=2, window_seconds=10)

    assert await limiter.allow("client", now=100) == (True, 0)
    assert await limiter.allow("client", now=101) == (True, 0)
    assert await limiter.allow("client", now=102) == (False, 8)
    assert await limiter.allow("other", now=102) == (True, 0)
    assert await limiter.allow("client", now=111) == (True, 0)
