import json

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import Settings
from app.rate_limit import (
    AnalysisRateLimitMiddleware,
    ConcurrentRequestGate,
    SlidingWindowLimiter,
)


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


def test_fast_batch_defaults_keep_network_and_inference_limits_separate() -> None:
    settings = Settings()

    assert settings.fetch_timeout_seconds == 8.0
    assert settings.fetch_max_retries == 0
    assert settings.fetch_concurrency == 6
    assert settings.inference_batch_size == 14
    assert settings.inference_batch_wait_ms == 40
    assert settings.rate_limit_requests == 60
    assert settings.rate_limit_global_requests == 240
    assert settings.max_concurrent_analysis_requests == 24
    assert settings.max_request_body_bytes == 512_000


@pytest.mark.parametrize("field", ["fetch_concurrency", "inference_batch_size"])
def test_concurrency_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError):
        Settings(**{field: 0})


def test_inference_batch_wait_must_not_be_negative() -> None:
    with pytest.raises(ValueError):
        Settings(inference_batch_wait_ms=-1)


@pytest.mark.parametrize(
    "field",
    [
        "rate_limit_requests",
        "rate_limit_global_requests",
        "rate_limit_window_seconds",
        "max_concurrent_analysis_requests",
    ],
)
def test_public_api_limits_must_not_be_negative(field: str) -> None:
    with pytest.raises(ValueError):
        Settings(**{field: -1})


def test_request_body_limit_cannot_be_disabled_accidentally() -> None:
    with pytest.raises(ValueError):
        Settings(max_request_body_bytes=1_024)


@pytest.mark.asyncio
async def test_sliding_window_rate_limiter_recovers_after_window() -> None:
    limiter = SlidingWindowLimiter(limit=2, window_seconds=10)

    assert await limiter.allow("client", now=100) == (True, 0)
    assert await limiter.allow("client", now=101) == (True, 0)
    assert await limiter.allow("client", now=102) == (False, 8)
    assert await limiter.allow("other", now=102) == (True, 0)
    assert await limiter.allow("client", now=111) == (True, 0)


@pytest.mark.asyncio
async def test_sliding_window_rate_limiter_counts_batch_items() -> None:
    limiter = SlidingWindowLimiter(limit=6, window_seconds=60)

    assert await limiter.allow("client", cost=5, now=100) == (True, 0)
    assert await limiter.allow("client", cost=2, now=101) == (False, 59)
    assert await limiter.allow("client", cost=1, now=101) == (True, 0)


@pytest.mark.asyncio
async def test_sliding_window_rate_limiter_bounds_client_buckets() -> None:
    limiter = SlidingWindowLimiter(
        limit=10,
        window_seconds=60,
        max_client_keys=2,
    )

    assert await limiter.allow("first", now=100) == (True, 0)
    assert await limiter.allow("second", now=100) == (True, 0)
    assert await limiter.allow("third", now=100) == (True, 0)
    assert list(limiter._requests) == ["second", "third"]


@pytest.mark.asyncio
async def test_batch_request_cost_matches_number_of_urls() -> None:
    body = json.dumps({"urls": [f"https://example.com/{index}" for index in range(6)]}).encode()
    sent = False

    async def receive() -> dict[str, object]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "path": "/api/v1/analyze/batch",
            "raw_path": b"/api/v1/analyze/batch",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("10.0.0.1", 1234),
            "server": ("testserver", 443),
        },
        receive,
    )
    middleware = AnalysisRateLimitMiddleware(
        lambda *_args, **_kwargs: None,
        limit=60,
        window_seconds=60,
    )

    assert await middleware.request_cost(request) == 6


@pytest.mark.asyncio
async def test_forced_batch_costs_twice_as_much() -> None:
    body = json.dumps(
        {
            "urls": [f"https://example.com/{index}" for index in range(6)],
            "force": True,
        }
    ).encode()
    sent = False

    async def receive() -> dict[str, object]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "path": "/api/v1/analyze/batch",
            "raw_path": b"/api/v1/analyze/batch",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("10.0.0.1", 1234),
            "server": ("testserver", 443),
        },
        receive,
    )
    middleware = AnalysisRateLimitMiddleware(
        lambda *_args, **_kwargs: None,
        limit=60,
        window_seconds=60,
    )

    assert await middleware.request_cost(request) == 12


@pytest.mark.asyncio
async def test_concurrency_gate_rejects_excess_work_and_recovers() -> None:
    gate = ConcurrentRequestGate(limit=2)

    assert await gate.enter() is True
    assert await gate.enter() is True
    assert await gate.enter() is False
    await gate.leave()
    assert await gate.enter() is True
    await gate.leave()
    await gate.leave()


@pytest.mark.asyncio
async def test_global_rate_limit_is_shared_across_clients() -> None:
    middleware = AnalysisRateLimitMiddleware(
        lambda *_args, **_kwargs: None,
        limit=10,
        global_limit=2,
        window_seconds=60,
    )

    async def call(client_ip: str):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "scheme": "https",
                "path": "/api/v1/analyze/text",
                "raw_path": b"/api/v1/analyze/text",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": (client_ip, 1234),
                "server": ("testserver", 443),
            }
        )

        async def call_next(_request: Request):
            return JSONResponse({"ok": True})

        return await middleware.dispatch(request, call_next)

    assert (await call("203.0.113.1")).status_code == 200
    assert (await call("203.0.113.2")).status_code == 200
    blocked = await call("203.0.113.3")
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"]


def test_trusted_proxy_uses_valid_railway_real_ip() -> None:
    middleware = AnalysisRateLimitMiddleware(
        lambda *_args, **_kwargs: None,
        limit=60,
        window_seconds=60,
        trust_proxy_headers=True,
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "path": "/api/v1/analyze/text",
            "raw_path": b"/api/v1/analyze/text",
            "query_string": b"",
            "headers": [
                (b"x-real-ip", b"203.0.113.8"),
                (b"x-forwarded-for", b"198.51.100.4, 10.0.0.1"),
            ],
            "client": ("10.0.0.2", 1234),
            "server": ("testserver", 443),
        }
    )

    assert middleware.client_key(request) == "203.0.113.8"
