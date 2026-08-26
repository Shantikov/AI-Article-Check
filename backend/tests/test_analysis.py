import asyncio

import pytest
from fastapi.testclient import TestClient

from app import main
from app.cache import ResultCache
from app.fetcher import FetchedPage
from app.models import AnalyzeTextRequest, BatchAnalyzeRequest


def article_html(word: str) -> str:
    text = " ".join([word] * 100)
    return f"<html><article><p>{text}</p></article></html>"


@pytest.mark.asyncio
async def test_batch_fetches_six_pages_concurrently_with_fast_limits(
    monkeypatch,
) -> None:
    active_fetches = 0
    peak_fetches = 0
    started_fetches = 0
    all_started = asyncio.Event()
    received_options: list[dict[str, object]] = []

    async def fake_fetch_html(url: str, **kwargs) -> FetchedPage:
        nonlocal active_fetches, peak_fetches, started_fetches
        received_options.append(kwargs)
        active_fetches += 1
        started_fetches += 1
        peak_fetches = max(peak_fetches, active_fetches)
        if started_fetches == 6:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=0.5)
        active_fetches -= 1
        return FetchedPage(final_url=url, html=article_html("article"))

    monkeypatch.setattr(main, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(main, "fetch_semaphore", asyncio.Semaphore(6))
    monkeypatch.setattr(main, "local_model", None)
    monkeypatch.setattr(main, "external_detector", None)
    monkeypatch.setattr(main, "cache", ResultCache(3_600))

    response = await asyncio.wait_for(
        main.analyze_batch(
            BatchAnalyzeRequest(
                urls=[f"https://example.com/article-{index}" for index in range(6)]
            )
        ),
        timeout=1,
    )

    assert peak_fetches == 6
    assert len(response.results) == 6
    assert all(option["timeout_seconds"] == 8.0 for option in received_options)
    assert all(option["max_retries"] == 0 for option in received_options)


@pytest.mark.asyncio
async def test_forced_recheck_bypasses_cache_and_updates_content_version(
    monkeypatch,
) -> None:
    calls = 0

    async def fake_fetch_html(*_args, **_kwargs) -> FetchedPage:
        nonlocal calls
        calls += 1
        return FetchedPage(
            final_url="https://example.com/article",
            html=article_html("alpha" if calls == 1 else "beta"),
        )

    monkeypatch.setattr(main, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(main, "local_model", None)
    monkeypatch.setattr(main, "external_detector", None)
    monkeypatch.setattr(main, "cache", ResultCache(3_600))

    first = await main.analyze_url("https://example.com/article")
    cached = await main.analyze_url("https://example.com/article")
    refreshed = await main.analyze_url(
        "https://example.com/article",
        force=True,
    )

    assert calls == 2
    assert first.cache_hit is False
    assert cached.cache_hit is True
    assert refreshed.cache_hit is False
    assert first.content_fingerprint != refreshed.content_fingerprint


@pytest.mark.asyncio
async def test_long_block_page_is_not_analyzed_as_an_article(monkeypatch) -> None:
    async def fake_fetch_html(*_args, **_kwargs) -> FetchedPage:
        return FetchedPage(
            final_url="https://example.com/protected",
            html=(
                "<html><title>Security check</title>"
                "<p>Verify you are human. "
                + "blocked " * 120
                + "</p></html>"
            ),
        )

    monkeypatch.setattr(main, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(main, "local_model", None)
    monkeypatch.setattr(main, "external_detector", None)
    monkeypatch.setattr(main, "cache", ResultCache(3_600))

    result = await main.analyze_url("https://example.com/protected", force=True)

    assert result.status == "error"
    assert result.label == "unavailable"
    assert result.error_code == "access_blocked"


@pytest.mark.asyncio
async def test_sitewide_captcha_script_does_not_block_real_article(monkeypatch) -> None:
    async def fake_fetch_html(*_args, **_kwargs) -> FetchedPage:
        return FetchedPage(
            final_url="https://en.wikipedia.org/wiki/Example",
            html=(
                "<html><head><title>Example - Wikipedia</title>"
                "<script>const captcha = 'cf-chl- hcaptcha access denied';</script>"
                "</head><body><h1>Example</h1><article><p>"
                + "encyclopedia " * 120
                + "</p></article></body></html>"
            ),
        )

    monkeypatch.setattr(main, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(main, "local_model", None)
    monkeypatch.setattr(main, "external_detector", None)
    monkeypatch.setattr(main, "cache", ResultCache(3_600))

    result = await main.analyze_url(
        "https://en.wikipedia.org/wiki/Example",
        force=True,
    )

    assert result.status == "ok"
    assert result.error_code is None
    assert result.word_count == 120


@pytest.mark.asyncio
async def test_browser_text_analysis_is_cached_for_page_and_canonical_url(
    monkeypatch,
) -> None:
    monkeypatch.setattr(main, "local_model", None)
    monkeypatch.setattr(main, "external_detector", None)
    monkeypatch.setattr(main, "cache", ResultCache(3_600))
    request = AnalyzeTextRequest(
        url="https://example.com/story?view=browser",
        canonical_url="https://example.com/story",
        title="Rendered story",
        text=" ".join(["rendered"] * 120),
        has_author=True,
    )

    result = await main.analyze_browser_text(request)
    page_cached = await main.cache.get("https://example.com/story?view=browser")
    canonical_cached = await main.cache.get("https://example.com/story")

    assert result.status == "ok"
    assert result.analysis_source == "browser_page"
    assert result.word_count == 120
    assert page_cached is not None
    assert canonical_cached is not None
    assert page_cached.content_fingerprint == result.content_fingerprint
    assert canonical_cached.content_fingerprint == result.content_fingerprint


@pytest.mark.asyncio
async def test_browser_text_analysis_rejects_short_open_page(monkeypatch) -> None:
    monkeypatch.setattr(main, "cache", ResultCache(3_600))
    request = AnalyzeTextRequest(
        url="https://example.com/short",
        text="Only a short page was rendered.",
    )

    result = await main.analyze_browser_text(request)

    assert result.status == "error"
    assert result.error_code == "too_little_text"
    assert result.analysis_source == "browser_page"


def test_browser_text_endpoint_uses_same_analysis_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(main, "local_model", None)
    monkeypatch.setattr(main, "external_detector", None)
    monkeypatch.setattr(main, "cache", ResultCache(3_600))

    with TestClient(main.app) as client:
        response = client.post(
            "/api/v1/analyze/text",
            json={
                "url": "https://example.com/rendered",
                "title": "Rendered article",
                "text": " ".join(["browser"] * 120),
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["analysis_source"] == "browser_page"
    assert payload["word_count"] == 120


def test_cors_allows_extension_origin_but_not_arbitrary_websites() -> None:
    with TestClient(main.app) as client:
        allowed = client.options(
            "/api/v1/analyze/text",
            headers={
                "Origin": f"chrome-extension://{'a' * 32}",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        blocked = client.options(
            "/api/v1/analyze/text",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"].startswith(
        "chrome-extension://"
    )
    assert "access-control-allow-origin" not in blocked.headers
