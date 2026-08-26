import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .cache import ResultCache
from .config import get_settings
from .detector import (
    analyze_heuristically,
    combine_with_external,
    combine_with_local_model,
)
from .external_detector import ExternalDetector
from .extractor import (
    article_from_text,
    content_fingerprint,
    diagnose_extraction_failure,
    extract_article,
)
from .fetcher import FetchError, canonicalize_url, fetch_html
from .models import (
    AnalysisResult,
    AnalyzeTextRequest,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    Evidence,
    ExtractedArticle,
)
from .onnx_detector import LocalOnnxDetector, ModelUnavailableError, is_likely_english
from .rate_limit import AnalysisRateLimitMiddleware


settings = get_settings()
cache = ResultCache(settings.cache_ttl_seconds)
external_detector = (
    ExternalDetector(
        settings.external_detector_url,
        settings.external_detector_api_key,
    )
    if settings.external_detector_url
    else None
)
local_model = (
    LocalOnnxDetector(
        settings.local_model_id,
        settings.local_model_filename,
        settings.local_model_cache_dir,
        settings.local_calibration_path,
        revision=settings.local_model_revision,
    )
    if settings.local_model_enabled
    else None
)
fetch_semaphore = asyncio.Semaphore(settings.fetch_concurrency)
inference_semaphore = asyncio.Semaphore(settings.inference_concurrency)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.preload_model and local_model:
        await asyncio.to_thread(local_model.prepare)
    yield


app = FastAPI(
    title="AI Article Check API",
    version="0.9.1",
    lifespan=lifespan,
)
app.add_middleware(
    AnalysisRateLimitMiddleware,
    limit=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
    trust_proxy_headers=settings.trust_proxy_headers,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_extension_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def analyze_article(
    raw_url: str,
    *,
    final_url: str,
    article: ExtractedArticle,
    content_truncated: bool = False,
    analysis_source: Literal["backend_fetch", "browser_page"] = "backend_fetch",
) -> AnalysisResult:
    fingerprint = content_fingerprint(article.text)

    if local_model and not is_likely_english(article.text):
        return AnalysisResult(
            url=raw_url,
            final_url=final_url,
            status="ok",
            label="unsupported",
            word_count=article.word_count,
            title=article.title,
            evidence=[
                Evidence(
                    kind="info",
                    message=(
                        "Only English-language articles are supported "
                        "by the current model."
                    ),
                )
            ],
            content_fingerprint=fingerprint,
            content_truncated=content_truncated,
            analysis_source=analysis_source,
            analyzed_at=now_iso(),
        )

    detector_output = analyze_heuristically(article)
    if local_model:
        try:
            async with inference_semaphore:
                model_output = await asyncio.to_thread(
                    local_model.analyze,
                    article.text,
                )
            detector_output = combine_with_local_model(
                detector_output,
                model_output,
            )
        except ModelUnavailableError as exc:
            raise FetchError(
                "The local detector is unavailable",
                code="detector_unavailable",
                retryable=True,
            ) from exc

    if content_truncated:
        detector_output.evidence.insert(
            0,
            Evidence(
                kind="info",
                message=(
                    "The page was large, so only the downloaded "
                    "portion was analyzed."
                ),
            ),
        )

    if external_detector:
        try:
            probability, confidence = await external_detector.analyze(
                article.text[:30_000]
            )
            detector_output = combine_with_external(
                detector_output,
                probability,
                confidence,
            )
        except Exception:
            detector_output.evidence.insert(
                0,
                Evidence(
                    kind="info",
                    message=(
                        "The external classifier was unavailable; "
                        "local analysis was used."
                    ),
                ),
            )

    return AnalysisResult(
        url=raw_url,
        final_url=final_url,
        status="ok",
        label=detector_output.label,
        word_count=article.word_count,
        sampled_word_count=detector_output.sampled_word_count,
        segments_checked=detector_output.segments_checked,
        ai_segments=detector_output.ai_segments,
        non_ai_segments=detector_output.non_ai_segments,
        ai_probability=detector_output.ai_probability,
        title=article.title,
        evidence=detector_output.evidence,
        content_fingerprint=fingerprint,
        content_truncated=content_truncated,
        analysis_source=analysis_source,
        analyzed_at=now_iso(),
    )


async def analyze_url(raw_url: str, *, force: bool = False) -> AnalysisResult:
    try:
        cache_key = canonicalize_url(raw_url)
    except FetchError as exc:
        return AnalysisResult(
            url=raw_url,
            status="error",
            label="unavailable",
            error=str(exc),
            error_code=exc.code,
            retryable=exc.retryable,
            analyzed_at=now_iso(),
        )

    if not force:
        cached = await cache.get(cache_key)
        if cached:
            return cached.model_copy(update={"url": raw_url})

    final_url: str | None = None
    try:
        async with fetch_semaphore:
            page = await fetch_html(
                cache_key,
                timeout_seconds=settings.fetch_timeout_seconds,
                max_bytes=settings.max_download_bytes,
                max_retries=settings.fetch_max_retries,
            )
        final_url = page.final_url
        article = extract_article(page.html)
        failure = diagnose_extraction_failure(
            page.html,
            word_count=article.word_count,
        )
        if failure:
            raise FetchError(
                failure.message,
                code=failure.code,
                retryable=failure.retryable,
            )
        result = await analyze_article(
            raw_url,
            final_url=page.final_url,
            article=article,
            content_truncated=page.truncated,
        )
    except FetchError as exc:
        result = AnalysisResult(
            url=raw_url,
            final_url=final_url,
            status="error",
            label="unavailable",
            error=str(exc),
            error_code=exc.code,
            retryable=exc.retryable,
            analyzed_at=now_iso(),
        )
    except Exception:
        result = AnalysisResult(
            url=raw_url,
            status="error",
            label="unavailable",
            error="Internal analysis error",
            error_code="internal_error",
            retryable=True,
            analyzed_at=now_iso(),
        )

    if result.status == "ok":
        await cache.set(cache_key, result)
    return result


async def analyze_browser_text(request: AnalyzeTextRequest) -> AnalysisResult:
    try:
        cache_key = canonicalize_url(request.url)
        canonical_key = (
            canonicalize_url(request.canonical_url)
            if request.canonical_url
            else cache_key
        )
    except FetchError as exc:
        return AnalysisResult(
            url=request.url,
            status="error",
            label="unavailable",
            error=str(exc),
            error_code=exc.code,
            retryable=exc.retryable,
            analysis_source="browser_page",
            analyzed_at=now_iso(),
        )

    article = article_from_text(
        request.text,
        title=request.title,
        has_author=request.has_author,
        has_citations=request.has_citations,
    )
    if article.word_count < 80:
        return AnalysisResult(
            url=request.url,
            final_url=canonical_key,
            status="error",
            label="unavailable",
            word_count=article.word_count,
            title=article.title,
            error="The open page does not contain enough article text",
            error_code="too_little_text",
            retryable=True,
            analysis_source="browser_page",
            analyzed_at=now_iso(),
        )

    try:
        result = await analyze_article(
            request.url,
            final_url=canonical_key,
            article=article,
            analysis_source="browser_page",
        )
    except FetchError as exc:
        result = AnalysisResult(
            url=request.url,
            final_url=canonical_key,
            status="error",
            label="unavailable",
            error=str(exc),
            error_code=exc.code,
            retryable=exc.retryable,
            analysis_source="browser_page",
            analyzed_at=now_iso(),
        )
    except Exception:
        result = AnalysisResult(
            url=request.url,
            final_url=canonical_key,
            status="error",
            label="unavailable",
            error="Internal analysis error",
            error_code="internal_error",
            retryable=True,
            analysis_source="browser_page",
            analyzed_at=now_iso(),
        )

    if result.status == "ok":
        for key in {cache_key, canonical_key}:
            await cache.set(key, result)
    return result


@app.get("/health")
async def health() -> dict[str, str | bool]:
    if settings.app_environment == "production":
        return {
            "status": "ok",
            "version": app.version,
        }
    detector_name = "onnx+heuristic" if local_model else "heuristic"
    if external_detector:
        detector_name = f"external+{detector_name}"
    return {
        "status": "ok",
        "version": app.version,
        "detector": detector_name,
        "model_configured": local_model is not None,
        "model_revision": settings.local_model_revision,
        "model_loaded": local_model.loaded if local_model else False,
        "calibrated": bool(local_model and local_model.calibration),
        "calibration_dataset": (
            local_model.calibration.dataset
            if local_model and local_model.calibration
            else "none"
        ),
    }


@app.post("/api/v1/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    results = await asyncio.gather(
        *(analyze_url(url, force=request.force) for url in request.urls)
    )
    return BatchAnalyzeResponse(results=list(results))


@app.post("/api/v1/analyze/text", response_model=AnalysisResult)
async def analyze_text(request: AnalyzeTextRequest) -> AnalysisResult:
    return await analyze_browser_text(request)
