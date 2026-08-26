from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from math import ceil
from time import monotonic

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response


class SlidingWindowLimiter:
    """Small per-process request limiter for the public analysis endpoints."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def allow(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        if self.limit == 0 or self.window_seconds == 0:
            return True, 0

        current = monotonic() if now is None else now
        cutoff = current - self.window_seconds
        async with self._lock:
            bucket = self._requests[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry_after = max(1, ceil(bucket[0] + self.window_seconds - current))
                return False, retry_after
            bucket.append(current)
            if not bucket:
                self._requests.pop(key, None)
        return True, 0


class AnalysisRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        limit: int,
        window_seconds: int,
        trust_proxy_headers: bool = False,
    ) -> None:
        super().__init__(app)
        self.limiter = SlidingWindowLimiter(limit, window_seconds)
        self.limit = limit
        self.trust_proxy_headers = trust_proxy_headers

    def client_key(self, request: Request) -> str:
        if self.trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for", "")
            first = forwarded.split(",", 1)[0].strip()
            if first:
                return first
        return request.client.host if request.client else "unknown"

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        is_analysis = (
            request.method == "POST"
            and request.url.path.startswith("/api/v1/analyze/")
        )
        if not is_analysis:
            return await call_next(request)

        allowed, retry_after = await self.limiter.allow(self.client_key(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many analysis requests. Try again shortly."
                },
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)
        if self.limit:
            response.headers["X-RateLimit-Limit"] = str(self.limit)
        return response
