from __future__ import annotations

import asyncio
import ipaddress
import json
from collections import OrderedDict, deque
from math import ceil
from time import monotonic

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response


class SlidingWindowLimiter:
    """Small per-process request limiter for the public analysis endpoints."""

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        *,
        max_client_keys: int = 10_000,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_client_keys = max(1, max_client_keys)
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def allow(
        self,
        key: str,
        *,
        cost: int = 1,
        now: float | None = None,
    ) -> tuple[bool, int]:
        if self.limit == 0 or self.window_seconds == 0:
            return True, 0
        cost = max(1, cost)

        current = monotonic() if now is None else now
        cutoff = current - self.window_seconds
        async with self._lock:
            bucket = self._requests.setdefault(key, deque())
            self._requests.move_to_end(key)
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) + cost > self.limit:
                retry_after = (
                    max(1, ceil(bucket[0] + self.window_seconds - current))
                    if bucket
                    else self.window_seconds
                )
                if not bucket:
                    self._requests.pop(key, None)
                return False, retry_after
            bucket.extend([current] * cost)
            while len(self._requests) > self.max_client_keys:
                self._requests.popitem(last=False)
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
            candidates = [
                request.headers.get("x-real-ip", "").strip(),
                request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip(),
            ]
            for candidate in candidates:
                try:
                    return str(ipaddress.ip_address(candidate))
                except ValueError:
                    continue
        return request.client.host if request.client else "unknown"

    async def request_cost(self, request: Request) -> int:
        if not request.url.path.endswith("/batch"):
            return 1
        try:
            payload = json.loads(await request.body())
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return 1
        urls = payload.get("urls") if isinstance(payload, dict) else None
        return min(10, max(1, len(urls))) if isinstance(urls, list) else 1

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

        cost = await self.request_cost(request)
        allowed, retry_after = await self.limiter.allow(
            self.client_key(request),
            cost=cost,
        )
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
