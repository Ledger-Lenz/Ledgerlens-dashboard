"""
Simple in-memory rate limiter middleware for the LedgerLens API.

Limits each client IP to MAX_REQUESTS requests per WINDOW_SECONDS.
Intended for the public-facing /score endpoint to prevent abuse.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

MAX_REQUESTS = 60
WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = MAX_REQUESTS, window: int = WINDOW_SECONDS) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next) -> Response:
        if not request.url.path.startswith("/score"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = self._buckets[client_ip]

        # Evict timestamps outside the window
        while bucket and bucket[0] < now - self.window:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            return Response(
                content='{"detail":"Rate limit exceeded. Max 60 requests per minute."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(self.window)},
            )

        bucket.append(now)
        return await call_next(request)
