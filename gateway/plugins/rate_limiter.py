"""Rate limiting plugin using sliding window algorithm."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from gateway.api.schemas import ChatCompletionRequest
from gateway.core.context import RequestContext
from gateway.core.exceptions import RateLimitError
from gateway.plugins.base import Plugin
from gateway.config.settings import get_settings

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter — async-safe, checks both RPM and TPM."""

    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        self._lock = asyncio.Lock()
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._tokens: dict[str, list[tuple[float, int]]] = defaultdict(list)

    @staticmethod
    def _clean(times: list, now: float, window: float = 60.0) -> None:
        """Remove expired entries from the window.

        Supports both plain timestamps (RPM tracking) and
        (timestamp, count) tuples (TPM tracking).
        """
        while times:
            first = times[0]
            ts = first[0] if isinstance(first, (list, tuple)) else first
            if ts < now - window:
                times.pop(0)
            else:
                break

    async def acquire(
        self, key: str, estimated_tokens: int = 100, timeout: float = 5.0
    ) -> None:
        """Acquire a rate limit slot. Checks both RPM and TPM. Raises RateLimitError on timeout."""
        deadline = time.time() + timeout
        while True:
            acquired = await self._try_acquire(key, estimated_tokens)
            if acquired:
                return
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RateLimitError(
                    f"Rate limit exceeded for key '{key}' (RPM={self.rpm}, TPM={self.tpm})",
                    retry_after=int(timeout),
                )
            await asyncio.sleep(min(0.1, remaining))

    async def _try_acquire(self, key: str, estimated_tokens: int) -> bool:
        now = time.time()
        async with self._lock:
            # RPM check
            self._clean(self._requests[key], now)
            if len(self._requests[key]) >= self.rpm:
                return False
            # TPM check
            self._clean(self._tokens[key], now)
            current_tpm = sum(tc for _, tc in self._tokens[key])
            if current_tpm + estimated_tokens > self.tpm:
                return False
            self._requests[key].append(now)
            self._tokens[key].append((now, estimated_tokens))
            return True


class RateLimiterPlugin(Plugin):
    """Rate limiting plugin using sliding window."""

    def __init__(self, rpm: int | None = None, tpm: int | None = None):
        settings = get_settings()
        self.rpm = rpm or settings.rate_limit_rpm
        self.tpm = tpm or settings.rate_limit_tpm
        self._limiter = SlidingWindowRateLimiter(self.rpm, self.tpm)

    @property
    def name(self) -> str:
        return "rate_limiter"

    async def pre_request(
        self, ctx: RequestContext, request: ChatCompletionRequest
    ) -> ChatCompletionRequest | None:
        # 租户 + 应用 双层分桶：不同项目配额互不影响
        key = f"{(ctx.project or 'default')}:{ctx.source_app or 'default'}"
        estimated_tokens = sum(len(m.content) // 4 for m in request.messages)
        await self._limiter.acquire(key, estimated_tokens)
        return request

    async def post_response(
        self, ctx: RequestContext, response: dict[str, Any]
    ) -> dict[str, Any]:
        return response

    async def on_error(self, ctx: RequestContext, error: Exception) -> None:
        pass
