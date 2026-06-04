"""Circuit breaker plugin — protects against cascading failures."""

from __future__ import annotations

import enum
import time
from collections import deque
from threading import Lock
from typing import Any

from gateway.api.schemas import ChatCompletionRequest
from gateway.core.context import RequestContext
from gateway.core.exceptions import CircuitBreakerOpenError
from gateway.plugins.base import Plugin
from gateway.config.settings import get_settings


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """State machine circuit breaker per provider."""

    def __init__(
        self,
        provider: str,
        failure_threshold: int = 3,
        error_rate_threshold: float = 0.3,
        window_seconds: int = 300,
        cooldown_seconds: int = 30,
    ):
        self.provider = provider
        self.failure_threshold = failure_threshold
        self.error_rate_threshold = error_rate_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self._state = CircuitState.CLOSED
        self._lock = Lock()
        self._failures: deque[float] = deque()
        self._successes: deque[float] = deque()
        self._open_at: float | None = None
        self._half_open_attempts: int = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    def _clean_window(self, events: deque[float], now: float) -> None:
        while events and events[0] < now - self.window_seconds:
            events.popleft()

    def _should_open(self) -> bool:
        now = time.time()
        self._clean_window(self._failures, now)
        total = len(self._failures) + len(self._successes)
        if total == 0:
            return False
        if len(self._failures) >= self.failure_threshold:
            return True
        if len(self._failures) / total >= self.error_rate_threshold:
            return True
        return False

    def record_success(self) -> None:
        with self._lock:
            now = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_attempts += 1
                self._successes.append(now)
                self._state = CircuitState.CLOSED
                self._failures.clear()
                self._successes.clear()
                self._half_open_attempts = 0
            elif self._state == CircuitState.CLOSED:
                self._successes.append(now)
                self._clean_window(self._successes, now)

    def record_failure(self) -> None:
        with self._lock:
            now = time.time()
            self._failures.append(now)
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._open_at = now
                self._half_open_attempts = 0
            elif self._state == CircuitState.CLOSED and self._should_open():
                self._state = CircuitState.OPEN
                self._open_at = now

    def check(self) -> None:
        """Raise CircuitBreakerOpenError if circuit is open."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._open_at and time.time() - self._open_at >= self.cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_attempts = 0
                else:
                    raise CircuitBreakerOpenError(
                        self.provider, self.cooldown_seconds
                    )
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_attempts >= 3:
                    raise CircuitBreakerOpenError(
                        self.provider, self.cooldown_seconds
                    )


class CircuitBreakerPlugin(Plugin):
    """Circuit breaker plugin managing per-provider breakers."""

    def __init__(
        self,
        failure_threshold: int | None = None,
        error_rate_threshold: float | None = None,
        window_seconds: int | None = None,
        cooldown_seconds: int | None = None,
    ):
        settings = get_settings()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._settings = settings

    def _get_breaker(self, provider: str) -> CircuitBreaker:
        if provider not in self._breakers:
            s = self._settings
            self._breakers[provider] = CircuitBreaker(
                provider,
                failure_threshold=s.cb_failure_threshold,
                error_rate_threshold=s.cb_error_rate_threshold,
                window_seconds=s.cb_window_seconds,
                cooldown_seconds=s.cb_cooldown_seconds,
            )
        return self._breakers[provider]

    @property
    def name(self) -> str:
        return "circuit_breaker"

    async def pre_request(
        self, ctx: RequestContext, request: ChatCompletionRequest
    ) -> ChatCompletionRequest | None:
        return request

    def check_provider(self, provider: str) -> None:
        """Call before invoking a provider. Raises CircuitBreakerOpenError."""
        self._get_breaker(provider).check()

    def record_success(self, provider: str) -> None:
        self._get_breaker(provider).record_success()

    def record_failure(self, provider: str) -> None:
        self._get_breaker(provider).record_failure()

    async def post_response(
        self, ctx: RequestContext, response: dict[str, Any]
    ) -> dict[str, Any]:
        if ctx.provider:
            self.record_success(ctx.provider)
        return response

    async def on_error(self, ctx: RequestContext, error: Exception) -> None:
        if ctx.provider:
            self.record_failure(ctx.provider)
