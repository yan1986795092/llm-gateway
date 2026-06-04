"""Observability plugin — structured logging, metrics, tracing."""

from __future__ import annotations

import logging
import time
from typing import Any

import structlog

from gateway.api.schemas import ChatCompletionRequest
from gateway.core.context import RequestContext
from gateway.plugins.base import Plugin

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


class ObservabilityPlugin(Plugin):
    """Observability plugin — logs, metrics counters."""

    def __init__(self):
        self._request_count = 0
        self._error_count = 0

    @property
    def name(self) -> str:
        return "observability"

    async def pre_request(
        self, ctx: RequestContext, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        ctx.started_at = time.perf_counter()
        self._request_count += 1
        logger.info(
            "request_start",
            trace_id=ctx.trace_id,
            model=request.model,
            message_count=len(request.messages),
            source_app=ctx.source_app,
            total_requests=self._request_count,
        )
        return request

    async def post_response(
        self, ctx: RequestContext, response: dict[str, Any]
    ) -> dict[str, Any]:
        elapsed_ms = round((time.perf_counter() - ctx.started_at) * 1000, 2)
        logger.info(
            "request_end",
            trace_id=ctx.trace_id,
            model=ctx.model,
            provider=ctx.provider,
            elapsed_ms=elapsed_ms,
            status="success",
        )
        return response

    async def on_error(self, ctx: RequestContext, error: Exception) -> None:
        self._error_count += 1
        elapsed_ms = round((time.perf_counter() - ctx.started_at) * 1000, 2)
        logger.error(
            "request_error",
            trace_id=ctx.trace_id,
            model=ctx.model,
            provider=ctx.provider,
            elapsed_ms=elapsed_ms,
            error_type=type(error).__name__,
            error_message=str(error),
            total_errors=self._error_count,
        )
