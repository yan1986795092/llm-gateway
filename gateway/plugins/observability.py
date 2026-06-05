"""可观测性插件 — 使用统一日志记录 LLM 请求详情"""

import time
from typing import Any

from gateway.plugins.base import Plugin
from gateway.core.context import RequestContext
from gateway.api.schemas import ChatCompletionRequest
from gateway.logging_shared.config import get_logger
from gateway.logging_shared.fields import (
    TRACE_ID, EVENT, LLM_MODEL, LLM_PROVIDER, LLM_LATENCY_MS,
    LLM_PROMPT_TOKENS, LLM_COMPLETION_TOKENS, LLM_TOTAL_TOKENS,
    LLM_STREAMING, DURATION_MS, SOURCE_APP, LLM_REQUEST_START,
    LLM_REQUEST_END, LLM_REQUEST_ERROR, STATUS_CODE,
)

logger = get_logger("llm")


class ObservabilityPlugin(Plugin):
    """可观测性插件 — 记录 LLM 请求的详细信息"""

    def __init__(self):
        self._request_count = 0
        self._error_count = 0

    @property
    def name(self) -> str:
        return "observability"

    async def pre_request(
        self, ctx: RequestContext, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        self._request_count += 1
        logger.info(
            EVENT,
            **{
                EVENT: LLM_REQUEST_START,
                TRACE_ID: ctx.trace_id,
                LLM_MODEL: request.model,
                LLM_STREAMING: request.stream,
                SOURCE_APP: ctx.source_app or None,
                "message_count": len(request.messages),
                "total_requests": self._request_count,
            },
        )
        return request

    async def post_response(
        self, ctx: RequestContext, response: dict[str, Any]
    ) -> dict[str, Any]:
        elapsed_ms = round((time.perf_counter() - ctx.started_at) * 1000, 2)

        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        logger.info(
            EVENT,
            **{
                EVENT: LLM_REQUEST_END,
                TRACE_ID: ctx.trace_id,
                LLM_MODEL: ctx.model,
                LLM_PROVIDER: ctx.provider,
                LLM_LATENCY_MS: elapsed_ms,
                LLM_PROMPT_TOKENS: prompt_tokens,
                LLM_COMPLETION_TOKENS: completion_tokens,
                LLM_TOTAL_TOKENS: total_tokens,
                STATUS_CODE: 200,
                DURATION_MS: elapsed_ms,
            },
        )
        return response

    async def on_error(self, ctx: RequestContext, error: Exception) -> None:
        self._error_count += 1
        elapsed_ms = round((time.perf_counter() - ctx.started_at) * 1000, 2)
        logger.error(
            EVENT,
            **{
                EVENT: LLM_REQUEST_ERROR,
                TRACE_ID: ctx.trace_id,
                LLM_MODEL: ctx.model,
                LLM_PROVIDER: ctx.provider,
                LLM_LATENCY_MS: elapsed_ms,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "total_errors": self._error_count,
                STATUS_CODE: 500,
            },
        )
