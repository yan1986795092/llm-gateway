"""OpenAI-compatible API routes."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from gateway.api.dependencies import (
    build_adapters,
    build_cache_plugin,
    build_circuit_breaker,
    build_rate_limiter,
    build_router,
)
from gateway.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelInfo,
    ModelList,
    UsageInfo,
)
from gateway.core.context import RequestContext
from gateway.core.exceptions import (
    GatewayError,
    ProviderError,
    RateLimitError,
)
from gateway.plugins.base import Plugin

router = APIRouter()


def _get_ctx(request: Request) -> RequestContext:
    return RequestContext(
        source_app=request.headers.get("X-Source-App", "unknown"),
    )


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    ctx = _get_ctx(http_request)
    router_dep = build_router()
    rate_limiter = build_rate_limiter()
    cache = build_cache_plugin()
    cb_plugin = build_circuit_breaker()
    adapters = build_adapters()

    # Run plugin pre_request chain
    plugins: list[Plugin] = [rate_limiter, cache]
    req = request
    for plugin in plugins:
        req = await plugin.pre_request(ctx, req)
        if req is None:
            raise GatewayError("Request blocked by plugin")

    # Select adapter
    if request.model == "auto":
        selected = router_dep.select(
            [{"role": m.role, "content": m.content} for m in req.messages]
        )
        adapter = selected.adapter
        ctx.model = adapter.model
        ctx.provider = adapter.provider_name
    else:
        adapter = next(
            (a for a in adapters if a.model == request.model),
            adapters[0] if adapters else None,
        )
        if not adapter:
            raise GatewayError(f"Model '{request.model}' not found", status_code=404)
        ctx.model = adapter.model
        ctx.provider = adapter.provider_name

    # Circuit breaker check
    cb_plugin.check_provider(adapter.provider_name)

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if request.stream:
        return StreamingResponse(
            _stream_response(adapter, messages, req, ctx, cb_plugin, cache),
            media_type="text/event-stream",
        )

    try:
        response = await adapter.chat_completions(
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=False,
        )
        cb_plugin.record_success(adapter.provider_name)
        router_dep.record_latency(adapter.provider_name, response.latency_ms)

        # Store in cache
        cache_key = f"llm:cache:{req.hash_key()}"
        await cache.store_response(
            cache_key,
            {
                "id": f"chatcmpl-{ctx.trace_id}",
                "model": response.model,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": response.content},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "usage": {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.prompt_tokens + response.completion_tokens,
                },
            },
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{ctx.trace_id}",
            created=int(time.time()),
            model=response.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.content},
                    "finish_reason": "stop",
                }
            ],
            usage=UsageInfo(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.prompt_tokens + response.completion_tokens,
            ),
        )
    except Exception as e:
        cb_plugin.record_failure(adapter.provider_name)
        raise


async def _stream_response(
    adapter: Any,
    messages: list[dict[str, str]],
    request: ChatCompletionRequest,
    ctx: RequestContext,
    cb_plugin: Any,
    cache: Any,
) -> AsyncIterator[str]:
    """Yield SSE stream chunks."""
    chunk_id = f"chatcmpl-{ctx.trace_id}"
    try:
        async for chunk in adapter.chat_completions_stream(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            yield f"data: {json.dumps({'id': chunk_id, 'choices': [{'index': 0, 'delta': {'content': chunk}}]})}\n\n"
        yield f"data: {json.dumps({'id': chunk_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
        cb_plugin.record_success(adapter.provider_name)
    except Exception as e:
        cb_plugin.record_failure(adapter.provider_name)
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'api_error'}})}\n\n"


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """List available models."""
    adapters = build_adapters()
    models: list[ModelInfo] = []
    for adapter in adapters:
        try:
            provider_models = await adapter.list_models()
            models.extend(provider_models)
        except Exception:
            pass
    return ModelList(data=models)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Placeholder for Prometheus metrics."""
    return {"status": "ok"}
