"""OpenAI-compatible API routes."""

from __future__ import annotations

import asyncio
import json
import logging
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
from gateway.plugins.observability import ObservabilityPlugin

router = APIRouter()

# 模块级单例
_obs_plugin: ObservabilityPlugin | None = None

logger = logging.getLogger(__name__)


def _get_obs_plugin() -> ObservabilityPlugin:
    global _obs_plugin
    if _obs_plugin is None:
        _obs_plugin = ObservabilityPlugin()
    return _obs_plugin


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
    plugins: list[Plugin] = [rate_limiter, cache, _get_obs_plugin()]
    req = request
    for plugin in plugins:
        req = await plugin.pre_request(ctx, req)
        if req is None:
            raise GatewayError("Request blocked by plugin")

    # Select adapter — 优先使用 AutoRouter
    # 当请求携带 tools 参数时，强制过滤不支持 function calling 的模型
    route_hint = http_request.headers.get("X-Route-Hint", "")
    _require_fc = bool(req.tools or req.tool_choice)

    if request.model == "auto" or not request.model or route_hint:
        # 从 header 获取任务类型提示
        hint = route_hint or "balanced"
        selected = router_dep.select(
            [{"role": m.role, "content": m.content} for m in req.messages],
            hint=hint,
            require_function_calling=_require_fc,
        )
        adapter = selected.adapter
        ctx.model = adapter.model
        ctx.provider = adapter.provider_name
        ctx.route_hint = hint
    else:
        # 精确匹配模式（向后兼容）
        adapter = next(
            (a for a in adapters if a.model == request.model),
            None,
        )
        if not adapter:
            # 精确匹配失败时尝试 AutoRouter 兜底
            obs = _get_obs_plugin()
            logger.warning("Model '%s' not found, falling back to auto-router", request.model)
            selected = router_dep.select(
                [{"role": m.role, "content": m.content} for m in req.messages],
                hint="balanced",
                require_function_calling=_require_fc,
            )
            adapter = selected.adapter
            ctx.model = adapter.model
            ctx.provider = adapter.provider_name
        else:
            ctx.model = adapter.model
            ctx.provider = adapter.provider_name

    # ──────────────────────────────────────────────
    # 适配器调用 + Provider 自动 Fallback
    # 主 provider 失败时自动排除并重试下一个
    # ──────────────────────────────────────────────
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    failed_providers: set[str] = set()
    last_error: Exception | None = None

    for attempt in range(len(adapters)):
        try:
            # 检查 Circuit Breaker（如果当前 provider 已被熔断，这里直接抛异常）
            cb_plugin.check_provider(adapter.provider_name)

            # ── 流式路径 ──
            if request.stream:
                return StreamingResponse(
                    _stream_response(adapter, messages, req, ctx, cb_plugin, cache),
                    media_type="text/event-stream",
                )

            # ── 非流式路径 ──
            effective_max_tokens = req.effective_max_tokens()
            extra_payload = req.extra_payload()
            response = await adapter.chat_completions(
                messages=messages,
                temperature=req.temperature,
                max_tokens=effective_max_tokens,
                stream=False,
                **extra_payload,
            )
            cb_plugin.record_success(adapter.provider_name)
            router_dep.record_latency(adapter.provider_name, response.latency_ms)

            # 记录日志（ObservabilityPlugin.post_response）
            obs = _get_obs_plugin()
            response_dict = {
                "usage": {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.prompt_tokens + response.completion_tokens,
                },
            }
            await obs.post_response(ctx, response_dict)

            # 从原始响应提取 tool_calls（非流式路径）
            tool_calls = response.raw.get("choices", [{}])[0].get("message", {}).get("tool_calls")

            # 写入缓存
            cache_key = f"llm:cache:{req.hash_key()}"
            cached_choice: dict[str, Any] = {
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": "stop",
                "index": 0,
            }
            if tool_calls:
                cached_choice["message"]["tool_calls"] = tool_calls
            await cache.store_response(
                cache_key,
                {
                    "id": f"chatcmpl-{ctx.trace_id}",
                    "model": response.model,
                    "choices": [cached_choice],
                    "usage": {
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.prompt_tokens + response.completion_tokens,
                    },
                },
            )

            choice_payload: dict[str, Any] = {
                "index": 0,
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": "stop",
            }
            if tool_calls:
                choice_payload["message"]["tool_calls"] = tool_calls

            return ChatCompletionResponse(
                id=f"chatcmpl-{ctx.trace_id}",
                created=int(time.time()),
                model=response.model,
                choices=[choice_payload],
                usage=UsageInfo(
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.prompt_tokens + response.completion_tokens,
                ),
            )

        except Exception as e:
            cb_plugin.record_failure(adapter.provider_name)
            failed_providers.add(adapter.provider_name)
            last_error = e

            # 尝试 fallback 到下一个可用 provider
            remaining = [a for a in adapters if a.provider_name not in failed_providers]
            if not remaining:
                break  # 全部 provider 都失败了，抛出原始异常

            # 用 AutoRouter 选下一个（排除已失败的 provider）
            selected = router_dep.select(
                messages,
                hint="balanced",
                exclude_providers=failed_providers,
                require_function_calling=_require_fc,
            )
            failed_name = adapter.provider_name
            adapter = selected.adapter
            ctx.model = adapter.model
            ctx.provider = adapter.provider_name
            logger.warning(
                "Provider '%s' 失败，切换到 '%s'（第 %d/%d 次尝试）",
                failed_name,
                adapter.provider_name,
                attempt + 1,
                len(adapters),
            )
            continue

    # 所有 provider 都失败了
    obs = _get_obs_plugin()
    await obs.on_error(ctx, last_error)
    raise last_error


async def _stream_response(
    adapter: Any,
    messages: list[dict[str, str]],
    request: ChatCompletionRequest,
    ctx: RequestContext,
    cb_plugin: Any,
    cache: Any,
) -> AsyncIterator[str]:
    """Yield SSE stream chunks.

    Handles content chunks, tool_calls deltas, and usage metadata from the adapter.
    Tool calls are forwarded in the standard OpenAI streaming format
    (delta.tool_calls) so that LangChain's ChatOpenAI can parse them.
    Usage info is forwarded as choices=[] with usage field.
    """
    chunk_id = f"chatcmpl-{ctx.trace_id}"
    try:
        async for chunk in adapter.chat_completions_stream(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.effective_max_tokens(),
            **request.extra_payload(),
        ):
            # Try to parse as a structured marker (tool_calls or usage)
            if isinstance(chunk, str) and chunk.startswith("{"):
                try:
                    data = json.loads(chunk)
                    if data.get("type") == "tool_calls":
                        # Forward tool_calls delta in OpenAI streaming format
                        delta = {"tool_calls": data["tool_calls"]}
                        yield f"data: {json.dumps({'id': chunk_id, 'choices': [{'index': data.get('index', 0), 'delta': delta}]})}\n\n"
                        continue
                    elif data.get("type") == "usage":
                        usage_payload = {
                            "prompt_tokens": data.get("prompt_tokens", 0),
                            "completion_tokens": data.get("completion_tokens", 0),
                        }
                        yield f"data: {json.dumps({'id': chunk_id, 'choices': [], 'usage': usage_payload})}\n\n"
                        continue
                except json.JSONDecodeError:
                    pass
            # Plain text chunk — wrap as delta.content
            yield f"data: {json.dumps({'id': chunk_id, 'choices': [{'index': 0, 'delta': {'content': chunk}}]})}\n\n"
        yield f"data: {json.dumps({'id': chunk_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
        cb_plugin.record_success(adapter.provider_name)
    except Exception as e:
        cb_plugin.record_failure(adapter.provider_name)
        logger.warning(
            "SSE 流错误（provider: %s）: %s",
            adapter.provider_name,
            e,
            exc_info=True,
        )
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
        except Exception as e:
            logger.warning(
                "list_models 失败（adapter: %s）: %s",
                adapter.__class__.__name__,
                e,
            )
    return ModelList(data=models)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Placeholder for Prometheus metrics."""
    return {"status": "ok"}
