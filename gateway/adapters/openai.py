"""OpenAI-compatible adapter using httpx."""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from gateway.adapters.base import BaseAdapter, ChatResponse, ModelCapabilities
from gateway.core.exceptions import ProviderError
from gateway.config.settings import get_settings


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI-compatible APIs (OpenAI, Zhipu, Azure, local proxies)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        provider_name: str = "openai",
        **kwargs: Any,
    ):
        settings = get_settings()
        super().__init__(
            api_key=api_key or settings.openai_api_key or "",
            model=model or settings.openai_model,
        )
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self._provider_name = provider_name
        self._client: httpx.AsyncClient | None = None

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._client

    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        start = time.perf_counter()
        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise ProviderError(self.provider_name, str(e)) from e
        except httpx.RequestError as e:
            raise ProviderError(self.provider_name, f"Request failed: {e}") from e

        latency_ms = (time.perf_counter() - start) * 1000

        if stream:
            content = ""
        else:
            choice = data["choices"][0]["message"]
            content = choice.get("content") or ""

        usage = data.get("usage", {})
        return ChatResponse(
            content=content,
            model=data.get("model", self.model),
            provider=self.provider_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            raw=data,
        )

    async def chat_completions_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield streamed response chunks.

        Yields:
            - Content text chunks (plain str) for normal text deltas.
            - JSON marker strings for structured data:
              * {"type": "tool_calls", "tool_calls": [...], "index": N}
                when the delta contains tool_calls.
              * {"type": "usage", "prompt_tokens": N, "completion_tokens": M}
                as a final marker from the upstream API.

        Tool calls from the upstream are forwarded so that _stream_response
        can reconstruct proper SSE events with delta.tool_calls — this is
        required for LangChain's ChatOpenAI to parse tool calls from streamed
        responses.
        """
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        usage_info: dict[str, int] = {}

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    # Capture usage info from final chunk (OpenAI format:
                    # choices=[] with usage field, or choices[n].finish_reason=stop with delta={})
                    if "usage" in chunk and chunk["usage"]:
                        usage_info = {
                            "prompt_tokens": chunk["usage"].get("prompt_tokens", 0),
                            "completion_tokens": chunk["usage"].get("completion_tokens", 0),
                        }
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                        # Forward tool_calls delta as a JSON marker so that
                        # _stream_response can build proper SSE with delta.tool_calls.
                        tool_calls = delta.get("tool_calls")
                        if tool_calls:
                            yield json.dumps({
                                "type": "tool_calls",
                                "tool_calls": tool_calls,
                                "index": choices[0].get("index", 0),
                            })

        # Emit usage as final marker if available
        if usage_info.get("prompt_tokens") or usage_info.get("completion_tokens"):
            yield json.dumps({"type": "usage", **usage_info})

    async def list_models(self) -> list[Any]:
        from gateway.api.schemas import ModelInfo
        client = self._get_client()
        try:
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()
            return [
                ModelInfo(id=m["id"], owned_by=m.get("owned_by", self.provider_name))
                for m in data.get("data", [])
                if m.get("id") and not m["id"].startswith("ft:")
            ]
        except httpx.HTTPStatusError as e:
            raise ProviderError(self.provider_name, f"list_models failed: {e}") from e

    def get_capabilities(self) -> ModelCapabilities:
        caps = _PROVIDER_CAPS.get(self._provider_name, _PROVIDER_CAPS["openai"])
        return ModelCapabilities(
            model_id=self.model,
            provider=self.provider_name,
            supports_streaming=True,
            supports_function_calling=caps["supports_function_calling"],
            max_tokens=caps["max_tokens"],
            cost_per_1k_input=caps["cost_per_1k_input"],
            cost_per_1k_output=caps["cost_per_1k_output"],
            typical_latency_ms=caps["typical_latency_ms"],
            quality_score=caps["quality_score"],
        )


# Per-provider model capabilities
# ⚠️ 注意：glm-4-flash（轻量版）不支持 function calling！
# 如需启用 zhipu function calling，换用 glm-4 / glm-4-plus / glm-4-alltools 等全功能模型。
_PROVIDER_CAPS: dict[str, dict[str, Any]] = {
    "zhipu": {
        # glm-4-flash 是轻量版，bind_tools 后会输出裸参数文本而非 tool_call，
        # 导致 conversation engine 崩溃。请换用 glm-4 / glm-4-plus。
        "supports_function_calling": False,
        "max_tokens": 8192,
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.01,
        "typical_latency_ms": 1200,
        "quality_score": 0.75,
    },
    "openai": {
        "supports_function_calling": True,
        "max_tokens": 16384,
        "cost_per_1k_input": 0.15,
        "cost_per_1k_output": 0.60,
        "typical_latency_ms": 800,
        "quality_score": 0.9,
    },
    "deepseek": {
        "supports_function_calling": True,
        "max_tokens": 8192,
        "cost_per_1k_input": 0.014,
        "cost_per_1k_output": 0.028,
        "typical_latency_ms": 900,
        "quality_score": 0.82,
    },
    "aihubmix": {
        "supports_function_calling": True,
        "max_tokens": 16384,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "typical_latency_ms": 1200,
        "quality_score": 0.8,
    },
    "private_qwen": {
        "supports_function_calling": False,
        "max_tokens": 8192,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "typical_latency_ms": 900,
        "quality_score": 0.7,
    },
}
