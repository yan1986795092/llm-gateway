"""Base adapter interface for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx


@dataclass
class ModelCapabilities:
    """Capabilities of a model."""

    model_id: str
    provider: str
    supports_streaming: bool = True
    supports_function_calling: bool = False
    max_tokens: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    typical_latency_ms: int = 1000
    quality_score: float = 0.5  # 0.0 - 1.0


@dataclass
class ChatResponse:
    """Unified chat completion response."""

    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    """Abstract base class for LLM provider adapters."""

    def __init__(self, api_key: str | None, model: str, **kwargs: Any):
        self.api_key = api_key or ""
        self.model = model
        self._client: httpx.AsyncClient | None = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier, e.g. 'openai', 'anthropic'."""
        raise NotImplementedError

    @abstractmethod
    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request."""
        raise NotImplementedError

    @abstractmethod
    async def list_models(self) -> list[Any]:
        """List available models for this provider."""
        raise NotImplementedError

    def get_capabilities(self) -> ModelCapabilities:
        """Return static capabilities for this adapter."""
        return ModelCapabilities(model_id=self.model, provider=self.provider_name)

    async def health_check(self) -> bool:
        """Return True if the provider is reachable."""
        try:
            await self.list_models()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "BaseAdapter":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
