"""Redis-backed cache plugin for deduplication."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis

from gateway.api.schemas import ChatCompletionRequest
from gateway.core.context import RequestContext
from gateway.core.exceptions import GatewayError
from gateway.plugins.base import Plugin
from gateway.config.settings import get_settings


class CacheHit(GatewayError):
    """Internal exception used to short-circuit with cached data."""

    def __init__(self, cached_response: dict[str, Any]):
        super().__init__("cache hit", status_code=200, error_code="cached")
        self.cached_response = cached_response


class CachePlugin(Plugin):
    """Redis cache plugin — stores response by request hash."""

    def __init__(self, ttl: int | None = None):
        settings = get_settings()
        self.ttl = ttl or settings.cache_ttl
        self._client: redis.Redis | None = None
        self._prefix = "llm:cache:"

    async def _get_client(self) -> redis.Redis:
        if self._client is None:
            settings = get_settings()
            self._client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    @property
    def name(self) -> str:
        return "cache"

    async def pre_request(
        self, ctx: RequestContext, request: ChatCompletionRequest
    ) -> ChatCompletionRequest | None:
        """Check cache. If hit, raise CacheHit to short-circuit."""
        if request.stream:
            return request

        cache_key = self._prefix + request.hash_key()
        try:
            client = await self._get_client()
            cached = await client.get(cache_key)
            if cached:
                raise CacheHit(json.loads(cached))
        except CacheHit:
            raise
        except Exception:
            pass
        return request

    async def post_response(
        self, ctx: RequestContext, response: dict[str, Any]
    ) -> dict[str, Any]:
        return response

    async def store_response(self, cache_key: str, response: dict[str, Any]) -> None:
        """Store a response in cache. Call this from the pipeline."""
        try:
            client = await self._get_client()
            await client.setex(cache_key, self.ttl, json.dumps(response))
        except Exception:
            pass

    async def on_error(self, ctx: RequestContext, error: Exception) -> None:
        pass

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
