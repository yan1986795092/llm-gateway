"""Redis-backed cache plugin for deduplication.

Graceful degradation: if Redis is unreachable, cache lookups silently miss
and writes are no-ops — the system continues without caching.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as redis

from gateway.api.schemas import ChatCompletionRequest
from gateway.core.context import RequestContext
from gateway.core.exceptions import GatewayError
from gateway.plugins.base import Plugin
from gateway.config.settings import get_settings

logger = logging.getLogger(__name__)


class CacheHit(GatewayError):
    """Internal exception used to short-circuit with cached data."""

    def __init__(self, cached_response: dict[str, Any]):
        super().__init__("cache hit", status_code=200, error_code="cached")
        self.cached_response = cached_response


class _NoopClient:
    """Stand-in when Redis is unreachable — all operations are no-ops."""

    async def get(self, _key: str) -> None:
        return None

    async def setex(self, _key: str, _ttl: int, _value: str) -> None:
        pass

    async def ping(self) -> bool:
        return False

    async def aclose(self) -> None:
        pass


class CachePlugin(Plugin):
    """Redis cache plugin — stores response by request hash.

    If Redis is unavailable at startup, the plugin degrades gracefully:
    all reads miss and all writes are silently dropped. A warning is
    emitted once so operators know caching is off.
    """

    def __init__(self, ttl: int | None = None):
        settings = get_settings()
        self.ttl = ttl or settings.cache_ttl
        self._client: redis.Redis | Any | None = None
        self._prefix = "llm:cache:"
        self._degraded = False

    async def _get_client(self) -> redis.Redis | Any:
        if self._client is not None:
            return self._client

        settings = get_settings()
        try:
            client = redis.from_url(
                settings.resolved_redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await client.ping()
            self._client = client
            logger.info("Redis cache connected at %s", settings.redis_url)
        except Exception as exc:
            logger.warning(
                "Redis unreachable at %s — cache degraded (reason: %s). "
                "Set REDIS_HOST=localhost in .env if not running in Docker.",
                settings.redis_url,
                exc,
            )
            self._client = _NoopClient()
            self._degraded = True
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

        # 租户 + 模型维度隔离：避免跨项目/跨模型命中旧答案（缓存污染）
        project = (ctx.project or "default")
        model = (request.model or "auto")
        cache_key = f"{self._prefix}{project}:{model}:{request.hash_key(model=model)}"
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
        if self._degraded:
            return
        try:
            client = await self._get_client()
            await client.setex(cache_key, self.ttl, json.dumps(response))
        except Exception:
            pass

    async def on_error(self, ctx: RequestContext, error: Exception) -> None:
        pass

    async def close(self) -> None:
        if self._client and not self._degraded:
            await self._client.aclose()
