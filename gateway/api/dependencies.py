"""FastAPI dependency injection."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from gateway.adapters.openai import OpenAIAdapter
from gateway.config.settings import Settings, get_settings
from gateway.core.router import AutoRouter
from gateway.plugins.cache import CachePlugin
from gateway.plugins.circuit_breaker import CircuitBreakerPlugin
from gateway.plugins.rate_limiter import RateLimiterPlugin


@lru_cache
def build_adapters() -> list[OpenAIAdapter]:
    """Build provider adapters from settings."""
    settings = get_settings()
    adapters: list[OpenAIAdapter] = []

    if settings.openai_api_key:
        adapters.append(
            OpenAIAdapter(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                base_url=settings.openai_base_url,
            )
        )
    return adapters


@lru_cache
def build_router() -> AutoRouter:
    adapters = build_adapters()
    return AutoRouter(adapters)


@lru_cache
def build_rate_limiter() -> RateLimiterPlugin:
    return RateLimiterPlugin()


@lru_cache
def build_cache_plugin() -> CachePlugin:
    return CachePlugin()


@lru_cache
def build_circuit_breaker() -> CircuitBreakerPlugin:
    return CircuitBreakerPlugin()
