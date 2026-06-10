"""FastAPI dependency injection."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from gateway.adapters.base import BaseAdapter
from gateway.adapters.openai import OpenAIAdapter
from gateway.config.settings import Settings, get_settings
from gateway.core.router import AutoRouter
from gateway.plugins.cache import CachePlugin
from gateway.plugins.circuit_breaker import CircuitBreakerPlugin
from gateway.plugins.rate_limiter import RateLimiterPlugin


@lru_cache
def build_adapters() -> list[BaseAdapter]:
    """Build all provider adapters from settings."""
    settings = get_settings()
    adapters: list[BaseAdapter] = []

    # Zhipu AI（智谱，国内可用）
    if settings.zhipu_api_key:
        adapters.append(
            OpenAIAdapter(
                api_key=settings.zhipu_api_key,
                model=settings.zhipu_model,
                base_url=settings.zhipu_base_url,
                provider_name="zhipu",
            )
        )

    # Qwen（阿里通义千问，国内可用）
    if settings.qwen_api_key:
        adapters.append(
            OpenAIAdapter(
                api_key=settings.qwen_api_key,
                model=settings.qwen_model,
                base_url=settings.qwen_base_url,
                provider_name="qwen",
            )
        )

    # DeepSeek（OpenAI 协议兼容，国内可用）
    if settings.deepseek_api_key:
        adapters.append(
            OpenAIAdapter(
                api_key=settings.deepseek_api_key,
                model=settings.deepseek_model,
                base_url=settings.deepseek_base_url,
                provider_name="deepseek",
            )
        )

    # OpenAI（需代理）
    if settings.openai_api_key:
        adapters.append(
            OpenAIAdapter(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                base_url=settings.openai_base_url,
                provider_name="openai",
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
