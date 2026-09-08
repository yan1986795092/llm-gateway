"""FastAPI dependency injection."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from gateway.adapters.base import BaseAdapter
from gateway.adapters.openai import OpenAIAdapter
from gateway.config.settings import Settings, get_settings
from gateway.core.exceptions import AuthenticationError
from gateway.core.router import AutoRouter
from gateway.plugins.cache import CachePlugin
from gateway.plugins.circuit_breaker import CircuitBreakerPlugin
from gateway.plugins.rate_limiter import RateLimiterPlugin


def _load_providers_file(settings) -> list | None:
    """读取 providers.json（配置驱动）。文件不存在/解析失败返回 None（回退 .env）。"""
    import json as _json
    import os as _os
    candidates = [
        settings.gateway_providers_file,
        # providers.json 与 gateway/ 同目录（gateway/api/dependencies.py 上溯 3 层即仓库根）
        _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), settings.gateway_providers_file),
    ]
    for fp in candidates:
        if _os.path.isfile(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                return data if isinstance(data, list) else None
            except Exception:
                return None
    return None


@lru_cache
def build_adapters() -> list[BaseAdapter]:
    """Build all provider adapters.

    方案A（配置驱动）：优先读 providers.json（新增供应商只改该文件+重启，
    代码零触碰）；文件缺失时回退 .env 逐字段注册（legacy 兼容）。
    """
    settings = get_settings()
    adapters: list[BaseAdapter] = []

    file_cfgs = _load_providers_file(settings)
    if file_cfgs is not None:
        for p in file_cfgs:
            provider = (p.get("provider") or "").strip()
            api_key = (p.get("api_key") or "").strip()
            if not provider or not api_key:
                continue
            adapters.append(OpenAIAdapter(
                api_key=api_key,
                model=(p.get("model") or "").strip() or None,
                base_url=(p.get("base_url") or "").strip() or None,
                provider_name=provider,
            ))
        return adapters

    # ── legacy：.env 逐字段注册（providers.json 缺失时） ──

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

    # 私有化微调模型（本地 llama.cpp）
    if settings.private_qwen_api_key:
        adapters.append(
            OpenAIAdapter(
                api_key=settings.private_qwen_api_key,
                model=settings.private_qwen_model,
                base_url=settings.private_qwen_base_url,
                provider_name="private_qwen",
            )
        )

    # AiHubMix（OpenAI 兼容）
    if settings.aihubmix_api_key:
        adapters.append(
            OpenAIAdapter(
                api_key=settings.aihubmix_api_key,
                model=settings.aihubmix_model,
                base_url=settings.aihubmix_base_url,
                provider_name="aihubmix",
            )
        )

    return adapters


def resolve_project(request: Request) -> str | None:
    """通过 Authorization Bearer key 解析项目。

    返回项目 id；未配置 gateway_projects 时恒为 "default"。
    """
    settings = get_settings()
    if not settings.gateway_projects:
        return "default"
    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token:
        return None
    for entry in settings.gateway_projects.split(","):
        entry = entry.strip()
        if "=" in entry:
            proj, key = entry.split("=", 1)
            if key.strip() == token:
                return proj.strip()
    return None


def require_project(request: Request) -> str:
    """鉴权入口：未配置鉴权则默认项目；配置后校验 key，非法抛 401。"""
    settings = get_settings()
    project = resolve_project(request)
    if settings.gateway_auth_required and settings.gateway_projects and project is None:
        raise AuthenticationError("invalid or missing API key")
    return project or "default"


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
