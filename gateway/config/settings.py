"""Configuration management using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # App
    app_name: str = "llm-gateway"
    app_host: str = "0.0.0.0"
    app_port: int = 9010
    debug: bool = False

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # Cache TTL (seconds)
    cache_ttl: int = 300

    # OpenAI
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Zhipu AI（智谱，国内可用）
    zhipu_api_key: str | None = None
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    zhipu_model: str = "glm-4-flash"

    # Qwen（阿里通义千问，国内可用）
    qwen_api_key: str | None = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    # DeepSeek（OpenAI 协议兼容）
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # Anthropic
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-haiku-20241022"

    # Gemini
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Rate Limiting
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000

    # Circuit Breaker
    cb_failure_threshold: int = 3
    cb_error_rate_threshold: float = 0.3
    cb_window_seconds: int = 300
    cb_cooldown_seconds: int = 30

    # Router weights (simple task)
    router_w_cost_simple: float = 0.7
    router_w_latency_simple: float = 0.2
    router_w_quality_simple: float = 0.1

    # Router weights (complex task)
    router_w_cost_complex: float = 0.1
    router_w_latency_complex: float = 0.4
    router_w_quality_complex: float = 0.5

    # Logging
    log_level: str = "INFO"
    log_json_format: bool = False

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
