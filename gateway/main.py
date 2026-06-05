"""FastAPI 入口"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gateway.api.routes import router
from gateway.core.exceptions import GatewayError, CircuitBreakerOpenError, RateLimitError
from gateway.logging_shared.config import setup, get_logger
from gateway.logging_shared.middleware import LoggingMiddleware
from gateway.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup(
        log_level=settings.log_level,
        log_json=settings.log_json_format,
        service_name="llm-gateway",
    )
    logger = get_logger("gateway")
    logger.info("gateway_startup", **({"message": "LLM Gateway 启动"}))
    yield
    logger = get_logger("gateway")
    logger.info("gateway_shutdown", **({"message": "LLM Gateway 关闭"}))


app = FastAPI(
    title="LLM Gateway",
    description="生产级 LLM 网关",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    LoggingMiddleware,
    service_name="llm-gateway",
)

app.include_router(router)


@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
    headers = {}
    if isinstance(exc, RateLimitError) and exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    if isinstance(exc, CircuitBreakerOpenError):
        headers["Retry-After"] = str(exc.cooldown)

    return JSONResponse(
        status_code=exc.status_code,
        headers=headers,
        content={
            "error": {
                "message": exc.message,
                "code": exc.error_code,
            }
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger = get_logger("gateway")
    logger.error(
        "unhandled_exception",
        **{
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "内部服务错误",
                "code": "internal_error",
            }
        },
    )
