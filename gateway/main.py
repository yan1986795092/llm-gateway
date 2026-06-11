"""FastAPI 入口"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from gateway.api.routes import router
from gateway.core.exceptions import GatewayError, CircuitBreakerOpenError, RateLimitError
from gateway.logging_shared.config import setup, get_logger
from gateway.logging_shared.middleware import LoggingMiddleware
from gateway.logging_shared.unified_writer import UnifiedEsWriter
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

    # 启动 UnifiedEsWriter
    try:
        writer = UnifiedEsWriter.get_instance()
        await writer.start()
    except Exception as exc:
        logger.warning("UnifiedEsWriter start failed (non-fatal): %s", exc)

    yield

    # 关闭 UnifiedEsWriter
    try:
        writer = UnifiedEsWriter.get_instance()
        await writer.stop()
    except Exception as exc:
        logger.warning("UnifiedEsWriter stop error: %s", exc)

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

    logger = get_logger("gateway")
    logger.warning(
        "网关异常",
        error_code=exc.error_code,
        status=exc.status_code,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        headers=headers,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "GatewayError",
                "code": exc.error_code,
                "details": getattr(exc, "details", {}),
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    field_errors = {}
    for err in errors:
        loc = ".".join(str(l) for l in err["loc"])
        field_errors[loc] = err["msg"]

    logger = get_logger("gateway")
    logger.warning("参数校验失败: %s", field_errors, path=request.url.path)

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "message": "Request validation failed",
                "type": "RequestValidationError",
                "code": "VALIDATION_ERROR",
                "details": {"fields": field_errors},
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger = get_logger("gateway")
    logger.warning("HTTP 异常 %d: %s", exc.status_code, exc.detail, path=request.url.path)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.detail,
                "type": "HTTPException",
                "code": f"HTTP_{exc.status_code}",
                "details": {},
            }
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger = get_logger("gateway")
    logger.error(
        "未处理异常: %s",
        exc,
        exc_info=True,
        path=request.url.path,
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "message": "发生内部错误，请稍后重试。",
                "type": type(exc).__name__,
                "code": "INTERNAL_ERROR",
                "details": {},
            }
        },
    )
