"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gateway.api.routes import router
from gateway.core.exceptions import (
    CircuitBreakerOpenError,
    GatewayError,
    RateLimitError,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="LLM Gateway",
    description="Production-grade LLM gateway with auto-routing, rate limiting, circuit breaker, and Redis cache",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
    """Convert GatewayError to proper HTTP responses."""
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
    """Catch-all for unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "code": "internal_error",
            }
        },
    )
