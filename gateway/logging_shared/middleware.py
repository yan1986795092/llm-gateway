"""FastAPI 中间件: per-request access-log + trace span (LLM Gateway 版).

每个 HTTP 请求:
1. 生成/提取 trace_id
2. 执行请求
3. 输出一行 access-log
4. 通过 UnifiedEsWriter 写 traces-* span
"""

import time
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .fields import TRACE_ID
from .config import get_logger
from .console import log_http_line
from .unified_writer import UnifiedEsWriter

TRACE_ID_HEADER = "X-Trace-Id"


def _get_or_create_trace_id(request: Request) -> str:
    trace_id = request.headers.get(TRACE_ID_HEADER)
    if trace_id:
        return trace_id
    return uuid.uuid4().hex[:16]


class LoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI 中间件: 单行 access-log + Jaeger span 写入 UnifiedEsWriter."""

    def __init__(self, app: ASGIApp, service_name: str):
        super().__init__(app)
        self.service_name = service_name
        self.logger = get_logger("http")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = _get_or_create_trace_id(request)
        span_id = uuid.uuid4().hex[:8]
        start_ns = time.perf_counter_ns()
        method = request.method
        path = request.url.path

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            **{TRACE_ID: trace_id, "service": self.service_name}
        )

        try:
            response = await call_next(request)
            elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2)
            elapsed_us = int((time.perf_counter_ns() - start_ns) / 1000)
            status_code = response.status_code

            # Single access-log line (replaces http_request_start + http_request_end)
            log_http_line(
                level="ERROR" if status_code >= 400 else "INFO",
                method=method,
                path=path,
                status=status_code,
                duration_ms=elapsed_ms,
                trace_id=trace_id,
            )

            # Write Jaeger span via UnifiedEsWriter
            span = {
                "traceID": trace_id,
                "spanID": span_id,
                "operationName": f"{method} {path}",
                "references": [],
                "tags": [
                    {"key": "http.method", "vType": "STRING", "vStr": method},
                    {"key": "http.url", "vType": "STRING", "vStr": str(request.url)},
                    {"key": "http.status_code", "vType": "INT64", "vInt": status_code},
                    {"key": "service.name", "vType": "STRING", "vStr": self.service_name},
                ],
                "logs": [],
                "process": {"serviceName": self.service_name, "tags": []},
                "startTime": int((time.time() * 1_000_000) - elapsed_us),
                "startTimeMillis": int((time.time() * 1000) - (elapsed_us // 1000)),
                "durationMicros": elapsed_us,
                "fields": {"traceID": trace_id, "spanID": span_id},
            }
            UnifiedEsWriter.get_instance().write("traces", span)

            response.headers[TRACE_ID_HEADER] = trace_id
            return response

        except Exception as exc:
            elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2)
            elapsed_us = int((time.perf_counter_ns() - start_ns) / 1000)

            log_http_line(
                level="ERROR",
                method=method,
                path=path,
                status=500,
                duration_ms=elapsed_ms,
                trace_id=trace_id,
                error=f"{type(exc).__name__}: {exc}",
            )

            span = {
                "traceID": trace_id,
                "spanID": span_id,
                "operationName": f"{method} {path}",
                "startTime": int((time.time() * 1_000_000) - elapsed_us),
                "startTimeMillis": int((time.time() * 1000) - (elapsed_us // 1000)),
                "durationMicros": elapsed_us,
                "tags": [
                    {"key": "http.method", "vType": "STRING", "vStr": method},
                    {"key": "http.url", "vType": "STRING", "vStr": str(request.url)},
                    {"key": "http.status_code", "vType": "INT64", "vInt": 500},
                    {"key": "error", "vType": "BOOL", "vBool": True},
                    {"key": "error.kind", "vType": "STRING", "vStr": type(exc).__name__},
                    {"key": "error.message", "vType": "STRING", "vStr": str(exc)[:500]},
                    {"key": "service.name", "vType": "STRING", "vStr": self.service_name},
                ],
                "logs": [],
                "process": {"serviceName": self.service_name, "tags": []},
                "fields": {"traceID": trace_id, "spanID": span_id},
            }
            UnifiedEsWriter.get_instance().write("traces", span)
            raise
