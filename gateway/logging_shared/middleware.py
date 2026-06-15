"""FastAPI middleware: access-log with request/response summary (LLM Gateway).

每个 HTTP 请求:
1. 生成/提取 trace_id
2. 读取请求体摘要（模型名、消息数等）
3. 执行请求
4. 输出一行 access-log 含入参出参摘要
5. 通过 UnifiedEsWriter 写 traces-* span
"""

from __future__ import annotations

import json
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


def _summarize_request_body(method: str, body: bytes) -> str:
    """提取请求体摘要：对 chat/completions 显示 model + messages 数，其余显示前 120 字符"""
    if method not in ("POST", "PUT", "PATCH") or not body:
        return ""
    try:
        data = json.loads(body)
        if "model" in data or "messages" in data:
            model = data.get("model", "")
            msg_count = len(data.get("messages", []))
            parts = []
            if model:
                parts.append(model)
            if msg_count:
                parts.append(f"{msg_count}msgs")
            return " ".join(parts)
        text = body.decode("utf-8", errors="replace")[:120].replace("\n", " ").strip()
        return text
    except (json.JSONDecodeError, UnicodeDecodeError):
        text = body.decode("utf-8", errors="replace")[:120].replace("\n", " ").strip()
        return text


def _summarize_response(response: Response) -> str:
    """提取响应摘要（content-type + content-length）"""
    ct = response.headers.get("content-type", "")
    cl = response.headers.get("content-length", "")
    parts = []
    if "text/event-stream" in ct:
        parts.append("SSE")
    elif "application/json" in ct:
        parts.append("json")
    elif ct:
        parts.append(ct.split(";")[0])
    if cl:
        size = int(cl)
        if size < 1024:
            parts.append(f"{size}B")
        elif size < 1024 * 1024:
            parts.append(f"{size/1024:.0f}KB")
        else:
            parts.append(f"{size/1024/1024:.1f}MB")
    return " ".join(parts)


class LoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI 中间件: access-log with request/response summary."""

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

        # 读取请求体摘要（跳过 SSE 流式路径，避免 BaseHTTPMiddleware 的 receive 冲突）
        request_summary = ""
        if method in ("POST", "PUT", "PATCH") and "/chat/completions" not in path:
            try:
                body_bytes = await request.body()
                request_summary = _summarize_request_body(method, body_bytes)
            except Exception:
                pass

        try:
            response = await call_next(request)
            elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2)
            elapsed_us = int((time.perf_counter_ns() - start_ns) / 1000)
            status_code = response.status_code

            response_summary = _summarize_response(response)

            log_http_line(
                level="ERROR" if status_code >= 500 else ("WARNING" if status_code >= 400 else "INFO"),
                method=method,
                path=path,
                status=status_code,
                duration_ms=elapsed_ms,
                trace_id=trace_id,
                request_summary=request_summary,
                response_summary=response_summary,
            )

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
            await UnifiedEsWriter.get_instance().write("traces", span)

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
                request_summary=request_summary,
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
            await UnifiedEsWriter.get_instance().write("traces", span)
            raise
