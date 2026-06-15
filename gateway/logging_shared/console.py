"""Access-log style console output — one line per HTTP request (中文标签版).

Usage:
    from .console import log_http_line
    log_http_line("INFO", "POST", "/api/chat", 200, 152.3, "abc123")
    # Output: INFO:  POST /api/chat 200 152ms [trace:abc123]
"""

from __future__ import annotations

import logging
import sys

_console_handler: logging.Handler | None = None
_console_logger: logging.Logger | None = None

# 中文标签映射
_LEVEL_LABELS = {
    "INFO": "信息",
    "WARNING": "警告",
    "ERROR": "错误",
}

_AGENT_LABELS = {
    "agent_start": "▶ 开始",
    "agent_finish": "✓ 完成",
    "tool_call": "⚙ 调用",
    "tool_result": "📎 结果",
    "thought": "💭 思考",
    "error": "✗ 错误",
}


def _ensure_logger() -> logging.Logger:
    """Lazy-init the access-log logger with a clean format."""
    global _console_logger, _console_handler
    if _console_logger is not None:
        return _console_logger

    _console_logger = logging.getLogger("access")
    _console_logger.setLevel(logging.INFO)
    _console_logger.propagate = False

    if not _console_logger.handlers:
        _console_handler = logging.StreamHandler(sys.stderr)
        _console_handler.setFormatter(
            logging.Formatter("%(levelname)s:  %(message)s")
        )
        _console_logger.addHandler(_console_handler)

    return _console_logger


def log_http_line(
    level: str,
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    trace_id: str = "",
    error: str = "",
    request_summary: str = "",
    response_summary: str = "",
) -> None:
    """Print one access-log-style line with Chinese labels."""
    parts = [f"{method} {path} {status} {duration_ms:.0f}ms"]
    if request_summary:
        parts.append(f"入参:{request_summary}")
    if response_summary:
        parts.append(f"出参:{response_summary}")
    if trace_id:
        parts.append(f"[trace:{trace_id[:12]}]")
    if error:
        parts.append(f"错误:{error}")

    line = " ".join(parts)
    logger = _ensure_logger()
    level_label = _LEVEL_LABELS.get(level, level)
    if level == "ERROR":
        logger.error(f"[{level_label}] {line}")
    elif level == "WARNING":
        logger.warning(f"[{level_label}] {line}")
    else:
        logger.info(f"[{level_label}] {line}")


def log_agent_line(event_type: str, agent_name: str = "", **details) -> None:
    """Print a compact one-line agent event summary with Chinese labels."""
    label = _AGENT_LABELS.get(event_type, event_type)
    parts = [label]
    if agent_name:
        parts.append(agent_name)
    for key, value in details.items():
        if value is not None:
            parts.append(f"{key}={value}")

    logger = _ensure_logger()
    logger.info("  ".join(parts))
