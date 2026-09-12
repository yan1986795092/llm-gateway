"""统一 structlog 配置（LLM Gateway 版）"""

import logging
import sys
from typing import Literal

import structlog

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_configured = False


def setup(
    log_level: LogLevel = "INFO",
    log_json: bool = False,
    service_name: str = "llm-gateway",
) -> None:
    """配置 structlog（两服务启动时各调用一次）。"""
    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        _add_service_context,
    ]

    if log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 追加文件输出：控制台没有时也可查 gateway.log（含 traceback）
    try:
        from pathlib import Path as _P
        _log_file = _P(__file__).resolve().parent.parent.parent / "gateway.log"
        _fh = logging.FileHandler(_log_file, encoding="utf-8", delay=True)
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(_fh)
    except Exception:
        pass
    for _name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        _l = logging.getLogger(_name)
        if not _l.handlers:
            _h = logging.StreamHandler(sys.stderr)
            _h.setFormatter(logging.Formatter("%(message)s"))
            _l.addHandler(_h)
        _l.setLevel(level)

    for _name in ["elasticsearch", "elastic_transport", "opentelemetry",
                  "urllib3", "httpx", "httpcore"]:
        logging.getLogger(_name).setLevel(logging.WARNING)


def _add_service_context(logger, method_name, event_dict):
    """注入 service_name 到每个日志事件。"""
    event_dict["service"] = event_dict.get("service", "llm-gateway")
    return event_dict


def get_logger(name: str | None = None):
    """获取 logger 实例。"""
    if name:
        return structlog.get_logger().bind(logger_name=name)
    return structlog.get_logger()
