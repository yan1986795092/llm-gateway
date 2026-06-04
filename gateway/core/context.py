"""Request context — trace_id, span_id, timing."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class RequestContext:
    """Per-request context propagated through the gateway."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started_at: float = field(default_factory=time.perf_counter)
    model: str | None = None
    provider: str | None = None
    source_app: str | None = None

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "elapsed_ms": round(self.elapsed_ms(), 2),
            "model": self.model,
            "provider": self.provider,
        }
