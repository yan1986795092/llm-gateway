"""Auto-routing engine with 3D weighted scoring."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

from gateway.adapters.base import BaseAdapter, ModelCapabilities
from gateway.config.settings import get_settings


@dataclass
class RoutingWeights:
    w_cost: float
    w_latency: float
    w_quality: float

    @classmethod
    def for_task(cls, task_type: str) -> "RoutingWeights":
        s = get_settings()
        if task_type == "simple":
            return cls(
                w_cost=s.router_w_cost_simple,
                w_latency=s.router_w_latency_simple,
                w_quality=s.router_w_quality_simple,
            )
        elif task_type == "complex":
            return cls(
                w_cost=s.router_w_cost_complex,
                w_latency=s.router_w_latency_complex,
                w_quality=s.router_w_quality_complex,
            )
        else:
            return cls(w_cost=0.4, w_latency=0.3, w_quality=0.3)


@dataclass
class ModelScore:
    adapter: BaseAdapter
    capabilities: ModelCapabilities
    score: float
    task_type: str


class AutoRouter:
    """3D weighted scoring router for automatic model selection."""

    def __init__(self, adapters: list[BaseAdapter]):
        self._adapters = adapters
        self._latency_windows: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()
        self._cost_max = 1.0

    def _normalize_cost(self, caps: ModelCapabilities) -> float:
        total = caps.cost_per_1k_input + caps.cost_per_1k_output
        if self._cost_max == 0:
            return 0.5
        return max(0.0, 1.0 - (total / self._cost_max))

    def _normalize_latency(self, caps: ModelCapabilities, provider: str) -> float:
        with self._lock:
            window = self._latency_windows.get(provider, [])
            if window:
                sorted_latencies = sorted(window)
                p95_idx = max(0, int(len(sorted_latencies) * 0.95) - 1)
                p95 = sorted_latencies[p95_idx]
            else:
                p95 = caps.typical_latency_ms
        max_latency = 5000.0
        return max(0.0, 1.0 - (p95 / max_latency))

    def _record_latency(self, provider: str, latency_ms: float) -> None:
        with self._lock:
            window = self._latency_windows[provider]
            window.append(latency_ms)
            if len(window) > 100:
                window.pop(0)

    def _detect_task_type(self, messages: list[dict[str, str]]) -> str:
        """Detect task type from messages content."""
        content = " ".join(m.get("content", "") for m in messages).lower()
        code_indicators = ["def ", "function", "class ", "import ", "```", "code", "python", "javascript"]
        complex_indicators = ["analyze", "compare", "evaluate", "explain in detail", "research"]
        simple_indicators = ["hi", "hello", "thanks", "what is", "simple", "briefly"]

        code_score = sum(1 for i in code_indicators if i in content)
        complex_score = sum(1 for i in complex_indicators if i in content)
        simple_score = sum(1 for i in simple_indicators if i in content)

        if code_score >= 2:
            return "complex"
        if complex_score >= 2:
            return "complex"
        if simple_score >= 2:
            return "simple"
        return "balanced"

    def _score_adapter(
        self, adapter: BaseAdapter, caps: ModelCapabilities, task_type: str
    ) -> float:
        weights = RoutingWeights.for_task(task_type)
        norm_cost = self._normalize_cost(caps)
        norm_latency = self._normalize_latency(caps, caps.provider)
        quality = caps.quality_score

        return (
            weights.w_cost * norm_cost
            + weights.w_latency * norm_latency
            + weights.w_quality * quality
        )

    _HINT_TO_TASK_TYPE = {
        "planning": "complex",
        "creative": "complex",
        "analysis": "complex",
        "code": "complex",
        "simple": "simple",
        "balanced": "balanced",
    }

    def select(
        self,
        messages: list[dict[str, str]],
        exclude_providers: set[str] | None = None,
        hint: str = "",
        require_function_calling: bool = False,
    ) -> ModelScore:
        """Select the best adapter for the given request.

        Args:
            messages: Chat messages to analyze.
            exclude_providers: Set of provider names to exclude.
            hint: Optional route hint from X-Route-Hint header.
                  Overrides task_type auto-detection.
                  Values: planning, creative, analysis, code, simple, balanced.
            require_function_calling: If True, only select adapters that support
                  function calling (tools). Prevents routing to models like
                  glm-4-flash that lack tool support, which would cause
                  bind_tools failures downstream.
        """
        task_type = self._HINT_TO_TASK_TYPE.get(hint, "")
        if not task_type:
            task_type = self._detect_task_type(messages)

        candidates: list[tuple[BaseAdapter, ModelCapabilities]] = []
        for adapter in self._adapters:
            if exclude_providers and adapter.provider_name in exclude_providers:
                continue
            caps = adapter.get_capabilities()
            if require_function_calling and not caps.supports_function_calling:
                continue
            candidates.append((adapter, caps))

        if not candidates:
            raise RuntimeError(
                "No available providers"
                if not require_function_calling
                else "No providers support function calling"
            )

        scores: list[ModelScore] = []
        for adapter, caps in candidates:
            score = self._score_adapter(adapter, caps, task_type)
            scores.append(ModelScore(adapter, caps, score, task_type))

        best = max(scores, key=lambda s: s.score)
        return best

    def record_latency(self, provider: str, latency_ms: float) -> None:
        self._record_latency(provider, latency_ms)
