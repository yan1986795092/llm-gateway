"""Pydantic schemas for OpenAI-compatible API."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[Message]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None

    def effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens if self.max_completion_tokens is not None else self.max_tokens

    def extra_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.n is not None:
            payload["n"] = self.n
        if self.stop is not None:
            payload["stop"] = self.stop
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.user is not None:
            payload["user"] = self.user
        if self.tools is not None:
            payload["tools"] = self.tools
        if self.tool_choice is not None:
            payload["tool_choice"] = self.tool_choice
        if self.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = self.parallel_tool_calls
        return payload

    def hash_key(self, model: str | None = None) -> str:
        """Generate a cache key from messages + params + model.

        model 必须纳入 key：同一批消息在不同 model（含 "auto"）下产出可能
        完全不同，缺 model 维度会跨模型命中旧答案（缓存污染，Z4 坑）。
        """
        data = {
            "model": model or self.model or "auto",
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.effective_max_tokens(),
            "tools": self.tools,
            "tool_choice": self.tool_choice,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int | None = None
    owned_by: str = "gateway"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
