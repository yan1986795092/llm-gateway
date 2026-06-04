"""Plugin system for request/response middleware."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gateway.core.context import RequestContext
    from gateway.api.schemas import ChatCompletionRequest


class Plugin(ABC):
    """Base class for gateway plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def pre_request(
        self, ctx: RequestContext, request: ChatCompletionRequest
    ) -> ChatCompletionRequest | None:
        """Called before the request is processed.
        
        Return None to short-circuit (block/redirect the request).
        Return the (possibly modified) request to continue.
        """
        raise NotImplementedError

    @abstractmethod
    async def post_response(
        self, ctx: RequestContext, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Called after a successful response. Allows modification of response."""
        raise NotImplementedError

    @abstractmethod
    async def on_error(
        self, ctx: RequestContext, error: Exception
    ) -> None:
        """Called when an error occurs in the pipeline."""
        raise NotImplementedError
