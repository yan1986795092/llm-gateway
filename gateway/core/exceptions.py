"""Gateway-specific exceptions."""

from __future__ import annotations


class GatewayError(Exception):
    """Base exception for all gateway errors."""

    def __init__(self, message: str, status_code: int = 500, error_code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or "gateway_error"


class RateLimitError(GatewayError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int | None = None):
        super().__init__(message, status_code=429, error_code="rate_limit_exceeded")
        self.retry_after = retry_after


class CircuitBreakerOpenError(GatewayError):
    """Circuit breaker is open."""

    def __init__(self, provider: str, cooldown: int):
        super().__init__(
            f"Circuit breaker open for provider '{provider}'",
            status_code=503,
            error_code="circuit_breaker_open",
        )
        self.provider = provider
        self.cooldown = cooldown


class AllProvidersUnavailableError(GatewayError):
    """All providers are unavailable."""

    def __init__(self):
        super().__init__(
            "All LLM providers are unavailable",
            status_code=503,
            error_code="all_providers_unavailable",
        )


class ProviderError(GatewayError):
    """Provider returned an error."""

    def __init__(self, provider: str, message: str, status_code: int = 502):
        super().__init__(f"{provider}: {message}", status_code=status_code, error_code="provider_error")
        self.provider = provider


class ValidationError(GatewayError):
    """Request validation failed."""

    def __init__(self, message: str):
        super().__init__(message, status_code=422, error_code="validation_error")


class AuthenticationError(GatewayError):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401, error_code="authentication_error")
