"""Centralised error taxonomy and helpers for Highest Volatility."""

from __future__ import annotations

from collections import Counter
from enum import Enum
from threading import Lock
from typing import Any, Mapping, MutableMapping, Type


class ErrorCode(str, Enum):
    """Stable identifiers for error categories used across the project."""

    DATA_SOURCE = "data_source"
    CACHE = "cache"
    VALIDATION = "validation"
    COMPUTE = "compute"
    INTEGRATION = "integration"
    CONFIG = "config"
    UNKNOWN = "unknown"


_SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "apikey",
    "api_key",
    "authorization",
    "auth",
    "credential",
}
_REDACTED = "***REDACTED***"


def _coerce(value: Any) -> Any:
    """Return a JSON-serialisable representation for ``value``."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce(v) for v in value]
    return repr(value)


def sanitize_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a shallow copy of ``context`` with sensitive values redacted."""

    if not context:
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in context.items():
        lowered = key.lower()
        if any(token in lowered for token in _SENSITIVE_KEYS):
            sanitized[key] = _REDACTED
        else:
            sanitized[key] = _coerce(value)
    return sanitized


class HVError(Exception):
    """Base class for structured application errors."""

    code: ErrorCode
    user_message: str
    context: MutableMapping[str, Any]

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | str = ErrorCode.UNKNOWN,
        user_message: str | None = None,
        context: Mapping[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.code = ErrorCode(code)
        self.user_message = user_message or message
        self.context = dict(context or {})
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def add_context(self, **context: Any) -> "HVError":
        """Attach additional context to the error in-place."""

        for key, value in context.items():
            if value is not None:
                self.context[key] = value
        return self

    def with_user_message(self, message: str) -> "HVError":
        """Override the safe user-facing message and return ``self``."""

        self.user_message = message
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable payload describing the error."""

        payload: dict[str, Any] = {
            "code": self.code.value,
            "message": self.user_message,
            "type": self.__class__.__name__,
        }
        if self.context:
            payload["context"] = sanitize_context(self.context)
        if self.cause is not None:
            payload["cause"] = type(self.cause).__name__
        return payload


class DataSourceError(HVError):
    def __init__(self, message: str = "Data source failure", **kwargs: Any) -> None:
        super().__init__(message, code=ErrorCode.DATA_SOURCE, **kwargs)


class CacheError(HVError):
    def __init__(self, message: str = "Cache operation failed", **kwargs: Any) -> None:
        super().__init__(message, code=ErrorCode.CACHE, **kwargs)


class ValidationError(HVError):
    def __init__(self, message: str = "Validation error", **kwargs: Any) -> None:
        super().__init__(message, code=ErrorCode.VALIDATION, **kwargs)


class ComputeError(HVError):
    def __init__(self, message: str = "Computation error", **kwargs: Any) -> None:
        super().__init__(message, code=ErrorCode.COMPUTE, **kwargs)


class IntegrationError(HVError):
    def __init__(self, message: str = "Integration failure", **kwargs: Any) -> None:
        super().__init__(message, code=ErrorCode.INTEGRATION, **kwargs)


class ConfigurationError(HVError):
    def __init__(self, message: str = "Configuration error", **kwargs: Any) -> None:
        super().__init__(message, code=ErrorCode.CONFIG, **kwargs)


def wrap_error(
    exc: Exception,
    error_cls: Type[HVError] = HVError,
    *,
    message: str,
    context: Mapping[str, Any] | None = None,
    user_message: str | None = None,
) -> HVError:
    """Return an :class:`HVError` instance wrapping ``exc``.

    Existing :class:`HVError` instances are enriched with ``context`` instead of
    being re-wrapped.
    """

    if isinstance(exc, HVError):
        if context:
            exc.add_context(**dict(context))
        if user_message:
            exc.with_user_message(user_message)
        return exc
    return error_cls(
        message,
        context=context,
        user_message=user_message,
        cause=exc,
    )


_error_counts: Counter[str] = Counter()
_counter_lock = Lock()


def record_error(error: HVError) -> None:
    """Increment in-memory metrics for ``error``."""

    with _counter_lock:
        _error_counts[error.code.value] += 1


def get_error_metrics() -> dict[str, int]:
    """Return a snapshot of error counts by :class:`ErrorCode`."""

    with _counter_lock:
        return dict(_error_counts)


def reset_error_metrics() -> None:
    """Reset the in-memory error metrics (intended for tests)."""

    with _counter_lock:
        _error_counts.clear()


__all__ = [
    "CacheError",
    "ComputeError",
    "ConfigurationError",
    "DataSourceError",
    "ErrorCode",
    "HVError",
    "IntegrationError",
    "ValidationError",
    "get_error_metrics",
    "record_error",
    "reset_error_metrics",
    "sanitize_context",
    "wrap_error",
]
