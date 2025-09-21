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


def _safe_str(value: Any) -> str:
    """Return a defensive string representation for diagnostics."""

    try:
        text = str(value)
    except Exception:  # pragma: no cover - extremely defensive
        text = repr(value)
    return text


def describe_exception(exc: Exception, *, max_depth: int = 3) -> dict[str, Any]:
    """Return a serialisable description of ``exc`` and its causes."""

    seen: set[int] = set()

    def _describe(err: Exception, depth: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": type(err).__name__,
            "message": _safe_str(err),
        }
        if hasattr(err, "errno") and getattr(err, "errno") is not None:
            payload["errno"] = getattr(err, "errno")
        os_error = getattr(err, "os_error", None)
        if os_error is not None:
            payload["os_error"] = {
                "type": type(os_error).__name__,
                "message": _safe_str(os_error),
            }
            errno = getattr(os_error, "errno", None)
            if errno is not None:
                payload["os_error"]["errno"] = errno

        identity = id(err)
        if identity in seen:
            payload["cycle"] = True
            return payload
        seen.add(identity)

        if depth >= max_depth:
            return payload

        if err.__cause__ is not None:
            payload["cause"] = _describe(err.__cause__, depth + 1)
        elif err.__context__ is not None and not err.__suppress_context__:
            payload["context"] = _describe(err.__context__, depth + 1)
        return payload

    return _describe(exc, 0)


def _coerce(value: Any) -> Any:
    """Return a JSON-serialisable representation for ``value``."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return sanitize_context(value)
    if isinstance(value, (list, tuple, set)):
        return [_coerce(v) for v in value]
    return repr(value)


def sanitize_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a shallow copy of ``context`` with sensitive values redacted."""

    if not context:
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in context.items():
        key_str = str(key)
        lowered = key_str.lower()
        if any(token in lowered for token in _SENSITIVE_KEYS):
            sanitized[key_str] = _REDACTED
        else:
            sanitized[key_str] = _coerce(value)
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
            payload["cause"] = describe_exception(self.cause)
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
    "describe_exception",
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
