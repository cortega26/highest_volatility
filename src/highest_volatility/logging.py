"""Structured logging helpers with contextual redaction."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Mapping

from .errors import HVError, record_error, sanitize_context


def _json_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    return repr(value)


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that serialises records as JSON strings."""

    def process(self, msg: Any, kwargs: Mapping[str, Any]):  # type: ignore[override]
        extra_context = kwargs.pop("context", None)
        context = dict(self.extra or {})
        if extra_context:
            context.update(extra_context)
        payload: dict[str, Any]
        if isinstance(msg, Mapping):
            payload = dict(msg)
        else:
            payload = {"message": str(msg)}
        sanitized_context = sanitize_context(context)
        if sanitized_context:
            payload.setdefault("context", {}).update(sanitized_context)
        payload.setdefault("logger", self.logger.name)
        payload.setdefault(
            "timestamp",
            datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )
        kwargs.setdefault("extra", {})["structured"] = payload
        return json.dumps(payload, default=_json_ready), dict(kwargs)


def get_logger(name: str, **context: Any) -> StructuredLoggerAdapter:
    """Return a structured logger adapter bound to ``name``."""

    base_logger = logging.getLogger(name)
    base_logger.setLevel(logging.INFO)
    return StructuredLoggerAdapter(base_logger, sanitize_context(context))


def log_exception(
    logger: logging.Logger,
    error: HVError,
    *,
    event: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Emit a structured error log and update error metrics."""

    combined_context: dict[str, Any] = {}
    if context:
        combined_context.update(context)
    combined_context.update(error.context)
    payload = {
        "event": event,
        "error": error.to_dict(),
    }
    if combined_context:
        payload["context"] = sanitize_context(combined_context)
    record_error(error)
    logger.error(payload)


__all__ = ["StructuredLoggerAdapter", "get_logger", "log_exception"]
