from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Iterator


# Correlation / request ID for the current chat turn. Backed by a ContextVar so
# every log line emitted while handling one turn can be tied together — which is
# what makes these logs usable once the pipeline is fronted by an API.
_request_id_var: ContextVar[str | None] = ContextVar("docintel_request_id", default=None)


def _use_json() -> bool:
    return os.getenv("LOG_FORMAT", "text").strip().lower() == "json"


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"


class _Formatter(logging.Formatter):
    """Human-readable, colorized single-line formatter (the default)."""

    COLORS = {
        logging.DEBUG: _Ansi.BLUE,
        logging.INFO: _Ansi.CYAN,
        logging.WARNING: _Ansi.YELLOW,
        logging.ERROR: _Ansi.RED,
        logging.CRITICAL: _Ansi.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        color = self.COLORS.get(record.levelno, _Ansi.CYAN)
        level_name = getattr(record, "log_level", record.levelname)
        request_id = getattr(record, "request_id", None)
        rid_part = f"[{request_id}] " if request_id else ""
        message = record.getMessage()
        return f"{color}[{timestamp}] {level_name:<8}{_Ansi.RESET} {rid_part}{message}"


class _JsonFormatter(logging.Formatter):
    """Structured JSON formatter (one JSON object per line) for LOG_FORMAT=json."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": getattr(record, "log_level", record.levelname),
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id

        event = getattr(record, "event", None)
        if event:
            payload["event"] = event

        data = getattr(record, "data", None)
        if data:
            payload["data"] = data

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class Logger:
    def __init__(self, name: str = "docintel") -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(_JsonFormatter() if _use_json() else _Formatter())
            self._logger.addHandler(handler)

    # -- Correlation / request ID -------------------------------------------------
    def new_request_id(self) -> str:
        """Mint a fresh correlation ID and bind it to the current context.

        Call once at the start of each chat turn (or, in an API, once per request
        — or pass the inbound request ID to :meth:`set_request_id` instead)."""
        request_id = uuid.uuid4().hex[:12]
        _request_id_var.set(request_id)
        return request_id

    def set_request_id(self, request_id: str | None) -> None:
        _request_id_var.set(request_id)

    def get_request_id(self) -> str | None:
        return _request_id_var.get()

    def clear_request_id(self) -> None:
        _request_id_var.set(None)

    @contextlib.contextmanager
    def request_context(self, request_id: str | None = None) -> Iterator[str]:
        """Bind a correlation ID for the duration of the block, then restore."""
        request_id = request_id or uuid.uuid4().hex[:12]
        token = _request_id_var.set(request_id)
        try:
            yield request_id
        finally:
            _request_id_var.reset(token)

    # -- Internal -----------------------------------------------------------------
    def _level_name(self, level: int) -> str:
        return {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARN",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRIT",
        }.get(level, "INFO")

    def _log(
        self,
        level: int,
        message: str,
        *,
        prefix: str | None = None,
        event: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        extra = {
            "log_level": prefix or self._level_name(level),
            "request_id": _request_id_var.get(),
        }
        if event is not None:
            extra["event"] = event
        if data is not None:
            extra["data"] = data
        self._logger.log(level, message, extra=extra)

    # -- Public logging API -------------------------------------------------------
    def info(self, message: str) -> None:
        self._log(logging.INFO, message)

    def success(self, message: str) -> None:
        self._log(logging.INFO, message, prefix="SUCCESS")

    def warning(self, message: str) -> None:
        self._log(logging.WARNING, message)

    def error(self, message: str) -> None:
        self._log(logging.ERROR, message)

    def debug(self, message: str) -> None:
        self._log(logging.DEBUG, message)

    def section(self, title: str) -> None:
        self.divider()
        self._log(logging.INFO, title, event="section")
        self.divider()

    def divider(self) -> None:
        self.info("-" * 72)

    def blank(self) -> None:
        sys.stderr.write("\n")

    def kv(self, key: str, value: Any) -> None:
        self._log(logging.INFO, f"{key}: {value}", event="kv", data={"key": key, "value": value})

    def list_item(self, text: str) -> None:
        self._log(logging.INFO, f"- {text}", event="list_item")


log = Logger()
