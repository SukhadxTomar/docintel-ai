from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Any


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
        message = record.getMessage()
        return f"{color}[{timestamp}] {level_name:<8}{_Ansi.RESET} {message}"


class Logger:
    def __init__(self, name: str = "docintel") -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(_Formatter())
            self._logger.addHandler(handler)

    def _level_name(self, level: int) -> str:
        return {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARN",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRIT",
        }.get(level, "INFO")

    def _log(self, level: int, message: str, *, prefix: str | None = None) -> None:
        extra = {
            "log_level": prefix or self._level_name(level)
        }
        self._logger.log(level, message, extra=extra)

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
        self.info(title)
        self.divider()

    def divider(self) -> None:
        self.info("-" * 72)

    def blank(self) -> None:
        sys.stderr.write("\n")

    def kv(self, key: str, value: Any) -> None:
        self.info(f"{key}: {value}")

    def list_item(self, text: str) -> None:
        self.info(f"- {text}")


log = Logger()