from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    level: str = "INFO"
    use_color: bool = True


_CONFIGURED = False


class _AnsiColorFormatter(logging.Formatter):
    _RESET = "\x1b[0m"
    _COLOR_BY_LEVEL = {
        logging.DEBUG: "\x1b[36m",
        logging.INFO: "\x1b[32m",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[31m",
        logging.CRITICAL: "\x1b[41m\x1b[97m",
    }

    def __init__(self, *, use_color: bool) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)s node=%(node)s action=%(action)s %(name)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "node"):
            record.node = "-"
        if not hasattr(record, "action"):
            record.action = "-"

        message = super().format(record)
        if not self._use_color:
            return message

        color = self._COLOR_BY_LEVEL.get(record.levelno)
        if not color:
            return message

        return f"{color}{message}{self._RESET}"


def configure_logging(*, settings: LoggingSettings | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = settings or LoggingSettings(
        level=os.getenv("RAG_AGENT_LOG_LEVEL", LoggingSettings.level),
        use_color=os.getenv("RAG_AGENT_LOG_COLOR", "true").strip().lower()
        in {"1", "true", "t", "yes", "y", "on"},
    )

    root_logger = logging.getLogger()
    if root_logger.handlers:
        _CONFIGURED = True
        return

    try:
        level = getattr(logging, settings.level.upper())
    except Exception:
        level = logging.INFO

    root_logger.setLevel(level)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)

    is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    formatter = _AnsiColorFormatter(use_color=settings.use_color and is_tty)
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str, *, node: str | None = None) -> logging.LoggerAdapter:
    configure_logging()
    logger = logging.getLogger(name)
    return logging.LoggerAdapter(logger, {"node": node or "-"})

