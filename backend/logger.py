"""
logger.py — Centralized logging for MedVisionQuery backend.

Provides a pre-configured logger with:
  - Timestamps (ISO-8601)
  - Level-coloring (via colorlog if installed, plain otherwise)
  - Consistent name prefixes per module / agent
  - Per-session log file named: medvqa_<YYYYMMDD_HHMMSS>_<short-id>.log
  - A get_logger() factory for per-component loggers
"""

import logging
import os
import sys
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

# ---------------------------------------------------------------------------
# Attempt to use colorlog for prettier terminal output.
# Falls back to plain logging if the package is not installed.
# ---------------------------------------------------------------------------
try:
    import colorlog  # type: ignore

    _HAVE_COLORLOG = True
except ImportError:
    _HAVE_COLORLOG = False

_ROOT_LOGGER_NAME = "medvqa"
_LOG_FORMAT_PLAIN = "%(asctime)s %(sample_ctx)s %(levelname)-8s  %(name)s  %(message)s"
_LOG_FORMAT_COLOR = (
    "%(log_color)s%(asctime)s %(reset)s%(yellow)s%(sample_ctx)s%(reset)s%(log_color)s%(levelname)-8s%(reset)s  "
    "%(cyan)s%(name)s%(reset)s  %(message)s"
)
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_COLOR_MAP = {
    "DEBUG": "white",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

from contextvars import ContextVar
sample_id_var: ContextVar[Optional[str]] = ContextVar('sample_id', default=None)

class SampleContextFilter(logging.Filter):
    """Injects sample_ctx into the log record."""
    def filter(self, record):
        sid = sample_id_var.get()
        record.sample_ctx = f"[Sample #{sid}]" if sid else ""
        return True

# Generate a unique session identity once at import time
_SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_SESSION_ID = uuid.uuid4().hex[:8]
_SESSION_LOG_FILENAME = f"medvqa_{_SESSION_TIMESTAMP}_{_SESSION_ID}.log"

# Log directory sits next to this file i.e. backend/logs/
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

_configured = False


def _configure_root() -> None:
    """Set up the root 'medvqa' logger exactly once."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)

    if root.handlers:
        # Already configured (e.g. imported twice)
        return

    plain_formatter = logging.Formatter(_LOG_FORMAT_PLAIN, datefmt=_DATE_FORMAT)

    # ── Console handler (coloured if colorlog is available) ──────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if _HAVE_COLORLOG:
        console_formatter = colorlog.ColoredFormatter(
            _LOG_FORMAT_COLOR,
            datefmt=_DATE_FORMAT,
            log_colors=_COLOR_MAP,
            reset=True,
            style="%",
        )
    else:
        console_formatter = plain_formatter

    filter_instance = SampleContextFilter()
    console_handler.addFilter(filter_instance)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    # ── File handler (plain text, one file per server session) ───────────────
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(_LOG_DIR, _SESSION_LOG_FILENAME)

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(filter_instance)
    file_handler.setFormatter(plain_formatter)
    root.addHandler(file_handler)

    # Emit the session header so it's easy to find session boundaries in logs
    root.info(
        "Session started  file=%s  session_id=%s",
        _SESSION_LOG_FILENAME,
        _SESSION_ID,
    )

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "haystack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a child logger under the 'medvqa' namespace.

    Usage:
        logger = get_logger(__name__)
        logger = get_logger("agents.image_interpreter")
    """
    _configure_root()
    if name:
        return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
    return logging.getLogger(_ROOT_LOGGER_NAME)
