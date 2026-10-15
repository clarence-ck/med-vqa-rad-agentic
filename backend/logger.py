"""
logger.py — Centralized logging for MedVisionQuery backend.

Provides a pre-configured logger with:
  - Timestamps (ISO-8601)
  - Level-coloring (via colorlog if installed, plain otherwise)
  - Consistent name prefixes per module / agent
  - Per-session log file named: medvqa_<YYYYMMDD_HHMMSS>_<short-id>.log
  - Lines logged while a dataset case runs are tagged "[sample <id>]", so one case's lines can be found with grep
  - A get_logger() factory for per-component loggers
"""

import logging
import os
import sys
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional
from contextvars import ContextVar

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
_LOG_FORMAT_PLAIN = "%(asctime)s  %(levelname)-8s  %(name)s%(sample)s  %(message)s"
_LOG_FORMAT_COLOR = (
    "%(log_color)s%(asctime)s  %(levelname)-8s%(reset)s  "
    "%(cyan)s%(name)s%(reset)s%(sample)s  %(message)s"
)
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_COLOR_MAP = {
    "DEBUG": "white",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

sample_id_var: ContextVar[Optional[str]] = ContextVar('sample_id', default=None)

# Generate a unique session identity once at import time
_SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_SESSION_ID = uuid.uuid4().hex[:8]
_SESSION_LOG_FILENAME = f"medvqa_{_SESSION_TIMESTAMP}_{_SESSION_ID}.log"

# Log directory sits next to this file i.e. backend/logs/
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

class SampleTagFilter(logging.Filter):
    """Adds `record.sample` (" [sample <id>]" while a dataset case runs, else empty) for the log format."""
    def filter(self, record: logging.LogRecord) -> bool:
        sid = sample_id_var.get()
        record.sample = f" [sample {sid}]" if sid else ""
        return True

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

    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(SampleTagFilter())
    root.addHandler(console_handler)

    # Tests and offline scripts can opt out of writing the session log file.
    if os.getenv("MEDVQA_DISABLE_FILE_LOGS", "").strip().lower() in {"1", "true", "yes"}:
        for noisy in ("httpx", "httpcore", "openai", "haystack"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        return

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
    file_handler.setFormatter(plain_formatter)
    file_handler.addFilter(SampleTagFilter())
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
