"""
utils/logger.py

Structured logger for the entire pipeline.

Design rules:
  - ONE logger instance per module, created via get_logger(__name__)
  - Hot-path calls (per-frame) use logger.debug() — compiled out at INFO level
  - No file I/O on the frame loop thread — log handler runs async
  - Timestamps in milliseconds for latency debugging
  - Color-coded console output so warnings/errors are instantly visible
"""

import logging
import sys
import time
from typing import Optional


# ── ANSI color codes (works on Linux/macOS terminals, Windows 10+) ──────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_COLORS = {
    "DEBUG"    : "\033[36m",    # cyan
    "INFO"     : "\033[32m",    # green
    "WARNING"  : "\033[33m",    # yellow
    "ERROR"    : "\033[31m",    # red
    "CRITICAL" : "\033[35m",    # magenta
}

# App start time — all log timestamps are relative (ms since start)
_T0 = time.monotonic()


class _RelativeFormatter(logging.Formatter):
    """
    Format: [  1234ms] INFO     hand_detector  : message
    Relative timestamps make latency spikes immediately obvious.
    """
    def format(self, record: logging.LogRecord) -> str:
        elapsed_ms = int((time.monotonic() - _T0) * 1000)
        level      = record.levelname
        color      = _COLORS.get(level, "")
        module     = record.name.split(".")[-1]          # last segment only
        prefix     = f"[{elapsed_ms:6d}ms] {level:<8} {module:<18}: "
        msg        = super().format(record)
        return f"{color}{_BOLD}{prefix}{_RESET}{color}{msg}{_RESET}"


class _PlainFormatter(logging.Formatter):
    """No ANSI — for file handlers or non-color terminals."""
    def format(self, record: logging.LogRecord) -> str:
        elapsed_ms = int((time.monotonic() - _T0) * 1000)
        level      = record.levelname
        module     = record.name.split(".")[-1]
        prefix     = f"[{elapsed_ms:6d}ms] {level:<8} {module:<18}: "
        return prefix + super().format(record)


# ── Root handler — set up once ───────────────────────────────────────────────
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_RelativeFormatter())

# Detect if terminal supports color
_supports_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
if not _supports_color:
    _handler.setFormatter(_PlainFormatter())

_root = logging.getLogger("hgmc")   # "hand gesture mouse control" namespace
_root.addHandler(_handler)
_root.propagate = False             # don't double-log via root logger


def setup(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Call once at startup (from main.py) to set global log level and
    optionally mirror output to a file.

    Args:
        level:    "DEBUG" | "INFO" | "WARNING" | "ERROR"
        log_file: Optional path. If given, plain-text copy written here.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    _root.setLevel(numeric)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(_PlainFormatter())
        fh.setLevel(numeric)
        _root.addHandler(fh)
        _root.info(f"Log file: {log_file}")

    _root.info(f"Logger ready — level={level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a module-level logger. Call at module import time:

        from utils.logger import get_logger
        log = get_logger(__name__)

    Then use:
        log.debug(...)   # per-frame — zero cost when level=INFO
        log.info(...)    # startup / state changes
        log.warning(...) # recoverable issues
        log.error(...)   # failures needing attention
    """
    return _root.getChild(name)


# ── Convenience: performance spike logger ────────────────────────────────────
class LatencyGuard:
    """
    Context manager that logs a WARNING if a block takes longer than budget_ms.
    Use on any block that runs inside the frame loop.

    Usage:
        with LatencyGuard("mediapipe_inference", budget_ms=20, log=log):
            results = hands.process(rgb)
    """
    __slots__ = ("label", "budget_ms", "_log", "_t")

    def __init__(self, label: str, budget_ms: float, log: logging.Logger):
        self.label     = label
        self.budget_ms = budget_ms
        self._log      = log

    def __enter__(self):
        self._t = time.monotonic()
        return self

    def __exit__(self, *_):
        elapsed = (time.monotonic() - self._t) * 1000
        if elapsed > self.budget_ms:
            self._log.warning(
                f"LATENCY SPIKE  {self.label}  took {elapsed:.1f}ms "
                f"(budget {self.budget_ms}ms)"
            )
        else:
            self._log.debug(f"{self.label} {elapsed:.1f}ms")