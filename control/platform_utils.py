"""
control/platform_utils.py

OS-specific mouse backend and screen query utilities.

Abstracts pyautogui behind a thin interface so:
  1. The rest of the codebase never imports pyautogui directly
  2. The backend is swappable (e.g. pynput, win32api) without touching
     mouse_controller.py or event_manager.py
  3. Import errors are handled in one place — the app degrades gracefully
     to "preview mode" (camera + gesture display, no actual mouse control)

Windows notes:
  - pyautogui.FAILSAFE = False — prevents the "move to corner = crash"
    safety feature from killing the app mid-session
  - pyautogui.PAUSE = 0 — removes the default 0.1s sleep after every
    mouse call (was the #1 latency killer in the original monolithic code)
  - SetProcessDpiAwareness(2) — makes pyautogui report real pixel
    coordinates on high-DPI displays (4K, Retina-via-Boot-Camp, etc.)
    Without this, screen size reads as 1920x1080 even on a 3840x2160
    display and cursor positions are off by 2x.

Multi-monitor notes:
  - screen_size() returns the bounding box of ALL monitors combined
    (pyautogui behaves this way on Windows; on Linux it's primary only)
  - Cursor can be moved to any pixel in the combined virtual desktop

Linux / macOS notes:
  - pyautogui works on both but requires additional system packages
    (python3-xlib on Linux, no extra deps on macOS)
  - DPI scaling is handled differently; SetProcessDpiAwareness is skipped
"""

from __future__ import annotations

import platform
import sys
from typing import Tuple

from utils.logger import get_logger

log = get_logger(__name__)

# ── Backend availability ───────────────────────────────────────────────────────
_MOUSE_AVAILABLE = False
_pyautogui = None

try:
    import pyautogui as _pag
    _pag.FAILSAFE = False   # don't crash when cursor hits corner
    _pag.PAUSE    = 0       # remove per-call sleep (was +100ms latency per action)
    _pyautogui       = _pag
    _MOUSE_AVAILABLE = True
    log.info("pyautogui loaded — mouse control ACTIVE")
except ImportError:
    log.warning(
        "pyautogui not installed — running in PREVIEW mode "
        "(camera + gesture display only, no mouse control). "
        "Install with:  pip install pyautogui"
    )

# ── Windows DPI awareness ─────────────────────────────────────────────────────
_OS = platform.system()   # "Windows" | "Darwin" | "Linux"

if _OS == "Windows" and _MOUSE_AVAILABLE:
    try:
        import ctypes
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        # Makes GetSystemMetrics / pyautogui.size() return real pixel counts
        # on scaled (HiDPI) displays. Critical for 4K monitors at 150%+ scaling.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        log.info("Windows: DPI awareness set to Per-Monitor v2")
    except Exception as e:
        log.debug(f"DPI awareness not set: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def is_available() -> bool:
    """True if mouse control is usable on this system."""
    return _MOUSE_AVAILABLE


def screen_size() -> Tuple[int, int]:
    """
    Returns (width, height) of the full virtual desktop in real pixels.
    Falls back to 1920×1080 if pyautogui is not available.
    """
    if _MOUSE_AVAILABLE:
        w, h = _pyautogui.size()
        log.info(f"Screen size: {w}×{h}")
        return int(w), int(h)
    log.info("Screen size: using fallback 1920×1080 (preview mode)")
    return 1920, 1080


def move_to(x: int, y: int) -> None:
    """Move cursor to absolute screen position. No-op in preview mode."""
    if _MOUSE_AVAILABLE:
        _pyautogui.moveTo(x, y)


def click(button: str = "left") -> None:
    """
    Fire a mouse click.
    button: "left" | "right" | "middle"
    No-op in preview mode.
    """
    if _MOUSE_AVAILABLE:
        _pyautogui.click(button=button)
        log.debug(f"Click: {button}")


def mouse_down(button: str = "left") -> None:
    """Press and hold mouse button. No-op in preview mode."""
    if _MOUSE_AVAILABLE:
        _pyautogui.mouseDown(button=button)
        log.debug(f"MouseDown: {button}")


def mouse_up(button: str = "left") -> None:
    """Release mouse button. No-op in preview mode."""
    if _MOUSE_AVAILABLE:
        _pyautogui.mouseUp(button=button)
        log.debug(f"MouseUp: {button}")


def scroll(ticks: int) -> None:
    """
    Scroll the mouse wheel.
    Positive ticks = scroll up, negative = scroll down.
    No-op in preview mode.
    """
    if _MOUSE_AVAILABLE:
        _pyautogui.scroll(ticks)


def current_position() -> Tuple[int, int]:
    """
    Current cursor position in screen pixels.
    Returns centre of screen in preview mode.
    """
    if _MOUSE_AVAILABLE:
        pos = _pyautogui.position()
        return int(pos.x), int(pos.y)
    w, h = screen_size()
    return w // 2, h // 2


def release_all() -> None:
    """
    Release all held mouse buttons.
    Call on app shutdown or when transitioning out of drag mode
    to prevent a stuck mouse button.
    """
    if _MOUSE_AVAILABLE:
        try:
            _pyautogui.mouseUp(button="left")
            _pyautogui.mouseUp(button="right")
        except Exception as e:
            log.warning(f"release_all failed: {e}")
    log.debug("All mouse buttons released")


def os_name() -> str:
    """Returns 'Windows', 'Darwin', or 'Linux'."""
    return _OS