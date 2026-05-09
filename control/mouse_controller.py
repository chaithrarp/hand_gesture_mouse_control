"""
control/mouse_controller.py

Translates GestureResult events → mouse actions (move, click, drag, scroll).

Design rules:
  - Consumes GestureEvent.ENTERED/HELD/EXITED — no gesture timing logic here
  - All pyautogui calls go through _execute() which catches exceptions
  - Action queue ensures no gesture event is silently dropped under load
  - Scroll accumulator prevents single-frame jitter from firing scroll ticks
  - Pause state (FIST gesture) blocks all output cleanly
  - Thread-safe: queue is filled on frame thread, could be drained on separate
    thread in future without API changes

Latency notes:
  - pyautogui.moveTo with _pause=0 is the fastest available on all platforms
  - Queue drain happens synchronously at end of each frame (sub-ms)
  - Scroll accumulation prevents OS scroll event spam (which causes lag)
"""

import time
import queue
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from config.settings import CursorSettings, GestureSettings, ScrollSettings
from config import constants as C
from core.gesture_recognizer import GestureType, GestureEvent, GestureResult
from utils.logger import get_logger

log = get_logger(__name__)

# ── Try importing pyautogui ────────────────────────────────────────────────────
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE    = 0          # critical — default 0.1s pause kills latency
    _MOUSE_AVAILABLE   = True
    log.info("pyautogui loaded — mouse control active")
except ImportError:
    _MOUSE_AVAILABLE = False
    log.warning("pyautogui not found — mouse control DISABLED (pip install pyautogui)")


# ── Action types for the queue ─────────────────────────────────────────────────
class ActionType(Enum):
    MOVE        = auto()
    LEFT_CLICK  = auto()
    RIGHT_CLICK = auto()
    MOUSE_DOWN  = auto()
    MOUSE_UP    = auto()
    SCROLL      = auto()


@dataclass
class MouseAction:
    kind:    ActionType
    x:       Optional[int]   = None
    y:       Optional[int]   = None
    ticks:   Optional[int]   = None   # for SCROLL


# ── Mouse Controller ───────────────────────────────────────────────────────────
class MouseController:
    """
    Translates gesture results into mouse actions.

    Usage (called once per frame per hand):
        controller = MouseController(screen_w, screen_h, settings)
        controller.handle(gesture_result, screen_x, screen_y)
        controller.flush()   # drain action queue — call at end of frame

    Two-hand usage:
        controller.handle(right_result, rx, ry)   # cursor + click
        controller.handle_scroll(left_result, ly)  # scroll only
        controller.flush()
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        cs: CursorSettings,
        gs: GestureSettings,
        ss: ScrollSettings,
    ):
        self._sw = screen_w
        self._sh = screen_h
        self._cs = cs
        self._gs = gs
        self._ss = ss

        # ── Cursor state ──────────────────────────────────────────────────
        self._cur_x: int = screen_w  // 2
        self._cur_y: int = screen_h // 2

        # ── Click / drag state ────────────────────────────────────────────
        self._pinch_start:  Optional[float] = None
        self._is_dragging:  bool            = False
        self._click_fired:  bool            = False

        # ── Scroll state ──────────────────────────────────────────────────
        self._scroll_ref_y:  Optional[int]  = None
        self._scroll_accum:  float          = 0.0

        # ── Right-click debounce ──────────────────────────────────────────
        self._last_right_click: float = 0.0

        # ── Pause (FIST) ──────────────────────────────────────────────────
        self.paused: bool = False

        # ── Action queue ──────────────────────────────────────────────────
        # Bounded at 16 — if we ever fill this the frame loop has bigger problems
        self._queue: queue.SimpleQueue = queue.SimpleQueue()

        if _MOUSE_AVAILABLE:
            self._sw, self._sh = pyautogui.size()
            log.info(f"Screen: {self._sw}x{self._sh}")

    # ── Public: per-frame entry points ────────────────────────────────────────

    def handle(
        self,
        result: GestureResult,
        screen_x: int,
        screen_y: int,
    ) -> None:
        """
        Process a gesture result for the primary (cursor-controlling) hand.
        Queues appropriate mouse actions. Call flush() after all hands processed.

        Args:
            result:   GestureResult from GestureStateMachine.update()
            screen_x: Final screen X from CoordinateMapper (after dead zone)
            screen_y: Final screen Y from CoordinateMapper (after dead zone)
        """
        g  = result.gesture
        ev = result.event

        # ── FIST: toggle pause on ENTERED only ───────────────────────────
        if g == GestureType.FIST:
            if ev == GestureEvent.ENTERED:
                self.paused = not self.paused
                if self.paused:
                    self._release_drag()
                    log.info("Mouse control PAUSED")
                else:
                    log.info("Mouse control RESUMED")
            return

        if self.paused:
            return

        # ── Route by gesture ──────────────────────────────────────────────
        if g == GestureType.POINT:
            self._handle_point(screen_x, screen_y, ev)

        elif g == GestureType.PINCH:
            self._handle_pinch(screen_x, screen_y, result.hold_s, ev)

        elif g == GestureType.VICTORY:
            self._handle_victory(ev)

        elif g == GestureType.OPEN_HAND:
            self._handle_open_hand(screen_x, screen_y, ev)

        elif g == GestureType.NONE:
            self._handle_none(screen_x, screen_y, ev)

    def handle_scroll(
        self,
        result: GestureResult,
        screen_y: int,
    ) -> None:
        """
        Scroll-only handler for secondary (non-dominant) hand.
        Only acts on OPEN_HAND gesture — ignores everything else.
        """
        if self.paused:
            return
        if result.gesture == GestureType.OPEN_HAND:
            self._accumulate_scroll(screen_y, active=True)
        else:
            self._scroll_ref_y = None
            self._scroll_accum = 0.0

    def flush(self) -> None:
        """
        Drain the action queue and execute all pending mouse actions.
        Call once at the end of each frame, after all handle() calls.
        """
        while not self._queue.empty():
            try:
                action: MouseAction = self._queue.get_nowait()
                self._execute(action)
            except queue.Empty:
                break

    # ── Gesture handlers (private) ────────────────────────────────────────────

    def _handle_point(self, sx: int, sy: int, ev: GestureEvent) -> None:
        """POINT → move cursor. Release any active drag/scroll state."""
        if ev == GestureEvent.ENTERED:
            self._release_drag()
            self._scroll_ref_y = None
        self._enqueue(MouseAction(ActionType.MOVE, sx, sy))

    def _handle_pinch(
        self, sx: int, sy: int, hold_s: float, ev: GestureEvent
    ) -> None:
        """
        PINCH state machine:
          ENTERED → start pinch timer
          HELD    → if held >= drag_lock_s: enter drag mode and move
                    else: wait (click will fire on EXITED)
          EXITED  → if dragging: mouseUp
                    elif hold within click window: left click
        """
        gs = self._gs

        if ev == GestureEvent.ENTERED:
            self._pinch_start = time.monotonic()
            self._click_fired = False

        elif ev == GestureEvent.HELD:
            if hold_s >= gs.drag_lock_s and not self._is_dragging:
                # Enter drag mode
                self._is_dragging = True
                self._enqueue(MouseAction(ActionType.MOUSE_DOWN, sx, sy))
                log.debug("Drag started")

            if self._is_dragging:
                self._enqueue(MouseAction(ActionType.MOVE, sx, sy))

        elif ev == GestureEvent.EXITED:
            if self._is_dragging:
                self._release_drag()
            elif (not self._click_fired and
                  gs.click_min_hold_s <= hold_s <= gs.click_max_hold_s):
                self._enqueue(MouseAction(ActionType.LEFT_CLICK, sx, sy))
                self._click_fired = True
                log.debug(f"Left click  (hold={hold_s*1000:.0f}ms)")
            self._pinch_start = None

    def _handle_victory(self, ev: GestureEvent) -> None:
        """VICTORY → right click on ENTERED only (debounced)."""
        if ev != GestureEvent.ENTERED:
            return
        now = time.monotonic()
        if now - self._last_right_click >= C.RIGHT_CLICK_DEBOUNCE_S:
            self._enqueue(MouseAction(ActionType.RIGHT_CLICK))
            self._last_right_click = now
            log.debug("Right click")

    def _handle_open_hand(self, sx: int, sy: int, ev: GestureEvent) -> None:
        """OPEN_HAND → scroll. Release drag if switching from pinch."""
        if ev == GestureEvent.ENTERED:
            self._release_drag()
            self._scroll_ref_y = None
            self._scroll_accum = 0.0
        self._accumulate_scroll(sy, active=True)

    def _handle_none(self, sx: int, sy: int, ev: GestureEvent) -> None:
        """NONE / ambiguous → release everything cleanly."""
        if ev == GestureEvent.ENTERED:
            self._release_drag()
            self._scroll_ref_y = None
            self._scroll_accum = 0.0

    # ── Scroll accumulator ────────────────────────────────────────────────────

    def _accumulate_scroll(self, screen_y: int, active: bool) -> None:
        """
        Accumulate vertical hand movement and fire scroll ticks when
        the accumulator crosses pixels_per_tick.

        Uses a fixed reference point (scroll_ref_y set on first call)
        so large hand movements = many ticks, not just velocity.
        The accumulator is smoothed to prevent single-frame jitter.
        """
        ss = self._ss

        if not active:
            self._scroll_ref_y = None
            self._scroll_accum = 0.0
            return

        if self._scroll_ref_y is None:
            self._scroll_ref_y = screen_y
            return

        delta = screen_y - self._scroll_ref_y  # positive = hand moved down

        # Smooth accumulation
        self._scroll_accum = (
            self._scroll_accum * (1.0 - ss.accumulation_smooth)
            + delta * ss.accumulation_smooth
        )

        if abs(self._scroll_accum) >= ss.pixels_per_tick:
            raw_ticks = int(self._scroll_accum / ss.pixels_per_tick)
            ticks = raw_ticks * ss.speed_multiplier

            # Natural direction: hand down → page scrolls down (negative for pyautogui)
            if ss.natural_direction:
                ticks = -ticks

            self._enqueue(MouseAction(ActionType.SCROLL, ticks=ticks))
            self._scroll_accum -= raw_ticks * ss.pixels_per_tick  # keep remainder
            log.debug(f"Scroll ticks={ticks}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _release_drag(self) -> None:
        if self._is_dragging:
            self._enqueue(MouseAction(ActionType.MOUSE_UP))
            self._is_dragging = False
            self._pinch_start = None
            log.debug("Drag released")

    def _enqueue(self, action: MouseAction) -> None:
        self._queue.put_nowait(action)

    def _execute(self, action: MouseAction) -> None:
        """Execute one mouse action. All pyautogui calls live here."""
        if not _MOUSE_AVAILABLE:
            return
        try:
            t = action.kind
            if t == ActionType.MOVE:
                pyautogui.moveTo(action.x, action.y)
            elif t == ActionType.LEFT_CLICK:
                pyautogui.click(action.x, action.y)
            elif t == ActionType.RIGHT_CLICK:
                pyautogui.rightClick()
            elif t == ActionType.MOUSE_DOWN:
                pyautogui.mouseDown()
            elif t == ActionType.MOUSE_UP:
                pyautogui.mouseUp()
            elif t == ActionType.SCROLL:
                pyautogui.scroll(action.ticks)
        except Exception as exc:
            log.error(f"Mouse action {action.kind.name} failed: {exc}")

    # ── Settings hot-reload ───────────────────────────────────────────────────

    def reconfigure(
        self,
        cs: CursorSettings,
        gs: GestureSettings,
        ss: ScrollSettings,
    ) -> None:
        self._cs = cs
        self._gs = gs
        self._ss = ss

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def release_all(self) -> None:
        """Call on shutdown or lost-hand to ensure clean mouse state."""
        self._release_drag()
        self.flush()
        log.info("MouseController: all released")

    @property
    def is_dragging(self) -> bool:
        return self._is_dragging

    @property
    def screen_pos(self) -> Tuple[int, int]:
        return self._cur_x, self._cur_y