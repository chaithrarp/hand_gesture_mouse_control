"""
control/mouse_controller.py

Translates GestureResult events → mouse actions (move, click, drag, scroll).

SCROLL FIX LOG (vs original):
  - ref_y now ROLLS (lerp toward current position) so scroll never stalls
    when hand drifts back to anchor. Controlled by ScrollSettings.rolling_ref
    and rolling_ref_lerp.
  - ticks now cast to int before pyautogui.scroll() — float ticks were
    silently ignored on Windows causing erratic/no scroll.
  - speed_multiplier is now float-aware (was rounded to int).
  - Accumulator remainder is preserved across ticks (no lost delta).

Everything else is unchanged — cursor movement, click, drag, right-click,
pause, action queue all untouched.
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

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE    = 0
    _MOUSE_AVAILABLE   = True
    log.info("pyautogui loaded — mouse control active")
except ImportError:
    _MOUSE_AVAILABLE = False
    log.warning("pyautogui not found — mouse control DISABLED (pip install pyautogui)")


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
    ticks:   Optional[int]   = None   # always int — pyautogui requires int


class MouseController:
    """
    Translates gesture results into mouse actions.

    Usage (called once per frame per hand):
        controller = MouseController(screen_w, screen_h, settings)
        controller.handle(gesture_result, screen_x, screen_y)
        controller.flush()   # drain action queue — call at end of frame
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

        self._cur_x: int = screen_w  // 2
        self._cur_y: int = screen_h // 2

        self._pinch_start:  Optional[float] = None
        self._is_dragging:  bool            = False
        self._click_fired:  bool            = False

        # ── Scroll state ───────────────────────────────────────────────────
        self._scroll_ref_y:  Optional[float] = None   # float for smooth rolling
        self._scroll_accum:  float            = 0.0

        self._last_right_click: float = 0.0

        self.paused: bool = False

        self._queue: queue.SimpleQueue = queue.SimpleQueue()

        if _MOUSE_AVAILABLE:
            self._sw, self._sh = pyautogui.size()
            log.info(f"Screen: {self._sw}x{self._sh}")

    # ── Public: per-frame entry points ────────────────────────────────────────

    def handle(self, result: GestureResult, screen_x: int, screen_y: int) -> None:
        g  = result.gesture
        ev = result.event

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

    def handle_scroll(self, result: GestureResult, screen_y: int) -> None:
        if self.paused:
            return
        if result.gesture == GestureType.OPEN_HAND:
            self._accumulate_scroll(screen_y, active=True)
        else:
            self._reset_scroll()

    def flush(self) -> None:
        while not self._queue.empty():
            try:
                action: MouseAction = self._queue.get_nowait()
                self._execute(action)
            except queue.Empty:
                break

    # ── Gesture handlers ─────────────────────────────────────────────────────

    def _handle_point(self, sx: int, sy: int, ev: GestureEvent) -> None:
        if ev == GestureEvent.ENTERED:
            self._release_drag()
            self._reset_scroll()
        self._enqueue(MouseAction(ActionType.MOVE, sx, sy))

    def _handle_pinch(self, sx: int, sy: int, hold_s: float, ev: GestureEvent) -> None:
        gs = self._gs

        if ev == GestureEvent.ENTERED:
            self._pinch_start = time.monotonic()
            self._click_fired = False

        elif ev == GestureEvent.HELD:
            if hold_s >= gs.drag_lock_s and not self._is_dragging:
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
        if ev != GestureEvent.ENTERED:
            return
        now = time.monotonic()
        if now - self._last_right_click >= C.RIGHT_CLICK_DEBOUNCE_S:
            self._enqueue(MouseAction(ActionType.RIGHT_CLICK))
            self._last_right_click = now
            log.debug("Right click")

    def _handle_open_hand(self, sx: int, sy: int, ev: GestureEvent) -> None:
        if ev == GestureEvent.ENTERED:
            self._release_drag()
            self._reset_scroll()
        self._accumulate_scroll(sy, active=True)

    def _handle_none(self, sx: int, sy: int, ev: GestureEvent) -> None:
        if ev == GestureEvent.ENTERED:
            self._release_drag()
            self._reset_scroll()

    # ── Scroll accumulator (FIXED) ────────────────────────────────────────────

    def _accumulate_scroll(self, screen_y: int, active: bool) -> None:
        """
        Fixed scroll accumulator.

        Key fixes vs original:
          1. ref_y rolls toward current position (controlled by rolling_ref_lerp)
             so scroll never stalls when hand drifts back to anchor.
          2. ticks is cast to int before enqueue — pyautogui.scroll() silently
             ignores floats on Windows which caused erratic / no-scroll behaviour.
          3. speed_multiplier is applied as float then rounded to int for OS call.
          4. Accumulator remainder is preserved so no delta is lost between ticks.
        """
        ss = self._ss

        if not active:
            self._reset_scroll()
            return

        if self._scroll_ref_y is None:
            self._scroll_ref_y = float(screen_y)
            return

        # ── Rolling reference (the main fix for "scroll stops") ────────────
        # Slowly lerp ref_y toward current position so even if the hand
        # drifts back toward the original anchor, the reference follows and
        # scroll keeps firing.  rolling_ref_lerp=0.08 means ref moves ~8% of
        # the gap per frame — slow enough to not kill the delta, fast enough
        # to prevent stall.
        if ss.rolling_ref:
            self._scroll_ref_y += (screen_y - self._scroll_ref_y) * ss.rolling_ref_lerp

        delta = screen_y - self._scroll_ref_y   # positive = hand moved down

        # Smooth accumulation — prevents single-frame spike from firing
        self._scroll_accum = (
            self._scroll_accum * (1.0 - ss.accumulation_smooth)
            + delta            * ss.accumulation_smooth
        )

        if abs(self._scroll_accum) >= ss.pixels_per_tick:
            # How many full ticks accumulated?
            raw_ticks = self._scroll_accum / ss.pixels_per_tick

            # Apply speed multiplier (float), then round to nearest int for OS
            ticks_float = raw_ticks * ss.speed_multiplier
            ticks_int   = int(round(ticks_float))   # FIX: was int(float) → truncation

            if ticks_int == 0:
                # rounding landed on 0 (very slow scroll) — skip this frame
                return

            # Natural direction: hand down → page scrolls down = negative for pyautogui
            if ss.natural_direction:
                ticks_int = -ticks_int

            self._enqueue(MouseAction(ActionType.SCROLL, ticks=ticks_int))

            # Keep remainder so no delta is lost between ticks
            full_ticks_used = int(round(raw_ticks))
            self._scroll_accum -= full_ticks_used * ss.pixels_per_tick

            log.debug(f"Scroll ticks={ticks_int}  accum_rem={self._scroll_accum:.1f}")

    def _reset_scroll(self) -> None:
        self._scroll_ref_y = None
        self._scroll_accum = 0.0

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
                pyautogui.scroll(action.ticks)   # always int now
        except Exception as exc:
            log.error(f"Mouse action {action.kind.name} failed: {exc}")

    def reconfigure(self, cs: CursorSettings, gs: GestureSettings, ss: ScrollSettings) -> None:
        self._cs = cs
        self._gs = gs
        self._ss = ss

    def release_all(self) -> None:
        self._release_drag()
        self.flush()
        log.info("MouseController: all released")

    @property
    def is_dragging(self) -> bool:
        return self._is_dragging

    @property
    def screen_pos(self) -> Tuple[int, int]:
        return self._cur_x, self._cur_y