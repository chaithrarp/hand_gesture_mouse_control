"""
Interactive settings panel — draw sliders on screen and let the user
drag them with the hand cursor (PINCH to grab, move hand, release to drop).

Design rules:
  - Panel is purely a RENDERER + HIT-TESTER. It never imports MouseController
    or touches any detection/filter code. Zero risk to hand tracking.
  - All interaction goes through two calls from main.py:
        panel.update(cursor_x, cursor_y, is_pinching, frame_w, frame_h)  → call every frame
        panel.draw(frame)                               → call every frame
  - Settings are mutated directly on the shared Settings instance so every
    other module sees the change immediately (live hot-reload).
  - Panel is toggled by pressing I (for "Interactive settings").
    P still opens the read-only sidebar in dashboard.py.
  - Sliders clamp values inside their own min/max — no need to call
    __post_init__ again (those clamps are for startup only).

Slider categories rendered:
  SCROLL    — the ones users care about most
  CURSOR    — speed, smoothing
  GESTURE   — pinch / extend sensitivity (advanced)
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

from config import constants as C
from config.settings import Settings


# ── Palette (BGR) ─────────────────────────────────────────────────────────────
_BG        = (12, 18, 28)
_PANEL_BOR = (60, 120, 60)
_TITLE     = (0, 220, 80)
_SECT      = (255, 200, 0)
_LABEL     = (200, 200, 200)
_TRACK     = (50, 50, 50)
_FILL      = (0, 180, 80)
_FILL_HOV  = (0, 220, 120)
_FILL_DRAG = (0, 120, 255)
_THUMB     = (255, 255, 255)
_HINT      = (90, 90, 90)
_VAL       = (0, 210, 255)


# ── Single slider descriptor ──────────────────────────────────────────────────
@dataclass
class _Slider:
    label:    str
    val_min:  float
    val_max:  float
    step:     float          # snapping resolution
    getter:   Callable[[], float]
    setter:   Callable[[float], None]
    fmt:      str  = "{:.2f}"   # display format
    section:  str  = ""         # section header (printed above if non-empty)

    # Runtime geometry (filled in by _layout)
    track_x0: int = 0
    track_x1: int = 0
    track_y:  int = 0          # vertical centre of track
    hovered:  bool = False
    dragging: bool = False

    @property
    def value(self) -> float:
        return self.getter()

    @value.setter
    def value(self, v: float) -> None:
        clamped = max(self.val_min, min(self.val_max, v))
        # Snap to step
        if self.step > 0:
            clamped = round(clamped / self.step) * self.step
        self.setter(clamped)

    def thumb_x(self) -> int:
        """Pixel x of the slider thumb."""
        t = (self.value - self.val_min) / max(self.val_max - self.val_min, 1e-6)
        return int(self.track_x0 + t * (self.track_x1 - self.track_x0))

    def value_from_x(self, x: int) -> float:
        t = (x - self.track_x0) / max(self.track_x1 - self.track_x0, 1)
        t = max(0.0, min(1.0, t))
        return self.val_min + t * (self.val_max - self.val_min)


# ── Panel ─────────────────────────────────────────────────────────────────────
THUMB_R   = 9    # thumb circle radius (px)
TRACK_H   = 4    # track bar height (px)
ROW_H     = 38   # vertical spacing between sliders
SECT_H    = 28   # extra space above section header
PANEL_PAD = 16   # inner horizontal padding

class SettingsPanel:
    """
    Interactive slider panel.

    One instance created in App.__init__() and kept alive for the session.
    Visibility toggled by pressing I.

    Usage in main.py frame loop (NORMAL mode only, after hand processing):

        # pass the latest cursor position and pinch state every frame
        self._settings_panel.update(cursor_x, cursor_y, is_pinching, frame_w, frame_h)
        if self._settings_panel.visible:
            self._settings_panel.draw(frame)

    The panel sits on the LEFT side so it doesn't overlap the right-side
    sidebar (P key) or the HUD.  Width is fixed at PANEL_W.
    """

    PANEL_W = 360

    def __init__(self, cfg: Settings):
        self._cfg     = cfg
        self.visible  = False          # toggled by App on I key
        self._sliders: List[_Slider]  = []
        self._active:  Optional[_Slider] = None   # currently dragged slider
        self._panel_h = 0              # computed by _build_sliders

        self._build_sliders()

    # ── Public API ────────────────────────────────────────────────────────────

    def toggle(self) -> None:
        self.visible = not self.visible

    def update(self, cursor_x: int, cursor_y: int, is_pinching: bool, frame_w: int = 0, frame_h: int = 0) -> None:
        """
        Call every frame with the current hand cursor position and pinch state.
        Handles hover detection and slider dragging.

        Args:
            cursor_x:    Screen-space X from CoordinateMapper (int).
            cursor_y:    Screen-space Y from CoordinateMapper (int).
            is_pinching: True while PINCH gesture is HELD.
            frame_w:     Frame width (used to convert screen coords to frame-relative).
            frame_h:     Frame height (used to convert screen coords to frame-relative).
        """
        if not self.visible:
            return

        # Convert screen coordinates to frame-relative coordinates
        # Panel sits at x = frame_w - PANEL_W, so subtract that offset
        frame_cursor_x = cursor_x - (frame_w - self.PANEL_W) if frame_w > 0 else cursor_x
        frame_cursor_y = cursor_y

        # ── Hover detection ───────────────────────────────────────────────
        for s in self._sliders:
            hit = self._thumb_hit(s, frame_cursor_x, frame_cursor_y)
            s.hovered = hit

        # ── Drag start ────────────────────────────────────────────────────
        if is_pinching and self._active is None:
            for s in self._sliders:
                if s.hovered:
                    s.dragging = True
                    self._active = s
                    break

        # ── Drag move ─────────────────────────────────────────────────────
        if self._active is not None:
            if is_pinching:
                new_val = self._active.value_from_x(frame_cursor_x)
                self._active.value = new_val
            else:
                # Pinch released — drop the slider
                self._active.dragging = False
                self._active = None

    def draw(self, frame: np.ndarray) -> None:
        """
        Render the panel onto frame in-place.
        Call AFTER hand skeleton drawing so the panel sits on top.
        """
        if not self.visible:
            return

        h, w = frame.shape[:2]
        self._layout_sliders(w, h)

        # Panel background
        px0 = w - self.PANEL_W
        px1 = w
        overlay = frame.copy()
        cv2.rectangle(overlay, (px0, 0), (px1, self._panel_h), _BG, -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        cv2.rectangle(frame, (px0, 0), (px1 - 1, self._panel_h - 1), _PANEL_BOR, 1)

        # Title bar
        cv2.rectangle(frame, (px0, 0), (px1, 30), (20, 35, 20), -1)
        self._put(frame, "SETTINGS  (I=close, PINCH slider to adjust)",
                  (px0 + 8, 20), 0.45, _TITLE, 1)

        # Sliders
        for s in self._sliders:
            self._draw_slider(frame, s)

        # Footer hint
        y_foot = self._panel_h - 12
        self._put(frame, "Hover thumb  ->  pinch + move hand to drag",
                  (px0 + 8, y_foot), 0.42, _HINT, 1)

    # ── Slider builder ────────────────────────────────────────────────────────

    def _build_sliders(self) -> None:
        """
        Define all sliders.  Getters/setters point directly at Settings
        fields — changing the slider value immediately updates the live setting.
        """
        cfg = self._cfg
        ss  = cfg.scroll
        cs  = cfg.cursor
        gs  = cfg.gesture

        self._sliders = [
            # ── SCROLL ────────────────────────────────────────────────────
            _Slider(
                label="px per tick (lower=faster)",
                val_min=3.0, val_max=40.0, step=0.5,
                getter=lambda: cfg.scroll.pixels_per_tick,
                setter=lambda v: setattr(cfg.scroll, "pixels_per_tick", v),
                fmt="{:.1f}",
                section="SCROLL",
            ),
            _Slider(
                label="speed multiplier",
                val_min=0.5, val_max=8.0, step=0.25,
                getter=lambda: cfg.scroll.speed_multiplier,
                setter=lambda v: setattr(cfg.scroll, "speed_multiplier", v),
                fmt="{:.2f}",
            ),
            _Slider(
                label="accumulation smooth",
                val_min=0.05, val_max=0.8, step=0.01,
                getter=lambda: cfg.scroll.accumulation_smooth,
                setter=lambda v: setattr(cfg.scroll, "accumulation_smooth", v),
                fmt="{:.2f}",
            ),
            _Slider(
                label="rolling ref lerp",
                val_min=0.01, val_max=0.3, step=0.01,
                getter=lambda: cfg.scroll.rolling_ref_lerp,
                setter=lambda v: setattr(cfg.scroll, "rolling_ref_lerp", v),
                fmt="{:.2f}",
            ),

            # ── CURSOR ────────────────────────────────────────────────────
            _Slider(
                label="cursor speed",
                val_min=0.3, val_max=3.0, step=0.05,
                getter=lambda: cfg.cursor.speed,
                setter=lambda v: setattr(cfg.cursor, "speed", v),
                fmt="{:.2f}",
                section="CURSOR",
            ),
            _Slider(
                label="smoothing",
                val_min=0.05, val_max=0.95, step=0.01,
                getter=lambda: cfg.cursor.smoothing,
                setter=lambda v: setattr(cfg.cursor, "smoothing", v),
                fmt="{:.2f}",
            ),
            _Slider(
                label="hover lock ms",
                val_min=50, val_max=500, step=10,
                getter=lambda: float(cfg.cursor.hover_lock_ms),
                setter=lambda v: setattr(cfg.cursor, "hover_lock_ms", int(v)),
                fmt="{:.0f}ms",
            ),

            # ── GESTURE ───────────────────────────────────────────────────
            _Slider(
                label="pinch sensitivity",
                val_min=0.10, val_max=0.50, step=0.01,
                getter=lambda: cfg.gesture.pinch_sensitivity,
                setter=lambda v: setattr(cfg.gesture, "pinch_sensitivity", v),
                fmt="{:.2f}",
                section="GESTURE",
            ),
            _Slider(
                label="extend sensitivity",
                val_min=0.05, val_max=0.30, step=0.01,
                getter=lambda: cfg.gesture.extend_sensitivity,
                setter=lambda v: setattr(cfg.gesture, "extend_sensitivity", v),
                fmt="{:.2f}",
            ),
            _Slider(
                label="debounce (s)",
                val_min=0.03, val_max=0.25, step=0.01,
                getter=lambda: cfg.gesture.debounce_s,
                setter=lambda v: setattr(cfg.gesture, "debounce_s", v),
                fmt="{:.2f}",
            ),
        ]

        # Compute panel height
        y = 40
        for s in self._sliders:
            if s.section:
                y += SECT_H
            y += ROW_H
        y += 24   # footer
        self._panel_h = y

    # ── Layout (called each draw — panel width may change on resize) ──────────

    def _layout_sliders(self, frame_w: int, frame_h: int) -> None:
        px0 = frame_w - self.PANEL_W
        track_x0 = px0 + PANEL_PAD + 10
        track_x1 = frame_w - PANEL_PAD - 50   # leave room for value label

        y = 48
        for s in self._sliders:
            if s.section:
                y += SECT_H
            s.track_x0 = track_x0
            s.track_x1 = track_x1
            s.track_y   = y + ROW_H // 2
            y += ROW_H

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_slider(self, frame: np.ndarray, s: _Slider) -> None:
        tx0 = s.track_x0
        tx1 = s.track_x1
        ty  = s.track_y
        px0 = frame.shape[1] - self.PANEL_W

        # Section header
        if s.section:
            sy = ty - ROW_H // 2 - SECT_H + 10
            cv2.line(frame, (px0 + 6, sy + 2), (frame.shape[1] - 6, sy + 2),
                     (40, 60, 40), 1)
            self._put(frame, s.section, (px0 + 8, sy + 14), 0.52, _SECT, 1)

        # Label
        self._put(frame, s.label, (tx0, ty - 10), 0.44, _LABEL, 1)

        # Track background
        cv2.line(frame, (tx0, ty), (tx1, ty), _TRACK, TRACK_H)

        # Fill up to thumb
        thumb_x = s.thumb_x()
        if thumb_x > tx0:
            fill_col = _FILL_DRAG if s.dragging else (_FILL_HOV if s.hovered else _FILL)
            cv2.line(frame, (tx0, ty), (thumb_x, ty), fill_col, TRACK_H)

        # Thumb circle
        thumb_col = _FILL_DRAG if s.dragging else (_FILL_HOV if s.hovered else _THUMB)
        cv2.circle(frame, (thumb_x, ty), THUMB_R, thumb_col, -1, cv2.LINE_AA)
        cv2.circle(frame, (thumb_x, ty), THUMB_R, C.C_WHITE, 1, cv2.LINE_AA)

        # Value label to the right of the track
        val_str = s.fmt.format(s.value)
        self._put(frame, val_str, (tx1 + 5, ty + 5), 0.50, _VAL, 1)

    @staticmethod
    def _put(frame, text, pos, scale, color, thick):
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def _thumb_hit(s: _Slider, cx: int, cy: int) -> bool:
        """True if cursor is within THUMB_R*2.5 of the thumb centre."""
        tx = s.thumb_x()
        ty = s.track_y
        hit_r = THUMB_R * 2.5
        return abs(cx - tx) <= hit_r and abs(cy - ty) <= hit_r