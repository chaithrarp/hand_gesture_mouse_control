"""
ui/dashboard.py

On-screen dashboard and debug overlay for Hand Gesture Mouse Control.

Draws directly onto the OpenCV frame (in-place mutation, no copy).
Called once per frame from main.py after all processing is done.

Panels:
  ── HUD (always visible) ──────────────────────────────────────
  Top-left:   FPS counter + frame latency (ms)
  Top-left:   Control status  (ACTIVE / PAUSED / DRAG)
  Top-left:   Per-hand gesture name + confidence bar
  Bottom bar: Hotkey hints

  ── Guide overlay (toggle with T) ────────────────────────────
  Semi-transparent full-screen panel listing all gestures.

  ── Settings sidebar (shown when show_guide=True) ─────────────
  Current Settings values so you can verify hot-reloads.

Design rules:
  - ASCII-only text — cv2 built-in fonts have no Unicode support.
    Any non-ASCII char renders as '???'. All labels here are plain ASCII.
  - No allocations in hot path — reuse pre-built strings where possible.
  - All drawing helpers are private staticmethods (no state needed).
"""

import cv2
import numpy as np
from typing import Optional

from config import constants as C
from config.settings import Settings
from control.mouse_controller import MouseController
from core.gesture_recognizer import GestureType, GestureResult


# ── Gesture display names (ASCII-only, cv2-safe) ──────────────────────────────
_GESTURE_LABEL = {
    GestureType.NONE:      "None",
    GestureType.POINT:     "POINT",
    GestureType.PINCH:     "PINCH",
    GestureType.OPEN_HAND: "OPEN HAND",
    GestureType.FIST:      "FIST",
    GestureType.VICTORY:   "VICTORY",
}

_GESTURE_COLORS = {
    GestureType.NONE:      C.C_GRAY,
    GestureType.POINT:     C.C_GREEN,
    GestureType.PINCH:     C.C_ORANGE,
    GestureType.OPEN_HAND: C.C_CYAN,
    GestureType.FIST:      C.C_RED,
    GestureType.VICTORY:   C.C_MAGENTA,
}


class Dashboard:
    """
    Stateless-ish overlay renderer. Holds a reference to Settings
    so it can display current parameter values without being passed
    them every frame.

    Usage (from main.py, once per frame):
        dashboard.draw(
            frame      = bgr_frame,
            fps        = current_fps,
            frame_ms   = last_frame_latency_ms,
            results    = {"Left": (result, sx, sy, color), ...},
            mouse      = mouse_controller_instance,
            show_guide = bool,
        )
    """

    def __init__(self, cfg: Settings):
        self._cfg = cfg

    # ── Public entry point ────────────────────────────────────────────────────

    def draw(
        self,
        frame:      np.ndarray,
        fps:        float,
        frame_ms:   float,
        results:    dict,   # label -> (GestureResult, sx, sy, color)
        mouse:      MouseController,
        show_guide: bool,
    ) -> None:
        """
        Render all dashboard elements onto frame in-place.

        Args:
            frame:      BGR frame to draw on (mutated in place).
            fps:        Current frames-per-second.
            frame_ms:   Time for last frame in milliseconds.
            results:    Per-hand recognition results from main.py.
            mouse:      MouseController for status flags (paused, dragging).
            show_guide: Whether to render the full gesture guide overlay.
        """
        h, w = frame.shape[:2]

        self._draw_hud(frame, fps, frame_ms, results, mouse, w, h)

        if show_guide:
            self._draw_guide_overlay(frame, w, h)
            self._draw_settings_sidebar(frame, w, h)

        self._draw_bottom_bar(frame, w, h)

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        frame:    np.ndarray,
        fps:      float,
        frame_ms: float,
        results:  dict,
        mouse:    MouseController,
        w: int, h: int,
    ) -> None:
        """Top-left HUD: FPS, latency, control status, per-hand gesture."""

        # Semi-transparent backing for readability
        panel_h = 40 + len(results) * 32 + 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (340, panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        y = 28

        # ── FPS + latency ──
        fps_color = C.C_GREEN if fps >= 20 else (C.C_YELLOW if fps >= 12 else C.C_RED)
        self._text(frame, f"FPS: {fps:.1f}   {frame_ms:.1f}ms/frame",
                   (10, y), 0.65, fps_color, 2)
        y += 28

        # ── Control status ──
        if mouse.paused:
            status, sc = "CONTROL: PAUSED", C.C_RED
        elif mouse.is_dragging:
            status, sc = "CONTROL: DRAG", C.C_ORANGE
        else:
            status, sc = "CONTROL: ACTIVE", C.C_GREEN
        self._text(frame, status, (10, y), 0.65, sc, 2)
        y += 28

        # ── Per-hand gesture ──
        if results:
            self._text(frame, "Hands:", (10, y), 0.58, C.C_WHITE, 1)
            y += 24
            for label, (result, sx, sy, color) in sorted(results.items()):
                g_name = _GESTURE_LABEL.get(result.gesture, result.gesture.value)
                line   = f"  {label:5s}: {g_name}"
                self._text(frame, line, (10, y), 0.60, color, 2)

                # Confidence bar (50px wide, colour-matched)
                bar_x   = 220
                bar_y   = y - 12
                bar_w   = int(result.confidence * 50)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 50, bar_y + 10),
                              C.C_GRAY, 1)
                if bar_w > 0:
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10),
                                  color, -1)
                conf_txt = f"{result.confidence:.2f}"
                self._text(frame, conf_txt, (bar_x + 55, y), 0.45, C.C_WHITE, 1)
                y += 28
        else:
            self._text(frame, "  No hands detected", (10, y), 0.58, C.C_GRAY, 1)
            y += 28

    # ── Guide overlay ─────────────────────────────────────────────────────────

    def _draw_guide_overlay(self, frame: np.ndarray, w: int, h: int) -> None:
        """Full-screen semi-transparent gesture reference card. Press T to toggle."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 30), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Title
        title = "GESTURE GUIDE  (press T to close)"
        ts    = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.05, 2)[0]
        cv2.putText(frame, title, ((w - ts[0]) // 2, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.05, C.C_CYAN, 2, cv2.LINE_AA)

        # Divider
        cv2.line(frame, (60, 80), (w - 60, 80), C.C_GRAY, 1)

        # Gesture rows
        rows = [
            (GestureType.POINT,     "[POINT]     Index finger only        ->  Move cursor"),
            (GestureType.PINCH,     "[PINCH]     Thumb + Index tip touch  ->  Click / Drag"),
            (GestureType.VICTORY,   "[VICTORY]   Index + Middle V shape   ->  Right click"),
            (GestureType.OPEN_HAND, "[OPEN HAND] All 5 fingers spread     ->  Scroll"),
            (GestureType.FIST,      "[FIST]      All fingers curled       ->  Pause / Resume"),
        ]
        y = 125
        for gesture, desc in rows:
            color = _GESTURE_COLORS[gesture]
            cv2.putText(frame, desc, (80, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2, cv2.LINE_AA)
            y += 52

        # Divider + hotkeys
        cv2.line(frame, (60, y + 5), (w - 60, y + 5), C.C_GRAY, 1)
        y += 28
        hotkeys = "Q=quit   T=guide   P=settings   C=calibrate   R=tutorial   FIST=pause"
        ts2 = cv2.getTextSize(hotkeys, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0]
        cv2.putText(frame, hotkeys, ((w - ts2[0]) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, C.C_GRAY, 1, cv2.LINE_AA)

    # ── Settings sidebar ─────────────────────────────────────────────────────

    def _draw_settings_sidebar(self, frame: np.ndarray, w: int, h: int) -> None:
        """
        Right-side panel showing active Settings values.
        Only visible when guide overlay is open.
        Lets you verify that settings are being picked up correctly.
        """
        cfg   = self._cfg
        panel = 280
        px    = w - panel - 10
        py    = 10

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 8, py), (w - 2, py + 310), (20, 10, 10), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        def row(label: str, val, y_off: int, color=C.C_WHITE):
            txt = f"{label}: {val}"
            cv2.putText(frame, txt, (px, py + y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        cv2.putText(frame, "-- Active Settings --", (px, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C.C_CYAN, 1, cv2.LINE_AA)

        cs = cfg.cursor
        ss = cfg.scroll
        ks = cfg.kalman
        gs = cfg.gesture
        ps = cfg.performance

        row("Speed",         f"{cs.speed:.2f}",       45,  C.C_GREEN)
        row("Smoothing",     f"{cs.smoothing:.2f}",   67,  C.C_GREEN)
        row("Adaptive smth", cs.adaptive_smoothing,   89,  C.C_GREEN)
        row("Dead zone",     f"{cs.dead_zone_px}px",  111, C.C_GREEN)
        row("Hover lock",    cs.hover_lock_enabled,   133, C.C_GREEN)
        row("Hover ms",      cs.hover_lock_ms,        155, C.C_GREEN)

        row("Scroll px/tick",f"{ss.pixels_per_tick:.0f}", 183, C.C_CYAN)
        row("Scroll mult",   ss.speed_multiplier,     205, C.C_CYAN)
        row("Scroll natural",ss.natural_direction,    227, C.C_CYAN)

        row("Kalman Q",      f"{ks.process_noise:.3f}",    255, C.C_ORANGE)
        row("Kalman R",      f"{ks.measurement_noise:.3f}", 277, C.C_ORANGE)

        row("Infer size",    f"{ps.inference_width}x{ps.inference_height}", 305, C.C_GRAY)

    # ── Bottom bar ────────────────────────────────────────────────────────────

    def _draw_bottom_bar(self, frame: np.ndarray, w: int, h: int) -> None:
        """Persistent hotkey reminder at the very bottom of the frame."""
        cv2.rectangle(frame, (0, h - 28), (w, h), (18, 18, 18), -1)
        hint = "Q=quit   T=guide   P=settings   C=calibrate   R=tutorial   FIST=pause"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(frame, hint, ((w - hw) // 2, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C.C_CYAN, 1, cv2.LINE_AA)

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _text(
        frame:  np.ndarray,
        text:   str,
        pos:    tuple,
        scale:  float,
        color:  tuple,
        thick:  int,
    ) -> None:
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)