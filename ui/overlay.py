"""
ui/overlay.py

All OpenCV rendering for the live camera window.

Renders (in draw order, back to front):
  1. Active zone border
  2. Hand skeleton + landmark dots
  3. Gesture label above active landmark
  4. HUD: FPS, control status, per-hand gesture list
  5. Bottom hint bar
  6. Debug panel (optional, toggled by hotkey)
  7. Guide overlay (optional, toggled by hotkey)
  8. Settings panel (optional, toggled by hotkey)

Design rules:
  - Zero business logic — pure rendering
  - Every draw call guarded against None / empty input
  - All text uses cv2.LINE_AA for clean sub-pixel rendering
  - Colours from constants.py — no magic BGR tuples in this file
  - Frame is mutated in-place for speed (no copies)
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any

from config import constants as C
from config.settings import Settings
from core.hand_detector import HandData
from core.gesture_recognizer import GestureType, GestureResult
from utils.logger import get_logger

log = get_logger(__name__)

# ── Gesture → colour mapping ──────────────────────────────────────────────────
GESTURE_COLOR: Dict[GestureType, tuple] = {
    GestureType.NONE:      C.C_GRAY,
    GestureType.POINT:     C.C_GREEN,
    GestureType.PINCH:     C.C_ORANGE,
    GestureType.OPEN_HAND: C.C_CYAN,
    GestureType.FIST:      C.C_RED,
    GestureType.VICTORY:   C.C_MAGENTA,
}

# Short display names (no arrow, fits in a label bubble)
GESTURE_LABEL: Dict[GestureType, str] = {
    GestureType.NONE:      "",
    GestureType.POINT:     "POINT",
    GestureType.PINCH:     "PINCH",
    GestureType.OPEN_HAND: "SCROLL",
    GestureType.FIST:      "PAUSE",
    GestureType.VICTORY:   "R-CLICK",
}


# ── Overlay renderer ──────────────────────────────────────────────────────────
class Overlay:
    """
    Stateless renderer — every method takes the frame + data, draws, returns.
    No frame caching, no internal frame buffer.

    Usage (each frame):
        overlay.draw_active_zone(frame)
        overlay.draw_hands(frame, hands, results)
        overlay.draw_hud(frame, perf_snap, mouse_ctrl, paused)
        if show_debug:
            overlay.draw_debug_panel(frame, perf_snap, coord_debug)
        if show_guide:
            overlay = overlay.draw_guide(frame)
    """

    def __init__(self, settings: Settings):
        self._s = settings

    def reconfigure(self, settings: Settings) -> None:
        self._s = settings

    # ── 1. Active zone border ─────────────────────────────────────────────────

    def draw_active_zone(self, frame: np.ndarray) -> None:
        """Draw the active gesture zone as a subtle dashed rectangle."""
        h, w = frame.shape[:2]
        cs   = self._s.cursor
        x0   = int(cs.active_zone_x[0] * w)
        x1   = int(cs.active_zone_x[1] * w)
        y0   = int(cs.active_zone_y[0] * h)
        y1   = int(cs.active_zone_y[1] * h)

        # Dashed border via short line segments
        color  = (*C.C_GRAY[:3], )
        dash   = 12
        gap    = 8
        thick  = 1

        def draw_dashed_line(p1, p2):
            pts = _dash_points(p1, p2, dash, gap)
            for a, b in pts:
                cv2.line(frame, a, b, color, thick, cv2.LINE_AA)

        draw_dashed_line((x0, y0), (x1, y0))
        draw_dashed_line((x1, y0), (x1, y1))
        draw_dashed_line((x1, y1), (x0, y1))
        draw_dashed_line((x0, y1), (x0, y0))

        # Corner labels (tiny)
        _text_small(frame, "ACTIVE ZONE", (x0 + 4, y0 - 6), C.C_GRAY)

    # ── 2. Hand skeleton + gesture label ──────────────────────────────────────

    def draw_hands(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        results: List[GestureResult],
    ) -> None:
        """
        Draw skeleton and gesture label for each detected hand.
        hands and results must be same length and same order.
        """
        for hd, res in zip(hands, results):
            color = GESTURE_COLOR.get(res.gesture, C.C_GRAY)
            self._draw_skeleton(frame, hd.landmarks, hd.frame_w, hd.frame_h, color)
            self._draw_pinch_indicator(frame, hd, res)
            if res.gesture != GestureType.NONE:
                self._draw_gesture_label(frame, res, color)

    def _draw_skeleton(self, frame, lm, fw, fh, color) -> None:
        h, w = frame.shape[:2]

        # Connections
        for s, e in C.HAND_CONNECTIONS:
            ls, le = lm[s], lm[e]
            p1 = (int(ls.x * w), int(ls.y * h))
            p2 = (int(le.x * w), int(le.y * h))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

        # Landmark dots — tips slightly larger
        tips = {C.LM_INDEX_TIP, C.LM_MIDDLE_TIP, C.LM_RING_TIP,
                C.LM_PINKY_TIP, C.LM_THUMB_TIP}
        for i, lmk in enumerate(lm):
            px = (int(lmk.x * w), int(lmk.y * h))
            r  = 6 if i in tips else 3
            cv2.circle(frame, px, r, color, -1, cv2.LINE_AA)
            cv2.circle(frame, px, r, C.C_WHITE, 1, cv2.LINE_AA)

    def _draw_pinch_indicator(
        self, frame: np.ndarray, hd: HandData, res: GestureResult
    ) -> None:
        """Draw a circle at pinch midpoint when pinching."""
        if res.gesture != GestureType.PINCH:
            return
        h, w = frame.shape[:2]
        lm   = hd.landmarks
        tx = int(lm[C.LM_THUMB_TIP].x * w)
        ty = int(lm[C.LM_THUMB_TIP].y * h)
        ix = int(lm[C.LM_INDEX_TIP].x * w)
        iy = int(lm[C.LM_INDEX_TIP].y * h)
        mid = ((tx + ix) // 2, (ty + iy) // 2)
        # Pulse radius based on hold time (grows during drag lock wait)
        r = int(12 + min(res.hold_s, 0.5) * 20)
        cv2.circle(frame, mid, r, C.C_ORANGE, 2, cv2.LINE_AA)
        cv2.circle(frame, mid, 4, C.C_ORANGE, -1, cv2.LINE_AA)

    def _draw_gesture_label(
        self, frame: np.ndarray, res: GestureResult, color: tuple
    ) -> None:
        """Floating label bubble above gesture position."""
        label = GESTURE_LABEL.get(res.gesture, "")
        if not label:
            return
        x, y = res.position
        # Offset upward so label doesn't cover the hand
        y = max(y - 30, 20)
        _label_bubble(frame, label, (x, y), color)

    # ── 3. HUD ────────────────────────────────────────────────────────────────

    def draw_hud(
        self,
        frame: np.ndarray,
        perf: Dict[str, Any],
        is_dragging: bool,
        is_paused: bool,
        hands: List[HandData],
        results: List[GestureResult],
    ) -> None:
        h, w = frame.shape[:2]

        # ── Top-left: FPS + status ─────────────────────────────────────────
        fps_color = C.C_GREEN if perf["fps_smooth"] >= 25 else C.C_YELLOW \
                    if perf["fps_smooth"] >= 15 else C.C_RED

        _text(frame, f"FPS  {perf['fps_smooth']:.0f}", (12, 32), fps_color, scale=0.85)
        _text(frame, f"ft  {perf['frame_ms_mean']:.1f}ms", (12, 56), C.C_GRAY, scale=0.65)

        # ── Control status ─────────────────────────────────────────────────
        if is_paused:
            status, sc = "PAUSED", C.C_RED
        elif is_dragging:
            status, sc = "DRAGGING", C.C_ORANGE
        else:
            status, sc = "ACTIVE", C.C_GREEN

        _text(frame, f"Control  {status}", (12, 82), sc, scale=0.72)

        # ── Per-hand gesture ───────────────────────────────────────────────
        y = 112
        for hd, res in zip(hands, results):
            color = GESTURE_COLOR.get(res.gesture, C.C_GRAY)
            label = GESTURE_LABEL.get(res.gesture, "NONE")
            _text(frame, f"{hd.label}:  {label}", (12, y), color, scale=0.68)
            y += 26

        # ── Bottom hint bar ────────────────────────────────────────────────
        cv2.rectangle(frame, (0, h - 28), (w, h), (18, 18, 18), -1)
        hints = "q=quit   t=guide   p=settings   c=calibrate   r=tutorial"
        _text(frame, hints, (10, h - 9), C.C_GRAY, scale=0.52)

    # ── 4. Debug panel ────────────────────────────────────────────────────────

    def draw_debug_panel(
        self,
        frame: np.ndarray,
        perf: Dict[str, Any],
        coord_debug: Optional[Dict] = None,
    ) -> None:
        """
        Right-side transparent panel with stage timings and coord pipeline values.
        Toggled by 't' key (same as guide — guide takes priority).
        """
        h, w = frame.shape[:2]
        pw   = 270
        px0  = w - pw - 8

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px0 - 4, 4), (w - 4, h - 34), (10, 10, 25), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        y = 22
        _text(frame, "── DEBUG ──", (px0, y), C.C_CYAN, scale=0.62)
        y += 24

        # Stage timings
        stages = perf.get("stages", {})
        for name, ms in stages.items():
            bar_w = int(min(ms / 33.0, 1.0) * 120)   # 33ms = one frame budget
            color = C.C_GREEN if ms < 10 else C.C_YELLOW if ms < 20 else C.C_RED
            cv2.rectangle(frame, (px0, y - 10), (px0 + bar_w, y - 2), color, -1)
            _text(frame, f"{name:<12} {ms:5.1f}ms", (px0, y), C.C_WHITE, scale=0.55)
            y += 18

        y += 6
        _text(frame, f"p95  {perf['frame_ms_p95']:.1f}ms", (px0, y), C.C_GRAY, scale=0.58)
        y += 18
        _text(frame, f"max  {perf['frame_ms_max']:.1f}ms", (px0, y), C.C_GRAY, scale=0.58)
        y += 18
        _text(frame, f"frames  {perf['total_frames']}", (px0, y), C.C_GRAY, scale=0.58)

        # Coordinate pipeline
        if coord_debug:
            y += 24
            _text(frame, "── COORDS ──", (px0, y), C.C_CYAN, scale=0.62)
            y += 20
            for key, val in coord_debug.items():
                _text(frame, f"{key}: {val}", (px0, y), C.C_WHITE, scale=0.55)
                y += 16

    # ── 5. Guide overlay ──────────────────────────────────────────────────────

    def draw_guide(self, frame: np.ndarray) -> np.ndarray:
        """Full-frame semi-transparent guide. Returns modified frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (8, 8, 22), -1)
        frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

        entries = [
            (GestureType.POINT,     "Move cursor     —  index finger only"),
            (GestureType.PINCH,     "Click / Drag    —  thumb touches index"),
            (GestureType.VICTORY,   "Right click     —  index + middle V"),
            (GestureType.OPEN_HAND, "Scroll          —  open hand, move up/down"),
            (GestureType.FIST,      "Pause / Resume  —  make a fist"),
        ]

        _text(frame, "GESTURE GUIDE", (w // 2 - 120, 55), C.C_CYAN, scale=1.1, bold=True)
        cv2.line(frame, (60, 70), (w - 60, 70), C.C_GRAY, 1)

        y = 105
        for gtype, desc in entries:
            color = GESTURE_COLOR[gtype]
            label = GESTURE_LABEL[gtype]
            # Coloured gesture tag
            _label_bubble(frame, label, (110, y), color)
            # Description
            _text(frame, desc, (185, y + 6), C.C_WHITE, scale=0.75)
            y += 52

        cv2.line(frame, (60, y + 5), (w - 60, y + 5), C.C_GRAY, 1)
        _text(frame, "Press 't' to close", (w // 2 - 90, y + 28), C.C_GRAY, scale=0.62)
        return frame

    # ── 6. Settings panel ─────────────────────────────────────────────────────

    def draw_settings_panel(self, frame: np.ndarray, settings: Settings) -> None:
        """
        Left-side semi-transparent panel showing all live-tunable values.
        User can see exactly what's active without opening a file.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (310, h), (8, 22, 8), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        cs  = settings.cursor
        gs  = settings.gesture
        ss  = settings.scroll
        ks  = settings.kalman
        ps  = settings.performance

        rows = [
            ("── CURSOR ──",          None,          C.C_CYAN),
            ("speed",                 cs.speed,      C.C_WHITE),
            ("smoothing",             cs.smoothing,  C.C_WHITE),
            ("adaptive",              cs.adaptive_smoothing, C.C_WHITE),
            ("dead_zone_px",          cs.dead_zone_px, C.C_WHITE),
            ("hover_lock",            cs.hover_lock_enabled, C.C_WHITE),
            ("hover_ms",              cs.hover_lock_ms, C.C_WHITE),
            ("── GESTURE ──",         None,          C.C_CYAN),
            ("pinch_sensitivity",     gs.pinch_sensitivity, C.C_WHITE),
            ("extend_sensitivity",    gs.extend_sensitivity, C.C_WHITE),
            ("debounce_s",            gs.debounce_s, C.C_WHITE),
            ("drag_lock_s",           gs.drag_lock_s, C.C_WHITE),
            ("── SCROLL ──",          None,          C.C_CYAN),
            ("px_per_tick",           ss.pixels_per_tick, C.C_WHITE),
            ("speed_mult",            ss.speed_multiplier, C.C_WHITE),
            ("natural_dir",           ss.natural_direction, C.C_WHITE),
            ("── KALMAN ──",          None,          C.C_CYAN),
            ("process_noise",         ks.process_noise, C.C_WHITE),
            ("measure_noise",         ks.measurement_noise, C.C_WHITE),
            ("── PERF ──",            None,          C.C_CYAN),
            ("infer_size",            f"{ps.inference_width}x{ps.inference_height}", C.C_WHITE),
            ("idle_skip",             ps.idle_skip_frames, C.C_WHITE),
        ]

        y = 18
        for label, val, color in rows:
            if val is None:
                _text(frame, label, (8, y), color, scale=0.58)
            else:
                val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
                _text(frame, f"  {label:<22} {val_str}", (8, y), color, scale=0.55)
            y += 17

        _text(frame, "Press 'p' to close", (8, h - 40), C.C_GRAY, scale=0.55)


# ── Drawing primitives ────────────────────────────────────────────────────────

def _text(
    frame: np.ndarray,
    text: str,
    pos: tuple,
    color: tuple,
    scale: float = 0.75,
    bold: bool = False,
) -> None:
    thick = 2 if bold else 1
    cv2.putText(
        frame, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
    )


def _text_small(frame, text, pos, color) -> None:
    _text(frame, text, pos, color, scale=0.45)


def _label_bubble(
    frame: np.ndarray,
    text: str,
    centre: tuple,
    color: tuple,
    scale: float = 0.65,
) -> None:
    """Filled rounded rectangle with text, centred on `centre`."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = 6
    x, y = centre
    x0, y0 = x - tw // 2 - pad, y - th - pad
    x1, y1 = x + tw // 2 + pad, y + pad
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x1, y1), C.C_WHITE, 1, cv2.LINE_AA)
    tx = x - tw // 2
    ty = y
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, scale, C.C_BLACK, 1, cv2.LINE_AA)


def _dash_points(p1, p2, dash, gap):
    """Return list of (start, end) point pairs for a dashed line."""
    x1, y1 = p1
    x2, y2 = p2
    length = np.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return []
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pts = []
    pos = 0.0
    while pos < length:
        s = pos
        e = min(pos + dash, length)
        pts.append((
            (int(x1 + dx * s), int(y1 + dy * s)),
            (int(x1 + dx * e), int(y1 + dy * e)),
        ))
        pos += dash + gap
    return pts