"""
Hand Gesture Mouse Control — main.py
=====================================
Entry point. Wires together every production module:

    config.settings      → single Settings instance, passed everywhere
    core.gesture_recognizer → GestureStateMachine (debounce + events)
    processing.filters   → FilterPipeline (outlier → Kalman → adaptive smooth → hover lock)
    control.mouse_controller → MouseController (action queue, scroll untouched)
    utils.calibration    → CalibrationManager (4-point homography)
    ui.dashboard         → Dashboard (HUD, guide overlay, settings sidebar)

Hotkeys
-------
  Q / ESC   quit
  T         toggle gesture guide overlay
  R         restart tutorial
  S         skip tutorial (during tutorial only)
  P         toggle settings sidebar
  C         start/cancel calibration

Scroll behaviour is handled entirely inside MouseController._accumulate_scroll()
and has NOT been modified.  Do not move scroll logic out of that method.
"""

from __future__ import annotations

import sys
import time
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── stdlib shim so we can run even without the full package tree ───────────────
# If the module tree isn't on PYTHONPATH, fall back gracefully.
def _try_import(module: str):
    try:
        return __import__(module, fromlist=[""])
    except ImportError:
        return None

# ── Core imports ───────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
except ImportError:
    sys.exit("ERROR: mediapipe not installed.  Run: pip install mediapipe")

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE    = 0
    _MOUSE_OK = True
except ImportError:
    _MOUSE_OK = False

# ── Project modules ────────────────────────────────────────────────────────────
# We import from the package tree.  If the file is run from the project root
# (python main.py) all imports below resolve normally.

from config.settings import Settings
from config import constants as C

from core.gesture_recognizer import (
    GestureStateMachine,
    GestureType,
    GestureEvent,
    GestureResult,
)

from processing.filters import FilterPipeline

from control.mouse_controller import MouseController

from utils.calibration import CalibrationManager
from utils.logger import get_logger

log = get_logger(__name__)

# ── Dashboard ──────────────────────────────────────────────────────────────────
# ui/dashboard.py is the canonical renderer; import it.
from ui.dashboard import Dashboard


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE MAPPER
# Inline here so we don't depend on core/coordinate_mapper.py existing yet.
# If that module is added later, swap this class for the import.
# ─────────────────────────────────────────────────────────────────────────────
class CoordinateMapper:
    """
    Maps camera-pixel positions → screen-pixel positions.

    Two modes:
      • Default   — linear scale using active_zone in CursorSettings.
      • Calibrated — homography (3×3 matrix) from CalibrationManager.

    Speed multiplier (cs.speed) is applied in default mode so the user
    can tune responsiveness from settings without touching any other code.
    """

    def __init__(self, screen_w: int, screen_h: int, settings: Settings):
        self._sw  = screen_w
        self._sh  = screen_h
        self._cfg = settings
        self._H: Optional[np.ndarray] = None   # calibration homography

    def set_calibration(self, H: Optional[np.ndarray]) -> None:
        self._H = H
        log.info("CoordinateMapper: calibration %s",
                 "applied" if H is not None else "cleared")

    def map(self, cam_x: float, cam_y: float,
            frame_w: int, frame_h: int) -> Tuple[int, int]:
        """Return (screen_x, screen_y) clamped to screen bounds."""
        if self._H is not None:
            return self._map_homography(cam_x, cam_y)
        return self._map_linear(cam_x, cam_y, frame_w, frame_h)

    def _map_linear(self, cx: float, cy: float,
                    fw: int, fh: int) -> Tuple[int, int]:
        cs   = self._cfg.cursor
        az_x = cs.active_zone_x
        az_y = cs.active_zone_y
        spd  = cs.speed

        nx = (cx / fw - az_x[0]) / (az_x[1] - az_x[0])
        ny = (cy / fh - az_y[0]) / (az_y[1] - az_y[0])

        # Apply speed: pivot around 0.5 so centre stays centred
        nx = 0.5 + (nx - 0.5) * spd
        ny = 0.5 + (ny - 0.5) * spd

        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        return int(nx * self._sw), int(ny * self._sh)

    def _map_homography(self, cx: float, cy: float) -> Tuple[int, int]:
        pt  = np.array([[[cx, cy]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, self._H)
        sx  = int(np.clip(dst[0][0][0], 0, self._sw - 1))
        sy  = int(np.clip(dst[0][0][1], 0, self._sh - 1))
        return sx, sy


# ─────────────────────────────────────────────────────────────────────────────
# TUTORIAL
# ASCII-only text throughout — cv2 built-in fonts cannot render emoji or
# Unicode beyond basic Latin.  All "???" issues are caused by emoji in text.
# ─────────────────────────────────────────────────────────────────────────────
TUTORIAL_STEPS = [
    {
        "title": "Welcome!",
        "subtitle": "Control your mouse with hand gestures.",
        "lines": [
            "No clicking, no touching — just move your hand.",
            "",
            "Your dominant hand (right by default) controls",
            "the cursor.  Both hands are tracked.",
            "",
            "You will learn 5 gestures in the next screens.",
            "",
            "  SPACE = next page",
            "  S     = skip tutorial",
        ],
    },
    {
        "title": "[POINT]  ->  Move Cursor",
        "subtitle": "Your index finger is the cursor.",
        "lines": [
            "Extend your INDEX finger only.",
            "Keep all other fingers curled into your palm.",
            "",
            "Tip: the fingertip drives the cursor — point",
            "at the screen area you want to reach.",
            "",
            "  Move hand slowly  ->  precise targeting",
            "  Move hand fast    ->  cursor keeps up",
            "",
            "  SPACE = next page",
        ],
    },
    {
        "title": "[PINCH]  ->  Left Click / Drag",
        "subtitle": "Touch thumb tip to index tip.",
        "lines": [
            "Quick pinch + release  =  Left Click",
            "Hold pinch (0.5s) + move hand  =  Drag",
            "",
            "During drag: cursor follows your hand.",
            "Release pinch to drop.",
            "",
            "Tip: move your whole hand to the target",
            "first, THEN pinch — more accurate.",
            "",
            "  SPACE = next page",
        ],
    },
    {
        "title": "[VICTORY V]  ->  Right Click",
        "subtitle": "Index + middle fingers in a V shape.",
        "lines": [
            "Extend INDEX and MIDDLE fingers.",
            "Keep ring and pinky fingers curled.",
            "Spread the two fingers apart slightly.",
            "",
            "A right-click fires at the current",
            "cursor position (1 second cooldown).",
            "",
            "Tip: hold the V for half a second so",
            "the gesture is read cleanly.",
            "",
            "  SPACE = next page",
        ],
    },
    {
        "title": "[OPEN HAND]  ->  Scroll",
        "subtitle": "All five fingers spread open.",
        "lines": [
            "Palm facing the camera, all fingers extended.",
            "",
            "Move hand UP    ->  Scroll Up",
            "Move hand DOWN  ->  Scroll Down",
            "",
            "The scroll reference is set when you first",
            "open your hand — distance from that point",
            "determines scroll speed.",
            "",
            "  SPACE = next page",
        ],
    },
    {
        "title": "[FIST]  ->  Pause / Resume",
        "subtitle": "Make a closed fist to pause mouse control.",
        "lines": [
            "Curl ALL fingers into your palm.",
            "",
            "First fist  ->  PAUSES the cursor",
            "Second fist ->  RESUMES control",
            "",
            "Use this when you need to rest your hand",
            "without accidentally moving the cursor.",
            "",
            "Any active drag is released on pause.",
            "",
            "  SPACE = start!",
        ],
    },
]

# Colour palette for tutorial (BGR)
_T_BG      = (12, 10, 28)
_T_TITLE   = (0, 220, 80)
_T_SUB     = (0, 195, 255)
_T_BODY    = (220, 220, 220)
_T_HINT    = (110, 110, 110)
_T_DOT_ON  = (0, 220, 80)
_T_DOT_OFF = (70, 70, 70)


def draw_tutorial(frame: np.ndarray, step: int) -> np.ndarray:
    """Render the tutorial overlay. Returns the composited frame."""
    h, w = frame.shape[:2]

    # Dark overlay
    bg = np.full_like(frame, _T_BG)
    frame = cv2.addWeighted(bg, 0.92, frame, 0.08, 0)

    data  = TUTORIAL_STEPS[step]
    total = len(TUTORIAL_STEPS)

    # ── Step counter (top-left) ────────────────────────────────────────────
    cv2.putText(frame, f"Step {step + 1} / {total}",
                (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.70, _T_HINT, 1, cv2.LINE_AA)

    # ── Progress dots (top-right) ──────────────────────────────────────────
    dot_r   = 7
    spacing = 22
    total_w = (total - 1) * spacing
    start_x = w - total_w - 30
    for i in range(total):
        col = _T_DOT_ON if i == step else _T_DOT_OFF
        cx  = start_x + i * spacing
        cv2.circle(frame, (cx, 26), dot_r, col, -1, cv2.LINE_AA)
        if i == step:
            cv2.circle(frame, (cx, 26), dot_r + 2, _T_DOT_ON, 1, cv2.LINE_AA)

    # ── Divider ────────────────────────────────────────────────────────────
    cv2.line(frame, (40, 52), (w - 40, 52), (40, 40, 60), 1)

    # ── Title ─────────────────────────────────────────────────────────────
    title = data["title"]
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.25, 2)
    cv2.putText(frame, title, ((w - tw) // 2, 105),
                cv2.FONT_HERSHEY_DUPLEX, 1.25, _T_TITLE, 2, cv2.LINE_AA)

    # ── Subtitle ───────────────────────────────────────────────────────────
    sub = data.get("subtitle", "")
    if sub:
        (sw2, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 1)
        cv2.putText(frame, sub, ((w - sw2) // 2, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, _T_SUB, 1, cv2.LINE_AA)

    # ── Body lines ─────────────────────────────────────────────────────────
    y = 200
    for line in data["lines"]:
        if line == "":
            y += 18
            continue
        # Distinguish hints (start with spaces / "  ")
        if line.startswith("  "):
            col   = _T_HINT
            scale = 0.72
        else:
            col   = _T_BODY
            scale = 0.82
        (lw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(frame, line, ((w - lw) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
        y += int(scale * 50)

    # ── Bottom hint ────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 32), (w, h), (18, 16, 36), -1)
    hint = "SPACE = continue    S = skip tutorial    Q = quit"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, hint, ((w - hw) // 2, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _T_HINT, 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# PER-HAND STATE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HandData:
    """Snapshot of one hand for this frame (passed to Dashboard)."""
    label:    str
    landmarks: object    # mediapipe landmark list
    frame_w:  int
    frame_h:  int


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
class AppMode(Enum):
    TUTORIAL    = auto()
    NORMAL      = auto()
    CALIBRATING = auto()


class App:
    """
    Main application.  Owns the camera, MediaPipe session, and all
    per-hand state machines.  Delegates rendering to Dashboard and
    mouse actions to MouseController.

    Frame pipeline (NORMAL mode):
        1. Capture + flip
        2. Resize to inference resolution → MediaPipe
        3. For each detected hand:
              a. FilterPipeline  (cam px → stable cam px)
              b. CoordinateMapper (cam px → screen px)
              c. GestureStateMachine.update() → GestureResult
              d. MouseController.handle()
        4. MouseController.flush()
        5. Dashboard.draw()
        6. imshow
    """

    CAL_FILE = "calibration.npy"

    def __init__(self):
        self._cfg  = Settings()
        log.info("\n" + self._cfg.summary())

        self._mode = AppMode.TUTORIAL
        self._tutorial_step = 0

        # ── Camera ────────────────────────────────────────────────────────
        cam = self._cfg.camera
        self._cap = cv2.VideoCapture(cam.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
        self._cap.set(cv2.CAP_PROP_FPS,          cam.fps)
        if not self._cap.isOpened():
            sys.exit("ERROR: Cannot open camera index %d" % cam.index)
        log.info("Camera ready (%dx%d @ %dfps)", cam.width, cam.height, cam.fps)

        # ── MediaPipe ─────────────────────────────────────────────────────
        self._mp_hands = mp.solutions.hands
        self._hands    = self._mp_hands.Hands(
            static_image_mode        = False,
            max_num_hands            = C.MP_MAX_NUM_HANDS,
            min_detection_confidence = self._cfg.detection.detection_confidence,
            min_tracking_confidence  = self._cfg.detection.tracking_confidence,
        )
        log.info("MediaPipe Hands ready")

        # ── Screen size ───────────────────────────────────────────────────
        if _MOUSE_OK:
            self._sw, self._sh = pyautogui.size()
        else:
            self._sw, self._sh = 1920, 1080
            log.warning("pyautogui not available — mouse control disabled")

        # ── Per-hand state machines (keyed by "Left" / "Right") ───────────
        # Each hand gets its own filter pipeline and gesture state machine
        # so state doesn't bleed between hands.
        self._filters : Dict[str, FilterPipeline]      = {}
        self._gestures: Dict[str, GestureStateMachine] = {}

        # ── Mouse controller ──────────────────────────────────────────────
        self._mouse = MouseController(
            screen_w = self._sw,
            screen_h = self._sh,
            cs       = self._cfg.cursor,
            gs       = self._cfg.gesture,
            ss       = self._cfg.scroll,
        )

        # ── Coordinate mapper ─────────────────────────────────────────────
        self._mapper = CoordinateMapper(self._sw, self._sh, self._cfg)

        # Load saved calibration if it exists
        H = CalibrationManager.load(self.CAL_FILE)
        if H is not None:
            self._mapper.set_calibration(H)
            log.info("Loaded saved calibration from %s", self.CAL_FILE)

        # ── Calibration manager ───────────────────────────────────────────
        self._cal = CalibrationManager(self._sw, self._sh)

        # ── Dashboard (UI renderer) ───────────────────────────────────────
        self._dashboard = Dashboard(self._cfg)

        # ── UI toggles ────────────────────────────────────────────────────
        self._show_guide    = False
        self._show_settings = False   # settings sidebar (press P)

        # ── Performance counters ──────────────────────────────────────────
        self._frame_count = 0
        self._t0          = time.monotonic()
        self._fps_smooth  = 0.0
        self._frame_ms    = 0.0

        # idle-skip counter (skip MediaPipe when no hand on previous frame)
        self._idle_frames   = 0
        self._last_had_hand = False

        log.info("Mouse: %s | Screen: %dx%d",
                 "ON" if _MOUSE_OK else "OFF (install pyautogui)",
                 self._sw, self._sh)
        log.info("Hotkeys: Q=quit  T=guide  R=tutorial  P=settings  C=calibrate")

    # ── Hand-state getters (lazy-init per label) ───────────────────────────────

    def _filter_for(self, label: str) -> FilterPipeline:
        if label not in self._filters:
            self._filters[label] = FilterPipeline(
                self._cfg.cursor, self._cfg.kalman
            )
        return self._filters[label]

    def _gesture_sm_for(self, label: str) -> GestureStateMachine:
        if label not in self._gestures:
            self._gestures[label] = GestureStateMachine(self._cfg.gesture)
        return self._gestures[label]

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow("Hand Gesture Mouse Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Gesture Mouse Control", 1280, 720)

        while True:
            t_frame_start = time.monotonic()

            ret, frame = self._cap.read()
            if not ret:
                log.warning("Camera read failed — retrying")
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)        # mirror — feels natural
            h, w  = frame.shape[:2]
            self._frame_count += 1

            # ── FPS (exponential moving average) ──────────────────────────
            elapsed = time.monotonic() - self._t0
            raw_fps = self._frame_count / max(elapsed, 1e-6)
            self._fps_smooth = self._fps_smooth * 0.92 + raw_fps * 0.08

            # ── Route to mode handler ──────────────────────────────────────
            if self._mode == AppMode.TUTORIAL:
                frame = self._run_tutorial(frame)
            elif self._mode == AppMode.CALIBRATING:
                frame = self._run_calibration(frame, w, h)
            else:
                frame = self._run_normal(frame, w, h)

            # ── Frame time ─────────────────────────────────────────────────
            self._frame_ms = (time.monotonic() - t_frame_start) * 1000.0

            cv2.imshow("Hand Gesture Mouse Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break

        self._cleanup()

    # ── Tutorial mode ──────────────────────────────────────────────────────────

    def _run_tutorial(self, frame: np.ndarray) -> np.ndarray:
        return draw_tutorial(frame, self._tutorial_step)

    # ── Calibration mode ───────────────────────────────────────────────────────

    def _run_calibration(self, frame: np.ndarray, w: int, h: int) -> np.ndarray:
        """
        Run one calibration frame.  Uses POINT gesture to drive the dwell target.
        Falls back to raw landmark position if no gesture is detected.
        """
        # Run MediaPipe on full frame (need accuracy during calibration)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        hand_pos: Optional[Tuple[int, int]] = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for lm_proto, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label   = handedness.classification[0].label
                lm_list = lm_proto.landmark

                # Draw skeleton (dim during calibration)
                self._draw_skeleton_simple(frame, lm_list, w, h, (80, 80, 80))

                # Use index fingertip as pointer
                tip = lm_list[C.LM_INDEX_TIP]
                hand_pos = (int(tip.x * w), int(tip.y * h))
                break   # calibrate with first detected hand only

        cal_result = self._cal.update(hand_pos)
        self._cal.draw(frame)

        if cal_result.complete and cal_result.homography is not None:
            self._mapper.set_calibration(cal_result.homography)
            self._cal.save(self.CAL_FILE)
            log.info("Calibration saved to %s", self.CAL_FILE)
            self._mode = AppMode.NORMAL

        return frame

    # ── Normal mode ────────────────────────────────────────────────────────────

    def _run_normal(self, frame: np.ndarray, w: int, h: int) -> np.ndarray:
        ps = self._cfg.performance

        # ── Idle skip (saves CPU when no hand on previous frame) ───────────
        skip_inference = (
            not self._last_had_hand
            and self._idle_frames < ps.idle_skip_frames
        )
        if skip_inference:
            self._idle_frames += 1
            results = None
        else:
            self._idle_frames = 0
            # Downscale to inference resolution before MediaPipe
            inf_w, inf_h = ps.inference_width, ps.inference_height
            if w != inf_w or h != inf_h:
                small = cv2.resize(frame, (inf_w, inf_h))
            else:
                small = frame
            rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)

        # ── Collect per-hand data ──────────────────────────────────────────
        hands_data: List[HandData]      = []
        gesture_results: List[GestureResult] = []
        result_map: dict = {}   # label → (GestureResult, sx, sy, color)

        detected_labels = set()

        if results and results.multi_hand_landmarks and results.multi_handedness:
            self._last_had_hand = True

            for lm_proto, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label   = handedness.classification[0].label
                lm_list = lm_proto.landmark
                detected_labels.add(label)

                # Raw landmark position (camera pixels, scaled to display frame)
                # MediaPipe returns coords normalised to inference frame —
                # multiply by display frame size (w, h) for correct pixel position.
                raw_x = lm_list[C.LM_INDEX_TIP].x * w
                raw_y = lm_list[C.LM_INDEX_TIP].y * h

                # ── Stage 1-4: FilterPipeline ─────────────────────────────
                fp     = self._filter_for(label)
                fx, fy = fp.update(raw_x, raw_y)

                # ── CoordinateMapper → screen coords ──────────────────────
                sx, sy = self._mapper.map(fx, fy, w, h)

                # ── GestureStateMachine ───────────────────────────────────
                sm     = self._gesture_sm_for(label)
                g_res  = sm.update(lm_list, w, h)

                # ── Colours for rendering ──────────────────────────────────
                GESTURE_COLOR = {
                    GestureType.NONE:      C.C_GRAY,
                    GestureType.POINT:     C.C_GREEN,
                    GestureType.PINCH:     C.C_ORANGE,
                    GestureType.OPEN_HAND: C.C_CYAN,
                    GestureType.FIST:      C.C_RED,
                    GestureType.VICTORY:   C.C_MAGENTA,
                }
                color = GESTURE_COLOR.get(g_res.gesture, C.C_GRAY)

                # ── Draw skeleton on display frame ─────────────────────────
                self._draw_skeleton_simple(frame, lm_list, w, h, color)
                self._draw_pinch_ring(frame, lm_list, w, h, g_res)
                self._draw_gesture_bubble(frame, g_res, color)

                # ── Collect for Dashboard ──────────────────────────────────
                hd = HandData(label=label, landmarks=lm_list,
                              frame_w=w, frame_h=h)
                hands_data.append(hd)
                gesture_results.append(g_res)
                result_map[label] = (g_res, sx, sy, color)

                # ── Mouse actions ──────────────────────────────────────────
                # Right hand → cursor + click + scroll
                # Left hand  → scroll only (when right hand is also doing something)
                if label == "Right" or len(results.multi_hand_landmarks) == 1:
                    self._mouse.handle(g_res, sx, sy)
                else:
                    # Left hand — only drive scroll
                    self._mouse.handle_scroll(g_res, sy)

        else:
            # No hands detected — release any held state cleanly
            self._last_had_hand = False
            for label, sm in self._gestures.items():
                sm.reset()
            for label, fp in self._filters.items():
                fp.reset()
            self._mouse.release_all()

        # Reset filters for labels that disappeared this frame
        for label in list(self._filters.keys()):
            if label not in detected_labels:
                self._filters[label].reset()
                self._gestures[label].reset()

        # Flush all queued mouse actions (done ONCE per frame, after all hands)
        self._mouse.flush()

        # ── Dashboard (HUD + overlays) ─────────────────────────────────────
        perf_snap = {
            "fps_smooth":     self._fps_smooth,
            "frame_ms_mean":  self._frame_ms,
            "frame_ms_p95":   self._frame_ms,   # simplified — add RingBuffer later
            "frame_ms_max":   self._frame_ms,
            "total_frames":   self._frame_count,
            "stages":         {},
        }

        self._dashboard.draw(
            frame      = frame,
            fps        = self._fps_smooth,
            frame_ms   = self._frame_ms,
            results    = result_map,
            mouse      = self._mouse,
            show_guide = self._show_guide,
        )

        # ── Active zone border ─────────────────────────────────────────────
        self._draw_active_zone(frame, w, h)

        # ── Settings sidebar (press P) — drawn LAST so nothing covers it ──
        if self._show_settings:
            self._draw_settings_sidebar(frame)

        return frame

    # ── Key handler ────────────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> bool:
        """Return False to quit."""
        if key in (C.KEY_QUIT, C.KEY_ESC, ord('Q')):
            return False

        if self._mode == AppMode.TUTORIAL:
            if key == ord(' '):
                self._tutorial_step += 1
                if self._tutorial_step >= len(TUTORIAL_STEPS):
                    self._mode = AppMode.NORMAL
            elif key in (C.KEY_SKIP, ord('S')):
                self._mode = AppMode.NORMAL
            return True

        # Normal / calibrating mode keys  (accept both lowercase and uppercase)
        key_lc = key | 0x20 if 65 <= key <= 90 else key   # A-Z → a-z

        if key_lc == C.KEY_GUIDE:                           # T — gesture guide
            self._show_guide = not self._show_guide
        elif key_lc == C.KEY_TUTORIAL:                      # R — restart tutorial
            self._mode         = AppMode.TUTORIAL
            self._tutorial_step = 0
        elif key_lc == C.KEY_SETTINGS:                      # P — settings sidebar
            self._show_settings = not self._show_settings
        elif key_lc == C.KEY_CALIBRATE:                     # C — calibrate / cancel
            if self._mode == AppMode.CALIBRATING:
                self._cal.cancel()
                self._mode = AppMode.NORMAL
                log.info("Calibration cancelled by user")
            else:
                self._cal.start()
                self._mode = AppMode.CALIBRATING
                log.info("Calibration started")

        return True

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def _draw_skeleton_simple(
        self, frame: np.ndarray, lm, fw: int, fh: int, color: tuple
    ) -> None:
        h, w = frame.shape[:2]
        # Connections
        for s, e in C.HAND_CONNECTIONS:
            ls, le = lm[s], lm[e]
            p1 = (int(ls.x * w), int(ls.y * h))
            p2 = (int(le.x * w), int(le.y * h))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        # Landmark dots
        tips = {C.LM_INDEX_TIP, C.LM_MIDDLE_TIP, C.LM_RING_TIP,
                C.LM_PINKY_TIP, C.LM_THUMB_TIP}
        for i, lmk in enumerate(lm):
            px = (int(lmk.x * w), int(lmk.y * h))
            r  = 6 if i in tips else 3
            cv2.circle(frame, px, r, color, -1, cv2.LINE_AA)
            cv2.circle(frame, px, r, C.C_WHITE, 1, cv2.LINE_AA)

    def _draw_pinch_ring(
        self, frame: np.ndarray, lm, fw: int, fh: int,
        res: GestureResult
    ) -> None:
        """Pulsing ring at pinch midpoint — grows as drag lock approaches."""
        if res.gesture != GestureType.PINCH:
            return
        h, w = frame.shape[:2]
        tx = int(lm[C.LM_THUMB_TIP].x * w)
        ty = int(lm[C.LM_THUMB_TIP].y * h)
        ix = int(lm[C.LM_INDEX_TIP].x * w)
        iy = int(lm[C.LM_INDEX_TIP].y * h)
        mid = ((tx + ix) // 2, (ty + iy) // 2)
        r   = int(12 + min(res.hold_s, 0.5) * 24)
        cv2.circle(frame, mid, r,  C.C_ORANGE, 2, cv2.LINE_AA)
        cv2.circle(frame, mid, 5,  C.C_ORANGE, -1, cv2.LINE_AA)

    def _draw_gesture_bubble(
        self, frame: np.ndarray, res: GestureResult, color: tuple
    ) -> None:
        """Filled label bubble above the gesture anchor point."""
        LABELS = {
            GestureType.POINT:     "POINT",
            GestureType.PINCH:     "PINCH",
            GestureType.OPEN_HAND: "SCROLL",
            GestureType.FIST:      "PAUSE",
            GestureType.VICTORY:   "R-CLICK",
        }
        label = LABELS.get(res.gesture, "")
        if not label:
            return
        x, y = res.position
        y = max(y - 30, 20)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        pad = 6
        x0, y0 = x - tw // 2 - pad, y - th - pad
        x1, y1 = x + tw // 2 + pad, y + pad
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y0), (x1, y1), C.C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, label, (x - tw // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C.C_BLACK, 1, cv2.LINE_AA)

    def _draw_active_zone(self, frame: np.ndarray, w: int, h: int) -> None:
        """Dashed border showing the active gesture zone."""
        cs  = self._cfg.cursor
        x0  = int(cs.active_zone_x[0] * w)
        x1  = int(cs.active_zone_x[1] * w)
        y0  = int(cs.active_zone_y[0] * h)
        y1  = int(cs.active_zone_y[1] * h)
        col = C.C_GRAY

        def dashed(p1, p2, dash=10, gap=6):
            x1_, y1_ = p1; x2_, y2_ = p2
            L = max(((x2_ - x1_) ** 2 + (y2_ - y1_) ** 2) ** 0.5, 1e-6)
            dx, dy = (x2_ - x1_) / L, (y2_ - y1_) / L
            pos = 0.0
            while pos < L:
                s = pos; e = min(pos + dash, L)
                a = (int(x1_ + dx * s), int(y1_ + dy * s))
                b = (int(x1_ + dx * e), int(y1_ + dy * e))
                cv2.line(frame, a, b, col, 1, cv2.LINE_AA)
                pos += dash + gap

        dashed((x0, y0), (x1, y0))
        dashed((x1, y0), (x1, y1))
        dashed((x1, y1), (x0, y1))
        dashed((x0, y1), (x0, y0))
        cv2.putText(frame, "ACTIVE ZONE", (x0 + 4, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C.C_GRAY, 1, cv2.LINE_AA)

    def _draw_settings_sidebar(self, frame: np.ndarray) -> None:
        """
        RIGHT-side panel: all live-tunable Settings values with their names.
        Drawn last so the dashboard HUD (left side) never covers it.
        Toggled with P.
        """
        h, w = frame.shape[:2]
        panel_w = 310
        px = w - panel_w   # anchor to RIGHT edge

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, 0), (w, h), (8, 22, 8), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (px, 0), (w - 1, h - 1), C.C_CYAN, 1)

        cs = self._cfg.cursor
        gs = self._cfg.gesture
        ss = self._cfg.scroll
        ks = self._cfg.kalman
        ps = self._cfg.performance

        def row(label: str, val, y: int, col=C.C_WHITE):
            v = f"{val:.3f}" if isinstance(val, float) else str(val)
            cv2.putText(frame, f"{label:<22} {v}",
                        (px + 6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, col, 1, cv2.LINE_AA)

        def sect(title: str, y: int):
            cv2.putText(frame, title, (px + 6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, C.C_CYAN, 1, cv2.LINE_AA)
            cv2.line(frame, (px, y + 4), (w, y + 4), (40, 80, 40), 1)

        y = 20
        cv2.putText(frame, "-- SETTINGS  (P to close) --",
                    (px + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C.C_CYAN, 2, cv2.LINE_AA)
        y += 20

        sect("CURSOR", y);       y += 17
        row("speed",                    cs.speed,                   y, C.C_GREEN);  y += 15
        row("smoothing",                cs.smoothing,               y, C.C_GREEN);  y += 15
        row("adaptive_smoothing",       cs.adaptive_smoothing,      y, C.C_GREEN);  y += 15
        row("fast_threshold (px/f)",    cs.adaptive_fast_threshold, y, C.C_GREEN);  y += 15
        row("slow_threshold (px/f)",    cs.adaptive_slow_threshold, y, C.C_GREEN);  y += 15
        row("dead_zone_px",             cs.dead_zone_px,            y, C.C_GREEN);  y += 15
        row("hover_lock",               cs.hover_lock_enabled,      y, C.C_GREEN);  y += 15
        row("hover_velocity (px/f)",    cs.hover_velocity_threshold,y, C.C_GREEN);  y += 15
        row("hover_lock_ms",            cs.hover_lock_ms,           y, C.C_GREEN);  y += 17

        sect("GESTURE",  y);     y += 17
        row("pinch_sensitivity",        gs.pinch_sensitivity,       y, C.C_ORANGE); y += 15
        row("extend_sensitivity",       gs.extend_sensitivity,      y, C.C_ORANGE); y += 15
        row("debounce_s",               gs.debounce_s,              y, C.C_ORANGE); y += 15
        row("drag_lock_s",              gs.drag_lock_s,             y, C.C_ORANGE); y += 15
        row("click_min_hold_s",         gs.click_min_hold_s,        y, C.C_ORANGE); y += 15
        row("right_click_debounce_s",   gs.right_click_debounce_s,  y, C.C_ORANGE); y += 17

        sect("SCROLL",  y);      y += 17
        row("pixels_per_tick",          ss.pixels_per_tick,         y, C.C_CYAN);   y += 15
        row("speed_multiplier",         ss.speed_multiplier,        y, C.C_CYAN);   y += 15
        row("accumulation_smooth",      ss.accumulation_smooth,     y, C.C_CYAN);   y += 15
        row("natural_direction",        ss.natural_direction,       y, C.C_CYAN);   y += 17

        sect("KALMAN",  y);      y += 17
        row("process_noise",            ks.process_noise,           y, C.C_YELLOW); y += 15
        row("measurement_noise",        ks.measurement_noise,       y, C.C_YELLOW); y += 17

        sect("PERFORMANCE",  y); y += 17
        row("inference_size",
            f"{ps.inference_width}x{ps.inference_height}",          y, C.C_GRAY);   y += 15
        row("idle_skip_frames",         ps.idle_skip_frames,        y, C.C_GRAY);   y += 15

        # Calibration status
        cal_str = "loaded" if self._mapper._H is not None else "none (press C)"
        row("calibration",              cal_str,                    y, C.C_GRAY)

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        self._mouse.release_all()
        if self._cap:
            self._cap.release()
        self._hands.close()
        cv2.destroyAllWindows()
        log.info("Shutdown clean.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.run()