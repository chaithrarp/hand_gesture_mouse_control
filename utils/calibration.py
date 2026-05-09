"""
utils/calibration.py

Hand-to-screen calibration system.

Problem the active zone solves poorly:
  The default active zone (10-90% of frame) assumes the camera is
  perfectly centred and the user's hand moves symmetrically. In practice:
    - Camera may be tilted or offset
    - User may sit off-centre
    - Hand may not reach frame edges comfortably

What calibration fixes:
  The user taps 4 screen corners with their pointing finger.
  We record the hand position at each corner and compute a homography
  (a perspective transform) that maps those 4 hand positions exactly
  to the 4 screen corners.

  Result: any hand position → correct screen pixel, even if the
  camera is tilted 30° or the user sits far to the left.

Calibration flow:
    cal = CalibrationManager(screen_w, screen_h)
    cal.start()

    # Each frame while calibrating:
    result = cal.update(hand_x, hand_y, frame)
    if result.complete:
        H = result.homography   # 3×3 matrix
        mapper.set_calibration(H)

    cal.cancel()   # abort at any time

The 4 target points cycle: TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT
Each point requires the hand to stay within DWELL_RADIUS for DWELL_MS ms
to be recorded (hands-free confirmation — no key press needed).

Persistence:
    cal.save(path)   # saves H as .npy
    H = CalibrationManager.load(path)   # returns ndarray or None
"""

from __future__ import annotations

import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

from utils.logger import get_logger

log = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
DWELL_MS     = 1200    # ms hand must stay still at a corner to confirm
DWELL_RADIUS = 22      # px — hand must stay within this circle
MARGIN       = 0.08    # fraction of screen to inset corner targets from edge
CAL_FILE_DEFAULT = "calibration.npy"


# ── State ─────────────────────────────────────────────────────────────────────
class CalState(Enum):
    IDLE       = auto()
    COLLECTING = auto()   # waiting for user to hold at corner
    COMPLETE   = auto()
    CANCELLED  = auto()


@dataclass
class CalibrationResult:
    complete:    bool
    homography:  Optional[np.ndarray]   # 3×3 float64 or None
    message:     str                    # UI feedback string


# ── CalibrationManager ────────────────────────────────────────────────────────
class CalibrationManager:
    """
    Manages the 4-point calibration flow.

    Usage:
        cm = CalibrationManager(screen_w=1920, screen_h=1080)
        cm.start()

        while True:
            frame = camera.read()
            hand_pos = get_hand_pos()    # (x, y) in camera pixels or None
            result = cm.update(hand_pos, frame)
            cm.draw(frame)               # draws overlay on frame in-place
            if result.complete:
                mapper.set_calibration(result.homography)
                break
    """

    def __init__(self, screen_w: int, screen_h: int):
        self._sw = screen_w
        self._sh = screen_h

        m = MARGIN
        # Screen corners in screen pixel space (target positions)
        self._screen_pts: List[Tuple[float, float]] = [
            (screen_w * m,         screen_h * m),          # TOP-LEFT
            (screen_w * (1 - m),   screen_h * m),          # TOP-RIGHT
            (screen_w * (1 - m),   screen_h * (1 - m)),    # BOTTOM-RIGHT
            (screen_w * m,         screen_h * (1 - m)),    # BOTTOM-LEFT
        ]
        self._labels = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]

        self._state:     CalState          = CalState.IDLE
        self._cam_pts:   List[Tuple]       = []   # collected camera positions
        self._step:      int               = 0    # which corner we're on (0-3)
        self._dwell_start: Optional[float] = None
        self._dwell_pos:   Optional[Tuple] = None
        self._homography:  Optional[np.ndarray] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin calibration. Resets any previous state."""
        self._state      = CalState.COLLECTING
        self._cam_pts    = []
        self._step       = 0
        self._dwell_start = None
        self._dwell_pos   = None
        self._homography  = None
        log.info("Calibration started")

    def cancel(self) -> None:
        """Abort calibration without applying results."""
        self._state = CalState.CANCELLED
        log.info("Calibration cancelled")

    @property
    def active(self) -> bool:
        return self._state == CalState.COLLECTING

    @property
    def homography(self) -> Optional[np.ndarray]:
        return self._homography

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(
        self,
        hand_pos: Optional[Tuple[int, int]],
    ) -> CalibrationResult:
        """
        Call each frame with the current hand position (camera pixels).
        hand_pos=None means no hand detected — resets dwell timer.

        Returns CalibrationResult. Check result.complete to know when done.
        """
        if self._state != CalState.COLLECTING:
            return CalibrationResult(
                complete   = self._state == CalState.COMPLETE,
                homography = self._homography,
                message    = self._state.name,
            )

        if hand_pos is None:
            self._dwell_start = None
            return CalibrationResult(
                complete   = False,
                homography = None,
                message    = f"Point at {self._labels[self._step]} — no hand detected",
            )

        hx, hy = hand_pos

        # ── Dwell check ───────────────────────────────────────────────────
        if self._dwell_pos is None:
            # First frame at this position
            self._dwell_pos   = (hx, hy)
            self._dwell_start = time.monotonic()
        else:
            dx = hx - self._dwell_pos[0]
            dy = hy - self._dwell_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > DWELL_RADIUS:
                # Hand moved — reset dwell
                self._dwell_pos   = (hx, hy)
                self._dwell_start = time.monotonic()

        elapsed_ms = (time.monotonic() - self._dwell_start) * 1000.0
        remaining_ms = max(0.0, DWELL_MS - elapsed_ms)

        if elapsed_ms >= DWELL_MS:
            # ── Corner confirmed ──────────────────────────────────────────
            self._cam_pts.append((float(hx), float(hy)))
            log.info(
                f"Calibration point {self._step + 1}/4 recorded: "
                f"cam=({hx},{hy}) → screen={self._screen_pts[self._step]}"
            )
            self._step += 1
            self._dwell_start = None
            self._dwell_pos   = None

            if self._step == 4:
                return self._finish()

            return CalibrationResult(
                complete   = False,
                homography = None,
                message    = f"✓ Got it! Now point at {self._labels[self._step]}",
            )

        # Still dwelling
        pct = int(elapsed_ms / DWELL_MS * 100)
        return CalibrationResult(
            complete   = False,
            homography = None,
            message    = (
                f"Hold at {self._labels[self._step]}... "
                f"{remaining_ms / 1000:.1f}s  ({pct}%)"
            ),
        )

    def _finish(self) -> CalibrationResult:
        """Compute homography from the 4 collected point pairs."""
        src = np.array(self._cam_pts,    dtype=np.float32)
        dst = np.array(self._screen_pts, dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, method=0)

        if H is None:
            log.error("Calibration failed: findHomography returned None")
            self._state = CalState.CANCELLED
            return CalibrationResult(
                complete   = False,
                homography = None,
                message    = "Calibration failed — please try again",
            )

        self._homography = H
        self._state      = CalState.COMPLETE
        log.info(f"Calibration complete. Homography:\n{H}")

        return CalibrationResult(
            complete   = True,
            homography = H,
            message    = "Calibration complete ✓",
        )

    # ── Overlay drawing ───────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray) -> None:
        """
        Draw calibration overlay on frame in-place.
        Shows: darkened background, target crosshair, dwell progress arc,
        step counter, instruction text.
        Call after update() each frame.
        """
        if self._state != CalState.COLLECTING:
            return

        h, w = frame.shape[:2]

        # Darken background slightly
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Map screen corner → frame pixel (approximate reverse for display)
        # Screen pts are in screen space; for display we place them in frame
        # proportionally. This is just for visual guidance, not math-critical.
        m = MARGIN
        frame_targets = [
            (int(w * m),         int(h * m)),
            (int(w * (1 - m)),   int(h * m)),
            (int(w * (1 - m)),   int(h * (1 - m))),
            (int(w * m),         int(h * (1 - m))),
        ]

        # Completed corners — green dot
        for i in range(self._step):
            cv2.circle(frame, frame_targets[i], 14, (0, 220, 80), -1, cv2.LINE_AA)
            cv2.putText(frame, "✓", (frame_targets[i][0] - 8, frame_targets[i][1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Current target — pulsing crosshair
        if self._step < 4:
            tx, ty = frame_targets[self._step]

            # Progress arc
            elapsed_ms = 0.0
            if self._dwell_start is not None:
                elapsed_ms = (time.monotonic() - self._dwell_start) * 1000.0
            angle = int(360 * min(elapsed_ms / DWELL_MS, 1.0))

            cv2.circle(frame, (tx, ty), 28, (80, 80, 80), 2, cv2.LINE_AA)
            if angle > 0:
                cv2.ellipse(frame, (tx, ty), (28, 28), -90, 0, angle,
                            (0, 220, 255), 3, cv2.LINE_AA)

            # Crosshair lines
            cv2.line(frame, (tx - 18, ty), (tx + 18, ty), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (tx, ty - 18), (tx, ty + 18), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), 5, (0, 220, 255), -1, cv2.LINE_AA)

            # Label
            cv2.putText(frame, self._labels[self._step],
                        (tx - 45, ty + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 1, cv2.LINE_AA)

        # Instruction bar at top
        cv2.rectangle(frame, (0, 0), (w, 48), (10, 10, 30), -1)
        msg = (
            f"CALIBRATION  {self._step}/4  —  "
            f"Point your index finger at the {self._labels[min(self._step, 3)]} target and hold still"
        )
        cv2.putText(frame, msg, (14, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 1, cv2.LINE_AA)

        # ESC hint
        cv2.putText(frame, "Press ESC to cancel", (14, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 120, 120), 1, cv2.LINE_AA)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = CAL_FILE_DEFAULT) -> bool:
        """Save homography matrix to .npy file. Returns True on success."""
        if self._homography is None:
            log.warning("CalibrationManager.save: no homography to save")
            return False
        try:
            np.save(path, self._homography)
            log.info(f"Calibration saved to {path}")
            return True
        except Exception as e:
            log.error(f"Calibration save failed: {e}")
            return False

    @staticmethod
    def load(path: str = CAL_FILE_DEFAULT) -> Optional[np.ndarray]:
        """
        Load a previously saved homography matrix.
        Returns ndarray (3×3) or None if file missing/corrupt.
        """
        p = Path(path)
        if not p.exists():
            log.info(f"No calibration file at {path}")
            return None
        try:
            H = np.load(str(p))
            if H.shape != (3, 3):
                log.warning(f"Calibration file {path} has wrong shape {H.shape}")
                return None
            log.info(f"Calibration loaded from {path}")
            return H
        except Exception as e:
            log.error(f"Calibration load failed: {e}")
            return None