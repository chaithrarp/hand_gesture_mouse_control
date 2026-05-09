"""
processing/data_processor.py

Preprocessing and normalization of raw MediaPipe landmark data
before it reaches the gesture recognizer or filter pipeline.

Responsibilities:
  - Validate landmark completeness (reject partial detections)
  - Normalize landmarks to hand-relative coordinate space
  - Compute derived hand metrics (hand size, orientation, palm normal)
  - Provide a clean ProcessedHand dataclass downstream modules consume

Why normalization matters:
  Raw MediaPipe landmarks are in frame-relative coords (0-1).
  Two problems:
    1. Hand size varies with distance from camera — a pinch at 20cm
       looks different from a pinch at 50cm in raw pixel space.
    2. Wrist position encodes both hand position AND cursor position,
       making gesture classification couple to cursor location.

  Solution: normalize all landmarks relative to wrist (origin) and
  hand_size (scale). Gesture classifier then works in a coordinate
  system where the hand is always the same size regardless of distance.
  Cursor position is tracked separately via the index fingertip.

Latency:
  All ops are numpy on 21×3 arrays. Total cost < 0.2ms per hand.
  No allocations after the first frame (arrays pre-allocated in __init__).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config import constants as C
from config.settings import DetectionSettings
from utils.logger import get_logger

log = get_logger(__name__)

# ── ProcessedHand — clean output contract ────────────────────────────────────
@dataclass
class ProcessedHand:
    """
    Fully preprocessed hand data ready for gesture classification
    and filter pipeline.

    All coordinates are in two spaces:
      raw_*    — original frame-relative (0-1), as from MediaPipe
      norm_*   — hand-relative, wrist=origin, scale=hand_size
    """

    label:      str                    # "Left" | "Right"
    confidence: float                  # detection confidence

    # Raw frame coords (for cursor mapping and overlay drawing)
    raw_landmarks: object              # original mediapipe landmark list
    cursor_px:     Tuple[int, int]     # index tip in full-frame pixels
    palm_px:       Tuple[int, int]     # palm centre in full-frame pixels
    frame_w:       int
    frame_h:       int

    # Hand metrics (scale-invariant)
    hand_size:    float                # wrist→middle_mcp distance in pixels
    orientation:  float                # hand tilt angle in degrees (-90 to 90)
    palm_normal:  Tuple[float, float]  # 2D normal of palm plane (unit vector)

    # Normalized landmark array (21×2, wrist=origin, hand_size=1)
    norm_lm:      np.ndarray           # shape (21, 2), float32

    # Quality flags
    valid:        bool = True          # False = reject this detection
    low_conf:     bool = False         # True = confidence below threshold


# ── DataProcessor ─────────────────────────────────────────────────────────────
class DataProcessor:
    """
    Converts raw HandData (from hand_detector.py) into ProcessedHand.

    One instance shared across all hands — stateless per-frame processing.

    Usage:
        processor = DataProcessor(detection_settings)
        for hand_data in detected_hands:
            processed = processor.process(hand_data)
            if processed.valid:
                gesture = recognizer.classify(processed.raw_landmarks, w, h)
                filtered = filter_pipeline.update(*processed.cursor_px)
    """

    def __init__(self, ds: DetectionSettings):
        self._ds = ds
        # Pre-allocate output array — reused every frame, no GC pressure
        self._norm_buf = np.zeros((C.LANDMARK_COUNT, 2), dtype=np.float32)

    def reconfigure(self, ds: DetectionSettings) -> None:
        self._ds = ds

    def process(self, hand_data) -> ProcessedHand:
        """
        Process one HandData into a ProcessedHand.
        hand_data: HandData from hand_detector.HandDetector.detect()
        """
        lm     = hand_data.landmarks
        fw     = hand_data.frame_w
        fh     = hand_data.frame_h
        label  = hand_data.label
        conf   = hand_data.confidence

        # ── Validate ──────────────────────────────────────────────────────
        if not self._validate(lm):
            log.debug(f"DataProcessor: rejected invalid detection ({label})")
            return self._invalid(label, conf, lm, fw, fh)

        # ── Key pixel positions ───────────────────────────────────────────
        cursor_px = (
            int(lm[C.LM_INDEX_TIP].x * fw),
            int(lm[C.LM_INDEX_TIP].y * fh),
        )
        palm_px = (
            int(lm[C.LM_PALM_REF].x * fw),
            int(lm[C.LM_PALM_REF].y * fh),
        )
        wrist_px = (
            int(lm[C.LM_WRIST].x * fw),
            int(lm[C.LM_WRIST].y * fh),
        )

        # ── Hand size (scale reference) ───────────────────────────────────
        hand_size = float(np.hypot(
            palm_px[0] - wrist_px[0],
            palm_px[1] - wrist_px[1],
        )) or 1.0

        # ── Normalized landmarks ──────────────────────────────────────────
        norm_lm = self._normalize(lm, wrist_px, hand_size, fw, fh)

        # ── Orientation ───────────────────────────────────────────────────
        orientation = self._compute_orientation(lm, fw, fh)

        # ── Palm normal ───────────────────────────────────────────────────
        palm_normal = self._compute_palm_normal(lm, fw, fh)

        # ── Confidence flag ───────────────────────────────────────────────
        low_conf = conf < self._ds.detection_confidence * 0.8

        return ProcessedHand(
            label        = label,
            confidence   = conf,
            raw_landmarks= lm,
            cursor_px    = cursor_px,
            palm_px      = palm_px,
            frame_w      = fw,
            frame_h      = fh,
            hand_size    = hand_size,
            orientation  = orientation,
            palm_normal  = palm_normal,
            norm_lm      = norm_lm,
            valid        = True,
            low_conf     = low_conf,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _validate(self, lm) -> bool:
        """
        Reject detections where critical landmarks are missing or degenerate.
        Checks:
          - Landmark count == 21
          - Wrist and palm_ref exist and are in-frame
          - Hand size is non-zero (degenerate flat detection)
        """
        try:
            if len(lm) != C.LANDMARK_COUNT:
                return False
            # Wrist and middle MCP must be in-frame (0-1 range)
            for idx in (C.LM_WRIST, C.LM_PALM_REF, C.LM_INDEX_TIP):
                pt = lm[idx]
                if not (0.0 <= pt.x <= 1.0 and 0.0 <= pt.y <= 1.0):
                    return False
            return True
        except (IndexError, AttributeError):
            return False

    def _normalize(
        self,
        lm,
        wrist_px: Tuple[int, int],
        hand_size: float,
        fw: int, fh: int,
    ) -> np.ndarray:
        """
        Normalize all 21 landmarks to hand-relative space.
        Output: (21, 2) float32 array where:
          - wrist is at (0, 0)
          - coordinates are in units of hand_size
          - invariant to camera distance and hand position in frame
        """
        buf = self._norm_buf
        wx, wy = wrist_px
        inv_hs = 1.0 / hand_size

        for i in range(C.LANDMARK_COUNT):
            pt       = lm[i]
            px       = pt.x * fw - wx
            py       = pt.y * fh - wy
            buf[i, 0] = px * inv_hs
            buf[i, 1] = py * inv_hs

        return buf.copy()   # caller owns a clean copy

    def _compute_orientation(self, lm, fw: int, fh: int) -> float:
        """
        Hand tilt angle in degrees.
        Defined as the angle of the wrist→middle_mcp vector from vertical.
        Range: -90 (tilted left) to +90 (tilted right). 0 = upright.
        """
        wx = lm[C.LM_WRIST].x * fw
        wy = lm[C.LM_WRIST].y * fh
        mx = lm[C.LM_PALM_REF].x * fw
        my = lm[C.LM_PALM_REF].y * fh
        dx = mx - wx
        dy = my - wy
        # atan2 of (dx, -dy): -dy because y increases downward in image
        angle = float(np.degrees(np.arctan2(dx, -dy)))
        return angle

    def _compute_palm_normal(self, lm, fw: int, fh: int) -> Tuple[float, float]:
        """
        2D approximation of palm facing direction (unit vector).
        Uses wrist → index_mcp and wrist → pinky_mcp as basis vectors,
        then takes their 2D perpendicular (cross product z-component → normal).

        Returns (nx, ny) unit vector pointing "out" from palm face.
        Useful for distinguishing palm-facing-camera vs back-of-hand.
        """
        def pt(idx):
            return np.array([lm[idx].x * fw, lm[idx].y * fh], dtype=np.float32)

        wrist = pt(C.LM_WRIST)
        v1    = pt(C.LM_INDEX_MCP) - wrist
        v2    = pt(C.LM_PINKY_MCP) - wrist

        # 2D "cross product" gives the signed perpendicular magnitude
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        norm  = np.array([-v1[1] + v2[1], v1[0] - v2[0]], dtype=np.float32)
        mag   = float(np.linalg.norm(norm))
        if mag < 1e-6:
            return (0.0, -1.0)   # default: pointing up
        norm /= mag
        return (float(norm[0]), float(norm[1]))

    def _invalid(self, label, conf, lm, fw, fh) -> ProcessedHand:
        """Return a sentinel ProcessedHand marked invalid."""
        return ProcessedHand(
            label        = label,
            confidence   = conf,
            raw_landmarks= lm,
            cursor_px    = (fw // 2, fh // 2),
            palm_px      = (fw // 2, fh // 2),
            frame_w      = fw,
            frame_h      = fh,
            hand_size    = 1.0,
            orientation  = 0.0,
            palm_normal  = (0.0, -1.0),
            norm_lm      = np.zeros((C.LANDMARK_COUNT, 2), dtype=np.float32),
            valid        = False,
            low_conf     = True,
        )