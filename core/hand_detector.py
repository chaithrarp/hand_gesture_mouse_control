"""
core/hand_detector.py

MediaPipe Hands wrapper.

Responsibilities:
  - Owns the MediaPipe Hands model lifecycle (init / close)
  - Downscales frame to inference resolution before MP (big CPU win)
  - Idle-frame skipping: if no hand found last frame, skip every N frames
  - Packages raw MP output into clean HandData dataclasses
  - Exposes one method: detect(frame) -> list[HandData]

HandData carries everything downstream modules need:
  - Raw landmark list (for gesture classifier)
  - Label ("Left" / "Right") — mirrored frame so labels are intuitive
  - Palm centre pixel (stable anchor for scroll reference)
  - Detection confidence

Latency notes:
  - Inference runs on a COPY of the downscaled frame (no mutation)
  - RGB conversion is in-place on the small frame only
  - Idle skip saves ~8ms per skipped frame on a typical laptop CPU
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mediapipe as mp

from config.settings import DetectionSettings, PerformanceSettings
from config import constants as C
from utils.logger import get_logger, LatencyGuard

log = get_logger(__name__)


# ── HandData — output contract ────────────────────────────────────────────────
@dataclass
class HandData:
    """
    Everything downstream needs about one detected hand, per frame.
    Immutable after creation — downstream modules never mutate this.
    """
    label: str                        # "Left" or "Right" (post-mirror)
    landmarks: object                 # mediapipe NormalizedLandmarkList.landmark
    confidence: float                 # detection confidence 0-1
    palm_center: Tuple[int, int]      # pixel coords of LM_PALM_REF in full frame
    frame_w: int                      # full frame width (for coord unpacking)
    frame_h: int                      # full frame height


def _palm_center(lm, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """LM_PALM_REF (middle MCP) in full-frame pixel coords."""
    ref = lm[C.LM_PALM_REF]
    return (int(ref.x * frame_w), int(ref.y * frame_h))


# ── HandDetector ──────────────────────────────────────────────────────────────
class HandDetector:
    """
    Wraps MediaPipe Hands for the rest of the pipeline.

    Usage:
        detector = HandDetector(detection_settings, performance_settings)
        hands: list[HandData] = detector.detect(bgr_frame)
        detector.close()   # call on shutdown

    The detector is stateful only for idle-skip tracking.
    All MP state is internal to self._hands.
    """

    def __init__(
        self,
        ds: DetectionSettings,
        ps: PerformanceSettings,
    ):
        self._ds = ds
        self._ps = ps

        self._hands = mp.solutions.hands.Hands(
            static_image_mode        = C.MP_STATIC_IMAGE,
            max_num_hands            = C.MP_MAX_NUM_HANDS,
            min_detection_confidence = ds.detection_confidence,
            min_tracking_confidence  = ds.tracking_confidence,
        )

        # Idle-skip state
        self._last_had_hands = False
        self._skip_counter   = 0
        self._last_results   = []   # cached results for skipped frames

        log.info(
            f"HandDetector ready — "
            f"detect={ds.detection_confidence}  "
            f"track={ds.tracking_confidence}  "
            f"inference={ps.inference_width}x{ps.inference_height}  "
            f"idle_skip={ps.idle_skip_frames}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, bgr_frame: np.ndarray) -> List[HandData]:
        """
        Run hand detection on one BGR frame.

        Returns a list of HandData (0, 1, or 2 entries).
        On idle-skip frames returns cached result from last real detection.

        Args:
            bgr_frame: Full-resolution BGR frame from camera (already mirrored).
        """
        h, w = bgr_frame.shape[:2]

        # ── Idle-frame skip ───────────────────────────────────────────────
        if not self._last_had_hands:
            self._skip_counter += 1
            if self._skip_counter < self._ps.idle_skip_frames:
                return self._last_results   # return stale (empty) result
            self._skip_counter = 0

        # ── Downscale for inference ───────────────────────────────────────
        iw, ih = self._ps.inference_width, self._ps.inference_height
        needs_resize = (w != iw or h != ih)

        if needs_resize:
            small = cv2.resize(bgr_frame, (iw, ih), interpolation=cv2.INTER_LINEAR)
        else:
            small = bgr_frame

        # MediaPipe requires RGB, non-writeable for zero-copy optimisation
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb_small.flags.writeable = False

        # ── MediaPipe inference ───────────────────────────────────────────
        with LatencyGuard("mp_inference", budget_ms=25, log=log):
            results = self._hands.process(rgb_small)

        # ── Package results ───────────────────────────────────────────────
        hands: List[HandData] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for lm_proto, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                lm     = lm_proto.landmark
                label  = handedness.classification[0].label
                conf   = handedness.classification[0].score

                hands.append(HandData(
                    label       = label,
                    landmarks   = lm,
                    confidence  = round(conf, 3),
                    palm_center = _palm_center(lm, w, h),
                    frame_w     = w,
                    frame_h     = h,
                ))

            log.debug(f"Detected {len(hands)} hand(s): {[h.label for h in hands]}")

        self._last_had_hands = len(hands) > 0
        self._last_results   = hands
        return hands

    # ── Settings hot-reload ───────────────────────────────────────────────────

    def reconfigure(
        self,
        ds: DetectionSettings,
        ps: PerformanceSettings,
    ) -> None:
        """
        Rebuild the MediaPipe model with new confidence thresholds.
        Performance settings (inference size, skip) apply immediately without
        rebuilding MP — only confidence changes need a rebuild.
        """
        self._ps = ps   # performance settings apply immediately

        if (ds.detection_confidence != self._ds.detection_confidence or
                ds.tracking_confidence != self._ds.tracking_confidence):
            log.info("HandDetector: rebuilding MP model with new confidence values")
            self._hands.close()
            self._hands = mp.solutions.hands.Hands(
                static_image_mode        = C.MP_STATIC_IMAGE,
                max_num_hands            = C.MP_MAX_NUM_HANDS,
                min_detection_confidence = ds.detection_confidence,
                min_tracking_confidence  = ds.tracking_confidence,
            )
            self._ds = ds

    def close(self) -> None:
        """Release MediaPipe resources. Call on app shutdown."""
        self._hands.close()
        log.info("HandDetector closed")