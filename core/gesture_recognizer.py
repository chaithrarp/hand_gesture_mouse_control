"""
core/gesture_recognizer.py

Two-layer gesture system:

  Layer 1 — GestureClassifier  (stateless, pure geometry)
    Looks at current landmark positions, returns (GestureType, confidence).
    No time, no history. Fast, deterministic, easily unit-tested.

  Layer 2 — GestureStateMachine  (stateful, time-aware)
    Wraps the classifier. Applies debounce, hold-duration tracking,
    and emits clean gesture events (ENTERED / HELD / EXITED).

Design rules:
  - Classifier has zero side effects — same input always → same output
  - All geometry ratios are relative to hand_size (scale-invariant)
  - State machine owns all timing — no time.time() in the classifier
  - Confidence is a float 0-1 used downstream to weight gesture trust
"""

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

from config import constants as C
from config.settings import GestureSettings
from utils.logger import get_logger

log = get_logger(__name__)

# ── Gesture types ─────────────────────────────────────────────────────────────
class GestureType(Enum):
    NONE       = "None"
    POINT      = "Point"       # index only extended  → move cursor
    PINCH      = "Pinch"       # thumb+index touch    → click / drag
    OPEN_HAND  = "Open Hand"   # all fingers out      → scroll
    FIST       = "Fist"        # all fingers in       → pause
    VICTORY    = "Victory"     # index+middle out     → right click


class GestureEvent(Enum):
    NONE    = auto()
    ENTERED = auto()   # gesture just became active this frame
    HELD    = auto()   # gesture ongoing
    EXITED  = auto()   # gesture just ended this frame


@dataclass(frozen=True)
class GestureResult:
    gesture:    GestureType
    event:      GestureEvent
    confidence: float
    hold_s:     float          # seconds current gesture has been held
    position:   Tuple[int, int]  # representative pixel for this gesture


# ── Layer 1: Stateless Geometry Classifier ────────────────────────────────────
class GestureClassifier:
    """
    Pure function: landmark list → (GestureType, confidence, position_px).

    All checks use ratios relative to hand_size so the same gesture
    works whether the hand is 15cm or 40cm from the camera.

    Finger extension uses TWO independent checks (MCP-based + PIP-based)
    combined with OR — catches fast motion where one check may fail.
    """

    def __init__(self, gs: GestureSettings):
        self._gs = gs

    def reconfigure(self, gs: GestureSettings) -> None:
        self._gs = gs

    # ── Public ────────────────────────────────────────────────────────────

    def classify(
        self,
        lm,              # mediapipe landmark list (lm_proto.landmark)
        frame_w: int,
        frame_h: int,
    ) -> Tuple[GestureType, float, Tuple[int, int]]:
        """
        Returns (gesture, confidence, position_px).
        position_px is the most meaningful point for this gesture
        (fingertip for POINT, pinch midpoint for PINCH, etc.)
        """
        pts = self._extract_points(lm, frame_w, frame_h)
        return self._classify(pts)

    # ── Internal ──────────────────────────────────────────────────────────

    def _extract_points(self, lm, w: int, h: int) -> dict:
        """Convert normalised landmarks to pixel dicts. O(21) ops."""
        def px(idx):
            return (int(lm[idx].x * w), int(lm[idx].y * h))

        return {
            "wrist"      : px(C.LM_WRIST),
            "thumb_tip"  : px(C.LM_THUMB_TIP),
            "thumb_mcp"  : px(C.LM_THUMB_MCP),
            "index_tip"  : px(C.LM_INDEX_TIP),
            "index_pip"  : px(C.LM_INDEX_PIP),
            "index_mcp"  : px(C.LM_INDEX_MCP),
            "middle_tip" : px(C.LM_MIDDLE_TIP),
            "middle_pip" : px(C.LM_MIDDLE_PIP),
            "middle_mcp" : px(C.LM_MIDDLE_MCP),
            "ring_tip"   : px(C.LM_RING_TIP),
            "ring_pip"   : px(C.LM_RING_PIP),
            "ring_mcp"   : px(C.LM_RING_MCP),
            "pinky_tip"  : px(C.LM_PINKY_TIP),
            "pinky_pip"  : px(C.LM_PINKY_PIP),
            "pinky_mcp"  : px(C.LM_PINKY_MCP),
            "palm_ref"   : px(C.LM_PALM_REF),
        }

    @staticmethod
    def _dist(a: Tuple, b: Tuple) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _classify(self, p: dict) -> Tuple[GestureType, float, Tuple[int, int]]:
        gs   = self._gs
        dist = self._dist

        # Scale reference: wrist → middle_mcp (most stable hand measurement)
        hand_size = dist(p["wrist"], p["palm_ref"]) or 1.0

        # ── Pinch (highest priority — checked before extension) ────────────
        pinch_d   = dist(p["thumb_tip"], p["index_tip"])
        pinching  = pinch_d < hand_size * gs.pinch_sensitivity
        if pinching:
            mid = (
                (p["thumb_tip"][0] + p["index_tip"][0]) // 2,
                (p["thumb_tip"][1] + p["index_tip"][1]) // 2,
            )
            conf = 1.0 - (pinch_d / (hand_size * gs.pinch_sensitivity))
            return GestureType.PINCH, float(np.clip(conf, 0.5, 1.0)), mid

        # ── Finger extension ───────────────────────────────────────────────
        # Two independent checks, combined with OR for robustness:
        #   a) tip is above MCP by ratio × hand_size  (reliable when hand flat)
        #   b) tip y < pip y − 5px                    (reliable in motion)
        ratio = gs.extend_sensitivity

        def extended(tip, mcp, pip) -> bool:
            by_mcp = tip[1] < mcp[1] - hand_size * ratio
            by_pip = tip[1] < pip[1] - 5
            return by_mcp or by_pip

        idx_ext = extended(p["index_tip"],  p["index_mcp"],  p["index_pip"])
        mid_ext = extended(p["middle_tip"], p["middle_mcp"], p["middle_pip"])
        rng_ext = extended(p["ring_tip"],   p["ring_mcp"],   p["ring_pip"])
        pky_ext = extended(p["pinky_tip"],  p["pinky_mcp"],  p["pinky_pip"])
        thm_ext = dist(p["thumb_tip"], p["wrist"]) > hand_size * C.THUMB_EXTEND_RATIO

        fingers_up = sum([idx_ext, mid_ext, rng_ext, pky_ext])

        # ── POINT: index only ──────────────────────────────────────────────
        if idx_ext and not mid_ext and not rng_ext:
            return GestureType.POINT, 0.95, p["index_tip"]

        # ── VICTORY: index + middle, rest curled ──────────────────────────
        if idx_ext and mid_ext and not rng_ext and not pky_ext:
            spread = dist(p["index_tip"], p["middle_tip"])
            if spread > hand_size * C.VICTORY_SPREAD_RATIO:
                pos = (
                    (p["index_tip"][0] + p["middle_tip"][0]) // 2,
                    (p["index_tip"][1] + p["middle_tip"][1]) // 2,
                )
                return GestureType.VICTORY, 0.90, pos

        # ── OPEN HAND: all four fingers extended ───────────────────────────
        if fingers_up == 4:
            cx = (p["index_tip"][0] + p["middle_tip"][0] +
                  p["ring_tip"][0]  + p["pinky_tip"][0]) // 4
            cy = (p["index_tip"][1] + p["middle_tip"][1] +
                  p["ring_tip"][1]  + p["pinky_tip"][1]) // 4
            return GestureType.OPEN_HAND, 0.93, (cx, cy)

        # ── FIST: all fingers curled ───────────────────────────────────────
        if fingers_up == 0:
            return GestureType.FIST, 0.88, p["wrist"]

        # ── NONE: ambiguous transition posture ────────────────────────────
        return GestureType.NONE, 0.0, p["wrist"]


# ── Layer 2: Stateful State Machine ───────────────────────────────────────────
class GestureStateMachine:
    """
    Wraps GestureClassifier with:
      - Debounce: new gesture must hold for debounce_s before accepted
      - Event emission: ENTERED / HELD / EXITED per transition
      - Hold tracking: how long current gesture has been active

    One instance per tracked hand.

    Usage:
        sm = GestureStateMachine(settings.gesture)
        result = sm.update(lm, frame_w, frame_h)
        if result.event == GestureEvent.ENTERED:
            ...
    """

    def __init__(self, gs: GestureSettings):
        self._gs          = gs
        self._classifier  = GestureClassifier(gs)

        # Confirmed (debounced) state
        self._current     = GestureType.NONE
        self._hold_start  = 0.0          # when current gesture was confirmed

        # Candidate (not yet debounced)
        self._candidate   = GestureType.NONE
        self._cand_start  = 0.0          # when candidate first appeared

    def reconfigure(self, gs: GestureSettings) -> None:
        self._gs = gs
        self._classifier.reconfigure(gs)

    def update(self, lm, frame_w: int, frame_h: int) -> GestureResult:
        """
        Call once per frame per hand.
        Returns a GestureResult with gesture, event, confidence, hold_s, position.
        """
        now = time.monotonic()

        raw_gesture, confidence, position = self._classifier.classify(
            lm, frame_w, frame_h
        )

        # ── Debounce logic ────────────────────────────────────────────────
        if raw_gesture != self._candidate:
            # New candidate — start timer
            self._candidate  = raw_gesture
            self._cand_start = now

        cand_held = now - self._cand_start

        # Special case: PINCH bypasses debounce for click responsiveness
        # (debounce on pinch = missed fast clicks)
        debounce = (
            0.0
            if raw_gesture == GestureType.PINCH
            else self._gs.debounce_s
        )

        if cand_held >= debounce and raw_gesture != self._current:
            # Candidate promoted to confirmed
            prev             = self._current
            self._current    = raw_gesture
            self._hold_start = now

            event = GestureEvent.ENTERED
            log.debug(
                f"Gesture: {prev.value} → {raw_gesture.value}  "
                f"conf={confidence:.2f}"
            )
        elif raw_gesture == self._current:
            event = GestureEvent.HELD
        else:
            # Candidate not yet promoted — report current (hold it steady)
            event       = GestureEvent.HELD
            raw_gesture = self._current
            confidence  = 0.5   # lower confidence during transition

        hold_s = now - self._hold_start

        return GestureResult(
            gesture    = self._current,
            event      = event,
            confidence = confidence,
            hold_s     = hold_s,
            position   = position,
        )

    def reset(self) -> None:
        """Call when hand is lost."""
        self._current   = GestureType.NONE
        self._candidate = GestureType.NONE
        self._hold_start = 0.0
        self._cand_start = 0.0
        log.debug("GestureStateMachine reset")

    @property
    def current(self) -> GestureType:
        return self._current

    @property
    def hold_s(self) -> float:
        return time.monotonic() - self._hold_start