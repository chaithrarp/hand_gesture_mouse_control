"""
processing/metrics.py

Pipeline performance metrics — collected every frame, reported to HUD
and log.

What this tracks (per session):
  - Frame count, uptime, mean/p95/max frame time
  - Per-stage latency (inference, filtering, gesture, mouse)
  - Detection rate: % of frames with a hand present
  - Gesture distribution: how many frames each gesture was active
  - Drop rate: frames where inference was skipped (idle skip)
  - Action count: how many mouse events were fired

Why separate from utils/performance.py:
  utils/performance.py owns raw timing primitives (RollingStats,
  FrameTimer, StageTimer). This module owns DOMAIN metrics — things
  that only make sense in the context of the gesture pipeline, like
  detection rate and gesture distribution.

  utils/performance.py has no imports from the rest of the project.
  This module imports GestureType and ActionType to track by name.

Usage:
    metrics = PipelineMetrics()

    # Each frame:
    metrics.record_frame(has_hand=True, skipped=False)
    metrics.record_gesture("Right", GestureType.POINT)
    metrics.record_action(ActionType.CURSOR_MOVE)

    # For HUD:
    snap = metrics.snapshot()

    # On shutdown:
    metrics.log_summary()
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ── PipelineMetrics ───────────────────────────────────────────────────────────
class PipelineMetrics:
    """
    Lightweight domain-level metrics for the gesture pipeline.

    All counters are integers or deques — zero float math on the hot path.
    snapshot() does the division/formatting, called once per frame for HUD.
    """

    def __init__(self, rolling_window: int = 300):
        """
        Args:
            rolling_window: Number of recent frames used for rolling rates.
                            300 = ~10 seconds at 30fps.
        """
        self._window = rolling_window
        self._t_start = time.monotonic()

        # ── Lifetime counters ─────────────────────────────────────────────
        self._total_frames    = 0
        self._frames_with_hand = 0
        self._frames_skipped  = 0   # idle-skip frames

        # ── Gesture distribution (lifetime, per hand) ─────────────────────
        # {"Right": {"Point": 342, "Pinch": 89, ...}}
        self._gesture_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # ── Action counts (lifetime) ──────────────────────────────────────
        self._action_counts: Dict[str, int] = defaultdict(int)

        # ── Rolling detection presence (1=hand, 0=no hand) ────────────────
        self._detection_window: deque = deque(maxlen=rolling_window)

        # ── Rolling frame drop (1=skipped, 0=processed) ───────────────────
        self._skip_window: deque = deque(maxlen=rolling_window)

        # ── Latency spike counter ─────────────────────────────────────────
        self._latency_spikes = 0   # frames where any stage exceeded budget

    # ── Per-frame recording ───────────────────────────────────────────────────

    def record_frame(self, has_hand: bool, skipped: bool = False) -> None:
        """
        Call once at the top of each frame loop iteration.

        Args:
            has_hand: True if at least one hand was detected this frame.
            skipped:  True if MediaPipe inference was skipped (idle skip).
        """
        self._total_frames += 1
        if has_hand:
            self._frames_with_hand += 1
        if skipped:
            self._frames_skipped += 1

        self._detection_window.append(1 if has_hand else 0)
        self._skip_window.append(1 if skipped else 0)

    def record_gesture(self, hand_label: str, gesture_name: str) -> None:
        """
        Record which gesture a hand is showing this frame.

        Args:
            hand_label:   "Left" or "Right"
            gesture_name: GestureType.value string (e.g. "Point", "Pinch")
        """
        self._gesture_counts[hand_label][gesture_name] += 1

    def record_action(self, action_name: str) -> None:
        """
        Record a mouse action being fired.

        Args:
            action_name: ActionType.name string (e.g. "LEFT_CLICK", "SCROLL")
        """
        self._action_counts[action_name] += 1

    def record_latency_spike(self) -> None:
        """Call when any stage exceeds its latency budget."""
        self._latency_spikes += 1

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict:
        """
        Returns a dict of all metrics for HUD display.
        Call once per frame — O(window) work for rolling rates.

        Keys:
          total_frames       — lifetime frame count
          uptime_s           — seconds since start
          detection_rate     — rolling % of frames with hand (0-100)
          skip_rate          — rolling % of frames skipped (0-100)
          latency_spikes     — lifetime spike count
          gesture_dist       — {hand: {gesture: count}} (lifetime)
          action_counts      — {action: count} (lifetime)
          top_gesture        — most used gesture this session (str)
          clicks             — lifetime left click count
          scrolls            — lifetime scroll count
        """
        now    = time.monotonic()
        uptime = now - self._t_start

        # Rolling rates
        det_window = list(self._detection_window)
        skp_window = list(self._skip_window)

        det_rate = (sum(det_window) / len(det_window) * 100) if det_window else 0.0
        skp_rate = (sum(skp_window) / len(skp_window) * 100) if skp_window else 0.0

        # Top gesture across all hands
        all_gestures: Dict[str, int] = defaultdict(int)
        for hand_dict in self._gesture_counts.values():
            for gname, count in hand_dict.items():
                all_gestures[gname] += count
        top_gesture = max(all_gestures, key=all_gestures.get) if all_gestures else "None"

        return {
            "total_frames":    self._total_frames,
            "uptime_s":        round(uptime, 1),
            "detection_rate":  round(det_rate, 1),
            "skip_rate":       round(skp_rate, 1),
            "latency_spikes":  self._latency_spikes,
            "gesture_dist":    dict(self._gesture_counts),
            "action_counts":   dict(self._action_counts),
            "top_gesture":     top_gesture,
            "clicks":          self._action_counts.get("LEFT_CLICK", 0),
            "scrolls":         self._action_counts.get("SCROLL", 0),
            "frames_with_hand": self._frames_with_hand,
            "frames_skipped":   self._frames_skipped,
        }

    # ── Shutdown summary ──────────────────────────────────────────────────────

    def log_summary(self) -> None:
        """Print full session metrics to log on shutdown."""
        s      = self.snapshot()
        uptime = s["uptime_s"]
        fps    = round(s["total_frames"] / max(uptime, 1), 1)

        log.info("── Session metrics ───────────────────────────")
        log.info(f"  Uptime          : {uptime:.1f}s")
        log.info(f"  Total frames    : {s['total_frames']}  (~{fps} fps avg)")
        log.info(f"  Detection rate  : {s['detection_rate']:.1f}%")
        log.info(f"  Inference skips : {s['frames_skipped']}  ({s['skip_rate']:.1f}%)")
        log.info(f"  Latency spikes  : {s['latency_spikes']}")
        log.info(f"  Left clicks     : {s['clicks']}")
        log.info(f"  Scroll events   : {s['scrolls']}")
        log.info(f"  Top gesture     : {s['top_gesture']}")

        if s["gesture_dist"]:
            log.info("  Gesture distribution:")
            for hand, gdict in s["gesture_dist"].items():
                sorted_g = sorted(gdict.items(), key=lambda x: x[1], reverse=True)
                parts    = "  ".join(f"{g}={c}" for g, c in sorted_g)
                log.info(f"    {hand}: {parts}")

        if s["action_counts"]:
            log.info("  Actions fired:")
            for action, count in sorted(
                s["action_counts"].items(), key=lambda x: x[1], reverse=True
            ):
                log.info(f"    {action:<20} {count}")

        log.info("─────────────────────────────────────────────")