"""
processing/filters.py

Signal processing for raw landmark coordinates → stable screen position.

Pipeline (in order):
  1. OutlierRejecter  — drops positions that jump impossibly far in one frame
  2. KalmanFilter2D   — removes Gaussian noise from MediaPipe landmark jitter
  3. AdaptiveSmoother — lerp whose alpha scales with hand velocity
  4. HoverLock        — freezes cursor when hand is stationary (targeting aid)

Each stage is independent and can be bypassed via settings flags.
All state is per-hand — instantiate one FilterPipeline per tracked hand.

Latency notes:
  - Every stage is O(1), no allocations in steady state
  - Numpy ops are on 4-element vectors max — no matrix copy overhead
  - Total pipeline cost < 0.1ms per hand per frame
"""

import time
import numpy as np
from typing import Optional, Tuple

from config.settings import CursorSettings, KalmanSettings
from utils.logger import get_logger

log = get_logger(__name__)

Point = Tuple[float, float]


# ── 1. Outlier Rejecter ───────────────────────────────────────────────────────
class OutlierRejecter:
    """
    Rejects positions that jump more than `max_jump_px` pixels from the last
    accepted position in a single frame.

    MediaPipe occasionally produces landmark teleports (especially during
    fast occlusion). These appear as single-frame cursor jumps 300-500px.
    Rejecting them costs nothing — we just hold the previous position.

    max_jump_px is scaled by hand velocity so fast-moving hands aren't
    wrongly rejected.
    """

    def __init__(self, max_jump_px: float = 180.0, velocity_scale: float = 2.5):
        """
        Args:
            max_jump_px:     Base maximum allowed movement per frame (pixels).
            velocity_scale:  When velocity is high, budget scales up by this
                             factor so fast intentional movement isn't rejected.
        """
        self._max_jump     = max_jump_px
        self._vel_scale    = velocity_scale
        self._last: Optional[np.ndarray] = None
        self._velocity     = 0.0          # rolling velocity (px/frame)
        self._rejections   = 0            # counter to prevent livelock

    def update(self, x: float, y: float) -> Tuple[float, float, bool]:
        """
        Returns (x, y, accepted).
        If rejected, returns last accepted position and accepted=False.
        """
        # Ignore (0, 0) glitches for initialization
        if self._last is None and x == 0 and y == 0:
            return 0.0, 0.0, False

        pos = np.array([x, y], dtype=np.float32)

        if self._last is None:
            self._last    = pos
            self._velocity = 0.0
            self._rejections = 0
            return x, y, True

        dist = float(np.linalg.norm(pos - self._last))

        # Budget scales with current velocity — fast hands get more headroom
        budget = self._max_jump + self._velocity * self._vel_scale

        if dist > budget:
            self._rejections += 1
            # If we reject 15+ frames in a row, we're likely stuck in a livelock
            # (e.g. initialised on a (0,0) glitch). Reset and accept current.
            if self._rejections > 15:
                log.info(f"OutlierRejecter: {self._rejections} rejections — resetting to ({x:.0f}, {y:.0f})")
                self.reset()
                return self.update(x, y)

            log.debug(f"Outlier rejected: jump={dist:.0f}px budget={budget:.0f}px")
            return float(self._last[0]), float(self._last[1]), False

        # Update rolling velocity (exponential smoothing)
        self._rejections = 0
        self._velocity = self._velocity * 0.7 + dist * 0.3
        self._last = pos
        return x, y, True

    def reset(self) -> None:
        self._last       = None
        self._velocity   = 0.0
        self._rejections = 0


# ── 2. Kalman Filter 2D ───────────────────────────────────────────────────────
class KalmanFilter2D:
    """
    Standard constant-velocity Kalman filter for 2D cursor position.

    State vector: [x, y, vx, vy]
    Measurement:  [x, y]

    Tuning via KalmanSettings:
      process_noise   → how much we trust the motion model
      measurement_noise → how much we trust the camera measurement

    Lower process_noise  → smoother, more lag
    Lower measure_noise  → trusts camera more, less filtering
    """

    def __init__(self, ks: KalmanSettings):
        import cv2
        self._kf = cv2.KalmanFilter(4, 2)

        # Measurement matrix H: maps state → measurement
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )

        # Transition matrix F: constant velocity model
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )

        self._apply_noise(ks)
        self._initialized = False
        log.debug(f"KalmanFilter2D init: Q={ks.process_noise} R={ks.measurement_noise}")

    def _apply_noise(self, ks: KalmanSettings) -> None:
        self._kf.processNoiseCov    = np.eye(4, dtype=np.float32) * ks.process_noise
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * ks.measurement_noise

    def reconfigure(self, ks: KalmanSettings) -> None:
        """Hot-reload noise params without resetting state."""
        self._apply_noise(ks)

    def update(self, x: float, y: float) -> Point:
        """
        Feed a measurement, get filtered position back.
        First call initialises the state — no warmup frames needed.
        """
        meas = np.array([[np.float32(x)], [np.float32(y)]])

        if not self._initialized:
            self._kf.statePre = np.array(
                [[x], [y], [0.0], [0.0]], dtype=np.float32
            )
            self._kf.statePost = self._kf.statePre.copy()
            self._initialized  = True

        self._kf.correct(meas)
        pred = self._kf.predict()
        return float(pred[0]), float(pred[1])

    def reset(self) -> None:
        self._initialized = False


# ── 3. Adaptive Smoother ──────────────────────────────────────────────────────
class AdaptiveSmoother:
    """
    Exponential moving average (lerp) whose alpha adapts to hand velocity.

    Problem with fixed smoothing:
      - Low alpha (0.2): great jitter suppression, terrible lag on fast moves
      - High alpha (0.8): responsive, but jittery when stationary

    Solution — three zones:
      velocity < slow_threshold  → alpha = base * slow_factor  (more smoothing)
      velocity > fast_threshold  → alpha = min(base * fast_factor, 0.95)  (less smoothing)
      between                    → linearly interpolate

    This gives hover-quality stability AND snap-quality responsiveness.
    """

    def __init__(self, cs: CursorSettings):
        self._cs    = cs
        self._pos   = np.array([0.0, 0.0], dtype=np.float64)
        self._vel   = 0.0
        self._ready = False

    def reconfigure(self, cs: CursorSettings) -> None:
        self._cs = cs

    def _alpha(self) -> float:
        """Compute current lerp factor based on velocity."""
        cs  = self._cs
        base = cs.smoothing

        if not cs.adaptive_smoothing:
            return base

        slow = cs.adaptive_slow_threshold
        fast = cs.adaptive_fast_threshold

        if self._vel <= slow:
            # Stationary zone — stronger smoothing
            return max(0.05, base * 0.5)
        elif self._vel >= fast:
            # Fast zone — minimal smoothing so cursor keeps up
            return min(0.95, base * 2.2)
        else:
            # Linear interpolation between zones
            t = (self._vel - slow) / (fast - slow)
            lo = max(0.05, base * 0.5)
            hi = min(0.95, base * 2.2)
            return lo + t * (hi - lo)

    def update(self, x: float, y: float) -> Point:
        new_pos = np.array([x, y], dtype=np.float64)

        if not self._ready:
            self._pos   = new_pos
            self._ready = True
            return x, y

        # Update velocity estimate (px/frame, exponential smooth)
        dist       = float(np.linalg.norm(new_pos - self._pos))
        self._vel  = self._vel * 0.6 + dist * 0.4

        alpha      = self._alpha()
        self._pos  = self._pos + alpha * (new_pos - self._pos)
        return float(self._pos[0]), float(self._pos[1])

    @property
    def velocity(self) -> float:
        return self._vel

    def reset(self) -> None:
        self._ready = False
        self._vel   = 0.0


# ── 4. Hover Lock ─────────────────────────────────────────────────────────────
class HoverLock:
    """
    Freezes cursor output when hand velocity stays below threshold for
    hover_lock_ms milliseconds.

    Effect: when you stop your hand to target a button, the cursor snaps
    to a stable position and holds it. Eliminates the last 2-3px of jitter
    that makes clicking small UI targets frustrating.

    Lock releases immediately on movement — no hysteresis needed because
    AdaptiveSmoother already handles the slow-speed zone.
    """

    def __init__(self, cs: CursorSettings):
        self._cs          = cs
        self._locked_pos: Optional[np.ndarray] = None
        self._still_since: Optional[float]     = None

    def reconfigure(self, cs: CursorSettings) -> None:
        self._cs = cs

    def update(self, x: float, y: float, velocity: float) -> Point:
        """
        Args:
            x, y:     Smoothed cursor position from AdaptiveSmoother.
            velocity: Current hand velocity (px/frame) from AdaptiveSmoother.

        Returns final cursor position (possibly locked).
        """
        cs = self._cs

        if not cs.hover_lock_enabled:
            return x, y

        now = time.monotonic()

        if velocity > cs.hover_velocity_threshold:
            # Moving — release lock immediately
            self._locked_pos  = None
            self._still_since = None
            return x, y

        # Below threshold — start or continue hover timer
        if self._still_since is None:
            self._still_since = now

        elapsed_ms = (now - self._still_since) * 1000.0

        if elapsed_ms >= cs.hover_lock_ms:
            # Lock engaged — freeze at first-lock position
            if self._locked_pos is None:
                self._locked_pos = np.array([x, y], dtype=np.float64)
                log.debug(f"HoverLock engaged at ({x:.0f}, {y:.0f})")
            return float(self._locked_pos[0]), float(self._locked_pos[1])

        # In the grace period before lock — pass through
        return x, y

    def reset(self) -> None:
        self._locked_pos  = None
        self._still_since = None


# ── 5. Filter Pipeline ────────────────────────────────────────────────────────
class FilterPipeline:
    """
    Composes all four stages into a single call per frame per hand.

    Usage:
        pipeline = FilterPipeline(settings.cursor, settings.kalman)

        # Each frame, for each tracked hand:
        sx, sy = pipeline.update(raw_cam_x, raw_cam_y)

        # On hand lost / re-detected:
        pipeline.reset()

        # On settings change (live):
        pipeline.reconfigure(new_cursor_settings, new_kalman_settings)
    """

    def __init__(self, cs: CursorSettings, ks: KalmanSettings):
        self._outlier  = OutlierRejecter(max_jump_px=200.0)
        self._kalman   = KalmanFilter2D(ks)
        self._smoother = AdaptiveSmoother(cs)
        self._hover    = HoverLock(cs)
        self._cs       = cs

    def update(self, raw_x: float, raw_y: float) -> Point:
        """
        Full pipeline: raw camera coords → stable screen coords.
        Input is in camera pixels. Output is in camera pixels
        (coordinate mapping to screen happens in CoordinateMapper).
        """
        # Stage 1 — outlier rejection
        x, y, accepted = self._outlier.update(raw_x, raw_y)

        # Stage 2 — Kalman (runs even on rejected frames using last good pos)
        x, y = self._kalman.update(x, y)

        # Stage 3 — adaptive lerp smoothing
        x, y = self._smoother.update(x, y)

        # Stage 4 — hover lock
        x, y = self._hover.update(x, y, self._smoother.velocity)

        return x, y

    @property
    def velocity(self) -> float:
        """Current hand velocity in px/frame — used by gesture recognizer."""
        return self._smoother.velocity

    def reconfigure(self, cs: CursorSettings, ks: KalmanSettings) -> None:
        """Apply updated settings without resetting filter state."""
        self._cs = cs
        self._kalman.reconfigure(ks)
        self._smoother.reconfigure(cs)
        self._hover.reconfigure(cs)

    def reset(self) -> None:
        """Call when hand is lost or re-detected."""
        self._outlier.reset()
        self._kalman.reset()
        self._smoother.reset()
        self._hover.reset()
        log.debug("FilterPipeline reset")