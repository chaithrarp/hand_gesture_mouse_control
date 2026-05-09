"""
utils/performance.py

Lightweight performance instrumentation for the frame loop.

Provides:
  - FrameTimer  : per-frame timing, FPS, frame-time history
  - StageTimer  : named stage breakdown (capture / inference / gesture / mouse / draw)
  - RollingStats: min/mean/max/p95 over a sliding window — zero allocation on update

Design rules:
  - All updates O(1) — circular buffer, no list appends in steady state
  - No locks — single-threaded frame loop only
  - Snapshots returned as plain dicts — safe to read from UI thread
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ── Rolling statistics over a fixed-size window ───────────────────────────────
class RollingStats:
    """
    Tracks min / mean / max / p95 of a metric over the last `window` samples.
    Uses a circular deque — update() is O(1), snapshot() is O(n) but called
    only for display (not every frame).
    """
    __slots__ = ("_name", "_buf", "_window")

    def __init__(self, name: str, window: int = 90):
        self._name   = name
        self._window = window
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> None:
        self._buf.append(value)

    def snapshot(self) -> Dict[str, float]:
        if not self._buf:
            return {"min": 0.0, "mean": 0.0, "max": 0.0, "p95": 0.0, "n": 0}
        data   = sorted(self._buf)
        n      = len(data)
        p95_i  = min(int(n * 0.95), n - 1)
        return {
            "min"  : data[0],
            "mean" : sum(data) / n,
            "max"  : data[-1],
            "p95"  : data[p95_i],
            "n"    : n,
        }

    @property
    def mean(self) -> float:
        return sum(self._buf) / len(self._buf) if self._buf else 0.0

    @property
    def latest(self) -> float:
        return self._buf[-1] if self._buf else 0.0


# ── Per-pipeline-stage timer ──────────────────────────────────────────────────
@dataclass
class StageTimer:
    """
    Tracks time spent in each named pipeline stage per frame.

    Usage:
        st = StageTimer()
        st.begin("capture")
        frame = cap.read()
        st.end("capture")

        st.begin("inference")
        results = hands.process(rgb)
        st.end("inference")

        breakdown = st.snapshot()   # {stage: mean_ms, ...}
    """
    _stages: Dict[str, RollingStats] = field(default_factory=dict)
    _t:      Dict[str, float]        = field(default_factory=dict)

    def begin(self, stage: str) -> None:
        if stage not in self._stages:
            self._stages[stage] = RollingStats(stage, window=60)
        self._t[stage] = time.monotonic()

    def end(self, stage: str) -> float:
        """Returns elapsed ms for this stage."""
        if stage not in self._t:
            return 0.0
        elapsed_ms = (time.monotonic() - self._t.pop(stage)) * 1000.0
        self._stages[stage].update(elapsed_ms)
        return elapsed_ms

    def snapshot(self) -> Dict[str, float]:
        """Returns {stage_name: mean_ms} for HUD display."""
        return {name: stats.mean for name, stats in self._stages.items()}

    def snapshot_full(self) -> Dict[str, Dict]:
        """Returns full stats (min/mean/max/p95) per stage."""
        return {name: stats.snapshot() for name, stats in self._stages.items()}


# ── Main frame timer ──────────────────────────────────────────────────────────
class FrameTimer:
    """
    Central timing object for the main loop.

    Tracks:
      - Instantaneous FPS (last frame only)
      - Smoothed FPS (rolling 60-frame mean)
      - Frame time in ms
      - Total frames processed
      - Pipeline stage breakdown via embedded StageTimer

    Usage:
        timer = FrameTimer()
        while True:
            timer.tick()                    # call at TOP of every frame
            timer.stage.begin("capture")
            ...
            timer.stage.end("capture")
            hud_data = timer.snapshot()     # call once per frame for display
    """

    def __init__(self, fps_window: int = 60):
        self._fps_window  = fps_window
        self._frame_times: deque[float] = deque(maxlen=fps_window)
        self._last_tick:   Optional[float] = None
        self._total_frames = 0
        self._t_start      = time.monotonic()

        self.frame_ms  = RollingStats("frame_ms",  window=fps_window)
        self.stage     = StageTimer()

        log.debug("FrameTimer initialised")

    def tick(self) -> float:
        """
        Call at the very top of each frame loop iteration.
        Returns frame_time_ms since last tick (0.0 on first call).
        """
        now = time.monotonic()
        if self._last_tick is None:
            self._last_tick = now
            return 0.0

        dt_ms = (now - self._last_tick) * 1000.0
        self._last_tick = now
        self._total_frames += 1
        self._frame_times.append(dt_ms)
        self.frame_ms.update(dt_ms)
        return dt_ms

    # ── FPS helpers ───────────────────────────────────────────────────────
    @property
    def fps_instant(self) -> float:
        """FPS from the last single frame — spiky but lag-free."""
        ft = self.frame_ms.latest
        return 1000.0 / ft if ft > 0 else 0.0

    @property
    def fps_smooth(self) -> float:
        """Rolling mean FPS over the window — stable display value."""
        mean = self.frame_ms.mean
        return 1000.0 / mean if mean > 0 else 0.0

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self._t_start

    # ── Snapshot for HUD ─────────────────────────────────────────────────
    def snapshot(self) -> Dict:
        """
        Returns a plain dict safe to pass to the UI layer.
        Called once per frame — O(n) sort inside RollingStats is fine here
        because n=60 and it's not on the critical path.
        """
        fm = self.frame_ms.snapshot()
        return {
            "fps_instant"  : round(self.fps_instant,  1),
            "fps_smooth"   : round(self.fps_smooth,   1),
            "frame_ms_mean": round(fm["mean"],  2),
            "frame_ms_p95" : round(fm["p95"],   2),
            "frame_ms_max" : round(fm["max"],   2),
            "total_frames" : self._total_frames,
            "uptime_s"     : round(self.uptime_s, 1),
            "stages"       : self.stage.snapshot(),
        }

    def log_summary(self) -> None:
        """Call on shutdown — logs a full performance report."""
        snap = self.snapshot()
        log.info(
            f"Performance summary — "
            f"uptime={snap['uptime_s']}s  "
            f"frames={snap['total_frames']}  "
            f"fps={snap['fps_smooth']}  "
            f"p95={snap['frame_ms_p95']}ms"
        )
        full = self.stage.snapshot_full()
        for stage, stats in full.items():
            log.info(
                f"  stage [{stage:<12}] "
                f"mean={stats['mean']:.1f}ms  "
                f"p95={stats['p95']:.1f}ms  "
                f"max={stats['max']:.1f}ms"
            )