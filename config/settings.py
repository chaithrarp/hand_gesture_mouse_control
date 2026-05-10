"""
config/settings.py

Runtime-tunable parameters grouped into focused dataclasses.
These can be changed live without restarting — every module holds
a reference to the same Settings instance.

Design rules:
  - Every field has a default that works out-of-the-box
  - __post_init__ clamps values to safe ranges (no silent bad state)
  - No I/O here — loading/saving is handled by utils/config_io.py
  - All timing in seconds, all ratios 0-1, all pixels explicit
"""

from dataclasses import dataclass, field
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CameraSettings:
    index: int   = 0
    width: int   = 1280
    height: int  = 720
    fps: int     = 30

    def __post_init__(self):
        self.index  = max(0, self.index)
        self.width  = max(320, min(3840, self.width))
        self.height = max(240, min(2160, self.height))
        self.fps    = max(15,  min(120,  self.fps))


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION  — MediaPipe confidence thresholds
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DetectionSettings:
    detection_confidence: float = 0.60
    tracking_confidence: float  = 0.45
    visibility_threshold: float = 0.05

    def __post_init__(self):
        self.detection_confidence = max(0.1, min(1.0, self.detection_confidence))
        self.tracking_confidence  = max(0.1, min(1.0, self.tracking_confidence))
        self.visibility_threshold = max(0.0, min(0.5, self.visibility_threshold))


# ─────────────────────────────────────────────────────────────────────────────
# CURSOR  — movement feel
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CursorSettings:
    speed: float = 1.2
    smoothing: float = 0.45
    adaptive_smoothing: bool = True
    adaptive_fast_threshold: float = 18.0
    adaptive_slow_threshold: float = 4.0
    dead_zone_px: int = 3
    hover_lock_enabled: bool  = True
    hover_velocity_threshold: float = 3.5
    hover_lock_ms: int        = 180
    active_zone_x: Tuple[float, float] = (0.05, 0.95)
    active_zone_y: Tuple[float, float] = (0.05, 0.95)

    def __post_init__(self):
        self.speed    = max(0.3, min(3.0,  self.speed))
        self.smoothing = max(0.05, min(0.95, self.smoothing))
        self.dead_zone_px = max(0, min(20, self.dead_zone_px))
        self.adaptive_fast_threshold = max(5.0, self.adaptive_fast_threshold)
        self.adaptive_slow_threshold = max(0.5, min(self.adaptive_fast_threshold - 1, self.adaptive_slow_threshold))
        self.hover_lock_ms = max(50, min(500, self.hover_lock_ms))


# ─────────────────────────────────────────────────────────────────────────────
# GESTURES  — recognition sensitivity
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GestureSettings:
    pinch_sensitivity: float = 0.28
    extend_sensitivity: float = 0.12
    debounce_s: float = 0.08
    click_min_hold_s: float = 0.06
    click_max_hold_s: float = 1.4
    drag_lock_s: float = 0.45
    right_click_debounce_s: float = 0.9

    def __post_init__(self):
        self.pinch_sensitivity  = max(0.10, min(0.50, self.pinch_sensitivity))
        self.extend_sensitivity = max(0.05, min(0.30, self.extend_sensitivity))
        self.debounce_s         = max(0.03, min(0.30, self.debounce_s))
        self.click_min_hold_s   = max(0.03, min(0.20, self.click_min_hold_s))
        self.click_max_hold_s   = max(0.5,  min(3.0,  self.click_max_hold_s))
        self.drag_lock_s        = max(0.2,  min(1.5,  self.drag_lock_s))


# ─────────────────────────────────────────────────────────────────────────────
# SCROLL  — feel of open-hand scroll
# FIX LOG:
#   pixels_per_tick  lowered  22 → 12   (scroll fires more often = feels faster)
#   speed_multiplier float    1→3 range (was int, couldn't fine-tune)
#   accumulation_smooth raised 0.18→0.35 (delta bleeds in faster = less "stuck")
#   rolling_ref      NEW      True = ref_y follows hand so scroll doesn't stall
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ScrollSettings:
    # How many pixels of hand movement = one scroll tick.
    # Lower = more sensitive (scrolls faster with less movement).
    # Was 22.0 — lowered to 12.0 so scroll fires more readily.
    pixels_per_tick: float = 12.0

    # Multiplier on scroll ticks sent to OS.
    # Now FLOAT so you can dial between 1.0 and 5.0 precisely.
    # Was int (2), now float default 2.5.
    speed_multiplier: float = 2.5

    # Smoothing on scroll delta accumulation.
    # Higher = delta bleeds in faster = less "stuck" feeling.
    # Was 0.18 — raised to 0.35.
    accumulation_smooth: float = 0.35

    # Direction: True = natural (hand down → page down), False = inverted
    natural_direction: bool = True

    # Rolling reference: if True, ref_y slowly follows the hand so scroll
    # doesn't stall when you move beyond the initial anchor point.
    # This was the main cause of "scroll stops working mid-gesture".
    rolling_ref: bool = True

    # How fast ref_y follows hand (lerp factor, 0=never follows, 1=instant).
    # Keep low (0.05-0.12) so it trails, not snaps.
    rolling_ref_lerp: float = 0.08

    def __post_init__(self):
        self.pixels_per_tick      = max(3.0,  min(80.0, self.pixels_per_tick))
        self.speed_multiplier     = max(0.5,  min(10.0, self.speed_multiplier))
        self.accumulation_smooth  = max(0.05, min(0.8,  self.accumulation_smooth))
        self.rolling_ref_lerp     = max(0.01, min(0.3,  self.rolling_ref_lerp))


# ─────────────────────────────────────────────────────────────────────────────
# KALMAN  — filter aggressiveness
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class KalmanSettings:
    process_noise: float = 0.08
    measurement_noise: float = 0.08

    def __post_init__(self):
        self.process_noise    = max(0.001, min(1.0, self.process_noise))
        self.measurement_noise = max(0.001, min(1.0, self.measurement_noise))


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE  — pipeline tuning
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PerformanceSettings:
    inference_width: int  = 640
    inference_height: int = 360
    idle_skip_frames: int = 2
    mp_threads: int = 0

    def __post_init__(self):
        self.inference_width  = max(160, min(1280, self.inference_width))
        self.inference_height = max(120, min(720,  self.inference_height))
        self.idle_skip_frames = max(1,   min(5,    self.idle_skip_frames))


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SETTINGS  — single object passed everywhere
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Settings:
    camera:      CameraSettings      = field(default_factory=CameraSettings)
    detection:   DetectionSettings   = field(default_factory=DetectionSettings)
    cursor:      CursorSettings      = field(default_factory=CursorSettings)
    gesture:     GestureSettings     = field(default_factory=GestureSettings)
    scroll:      ScrollSettings      = field(default_factory=ScrollSettings)
    kalman:      KalmanSettings      = field(default_factory=KalmanSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)

    def summary(self) -> str:
        lines = ["── Settings ──────────────────────────────"]
        lines.append(f"  Camera      : {self.camera.width}x{self.camera.height} @ {self.camera.fps}fps  idx={self.camera.index}")
        lines.append(f"  Detection   : conf={self.detection.detection_confidence}  track={self.detection.tracking_confidence}")
        lines.append(f"  Cursor      : speed={self.cursor.speed}  smooth={self.cursor.smoothing}  adaptive={self.cursor.adaptive_smoothing}")
        lines.append(f"  Dead zone   : {self.cursor.dead_zone_px}px  hover_lock={self.cursor.hover_lock_enabled} ({self.cursor.hover_lock_ms}ms)")
        lines.append(f"  Gesture     : pinch={self.gesture.pinch_sensitivity}  extend={self.gesture.extend_sensitivity}  debounce={self.gesture.debounce_s}s")
        lines.append(f"  Scroll      : {self.scroll.pixels_per_tick}px/tick  x{self.scroll.speed_multiplier}  natural={self.scroll.natural_direction}  rolling={self.scroll.rolling_ref}")
        lines.append(f"  Kalman      : process={self.kalman.process_noise}  measure={self.kalman.measurement_noise}")
        lines.append(f"  Inference   : {self.performance.inference_width}x{self.performance.inference_height}  skip={self.performance.idle_skip_frames}")
        lines.append("──────────────────────────────────────────")
        return "\n".join(lines)


# ── Module-level default instance ─────────────────────────────────────────────
settings = Settings()