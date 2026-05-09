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
    # Lower detection_confidence → finds hand in more lighting conditions
    # but may produce false positives
    detection_confidence: float = 0.60

    # Lower tracking_confidence → keeps track through fast motion
    # but may drift slightly
    tracking_confidence: float  = 0.45

    # Ignore landmarks below this visibility (webcams often give ~0.1 even
    # for clearly visible points — keep this very low)
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
    # ── Speed ──────────────────────────────────────────────────────────────
    # Multiplier on raw camera-to-screen mapping.
    # 1.0 = hand must cross full active zone to cross full screen.
    # 1.5 = faster — smaller hand movements needed.
    speed: float = 1.2

    # ── Smoothing ──────────────────────────────────────────────────────────
    # Lerp factor applied AFTER Kalman. 0=frozen, 1=raw/instant.
    # Lower = smoother path but more lag. 0.35-0.55 is the sweet spot.
    smoothing: float = 0.45

    # ── Adaptive smoothing ─────────────────────────────────────────────────
    # When hand moves fast, reduce smoothing so cursor keeps up.
    # When hand is slow/still, increase smoothing to kill jitter.
    # Set False to use fixed smoothing above.
    adaptive_smoothing: bool = True

    # Pixel/frame velocity above which smoothing is halved (fast mode)
    adaptive_fast_threshold: float = 18.0

    # Pixel/frame velocity below which smoothing is doubled (hover mode)
    adaptive_slow_threshold: float = 4.0

    # ── Dead zone ──────────────────────────────────────────────────────────
    # Cursor won't move unless it would move more than this many pixels.
    # Kills jitter on a stationary hand. Keep small — large values make
    # cursor feel "sticky".
    dead_zone_px: int = 3

    # ── Hover lock ─────────────────────────────────────────────────────────
    # If hand velocity < hover_velocity_threshold for hover_lock_ms,
    # cursor position is frozen (eliminates jitter during targeting).
    hover_lock_enabled: bool  = True
    hover_velocity_threshold: float = 3.5   # px/frame — below this = hovering
    hover_lock_ms: int        = 180          # ms stationary before lock engages

    # ── Active zone mapping ────────────────────────────────────────────────
    # Fraction of frame that maps to full screen. Shrink to need less
    # hand movement; expand for more precision.
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
    # Pinch sensitivity: fraction of hand_size for thumb-index contact.
    # Lower = need to pinch tighter. Higher = triggers more easily.
    pinch_sensitivity: float = 0.28

    # How firmly fingers must extend to count as "up".
    # Lower = more lenient (better for fast motion).
    # Higher = stricter (fewer false extends).
    extend_sensitivity: float = 0.12

    # Seconds a gesture must be held before it's accepted.
    # Prevents flicker when transitioning between gestures.
    debounce_s: float = 0.08

    # ── Click ──────────────────────────────────────────────────────────────
    click_min_hold_s: float = 0.06   # below this = accidental brush
    click_max_hold_s: float = 1.4    # above this = drag, not click

    # ── Drag ───────────────────────────────────────────────────────────────
    drag_lock_s: float = 0.45        # pinch held this long → drag mode

    # ── Right click ────────────────────────────────────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ScrollSettings:
    # How many pixels of hand movement = one scroll tick.
    # Lower = more sensitive (scrolls faster with less movement).
    pixels_per_tick: float = 22.0

    # Multiplier on scroll ticks sent to OS.
    # 1 = one OS scroll event per tick. 3 = three events (faster page scroll).
    speed_multiplier: int = 2

    # Smoothing on scroll delta accumulation (0=instant, 1=never fires).
    # Prevents single-frame jitter from triggering a scroll.
    accumulation_smooth: float = 0.18

    # Direction: True = natural (hand down → page down), False = inverted
    natural_direction: bool = True

    def __post_init__(self):
        self.pixels_per_tick      = max(5.0,  min(80.0, self.pixels_per_tick))
        self.speed_multiplier     = max(1,     min(10,   self.speed_multiplier))
        self.accumulation_smooth  = max(0.05,  min(0.5,  self.accumulation_smooth))


# ─────────────────────────────────────────────────────────────────────────────
# KALMAN  — filter aggressiveness
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class KalmanSettings:
    # Process noise: how much we expect the hand to accelerate between frames.
    # Lower → smoother output, more lag on direction changes.
    # Higher → snappier, follows raw input more closely.
    process_noise: float = 0.08

    # Measurement noise: how much we trust the raw landmark position.
    # Lower → trust camera more (less filtering).
    # Higher → distrust camera more (more smoothing).
    measurement_noise: float = 0.08

    def __post_init__(self):
        self.process_noise    = max(0.001, min(1.0, self.process_noise))
        self.measurement_noise = max(0.001, min(1.0, self.measurement_noise))


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE  — pipeline tuning
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PerformanceSettings:
    # Resize frame before sending to MediaPipe.
    # Does NOT affect display frame — only the inference input.
    # Smaller = faster detection, less accuracy. 640x360 is the sweet spot.
    inference_width: int  = 640
    inference_height: int = 360

    # Skip MediaPipe inference every N frames when no hand was detected
    # last frame. Saves CPU on idle frames. 1 = never skip (always infer).
    idle_skip_frames: int = 2

    # Max CPU threads MediaPipe may use. 0 = auto.
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
        """Human-readable dump for logging on startup."""
        lines = ["── Settings ──────────────────────────────"]
        lines.append(f"  Camera      : {self.camera.width}x{self.camera.height} @ {self.camera.fps}fps  idx={self.camera.index}")
        lines.append(f"  Detection   : conf={self.detection.detection_confidence}  track={self.detection.tracking_confidence}")
        lines.append(f"  Cursor      : speed={self.cursor.speed}  smooth={self.cursor.smoothing}  adaptive={self.cursor.adaptive_smoothing}")
        lines.append(f"  Dead zone   : {self.cursor.dead_zone_px}px  hover_lock={self.cursor.hover_lock_enabled} ({self.cursor.hover_lock_ms}ms)")
        lines.append(f"  Gesture     : pinch={self.gesture.pinch_sensitivity}  extend={self.gesture.extend_sensitivity}  debounce={self.gesture.debounce_s}s")
        lines.append(f"  Scroll      : {self.scroll.pixels_per_tick}px/tick  x{self.scroll.speed_multiplier}  natural={self.scroll.natural_direction}")
        lines.append(f"  Kalman      : process={self.kalman.process_noise}  measure={self.kalman.measurement_noise}")
        lines.append(f"  Inference   : {self.performance.inference_width}x{self.performance.inference_height}  skip={self.performance.idle_skip_frames}")
        lines.append("──────────────────────────────────────────")
        return "\n".join(lines)


# ── Module-level default instance ─────────────────────────────────────────────
# Import this anywhere: `from config.settings import settings`
settings = Settings()