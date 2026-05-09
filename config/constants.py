"""
config/constants.py

All fixed magic numbers. These are NOT user-tunable — they reflect hard physical
and algorithmic constraints. User-facing knobs live in settings.py.
"""

# ── Camera ─────────────────────────────────────────────────────────────────
CAMERA_DEFAULT_INDEX  = 0
CAMERA_DEFAULT_WIDTH  = 1280
CAMERA_DEFAULT_HEIGHT = 720
CAMERA_DEFAULT_FPS    = 30

# ── MediaPipe ──────────────────────────────────────────────────────────────
MP_MAX_NUM_HANDS   = 2
MP_STATIC_IMAGE    = False
LANDMARK_COUNT     = 21

# Landmark indices — MediaPipe Hand Model spec (do not change)
LM_WRIST       = 0
LM_THUMB_CMC   = 1
LM_THUMB_MCP   = 2
LM_THUMB_IP    = 3
LM_THUMB_TIP   = 4
LM_INDEX_MCP   = 5
LM_INDEX_PIP   = 6
LM_INDEX_DIP   = 7
LM_INDEX_TIP   = 8
LM_MIDDLE_MCP  = 9
LM_MIDDLE_PIP  = 10
LM_MIDDLE_DIP  = 11
LM_MIDDLE_TIP  = 12
LM_RING_MCP    = 13
LM_RING_PIP    = 14
LM_RING_DIP    = 15
LM_RING_TIP    = 16
LM_PINKY_MCP   = 17
LM_PINKY_PIP   = 18
LM_PINKY_DIP   = 19
LM_PINKY_TIP   = 20

# Palm reference point (base of middle finger — most stable anchor)
LM_PALM_REF = LM_MIDDLE_MCP

# Skeleton connectivity for drawing
HAND_CONNECTIONS = [
    # Thumb
    (LM_WRIST, LM_THUMB_CMC), (LM_THUMB_CMC, LM_THUMB_MCP),
    (LM_THUMB_MCP, LM_THUMB_IP), (LM_THUMB_IP, LM_THUMB_TIP),
    # Index
    (LM_WRIST, LM_INDEX_MCP), (LM_INDEX_MCP, LM_INDEX_PIP),
    (LM_INDEX_PIP, LM_INDEX_DIP), (LM_INDEX_DIP, LM_INDEX_TIP),
    # Middle
    (LM_WRIST, LM_MIDDLE_MCP), (LM_MIDDLE_MCP, LM_MIDDLE_PIP),
    (LM_MIDDLE_PIP, LM_MIDDLE_DIP), (LM_MIDDLE_DIP, LM_MIDDLE_TIP),
    # Ring
    (LM_WRIST, LM_RING_MCP), (LM_RING_MCP, LM_RING_PIP),
    (LM_RING_PIP, LM_RING_DIP), (LM_RING_DIP, LM_RING_TIP),
    # Pinky
    (LM_WRIST, LM_PINKY_MCP), (LM_PINKY_MCP, LM_PINKY_PIP),
    (LM_PINKY_PIP, LM_PINKY_DIP), (LM_PINKY_DIP, LM_PINKY_TIP),
    # Palm cross-connections
    (LM_INDEX_MCP, LM_MIDDLE_MCP), (LM_MIDDLE_MCP, LM_RING_MCP),
    (LM_RING_MCP, LM_PINKY_MCP),
]

# ── Gesture timing (seconds) — physical limits, not preferences ────────────
CLICK_MIN_HOLD_S       = 0.06   # below this = accidental contact, ignore
CLICK_MAX_HOLD_S       = 1.4    # above this = drag intent, not click
DRAG_LOCK_S            = 0.45   # pinch held this long → enter drag mode
RIGHT_CLICK_DEBOUNCE_S = 0.9    # min gap between right-clicks (prevent spam)
GESTURE_DEBOUNCE_S     = 0.08   # min time a new gesture must hold before accepted

# ── Geometry ratios (relative to hand size — scale-invariant) ──────────────
# "Hand size" = wrist → middle_mcp distance in pixels
PINCH_RATIO          = 0.28   # thumb-index distance / hand_size → pinching
FINGER_EXTEND_RATIO  = 0.12   # tip must be this far above MCP (ratio) to count as extended
THUMB_EXTEND_RATIO   = 0.55   # thumb tip distance from wrist / hand_size
VICTORY_SPREAD_RATIO = 0.14   # min spread between index & middle tips for victory sign

# ── Kalman filter — process/measurement noise ──────────────────────────────
# Lower process noise  → smoother path but more lag on fast moves
# Lower measure noise  → trusts raw camera more (less smoothing)
KALMAN_PROCESS_NOISE = 0.08
KALMAN_MEASURE_NOISE = 0.08

# ── Coordinate mapping ─────────────────────────────────────────────────────
# Fraction of frame used as active gesture zone (edges are dead zone)
ACTIVE_ZONE_X = (0.10, 0.90)
ACTIVE_ZONE_Y = (0.10, 0.90)

# ── Scroll accumulator ─────────────────────────────────────────────────────
SCROLL_FIRE_THRESHOLD = 3.0   # accumulated delta units before one scroll tick fires

# ── UI / overlay colors (BGR) ──────────────────────────────────────────────
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0)
C_GREEN   = (0,   220, 80)
C_ORANGE  = (0,   165, 255)
C_CYAN    = (255, 220, 0)
C_MAGENTA = (200, 80,  255)
C_RED     = (50,  50,  255)
C_YELLOW  = (0,   210, 255)
C_GRAY    = (120, 120, 120)
C_BLUE    = (255, 100, 60)

# ── Hotkeys ────────────────────────────────────────────────────────────────
KEY_QUIT      = ord('q')
KEY_GUIDE     = ord('t')
KEY_TUTORIAL  = ord('r')
KEY_SKIP      = ord('s')
KEY_CALIBRATE = ord('c')
KEY_SETTINGS  = ord('p')   # p = params
KEY_ESC       = 27