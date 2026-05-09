"""
core/gesture_state.py

Higher-level gesture state machine that sits ABOVE GestureStateMachine.

Responsibilities:
  - Tracks gesture sequences across frames (e.g. double-pinch, swipe)
  - Manages per-hand application state (paused, dragging, scrolling)
  - Emits high-level ActionEvents that mouse_controller consumes directly
  - Decouples gesture recognition from mouse action logic

Layer diagram:
    MediaPipe landmarks
          ↓
    GestureClassifier          (stateless geometry)
          ↓
    GestureStateMachine        (debounce + ENTERED/HELD/EXITED events)
          ↓
    HandGestureState  ←── THIS FILE
          ↓
    ActionEvent → mouse_controller / event_manager

ActionEvent types:
    CURSOR_MOVE      — move cursor to position
    LEFT_CLICK       — fire left click
    RIGHT_CLICK      — fire right click
    DRAG_START       — begin drag (mouseDown)
    DRAG_MOVE        — move while dragging
    DRAG_END         — end drag (mouseUp)
    SCROLL           — scroll by delta
    PAUSE_TOGGLE     — toggle pause state
    DOUBLE_PINCH     — double-pinch detected (configurable action)
    SWIPE_LEFT/RIGHT/UP/DOWN  — directional swipe detected
    NONE             — no action this frame

Design rules:
  - One HandGestureState instance per tracked hand
  - All timing via time.monotonic() — no frame counting
  - Sequence detection (double-pinch, swipe) is non-blocking:
    if sequence times out, falls back to single gesture behaviour
  - ActionEvent carries everything needed — no shared mutable state
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from core.gesture_recognizer import (
    GestureEvent,
    GestureResult,
    GestureStateMachine,
    GestureType,
)
from config.settings import GestureSettings
from utils.logger import get_logger

log = get_logger(__name__)


# ── Action event types ────────────────────────────────────────────────────────
class ActionType(Enum):
    NONE         = auto()
    CURSOR_MOVE  = auto()
    LEFT_CLICK   = auto()
    RIGHT_CLICK  = auto()
    DRAG_START   = auto()
    DRAG_MOVE    = auto()
    DRAG_END     = auto()
    SCROLL       = auto()
    PAUSE_TOGGLE = auto()
    DOUBLE_PINCH = auto()
    SWIPE_LEFT   = auto()
    SWIPE_RIGHT  = auto()
    SWIPE_UP     = auto()
    SWIPE_DOWN   = auto()


@dataclass(frozen=True)
class ActionEvent:
    action:    ActionType
    position:  Optional[Tuple[int, int]] = None   # screen-space position hint
    delta:     Optional[Tuple[float, float]] = None  # for SCROLL / SWIPE
    hand:      str = "Right"                       # which hand triggered this
    hold_s:    float = 0.0                         # how long gesture was held


# ── Internal states ───────────────────────────────────────────────────────────
class _DragState(Enum):
    IDLE     = auto()
    PENDING  = auto()   # pinch held, waiting for drag_lock_s
    ACTIVE   = auto()   # drag in progress


class _PinchSeqState(Enum):
    IDLE        = auto()
    FIRST_UP    = auto()   # first pinch released, watching for second


# ── Swipe tracker ─────────────────────────────────────────────────────────────
class _SwipeTracker:
    """
    Detects directional swipes from OPEN_HAND gesture.

    A swipe is: hand moves > swipe_min_px in one axis
    in < swipe_max_s seconds with velocity > swipe_min_vel.

    Only fires once per open-hand gesture entry — not continuously.
    """

    SWIPE_MIN_PX  = 55    # minimum displacement to count as swipe
    SWIPE_MAX_S   = 0.55  # swipe must complete within this window
    SWIPE_MIN_VEL = 90    # minimum px/s average velocity

    def __init__(self):
        self._start_pos:  Optional[Tuple[int, int]] = None
        self._start_time: Optional[float]           = None
        self._fired:      bool                      = False

    def begin(self, pos: Tuple[int, int]) -> None:
        """Call on OPEN_HAND ENTERED."""
        self._start_pos  = pos
        self._start_time = time.monotonic()
        self._fired      = False

    def update(self, pos: Tuple[int, int]) -> Optional[ActionType]:
        """
        Call each frame while OPEN_HAND is HELD.
        Returns swipe ActionType or None.
        Once a swipe fires, returns None for the rest of the gesture.
        """
        if self._fired or self._start_pos is None:
            return None

        now     = time.monotonic()
        elapsed = now - self._start_time

        if elapsed > self.SWIPE_MAX_S:
            # Time window expired — not a swipe
            return None

        dx = pos[0] - self._start_pos[0]
        dy = pos[1] - self._start_pos[1]
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < self.SWIPE_MIN_PX:
            return None

        vel = dist / max(elapsed, 1e-6)
        if vel < self.SWIPE_MIN_VEL:
            return None

        # Determine primary axis
        if abs(dx) >= abs(dy):
            action = ActionType.SWIPE_RIGHT if dx > 0 else ActionType.SWIPE_LEFT
        else:
            action = ActionType.SWIPE_DOWN if dy > 0 else ActionType.SWIPE_UP

        self._fired = True
        log.debug(f"Swipe: {action.name}  dx={dx:.0f} dy={dy:.0f} vel={vel:.0f}px/s")
        return action

    def reset(self) -> None:
        self._start_pos  = None
        self._start_time = None
        self._fired      = False


# ── Per-hand gesture state ────────────────────────────────────────────────────
class HandGestureState:
    """
    Manages the full gesture→action pipeline for ONE hand.

    Instantiate one per tracked hand.  Feed it GestureResult each frame
    (from GestureStateMachine.update()), get back an ActionEvent.

    Usage:
        state = HandGestureState(settings.gesture, hand_label="Right")
        result: GestureResult = gesture_sm.update(lm, w, h)
        action: ActionEvent   = state.process(result)
        # hand action to event_manager
    """

    # Double-pinch: second pinch must start within this many seconds
    # of the first pinch being released
    DOUBLE_PINCH_WINDOW_S = 0.35

    def __init__(self, gs: GestureSettings, hand_label: str = "Right"):
        self._gs    = gs
        self._label = hand_label

        # Drag state machine
        self._drag:      _DragState     = _DragState.IDLE
        self._drag_enter_t: float       = 0.0

        # Double-pinch sequence
        self._pinch_seq:    _PinchSeqState = _PinchSeqState.IDLE
        self._first_up_t:   float          = 0.0

        # Scroll reference
        self._scroll_ref_y: Optional[int]  = None
        self._scroll_accum: float          = 0.0

        # Pause state (toggled by FIST)
        self._paused: bool = False

        # Swipe detector (for OPEN_HAND)
        self._swipe = _SwipeTracker()

        # Previous gesture for transition detection
        self._prev_gesture: GestureType = GestureType.NONE

    # ── Public ────────────────────────────────────────────────────────────────

    def reconfigure(self, gs: GestureSettings) -> None:
        self._gs = gs

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_dragging(self) -> bool:
        return self._drag == _DragState.ACTIVE

    def process(self, result: GestureResult) -> ActionEvent:
        """
        Main entry point. Call once per frame per hand.

        Args:
            result: GestureResult from GestureStateMachine.update()

        Returns:
            ActionEvent describing what mouse action to take this frame.
        """
        g     = result.gesture
        event = result.event
        pos   = result.position

        action = self._dispatch(g, event, result)
        self._prev_gesture = g
        return action

    def reset(self) -> None:
        """Call when hand is lost. Cleans up any in-progress state."""
        was_dragging = self._drag == _DragState.ACTIVE
        self._drag       = _DragState.IDLE
        self._drag_enter_t = 0.0
        self._pinch_seq  = _PinchSeqState.IDLE
        self._scroll_ref_y = None
        self._scroll_accum = 0.0
        self._swipe.reset()
        log.debug(f"HandGestureState reset ({self._label}), was_dragging={was_dragging}")

        if was_dragging:
            # Signal caller to fire mouseUp
            return ActionEvent(ActionType.DRAG_END, hand=self._label)
        return ActionEvent(ActionType.NONE, hand=self._label)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(
        self,
        g:      GestureType,
        event:  GestureEvent,
        result: GestureResult,
    ) -> ActionEvent:
        pos = result.position

        # ── FIST — pause toggle ───────────────────────────────────────────
        if g == GestureType.FIST:
            if event == GestureEvent.ENTERED:
                self._paused = not self._paused
                # Release any in-progress drag on pause
                if self._paused and self._drag == _DragState.ACTIVE:
                    self._drag = _DragState.IDLE
                    return ActionEvent(ActionType.DRAG_END, hand=self._label)
                log.info(f"{self._label} hand: paused={self._paused}")
                return ActionEvent(ActionType.PAUSE_TOGGLE, hand=self._label)
            return ActionEvent(ActionType.NONE, hand=self._label)

        if self._paused:
            return ActionEvent(ActionType.NONE, hand=self._label)

        # ── PINCH — click / drag / double-pinch ───────────────────────────
        if g == GestureType.PINCH:
            return self._handle_pinch(event, result)

        # If we were pinching and now we're not — handle release
        if self._prev_gesture == GestureType.PINCH and g != GestureType.PINCH:
            return self._handle_pinch_release(result)

        # ── POINT — cursor move ───────────────────────────────────────────
        if g == GestureType.POINT:
            self._reset_scroll()
            return ActionEvent(
                ActionType.CURSOR_MOVE,
                position=pos,
                hand=self._label,
                hold_s=result.hold_s,
            )

        # ── VICTORY — right click (on entry only) ────────────────────────
        if g == GestureType.VICTORY:
            self._reset_scroll()
            if event == GestureEvent.ENTERED:
                log.debug(f"{self._label}: right click")
                return ActionEvent(
                    ActionType.RIGHT_CLICK,
                    position=pos,
                    hand=self._label,
                )
            return ActionEvent(ActionType.NONE, hand=self._label)

        # ── OPEN_HAND — scroll or swipe ───────────────────────────────────
        if g == GestureType.OPEN_HAND:
            return self._handle_open_hand(event, result)

        # ── NONE — clean up ───────────────────────────────────────────────
        self._reset_scroll()
        return ActionEvent(ActionType.NONE, hand=self._label)

    # ── Pinch handling ────────────────────────────────────────────────────────

    def _handle_pinch(self, event: GestureEvent, result: GestureResult) -> ActionEvent:
        gs  = self._gs
        pos = result.position
        now = time.monotonic()

        if event == GestureEvent.ENTERED:
            # Check for double-pinch
            if (self._pinch_seq == _PinchSeqState.FIRST_UP and
                    now - self._first_up_t <= self.DOUBLE_PINCH_WINDOW_S):
                self._pinch_seq = _PinchSeqState.IDLE
                log.debug(f"{self._label}: double-pinch")
                return ActionEvent(
                    ActionType.DOUBLE_PINCH,
                    position=pos,
                    hand=self._label,
                )

            # Start tracking this pinch for drag
            self._drag       = _DragState.PENDING
            self._drag_enter_t = now
            self._pinch_seq  = _PinchSeqState.IDLE
            return ActionEvent(ActionType.NONE, hand=self._label)

        if event == GestureEvent.HELD:
            held = now - self._drag_enter_t

            if self._drag == _DragState.PENDING:
                if held >= gs.drag_lock_s:
                    # Promote to active drag
                    self._drag = _DragState.ACTIVE
                    log.debug(f"{self._label}: drag start (held {held:.2f}s)")
                    return ActionEvent(
                        ActionType.DRAG_START,
                        position=pos,
                        hand=self._label,
                        hold_s=held,
                    )
                # Still in pending — no action yet (waiting for drag threshold)
                return ActionEvent(ActionType.NONE, hand=self._label)

            if self._drag == _DragState.ACTIVE:
                return ActionEvent(
                    ActionType.DRAG_MOVE,
                    position=pos,
                    hand=self._label,
                    hold_s=held,
                )

        return ActionEvent(ActionType.NONE, hand=self._label)

    def _handle_pinch_release(self, result: GestureResult) -> ActionEvent:
        """Called the frame pinch exits. Decides click vs drag-end."""
        gs  = self._gs
        pos = result.position
        now = time.monotonic()

        if self._drag == _DragState.ACTIVE:
            self._drag = _DragState.IDLE
            log.debug(f"{self._label}: drag end")
            return ActionEvent(
                ActionType.DRAG_END,
                position=pos,
                hand=self._label,
            )

        if self._drag == _DragState.PENDING:
            held = now - self._drag_enter_t
            self._drag = _DragState.IDLE

            if held >= gs.click_min_hold_s:
                # Start double-pinch watch window
                self._pinch_seq = _PinchSeqState.FIRST_UP
                self._first_up_t = now
                log.debug(f"{self._label}: left click (held {held:.3f}s)")
                return ActionEvent(
                    ActionType.LEFT_CLICK,
                    position=pos,
                    hand=self._label,
                    hold_s=held,
                )

        return ActionEvent(ActionType.NONE, hand=self._label)

    # ── Open hand / scroll / swipe ────────────────────────────────────────────

    def _handle_open_hand(
        self, event: GestureEvent, result: GestureResult
    ) -> ActionEvent:
        gs  = self._gs
        pos = result.position
        py  = pos[1]

        if event == GestureEvent.ENTERED:
            self._scroll_ref_y = py
            self._scroll_accum = 0.0
            self._swipe.begin(pos)
            return ActionEvent(ActionType.NONE, hand=self._label)

        # Check for swipe first (takes priority over scroll)
        swipe_action = self._swipe.update(pos)
        if swipe_action is not None:
            self._reset_scroll()
            log.debug(f"{self._label}: {swipe_action.name}")
            return ActionEvent(
                swipe_action,
                position=pos,
                delta=(
                    float(pos[0] - (self._swipe._start_pos[0] if self._swipe._start_pos else pos[0])),
                    float(pos[1] - (self._swipe._start_pos[1] if self._swipe._start_pos else pos[1])),
                ),
                hand=self._label,
            )

        # Scroll accumulation
        if self._scroll_ref_y is None:
            self._scroll_ref_y = py

        raw_delta = py - self._scroll_ref_y
        ss = gs  # scroll params live in GestureSettings adjacently;
                  # pixels_per_tick sourced from settings.scroll in main

        # Accumulate with smoothing (prevents single-frame spike from firing)
        self._scroll_accum = self._scroll_accum * 0.82 + raw_delta * 0.18

        # Fire if accumulated enough (threshold set by ScrollSettings.pixels_per_tick
        # but we use a local constant here so gesture_state has no scroll dep)
        SCROLL_THRESHOLD = 20.0   # px — caller overrides via _set_scroll_threshold

        if abs(self._scroll_accum) >= self._scroll_threshold:
            ticks = int(self._scroll_accum / self._scroll_threshold)
            self._scroll_accum -= ticks * self._scroll_threshold
            log.debug(f"{self._label}: scroll ticks={ticks}")
            return ActionEvent(
                ActionType.SCROLL,
                position=pos,
                delta=(0.0, float(ticks)),
                hand=self._label,
            )

        return ActionEvent(ActionType.NONE, hand=self._label)

    def set_scroll_threshold(self, px: float) -> None:
        """
        Called by main/mouse_controller to sync scroll sensitivity
        from ScrollSettings.pixels_per_tick without creating a dependency.
        """
        self._scroll_threshold = max(5.0, px)

    def _reset_scroll(self) -> None:
        self._scroll_ref_y = None
        self._scroll_accum = 0.0

    # ── Init scroll threshold ─────────────────────────────────────────────────

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    # Override __init__ to set default scroll threshold
    _scroll_threshold: float = 20.0


# Fix: set _scroll_threshold as instance attr in __init__
_original_init = HandGestureState.__init__

def _patched_init(self, gs, hand_label="Right"):
    _original_init(self, gs, hand_label)
    self._scroll_threshold = 20.0

HandGestureState.__init__ = _patched_init


# ── Multi-hand coordinator ────────────────────────────────────────────────────
class GestureStateManager:
    """
    Manages HandGestureState instances for up to 2 hands.

    Routing rules (applied each frame):
      - RIGHT hand: cursor move, click, drag (primary control)
      - LEFT hand:  scroll, swipe (secondary control)
      - If only one hand present: that hand does everything
      - FIST on either hand toggles pause globally

    Usage:
        manager = GestureStateManager(settings.gesture)

        # Each frame, after gesture recognition:
        actions = manager.update(gesture_results)
        # gesture_results: dict {"Left": GestureResult, "Right": GestureResult}
        # actions: list[ActionEvent]

        for action in actions:
            event_manager.dispatch(action)
    """

    def __init__(self, gs: GestureSettings):
        self._gs = gs
        self._hands: dict[str, HandGestureState] = {}

    def reconfigure(self, gs: GestureSettings) -> None:
        self._gs = gs
        for state in self._hands.values():
            state.reconfigure(gs)

    def set_scroll_threshold(self, px: float) -> None:
        for state in self._hands.values():
            state.set_scroll_threshold(px)

    def update(
        self,
        gesture_results: dict,  # {"Left": GestureResult | None, "Right": GestureResult | None}
    ) -> list:
        """
        Process gesture results for all hands.
        Returns list[ActionEvent] (may be empty, may have 1-2 entries).
        """
        actions = []

        present_labels = {k for k, v in gesture_results.items() if v is not None}

        # Retire states for hands that disappeared
        for label in list(self._hands.keys()):
            if label not in present_labels:
                leftover = self._hands[label].reset()
                if leftover and leftover.action != ActionType.NONE:
                    actions.append(leftover)
                del self._hands[label]
                log.debug(f"Hand lost: {label}")

        # Process each present hand
        for label, result in gesture_results.items():
            if result is None:
                continue

            if label not in self._hands:
                self._hands[label] = HandGestureState(self._gs, label)
                log.debug(f"Hand gained: {label}")

            state  = self._hands[label]
            action = state.process(result)

            # ── Routing: restrict actions by hand role ─────────────────
            if len(present_labels) == 2:
                action = self._route_dual_hand(label, action)

            if action.action != ActionType.NONE:
                actions.append(action)

        return actions

    def _route_dual_hand(self, label: str, action: ActionEvent) -> ActionEvent:
        """
        When both hands are present, enforce role restrictions:
          Right → cursor / click / drag / pause
          Left  → scroll / swipe / pause
        """
        right_only = {
            ActionType.CURSOR_MOVE,
            ActionType.LEFT_CLICK,
            ActionType.RIGHT_CLICK,
            ActionType.DRAG_START,
            ActionType.DRAG_MOVE,
            ActionType.DRAG_END,
            ActionType.DOUBLE_PINCH,
        }
        left_only = {
            ActionType.SCROLL,
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
        }
        shared = {ActionType.PAUSE_TOGGLE, ActionType.NONE}

        if label == "Right" and action.action in left_only:
            return ActionEvent(ActionType.NONE, hand=label)
        if label == "Left" and action.action in right_only:
            return ActionEvent(ActionType.NONE, hand=label)
        return action

    @property
    def any_paused(self) -> bool:
        return any(s.is_paused for s in self._hands.values())

    @property
    def any_dragging(self) -> bool:
        return any(s.is_dragging for s in self._hands.values())