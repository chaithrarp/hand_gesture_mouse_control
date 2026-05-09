"""
control/event_manager.py

Thread-safe event queue between gesture processing and mouse execution.

Why this exists:
  Gesture recognition and mouse control run on the same thread (frame loop),
  but we still need an explicit queue to:
    1. Prevent re-entrancy: gesture handlers enqueue, flush() drains at
       end of frame — never mid-recognition
    2. Priority: RELEASE > CLICK > MOVE so a drag-release always fires
       before the next move, even if two events land in one frame
    3. Deduplication: coalesce multiple MOVE events in one frame into one
    4. Future-proof for threading: move flush() to a worker thread with
       zero API changes

Event types and priority (lower number = higher priority):
  MOUSE_UP     = 0   — always first (prevents stuck button)
  CLICK        = 1   — left / right click
  MOUSE_DOWN   = 2   — drag start
  SCROLL       = 3   — scroll ticks
  MOVE         = 4   — cursor movement (lowest — coalesced)

Usage:
    em = EventManager()
    em.enqueue_move(sx, sy)
    em.enqueue_click()
    em.flush()          # called once at end of each frame

Latency notes:
  - enqueue() is a list.append() — O(1), no locks needed (single thread)
  - flush() sorts a list of max ~5 events per frame — negligible
  - MOVE coalescing: only the last MOVE per flush cycle is kept
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ── Event types ───────────────────────────────────────────────────────────────
class EventType(IntEnum):
    MOUSE_UP   = 0   # release mouse button (highest priority)
    CLICK      = 1   # left click
    RCLICK     = 1   # right click (same priority as click)
    MOUSE_DOWN = 2   # press mouse button (drag start)
    SCROLL     = 3   # scroll ticks
    MOVE       = 4   # cursor move (lowest — coalesced)


@dataclass
class MouseEvent:
    etype:    EventType
    x:        Optional[int]   = None   # screen x (MOVE only)
    y:        Optional[int]   = None   # screen y (MOVE only)
    ticks:    Optional[int]   = None   # scroll ticks (SCROLL only)
    button:   str             = "left" # "left" | "right"
    # Priority for sorting — matches EventType value
    priority: int             = field(init=False)

    def __post_init__(self):
        self.priority = int(self.etype)


# ── EventManager ─────────────────────────────────────────────────────────────
class EventManager:
    """
    Collects mouse events during a frame, then executes them in
    priority order at the end of the frame via flush().

    Handlers are injected at construction (dependency injection)
    so this class has zero direct pyautogui dependency — fully testable.

    Args:
        on_move(x, y)       — called for cursor movement
        on_click(button)    — called for click ("left" | "right")
        on_mouse_down()     — called to start drag
        on_mouse_up()       — called to end drag
        on_scroll(ticks)    — called for scroll (positive=up, negative=down)
    """

    def __init__(
        self,
        on_move:       Callable[[int, int], None],
        on_click:      Callable[[str], None],
        on_mouse_down: Callable[[], None],
        on_mouse_up:   Callable[[], None],
        on_scroll:     Callable[[int], None],
    ):
        self._on_move       = on_move
        self._on_click      = on_click
        self._on_mouse_down = on_mouse_down
        self._on_mouse_up   = on_mouse_up
        self._on_scroll     = on_scroll

        self._queue: List[MouseEvent] = []
        self._total_flushed = 0

    # ── Enqueue helpers ───────────────────────────────────────────────────────

    def enqueue_move(self, x: int, y: int) -> None:
        """Enqueue a cursor move. Multiple moves per frame are coalesced."""
        self._queue.append(MouseEvent(EventType.MOVE, x=x, y=y))

    def enqueue_click(self, button: str = "left") -> None:
        etype = EventType.CLICK if button == "left" else EventType.RCLICK
        self._queue.append(MouseEvent(etype, button=button))
        log.debug(f"Queued {button} click")

    def enqueue_mouse_down(self) -> None:
        self._queue.append(MouseEvent(EventType.MOUSE_DOWN))
        log.debug("Queued mouseDown")

    def enqueue_mouse_up(self) -> None:
        self._queue.append(MouseEvent(EventType.MOUSE_UP))
        log.debug("Queued mouseUp")

    def enqueue_scroll(self, ticks: int) -> None:
        if ticks == 0:
            return
        self._queue.append(MouseEvent(EventType.SCROLL, ticks=ticks))

    # ── Flush ─────────────────────────────────────────────────────────────────

    def flush(self) -> int:
        """
        Execute all queued events in priority order.
        MOVE events are coalesced — only the last one fires.
        Returns number of events executed.

        Call exactly once at the end of each frame loop iteration.
        """
        if not self._queue:
            return 0

        # ── Coalesce MOVE events — keep only the last ──────────────────────
        moves = [e for e in self._queue if e.etype == EventType.MOVE]
        non_moves = [e for e in self._queue if e.etype != EventType.MOVE]

        events: List[MouseEvent] = non_moves
        if moves:
            events.append(moves[-1])   # only last move per frame

        # ── Sort by priority (stable — preserves enqueue order within tier) ──
        events.sort(key=lambda e: e.priority)

        # ── Execute ────────────────────────────────────────────────────────
        executed = 0
        for ev in events:
            try:
                if ev.etype == EventType.MOUSE_UP:
                    self._on_mouse_up()
                elif ev.etype in (EventType.CLICK, EventType.RCLICK):
                    self._on_click(ev.button)
                elif ev.etype == EventType.MOUSE_DOWN:
                    self._on_mouse_down()
                elif ev.etype == EventType.SCROLL:
                    self._on_scroll(ev.ticks)
                elif ev.etype == EventType.MOVE:
                    self._on_move(ev.x, ev.y)
                executed += 1
            except Exception as exc:
                # Never let a mouse call crash the frame loop
                log.warning(f"EventManager: handler raised {exc!r}")

        self._queue.clear()
        self._total_flushed += executed
        return executed

    # ── State ─────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Drop all queued events without executing. Call on pause/reset."""
        dropped = len(self._queue)
        self._queue.clear()
        if dropped:
            log.debug(f"EventManager: cleared {dropped} pending events")

    @property
    def pending(self) -> int:
        """Number of events waiting to be flushed."""
        return len(self._queue)

    @property
    def total_flushed(self) -> int:
        """Lifetime count of executed events — for metrics."""
        return self._total_flushed