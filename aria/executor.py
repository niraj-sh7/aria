"""Command executor for ARIA — dispatches LLM tool calls to servos.

Bridges the gap between the structured output of ``LLMController``
and the hardware-level ``ServoController``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from aria.servo_controller import (
    FINGER_CHANNELS,
    ServoController,
)

logger = logging.getLogger(__name__)


class CommandExecutor:
    """Dispatch parsed tool calls to the servo controller.

    Parameters
    ----------
    servo : ServoController
        Initialised servo controller instance.
    """

    def __init__(self, servo: ServoController) -> None:
        self.servo = servo

    # ── Public entry point ───────────────────────────────────────────

    def execute(self, tool_calls: list[dict[str, Any]]) -> None:
        """Execute a list of tool calls produced by the LLM.

        Parameters
        ----------
        tool_calls : list[dict[str, Any]]
            Each element has ``"name"`` (str) and ``"arguments"`` (dict).
        """
        for call in tool_calls:
            name: str = call.get("name", "")
            args: dict[str, Any] = call.get("arguments", {})
            self._dispatch(name, args)

    # ── Dispatcher ───────────────────────────────────────────────────

    def _dispatch(self, name: str, args: dict[str, Any]) -> None:
        """Route a single tool call to the appropriate handler.

        Parameters
        ----------
        name : str
            Tool / function name.
        args : dict[str, Any]
            Keyword arguments for the tool.
        """
        handler = {
            "move_finger": self._move_finger,
            "open_hand": self._open_hand,
            "close_hand": self._close_hand,
            "pinch": self._pinch,
            "point": self._point,
            "peace_sign": self._peace_sign,
            "wave": self._wave,
            "rotate_wrist": self._rotate_wrist,
        }.get(name)

        if handler is None:
            self._log("⚠️  Unknown tool", name)
            logger.warning("Unknown tool call: %s(%s)", name, args)
            return

        self._log("▶ Executing", f"{name}({args})")
        handler(**args)

    # ── Tool implementations ─────────────────────────────────────────

    def _move_finger(self, finger: str, position: float) -> None:
        """Move a named finger to *position*.

        Parameters
        ----------
        finger : str
            One of ``thumb``, ``index``, ``middle``, ``ring``, ``pinky``.
        position : float
            0.0 (open) → 1.0 (closed).
        """
        channel = FINGER_CHANNELS.get(finger.lower())
        if channel is None:
            logger.warning("Unknown finger name: %r", finger)
            return
        self.servo.set_finger(channel, float(position))

    def _open_hand(self) -> None:
        """Fully open all fingers."""
        self.servo.open_all()

    def _close_hand(self) -> None:
        """Fully close all fingers (fist)."""
        self.servo.close_all()

    def _pinch(self, strength: float = 0.8) -> None:
        """Pinch gesture — close thumb + index, open others.

        Parameters
        ----------
        strength : float
            Pinch closure level (0.0 — light, 1.0 — firm).
        """
        self.servo.set_finger(FINGER_CHANNELS["thumb"], float(strength))
        self.servo.set_finger(FINGER_CHANNELS["index"], float(strength))
        self.servo.open_finger(FINGER_CHANNELS["middle"])
        self.servo.open_finger(FINGER_CHANNELS["ring"])
        self.servo.open_finger(FINGER_CHANNELS["pinky"])

    def _point(self) -> None:
        """Point gesture — extend index, close others."""
        self.servo.open_finger(FINGER_CHANNELS["index"])
        self.servo.close_finger(FINGER_CHANNELS["thumb"])
        self.servo.close_finger(FINGER_CHANNELS["middle"])
        self.servo.close_finger(FINGER_CHANNELS["ring"])
        self.servo.close_finger(FINGER_CHANNELS["pinky"])

    def _peace_sign(self) -> None:
        """Peace / victory gesture — extend index + middle, close others."""
        self.servo.open_finger(FINGER_CHANNELS["index"])
        self.servo.open_finger(FINGER_CHANNELS["middle"])
        self.servo.close_finger(FINGER_CHANNELS["thumb"])
        self.servo.close_finger(FINGER_CHANNELS["ring"])
        self.servo.close_finger(FINGER_CHANNELS["pinky"])

    def _wave(self, repetitions: int = 3) -> None:
        """Wave — open and close the hand *repetitions* times.

        Parameters
        ----------
        repetitions : int
            Number of open/close cycles. Defaults to ``3``.
        """
        for i in range(int(repetitions)):
            self._log("  👋 Wave", f"{i + 1}/{repetitions}")
            self.servo.open_all()
            time.sleep(0.4)
            self.servo.close_all()
            time.sleep(0.4)
        # End open
        self.servo.open_all()

    def _rotate_wrist(self, angle: float = 0.0) -> None:
        """Rotate the wrist to *angle* degrees.

        Parameters
        ----------
        angle : float
            Target angle (−90° to +90°).
        """
        self.servo.set_wrist(float(angle))

    # ── Logging helper ───────────────────────────────────────────────

    @staticmethod
    def _log(prefix: str, message: str) -> None:
        """Print a timestamped, user-facing log line."""
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {prefix}: {message}")
