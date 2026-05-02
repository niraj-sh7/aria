"""Command executor for ARIA — dispatches LLM tool calls to servos.

Bridges the gap between the structured output of ``LLMController``
and the hardware-level ``ServoController``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from aria.servo_controller import (
    FINGER_CHANNELS,
    ServoController,
)

if TYPE_CHECKING:
    from aria.llm_controller import LLMController
    from aria.vision import VisionController

logger = logging.getLogger(__name__)

# Wrist rotation angles for directional reach
_REACH_ANGLES: dict[str, float] = {
    "left": -45.0,
    "right": 45.0,
    "center": 0.0,
    "up": 0.0,
    "down": 0.0,
}


class CommandExecutor:
    """Dispatch parsed tool calls to the servo controller.

    Parameters
    ----------
    servo : ServoController
        Initialised servo controller instance.
    llm : LLMController, optional
        LLM controller used by vision tool calls that need inference
        (``describe_scene``, ``track_and_grab``).
    vision : VisionController, optional
        Vision controller used by ``track_and_grab``.
    """

    def __init__(
        self,
        servo: ServoController,
        llm: Optional["LLMController"] = None,
        vision: Optional["VisionController"] = None,
    ) -> None:
        self.servo = servo
        self.llm = llm
        self.vision = vision

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
            # Vision tools
            "grab_object": self._grab_object,
            "reach_toward": self._reach_toward,
            "describe_scene": self._describe_scene,
            "track_and_grab": self._track_and_grab,
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

    # ── Vision tool handlers ───────────────────────────────────────────────

    def _grab_object(self, object_name: str = "object") -> None:
        """Close the hand firmly to grab a named object.

        Parameters
        ----------
        object_name : str
            Human-readable name of the target (used in log output only).
        """
        self._log("✊ Grabbing", object_name)
        self.servo.close_all()

    def _reach_toward(
        self,
        direction: str = "center",
        distance: str = "medium",
    ) -> None:
        """Rotate the wrist toward a target direction and open the hand to reach.

        Parameters
        ----------
        direction : str
            One of ``left``, ``right``, ``center``, ``up``, ``down``.
        distance : str
            One of ``near``, ``medium``, ``far`` (informational, logged only).
        """
        angle = _REACH_ANGLES.get(direction.lower(), 0.0)
        self._log("👉 Reaching", f"direction={direction}, distance={distance}, wrist={angle}°")
        self.servo.set_wrist(angle)
        self.servo.open_all()

    def _describe_scene(self) -> None:
        """Ask Gemma 4 to narrate the current camera frame.

        Requires both ``self.llm`` and ``self.vision`` to be set.
        Prints the description to the console; does not move any servos.
        """
        if self.llm is None or self.vision is None:
            print("  ⚠️  describe_scene requires --vision flag (camera + LLM not attached).")
            return
        if not self.vision.is_available:
            print("  ⚠️  Camera unavailable — cannot describe scene.")
            return
        try:
            image_b64 = self.vision.capture_frame_base64()
            prompt = (
                "Look at this image carefully. "
                "Describe all visible objects, their positions, colours, and "
                "whether any of them look graspable by a robotic hand."
            )
            description = self.llm.ask_vision(prompt, image_b64)
            print(f"\n  👁️  Scene: {description}\n")
        except Exception as exc:
            logger.error("describe_scene failed: %s", exc)
            print(f"  ❌  Could not describe scene: {exc}")

    def _track_and_grab(self, object_name: str = "object") -> None:
        """Capture up to 3 frames; grab on the first that shows the object centred.

        Sends each frame to Gemma 4 with a yes/no centering question.
        Falls back to an immediate grab if vision is unavailable.

        Parameters
        ----------
        object_name : str
            The object to track.
        """
        if self.llm is None or self.vision is None or not self.vision.is_available:
            self._log("🎯 Track+Grab (no vision)", object_name)
            self.servo.close_all()
            return

        self._log("🎯 Tracking", object_name)
        grabbed = False
        for attempt in range(1, 4):
            try:
                image_b64 = self.vision.capture_frame_base64()
                prompt = (
                    f"Is the '{object_name}' roughly centered in this image? "
                    "Answer only 'yes' or 'no'."
                )
                answer = self.llm.ask_vision(prompt, image_b64).lower()
                self._log(f"  Frame {attempt}/3", f"centred? {answer}")
                if "yes" in answer:
                    self._log("  ✔️ Centred — grabbing", object_name)
                    self.servo.close_all()
                    grabbed = True
                    break
                time.sleep(0.5)
            except Exception as exc:
                logger.warning("track_and_grab frame %d failed: %s", attempt, exc)

        if not grabbed:
            self._log("  ⚠️ Not centred after 3 frames — grabbing anyway", object_name)
            self.servo.close_all()

    # ── Logging helper ──────────────────────────────────────────────────────

    @staticmethod
    def _log(prefix: str, message: str) -> None:
        """Print a timestamped, user-facing log line."""
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {prefix}: {message}")
