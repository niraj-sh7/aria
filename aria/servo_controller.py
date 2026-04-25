"""Servo controller for the ARIA robotic hand.

Manages a PCA9685 16-channel PWM servo controller over I2C to drive
six MG996R servos (five fingers + wrist rotation).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Finger channel mapping ──────────────────────────────────────────
CHANNEL_THUMB: int = 0
CHANNEL_INDEX: int = 1
CHANNEL_MIDDLE: int = 2
CHANNEL_RING: int = 3
CHANNEL_PINKY: int = 4
CHANNEL_WRIST: int = 5

FINGER_CHANNELS: dict[str, int] = {
    "thumb": CHANNEL_THUMB,
    "index": CHANNEL_INDEX,
    "middle": CHANNEL_MIDDLE,
    "ring": CHANNEL_RING,
    "pinky": CHANNEL_PINKY,
}

ALL_FINGER_CHANNELS: list[int] = [
    CHANNEL_THUMB,
    CHANNEL_INDEX,
    CHANNEL_MIDDLE,
    CHANNEL_RING,
    CHANNEL_PINKY,
]

# ── Pulse tick limits (PCA9685 12-bit, 4096 steps) ──────────────────
PULSE_OPEN: int = 150      # Fully open position
PULSE_CLOSED: int = 600    # Fully closed position
PULSE_NEUTRAL: int = 375   # Neutral / safe resting position

# Wrist angle range mapped onto the same pulse window
WRIST_ANGLE_MIN: float = -90.0
WRIST_ANGLE_MAX: float = 90.0


class ServoError(Exception):
    """Raised when a servo or I2C communication error occurs."""


class ServoController:
    """High-level driver for the ARIA robotic hand servos.

    Wraps the Adafruit PCA9685 library to expose finger-level and
    wrist-level abstractions.

    Parameters
    ----------
    i2c_address : int
        I2C address of the PCA9685 board (default ``0x40``).
    pwm_frequency : int
        PWM frequency in Hz (default ``50`` for standard servos).
    """

    def __init__(
        self,
        i2c_address: int = 0x40,
        pwm_frequency: int = 50,
    ) -> None:
        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685

            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._pca = PCA9685(self._i2c, address=i2c_address)
            self._pca.frequency = pwm_frequency
            self._simulated = False
            logger.info(
                "PCA9685 initialised at address 0x%02X, frequency %d Hz",
                i2c_address,
                pwm_frequency,
            )
        except Exception as exc:
            # Allow the class to work in simulation mode when the real
            # hardware is not available (e.g. during development / demo).
            logger.warning(
                "Could not initialise PCA9685 hardware (%s). "
                "Running in SIMULATION mode.",
                exc,
            )
            self._i2c = None
            self._pca = None
            self._simulated = True

        self._address = i2c_address
        self._frequency = pwm_frequency

    # ── Low-level helpers ────────────────────────────────────────────

    def _set_pwm(self, channel: int, pulse: int) -> None:
        """Set a raw PWM pulse on *channel*.

        Parameters
        ----------
        channel : int
            PCA9685 channel number (0–15).
        pulse : int
            Duty-cycle tick count (0–4095).

        Raises
        ------
        ServoError
            If the I2C write fails.
        """
        pulse = max(0, min(4095, pulse))
        if self._simulated:
            logger.debug("[SIM] Channel %d → pulse %d", channel, pulse)
            return
        try:
            self._pca.channels[channel].duty_cycle = self._pulse_to_duty(pulse)
        except Exception as exc:
            raise ServoError(
                f"Failed to set channel {channel} to pulse {pulse}: {exc}"
            ) from exc

    @staticmethod
    def _pulse_to_duty(pulse: int) -> int:
        """Convert a 12-bit PCA9685 pulse tick to a 16-bit duty cycle.

        The Adafruit library expects a 16-bit value (0–65535).
        We scale our 12-bit tick (0–4095) accordingly.
        """
        return int(pulse / 4095 * 65535)

    @staticmethod
    def _position_to_pulse(position: float) -> int:
        """Map a normalised finger position [0.0, 1.0] to a pulse tick.

        0.0 → ``PULSE_OPEN``  (150)
        1.0 → ``PULSE_CLOSED`` (600)
        """
        position = max(0.0, min(1.0, position))
        return int(PULSE_OPEN + position * (PULSE_CLOSED - PULSE_OPEN))

    @staticmethod
    def _angle_to_pulse(angle: float) -> int:
        """Map a wrist angle (−90° … +90°) to a pulse tick.

        −90° → ``PULSE_OPEN``  (150)
        +90° → ``PULSE_CLOSED`` (600)
          0° → ``PULSE_NEUTRAL`` (375)
        """
        angle = max(WRIST_ANGLE_MIN, min(WRIST_ANGLE_MAX, angle))
        fraction = (angle - WRIST_ANGLE_MIN) / (WRIST_ANGLE_MAX - WRIST_ANGLE_MIN)
        return int(PULSE_OPEN + fraction * (PULSE_CLOSED - PULSE_OPEN))

    # ── Public API ───────────────────────────────────────────────────

    def set_finger(self, channel: int, position: float) -> None:
        """Move a finger servo to a normalised position.

        Parameters
        ----------
        channel : int
            Servo channel (use the ``CHANNEL_*`` constants).
        position : float
            0.0 (fully open) to 1.0 (fully closed).
        """
        pulse = self._position_to_pulse(position)
        logger.debug(
            "set_finger  channel=%d  position=%.2f  pulse=%d",
            channel, position, pulse,
        )
        self._set_pwm(channel, pulse)

    def open_finger(self, channel: int) -> None:
        """Fully open a single finger.

        Parameters
        ----------
        channel : int
            Servo channel to open.
        """
        self.set_finger(channel, 0.0)

    def close_finger(self, channel: int) -> None:
        """Fully close a single finger.

        Parameters
        ----------
        channel : int
            Servo channel to close.
        """
        self.set_finger(channel, 1.0)

    def open_all(self) -> None:
        """Fully open every finger (but not the wrist)."""
        for ch in ALL_FINGER_CHANNELS:
            self.open_finger(ch)

    def close_all(self) -> None:
        """Fully close every finger (make a fist)."""
        for ch in ALL_FINGER_CHANNELS:
            self.close_finger(ch)

    def set_wrist(self, angle_degrees: float) -> None:
        """Rotate the wrist to *angle_degrees*.

        Parameters
        ----------
        angle_degrees : float
            Target angle in the range −90° to +90°.
        """
        pulse = self._angle_to_pulse(angle_degrees)
        logger.debug(
            "set_wrist  angle=%.1f°  pulse=%d",
            angle_degrees, pulse,
        )
        self._set_pwm(CHANNEL_WRIST, pulse)

    def release(self) -> None:
        """Safe shutdown — move all servos to neutral and de-init."""
        logger.info("Releasing all servos to neutral position.")
        for ch in ALL_FINGER_CHANNELS + [CHANNEL_WRIST]:
            self._set_pwm(ch, PULSE_NEUTRAL)
        if self._pca is not None:
            try:
                self._pca.deinit()
            except Exception:
                pass
        if self._i2c is not None:
            try:
                self._i2c.deinit()
            except Exception:
                pass

    # ── Informational ────────────────────────────────────────────────

    @property
    def is_simulated(self) -> bool:
        """Return ``True`` when running without real hardware."""
        return self._simulated

    def __repr__(self) -> str:
        mode = "SIM" if self._simulated else "HW"
        return (
            f"<ServoController mode={mode} addr=0x{self._address:02X} "
            f"freq={self._frequency}Hz>"
        )
