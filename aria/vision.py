"""Vision controller for ARIA — USB camera capture + base64 encoding.

Provides frame capture and encoding utilities to feed live camera
images into Gemma 4's multimodal vision API via Ollama.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CAMERA_INDEX: int = 0            # Default USB camera device index
JPEG_QUALITY: int = 70           # JPEG compression (0–100); keeps frames <500 KB
CAPTURE_TIMEOUT: float = 2.0     # Seconds to wait for a valid frame


class VisionError(Exception):
    """Raised when the camera cannot be accessed or a frame cannot be read."""


class VisionController:
    """USB camera interface for ARIA's vision system.

    Wraps OpenCV ``VideoCapture`` to provide convenient frame capture,
    base64 encoding, preview, and snapshot utilities.

    Parameters
    ----------
    camera_index : int
        OpenCV device index for the USB camera (default ``0``).
    jpeg_quality : int
        JPEG compression quality used when encoding frames for the LLM.
        Lower values reduce file size at the cost of image quality.
    """

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        jpeg_quality: int = JPEG_QUALITY,
    ) -> None:
        self.camera_index = camera_index
        self.jpeg_quality = jpeg_quality
        self._cap: Optional[cv2.VideoCapture] = None
        self._available: bool = False
        self._open_camera()

    # ── Lifecycle ────────────────────────────────────────────────────

    def _open_camera(self) -> None:
        """Attempt to open the USB camera; sets ``_available`` flag."""
        cap = cv2.VideoCapture(self.camera_index)
        if cap is None or not cap.isOpened():
            logger.warning(
                "Camera index %d not found. Vision features disabled.",
                self.camera_index,
            )
            self._available = False
            return
        self._cap = cap
        self._available = True
        logger.info("Camera opened at index %d.", self.camera_index)

    def _ensure_open(self) -> None:
        """Raise ``VisionError`` if the camera is unavailable."""
        if not self._available or self._cap is None:
            raise VisionError(
                f"Camera (index {self.camera_index}) is not available. "
                "Check that the USB camera is connected."
            )

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._available = False
        logger.info("Camera released.")

    # ── Frame capture ────────────────────────────────────────────────

    def capture_frame(self) -> np.ndarray:
        """Capture a single BGR frame from the camera.

        Returns
        -------
        np.ndarray
            BGR image array with shape ``(H, W, 3)``.

        Raises
        ------
        VisionError
            If the camera is unavailable or the read times out.
        """
        self._ensure_open()
        deadline = time.monotonic() + CAPTURE_TIMEOUT
        while time.monotonic() < deadline:
            ret, frame = self._cap.read()  # type: ignore[union-attr]
            if ret and frame is not None:
                logger.debug(
                    "Frame captured: %dx%d", frame.shape[1], frame.shape[0]
                )
                return frame
            time.sleep(0.05)
        raise VisionError(
            f"Camera read timed out after {CAPTURE_TIMEOUT}s. "
            "Check USB connection."
        )

    def capture_frame_base64(self) -> str:
        """Capture a frame and return it as a base64-encoded JPEG string.

        The JPEG is compressed to stay well under 500 KB so that Ollama
        inference remains fast on the Raspberry Pi 5.

        Returns
        -------
        str
            Base64-encoded JPEG image (UTF-8 string, no data-URI prefix).

        Raises
        ------
        VisionError
            If the camera is unavailable or encoding fails.
        """
        frame = self.capture_frame()
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            raise VisionError("Failed to JPEG-encode the captured frame.")
        encoded = base64.b64encode(buffer).decode("utf-8")
        size_kb = len(encoded) * 3 / 4 / 1024  # approximate decoded bytes
        logger.debug("Encoded frame: %.1f KB (base64)", size_kb)
        return encoded

    # ── Utilities ────────────────────────────────────────────────────

    def show_preview(self, duration_seconds: float = 3.0) -> None:
        """Display a live camera preview in an OpenCV window.

        Parameters
        ----------
        duration_seconds : float
            How long to show the preview (default ``3.0`` seconds).
        """
        self._ensure_open()
        deadline = time.monotonic() + duration_seconds
        print(f"  📷 Showing camera preview for {duration_seconds}s … (press Q to quit early)")
        while time.monotonic() < deadline:
            ret, frame = self._cap.read()  # type: ignore[union-attr]
            if ret and frame is not None:
                cv2.imshow("ARIA Camera Preview", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    def save_snapshot(self, path: str = "snapshot.jpg") -> Path:
        """Capture and save a single frame to disk.

        Parameters
        ----------
        path : str
            Output file path (default ``"snapshot.jpg"``).

        Returns
        -------
        Path
            Resolved path of the saved file.

        Raises
        ------
        VisionError
            If the camera is unavailable or the file cannot be written.
        """
        frame = self.capture_frame()
        out = Path(path).resolve()
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success = cv2.imwrite(str(out), frame, encode_params)
        if not success:
            raise VisionError(f"Failed to write snapshot to {out}.")
        logger.info("Snapshot saved: %s", out)
        print(f"  📸 Snapshot saved → {out}")
        return out

    # ── Properties ───────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """``True`` if the camera was successfully opened."""
        return self._available

    def __repr__(self) -> str:
        status = "OK" if self._available else "UNAVAILABLE"
        return f"<VisionController index={self.camera_index} status={status}>"
