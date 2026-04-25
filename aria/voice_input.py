"""Voice input module for ARIA — microphone capture + Whisper STT.

Records audio from a USB microphone and transcribes it locally
using OpenAI Whisper (tiny model).
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Recording defaults ───────────────────────────────────────────────
DEFAULT_DURATION: float = 3.0        # seconds
SAMPLE_RATE: int = 16_000            # Whisper expects 16 kHz
CHANNELS: int = 1                    # mono
SILENCE_THRESHOLD: float = 0.01     # RMS below this → silence
SILENCE_WINDOW: float = 1.0         # seconds of silence to auto-stop


class VoiceInput:
    """Capture and transcribe voice commands from a USB microphone.

    Uses ``sounddevice`` for recording and ``openai-whisper`` (tiny model)
    for speech-to-text.  Falls back to ``pyaudio`` if ``sounddevice``
    is unavailable.

    Parameters
    ----------
    model_name : str
        Whisper model size (default ``"tiny"`` for speed on Pi 5).
    duration : float
        Maximum recording duration in seconds (default ``3.0``).
    language : str
        Expected spoken language (default ``"en"``).
    """

    def __init__(
        self,
        model_name: str = "tiny",
        duration: float = DEFAULT_DURATION,
        language: str = "en",
    ) -> None:
        self.model_name = model_name
        self.duration = duration
        self.language = language
        self._whisper_model = None  # lazy-loaded

    # ── Lazy model loading ───────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the Whisper model on first use."""
        if self._whisper_model is not None:
            return
        import whisper

        logger.info("Loading Whisper '%s' model …", self.model_name)
        self._whisper_model = whisper.load_model(self.model_name)
        logger.info("Whisper model loaded.")

    # ── Recording ────────────────────────────────────────────────────

    def _record_audio(self) -> np.ndarray:
        """Record audio from the default input device.

        Attempts ``sounddevice`` first; falls back to ``pyaudio``.

        Returns
        -------
        np.ndarray
            1-D float32 array of samples at ``SAMPLE_RATE``.
        """
        try:
            return self._record_sounddevice()
        except Exception as exc:
            logger.warning("sounddevice failed (%s), trying pyaudio.", exc)
            return self._record_pyaudio()

    def _record_sounddevice(self) -> np.ndarray:
        """Record using the ``sounddevice`` library."""
        import sounddevice as sd

        total_frames = int(self.duration * SAMPLE_RATE)
        print("🎤 Listening…")
        audio = sd.rec(
            total_frames,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()
        return audio.flatten()

    def _record_pyaudio(self) -> np.ndarray:
        """Record using ``pyaudio`` as a fallback backend."""
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=1024,
        )
        frames: list[bytes] = []
        total_chunks = int(SAMPLE_RATE / 1024 * self.duration)
        print("🎤 Listening…")
        for _ in range(total_chunks):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        pa.terminate()
        return np.frombuffer(b"".join(frames), dtype=np.float32)

    # ── Transcription ────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run Whisper on the captured audio.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 audio at 16 kHz.

        Returns
        -------
        str
            Transcribed text (stripped).
        """
        self._load_model()
        result = self._whisper_model.transcribe(
            audio,
            language=self.language,
            fp16=False,  # Pi 5 CPU — no FP16
        )
        text: str = result.get("text", "").strip()
        logger.debug("Whisper transcription: %r", text)
        return text

    # ── Public API ───────────────────────────────────────────────────

    def listen(self) -> str:
        """Record from the microphone and return the transcribed text.

        Returns
        -------
        str
            The transcribed user command.
        """
        audio = self._record_audio()
        return self._transcribe(audio)

    def listen_continuous(self, callback: Callable[[str], None]) -> None:
        """Continuously listen and invoke *callback* with each transcription.

        Runs forever until interrupted (``KeyboardInterrupt``).

        Parameters
        ----------
        callback : Callable[[str], None]
            Function to call with each non-empty transcription.
        """
        logger.info("Entering continuous listening mode.")
        while True:
            try:
                text = self.listen()
                if text:
                    callback(text)
            except KeyboardInterrupt:
                logger.info("Continuous listening stopped by user.")
                break
            except Exception as exc:
                logger.error("Error during listen cycle: %s", exc)
                time.sleep(0.5)

    def __repr__(self) -> str:
        return (
            f"<VoiceInput model={self.model_name!r} "
            f"duration={self.duration}s lang={self.language!r}>"
        )
