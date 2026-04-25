"""LLM controller for ARIA — natural language → tool calls via Gemma 4.

Connects to a locally-running Ollama server and uses structured
function-calling to translate free-form commands into servo actions.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import ollama

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT: str = (
    "You are ARIA, a robotic hand controller. "
    "Parse the user's natural language command into precise tool calls. "
    "Always respond with tool calls only, never plain text."
)

# ── Tool schema (Ollama function-calling format) ─────────────────────
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move_finger",
            "description": (
                "Move a specific finger to a given position. "
                "Position 0.0 is fully open, 1.0 is fully closed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "finger": {
                        "type": "string",
                        "enum": ["thumb", "index", "middle", "ring", "pinky"],
                        "description": "Which finger to move.",
                    },
                    "position": {
                        "type": "number",
                        "description": "Target position from 0.0 (open) to 1.0 (closed).",
                    },
                },
                "required": ["finger", "position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_hand",
            "description": "Fully open all fingers of the hand.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_hand",
            "description": "Fully close all fingers to make a fist.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pinch",
            "description": "Close the thumb and index finger together to pinch. Other fingers remain open.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strength": {
                        "type": "number",
                        "description": "Pinch strength from 0.0 (light) to 1.0 (firm).",
                    },
                },
                "required": ["strength"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "point",
            "description": "Extend the index finger while closing all other fingers (pointing gesture).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "peace_sign",
            "description": "Extend the index and middle fingers while closing the others (peace / victory sign).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wave",
            "description": "Open and close the hand repeatedly to wave.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repetitions": {
                        "type": "integer",
                        "description": "How many times to wave.",
                    },
                },
                "required": ["repetitions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate_wrist",
            "description": "Rotate the wrist to a given angle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "description": "Target angle in degrees from -90 (full left) to +90 (full right).",
                    },
                },
                "required": ["angle"],
            },
        },
    },
]


class LLMController:
    """Translates natural language into structured servo tool calls.

    Uses Ollama's function-calling interface with Gemma 4 to parse
    free-form user commands into a deterministic list of actions.

    Parameters
    ----------
    model : str
        Ollama model tag (default ``"gemma4:e2b"``).
    host : str
        Ollama server URL (default ``"http://localhost:11434"``).
    timeout : float
        Request timeout in seconds (default ``10.0``).
    max_retries : int
        Maximum number of retry attempts on failure (default ``1``).
    """

    def __init__(
        self,
        model: str = "gemma4:e2b",
        host: str = "http://localhost:11434",
        timeout: float = 10.0,
        max_retries: int = 1,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = ollama.Client(host=host, timeout=timeout)
        logger.info("LLMController ready — model=%s, host=%s", model, host)

    def parse_command(self, user_text: str) -> list[dict[str, Any]]:
        """Send *user_text* to the LLM and return parsed tool calls.

        Parameters
        ----------
        user_text : str
            The raw natural-language command from the user.

        Returns
        -------
        list[dict[str, Any]]
            Each dict has ``"name"`` (str) and ``"arguments"`` (dict).
            Returns an empty list if the LLM produces no tool calls.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):  # 1-indexed
            try:
                logger.debug(
                    "Ollama request attempt %d/%d for: %r",
                    attempt,
                    self.max_retries + 1,
                    user_text,
                )
                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                )
                return self._extract_tool_calls(response)

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Ollama request failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
                if attempt <= self.max_retries:
                    time.sleep(1.0)

        # All attempts exhausted
        logger.error("All Ollama attempts failed. Last error: %s", last_error)
        return []

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
        """Pull structured tool calls out of an Ollama chat response.

        Handles both the ``response.message.tool_calls`` attribute
        path and raw dict responses.
        """
        tool_calls: list[dict[str, Any]] = []

        message = getattr(response, "message", None)
        if message is None and isinstance(response, dict):
            message = response.get("message", {})

        raw_calls = None
        if hasattr(message, "tool_calls"):
            raw_calls = message.tool_calls
        elif isinstance(message, dict):
            raw_calls = message.get("tool_calls")

        if not raw_calls:
            logger.debug("No tool_calls found in LLM response.")
            return tool_calls

        for tc in raw_calls:
            # Ollama Python SDK returns objects with .function.name / .function.arguments
            if hasattr(tc, "function"):
                fn = tc.function
                name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
                args = getattr(fn, "arguments", None) or (fn.get("arguments", {}) if isinstance(fn, dict) else {})
            elif isinstance(tc, dict):
                fn = tc.get("function", tc)
                name = fn.get("name")
                args = fn.get("arguments", {})
            else:
                continue

            if name:
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append({"name": name, "arguments": args})

        logger.debug("Parsed %d tool call(s): %s", len(tool_calls), tool_calls)
        return tool_calls

    def __repr__(self) -> str:
        return f"<LLMController model={self.model!r} host={self.host!r}>"
