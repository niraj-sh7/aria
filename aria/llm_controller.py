"""LLM controller for ARIA — supports Ollama (local) and OpenRouter (cloud).

Translates natural language into structured tool calls or plain text
descriptions using either a local Ollama server or the OpenRouter API.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

import ollama
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT: str = (
    "You are ARIA, a robotic hand with vision. You can see through a camera. "
    "When given a vision-enabled command, analyze the image and determine the "
    "appropriate action. For grab commands, close the hand firmly. For reach "
    "commands, rotate the wrist toward the target. "
    "Always respond with tool calls only, never plain text."
)

OPENROUTER_SYSTEM_PROMPT: str = """You are ARIA, a robotic hand controller. Parse the user's natural language
command and respond with ONLY a valid JSON array of tool calls. No explanation,
no markdown, no code blocks — raw JSON only.

Available tools and their exact format:
[
{"tool": "move_finger", "args": {"finger": "thumb|index|middle|ring|pinky", "position": 0.0-1.0}},
{"tool": "open_hand", "args": {}},
{"tool": "close_hand", "args": {}},
{"tool": "pinch", "args": {"strength": 0.0-1.0}},
{"tool": "point", "args": {}},
{"tool": "peace_sign", "args": {}},
{"tool": "wave", "args": {"repetitions": 1-5}},
{"tool": "rotate_wrist", "args": {"angle": -90.0-90.0}}
]

Examples:
User: "give me a peace sign"
Response: [{"tool": "peace_sign", "args": {}}]

User: "grab the bottle"
Response: [{"tool": "close_hand", "args": {}}]

User: "point at the camera then wave"
Response: [{"tool": "point", "args": {}}, {"tool": "wave", "args": {"repetitions": 2}}]

User: "pinch gently"
Response: [{"tool": "pinch", "args": {"strength": 0.3}}]
"""

# ── Tool schema ──────────────────────────────────────────────────────
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move_finger",
            "description": "Move a specific finger to a given position (0.0=open, 1.0=closed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "finger": {"type": "string", "enum": ["thumb", "index", "middle", "ring", "pinky"]},
                    "position": {"type": "number"}
                },
                "required": ["finger", "position"]
            }
        }
    },
    {"type": "function", "function": {"name": "open_hand", "description": "Fully open all fingers."}},
    {"type": "function", "function": {"name": "close_hand", "description": "Fully close all fingers (fist)."}},
    {
        "type": "function",
        "function": {
            "name": "pinch",
            "description": "Close thumb and index together.",
            "parameters": {
                "type": "object",
                "properties": {"strength": {"type": "number"}},
                "required": ["strength"]
            }
        }
    },
    {"type": "function", "function": {"name": "point", "description": "Extend index, close others."}},
    {"type": "function", "function": {"name": "peace_sign", "description": "Extend index + middle, close others."}},
    {
        "type": "function",
        "function": {
            "name": "wave",
            "description": "Open and close hand repeatedly.",
            "parameters": {
                "type": "object",
                "properties": {"repetitions": {"type": "integer"}},
                "required": ["repetitions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rotate_wrist",
            "description": "Rotate wrist -90 to +90 degrees.",
            "parameters": {
                "type": "object",
                "properties": {"angle": {"type": "number"}},
                "required": ["angle"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grab_object",
            "description": "Close hand to grab a named object.",
            "parameters": {
                "type": "object",
                "properties": {"object_name": {"type": "string"}},
                "required": ["object_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reach_toward",
            "description": "Reach toward a target direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["left", "right", "center", "up", "down"]},
                    "distance": {"type": "string", "enum": ["near", "medium", "far"]}
                },
                "required": ["direction", "distance"]
            }
        }
    },
    {"type": "function", "function": {"name": "describe_scene", "description": "Describe the current camera frame."}},
    {
        "type": "function",
        "function": {
            "name": "track_and_grab",
            "description": "Center on object then grab.",
            "parameters": {
                "type": "object",
                "properties": {"object_name": {"type": "string"}},
                "required": ["object_name"]
            }
        }
    }
]

class ARIABackendError(Exception):
    """Raised when an LLM backend call fails."""

class LLMController:
    """Unified controller for Ollama and OpenRouter backends.

    Parameters
    ----------
    backend : str, optional
        "ollama" or "openrouter". If None, reads from ARIA_BACKEND env var.
    model : str, optional
        Model name. If None, uses backend-specific defaults.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: int = 1,
    ) -> None:
        self.backend = backend or os.getenv("ARIA_BACKEND", "ollama").lower()
        self.timeout = timeout
        self.max_retries = max_retries

        if self.backend == "ollama":
            self.model = model or "gemma4:e4b"
            self.host = "http://localhost:11434"
            self._client = ollama.Client(host=self.host, timeout=timeout)
            logger.info("LLMController using Ollama (local) — model=%s", self.model)
        elif self.backend == "openrouter":
            self.model = model or os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ARIABackendError(
                    "OpenRouter backend selected but OPENROUTER_API_KEY not set. "
                    "Run: export OPENROUTER_API_KEY=your_key_here"
                )
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/nirajshah/aria-robotic-hand",
                    "X-Title": "ARIA Robotic Hand",
                },
                timeout=timeout
            )
            logger.info("LLMController using OpenRouter (cloud) — model=%s", self.model)
        else:
            raise ARIABackendError(f"Unknown backend: {self.backend}")

    # ── Public API ───────────────────────────────────────────────────

    def parse_command(self, user_text: str) -> list[dict[str, Any]]:
        """Parse text into tool calls."""
        if self.backend == "ollama":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            return self._chat_with_retries(messages, tools=TOOLS)
        else:
            messages = [
                {"role": "system", "content": OPENROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            return self._chat_with_retries(messages, tools=None)

    def parse_command_with_vision(self, user_text: str, image_base64: str) -> list[dict[str, Any]]:
        """Parse text + image into tool calls."""
        if self.backend == "ollama":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text, "images": [image_base64]},
            ]
        else:  # openrouter / openai format
            messages = [
                {"role": "system", "content": OPENROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text + "\n\nAnalyze the image and include spatial context in your tool calls."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ]
            return self._chat_with_retries(messages, tools=None)

    def ask_vision(self, prompt: str, image_base64: str) -> str:
        """Ask a vision-related question and get a text response."""
        if self.backend == "ollama":
            messages = [{"role": "user", "content": prompt, "images": [image_base64]}]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ]
        
        for attempt in range(1, self.max_retries + 2):
            try:
                if self.backend == "ollama":
                    response = self._client.chat(model=self.model, messages=messages)
                    return response.message.content.strip()
                else:
                    response = self._client.chat.completions.create(model=self.model, messages=messages)
                    return response.choices[0].message.content.strip()
            except Exception as exc:
                if attempt > self.max_retries:
                    raise ARIABackendError(f"Backend {self.backend} failed after retries: {exc}")
                time.sleep(1.0)
        return ""

    # ── Internal helpers ─────────────────────────────────────────────

    def _chat_with_retries(self, messages: list[dict], tools: Optional[list] = None) -> list[dict]:
        """Core chat loop with retry logic and unified tool parsing."""
        for attempt in range(1, self.max_retries + 2):
            try:
                if self.backend == "ollama":
                    response = self._client.chat(model=self.model, messages=messages, tools=tools)
                    return self._extract_ollama_tools(response)
                else:
                    kwargs = {
                        "model": self.model,
                        "messages": messages,
                    }
                    if tools:
                        kwargs["tools"] = tools
                        kwargs["tool_choice"] = "auto"
                    response = self._client.chat.completions.create(**kwargs)
                    return self._extract_openai_tools(response)
            except Exception as exc:
                logger.warning("LLM attempt %d failed: %s", attempt, exc)
                if attempt > self.max_retries:
                    raise ARIABackendError(f"Backend {self.backend} failed: {exc}")
                time.sleep(1.0)
        return []

    def _extract_ollama_tools(self, response: Any) -> list[dict]:
        tool_calls = []
        raw = getattr(response.message, "tool_calls", [])
        for tc in raw:
            tool_calls.append({
                "name": tc.function.name,
                "arguments": tc.function.arguments
            })
        return tool_calls

    def _extract_openai_tools(self, response: Any) -> list[dict]:
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        try:
            tool_calls = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON response: %s", raw[:100])
            raise ARIABackendError(f"LLM returned non-JSON response: {raw[:100]}")

        normalized = []
        if isinstance(tool_calls, list):
            for call in tool_calls:
                normalized.append({
                    "name": call.get("tool", ""),
                    "arguments": call.get("args", {})
                })
        return normalized

    def __repr__(self) -> str:
        return f"<LLMController backend={self.backend} model={self.model}>"
