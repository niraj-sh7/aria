#!/usr/bin/env python3
"""ARIA — Autonomous Robotic Intelligence Arm.

Main entry point.  Connects voice input → LLM → command executor → servos.

Usage
-----
    python main.py              # full voice-controlled mode
    python main.py --text       # text-only mode (no microphone)
    python main.py --demo       # run a preset demo sequence
    python main.py --verbose    # enable debug logging
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from aria.servo_controller import (
    CHANNEL_INDEX,
    CHANNEL_MIDDLE,
    CHANNEL_PINKY,
    CHANNEL_RING,
    CHANNEL_THUMB,
    CHANNEL_WRIST,
    FINGER_CHANNELS,
    ServoController,
)
from aria.llm_controller import LLMController
from aria.executor import CommandExecutor

logger = logging.getLogger("aria")

# ── Banner ───────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      █████╗ ██████╗ ██╗ █████╗                                   ║
║     ██╔══██╗██╔══██╗██║██╔══██╗                                  ║
║     ███████║██████╔╝██║███████║                                  ║
║     ██╔══██║██╔══██╗██║██╔══██║                                  ║
║     ██║  ██║██║  ██║██║██║  ██║                                  ║
║     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝                                  ║
║                                                                  ║
║     Autonomous Robotic Intelligence Arm                          ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  LLM Model  │  gemma4:e2b  (Ollama @ localhost:11434)            ║
║  I2C Addr   │  0x40  (PCA9685 PWM driver)                        ║
║  Servos     │  CH0 Thumb · CH1 Index · CH2 Middle                ║
║             │  CH3 Ring  · CH4 Pinky · CH5 Wrist                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Commands   │  "open hand"  "make a fist"  "point at that"       ║
║             │  "peace sign"  "pinch gently"  "wave hello"        ║
║             │  "rotate wrist left"  "close the index finger"     ║
╚══════════════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aria",
        description="ARIA — Autonomous Robotic Intelligence Arm",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Text-only mode (type commands instead of using the mic).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a preset demo sequence and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Demo mode ────────────────────────────────────────────────────────

def run_demo(executor: CommandExecutor) -> None:
    """Execute a preset demo sequence to show off ARIA's capabilities."""
    print("\n  🤖 Running ARIA demo sequence …\n")
    steps: list[tuple[str, dict]] = [
        ("open_hand", {}),
        ("point", {}),
        ("peace_sign", {}),
        ("pinch", {"strength": 0.8}),
        ("close_hand", {}),
        ("wave", {"repetitions": 3}),
    ]
    for name, args in steps:
        executor.execute([{"name": name, "arguments": args}])
        time.sleep(1.0)
    print("\n  ✅ Demo complete.\n")


# ── Text input loop ─────────────────────────────────────────────────

def run_text_mode(llm: LLMController, executor: CommandExecutor) -> None:
    """Interactive text-command loop (no microphone required)."""
    print("\n  ⌨️  Text mode — type a command (or 'quit' to exit):\n")
    while True:
        try:
            user_input = input("  ARIA> ").strip()
        except EOFError:
            break
        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            break
        tool_calls = llm.parse_command(user_input)
        if tool_calls:
            executor.execute(tool_calls)
        else:
            print("  ❓ Could not parse that command. Try rephrasing.")


# ── Voice input loop ────────────────────────────────────────────────

def run_voice_mode(
    llm: LLMController,
    executor: CommandExecutor,
) -> None:
    """Continuous voice-command loop."""
    # Import here so --text and --demo don't require audio deps
    from aria.voice_input import VoiceInput

    voice = VoiceInput()
    print("\n  🎙️  Voice mode — speak a command (Ctrl+C to exit):\n")

    def handle(text: str) -> None:
        print(f"  📝 Heard: \"{text}\"")
        tool_calls = llm.parse_command(text)
        if tool_calls:
            executor.execute(tool_calls)
        else:
            print("  ❓ Could not parse command. Try again.")

    voice.listen_continuous(handle)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    """Application entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    print(BANNER)

    servo = ServoController()
    executor = CommandExecutor(servo)

    if servo.is_simulated:
        print("  ⚙️  Hardware not detected — running in simulation mode.\n")
    else:
        print("  ✅ PCA9685 connected — hardware mode.\n")

    try:
        if args.demo:
            run_demo(executor)
        elif args.text:
            llm = LLMController()
            run_text_mode(llm, executor)
        else:
            llm = LLMController()
            run_voice_mode(llm, executor)
    except KeyboardInterrupt:
        print("\n\n  🛑 Interrupted — shutting down safely …")
    finally:
        servo.release()
        print("  🔌 Servos released. Goodbye!\n")


if __name__ == "__main__":
    main()
