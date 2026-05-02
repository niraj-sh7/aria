#!/usr/bin/env python3
"""ARIA — Autonomous Robotic Intelligence Arm.

Main entry point. Connects voice/text input to LLM backends (Ollama/OpenRouter).

Usage
-----
    python main.py              # voice-controlled mode
    python main.py --text       # text-only mode
    python main.py --demo       # preset demo sequence
    python main.py --vision     # attach camera frame to every LLM call
    python main.py --watch      # continuous scene narration loop
    python main.py --snap       # single snapshot + describe scene
    python main.py --sim        # browser hardware simulator (no Pi needed)
    python main.py --backend openrouter  # override backend
    python main.py --verbose    # debug logging
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv

# Load .env before imports that might check env vars
load_dotenv()

from aria.servo_controller import ServoController
from aria.llm_controller import LLMController, ARIABackendError
from aria.executor import CommandExecutor

logger = logging.getLogger("aria")


# ── Banner ────────────────────────────────────────────────────────────

def build_banner(
    camera_index: int,
    camera_ok: bool,
    vision_active: bool,
    llm: LLMController,
) -> str:
    """Build the startup banner string with backend and camera info."""
    cam_status = f"CH{camera_index} {'✅ connected' if camera_ok else '⚠️ not found'}"
    vision_label = "ACTIVE 👁" if vision_active else "inactive"
    
    backend_info = "Ollama (Local)" if llm.backend == "ollama" else "OpenRouter (Cloud)"
    model_name = llm.model
    
    # Mask API key if using OpenRouter
    api_key_status = "N/A"
    if llm.backend == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY", "")
        if key:
            api_key_status = f"sk-...{key[-4:]}" if len(key) > 8 else "SET"
        else:
            api_key_status = "MISSING"

    return f"""
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
║  Backend    │  {backend_info:<47s}║
║  LLM Model  │  {model_name:<47s}║
║  API Key    │  {api_key_status:<47s}║
║  I2C Addr   │  0x40 (PCA9685) · Camera: {cam_status}           ║
║  Vision     │  {vision_label:<47s}║
╠══════════════════════════════════════════════════════════════════╣
║  Commands   │  "open hand"  "make a fist"  "point at that"       ║
║             │  "peace sign"  "pinch gently"  "wave hello"        ║
║             │  "grab the bottle"  "describe the scene"           ║
╚══════════════════════════════════════════════════════════════════╝
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aria")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--snap", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--backend", choices=["ollama", "openrouter"], help="Override ARIA_BACKEND")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Launch browser hardware simulator (no Pi or mic required).",
    )
    return parser.parse_args()

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-8s %(message)s")

# ... (rest of the runner functions remain the same logic, just ensure they handle ARIABackendError)

def _llm_call(llm: LLMController, text: str, vision_ctrl: Optional[object]) -> list[dict]:
    try:
        if vision_ctrl is not None:
            from aria.vision import VisionController
            vc: VisionController = vision_ctrl  # type: ignore
            if vc.is_available:
                return llm.parse_command_with_vision(text, vc.capture_frame_base64())
        return llm.parse_command(text)
    except ARIABackendError as exc:
        print(f"  ❌ LLM Error: {exc}")
        return []

# ── Runner functions (shortened for brevity but keeping logic) ───────
def run_text_mode(llm, executor, vision_ctrl):
    print("\n  ⌨️ Text mode (type 'q' to quit):\n")
    while True:
        user_input = input("  ARIA> ").strip()
        if user_input.lower() in ("q", "quit", "exit"): break
        if user_input: executor.execute(_llm_call(llm, user_input, vision_ctrl))

def run_voice_mode(llm, executor, vision_ctrl):
    from aria.voice_input import VoiceInput
    voice = VoiceInput()
    print("\n  🎙️ Voice mode active:\n")
    voice.listen_continuous(lambda t: executor.execute(_llm_call(llm, t, vision_ctrl)))

def run_watch_mode(llm, vision_ctrl):
    print("\n  👁️ Watch mode (Ctrl+C to stop):\n")
    while True:
        try:
            print(f"  🔍 {llm.ask_vision('Describe graspable objects.', vision_ctrl.capture_frame_base64())}\n")
            time.sleep(2.0)
        except KeyboardInterrupt: break

def run_snap_mode(llm, vision_ctrl):
    vision_ctrl.save_snapshot("snapshot.jpg")
    print(f"\n  👁️ Scene: {llm.ask_vision('Describe everything you see.', vision_ctrl.capture_frame_base64())}")

def run_demo(executor):
    for name, args in [("open_hand", {}), ("point", {}), ("peace_sign", {}), ("pinch", {"strength": 0.8}), ("close_hand", {}), ("wave", {"repetitions": 2})]:
        executor.execute([{"name": name, "arguments": args}])
        time.sleep(1.0)

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # ── Simulator mode: short-circuit before any hardware init ─────────
    if args.sim:
        from aria.simulator import launch
        launch()
        return

    try:
        llm = LLMController(backend=args.backend)
    except ARIABackendError as exc:
        print(f"Fatal error: {exc}")
        sys.exit(1)

    vision_ctrl = None
    if args.vision or args.watch or args.snap:
        from aria.vision import VisionController
        vision_ctrl = VisionController()

    print(build_banner(
        vision_ctrl.camera_index if vision_ctrl else 0,
        vision_ctrl.is_available if vision_ctrl else False,
        args.vision or args.watch or args.snap,
        llm
    ))

    servo = ServoController()
    executor = CommandExecutor(servo, llm=llm, vision=vision_ctrl)

    try:
        if args.demo: run_demo(executor)
        elif args.snap: run_snap_mode(llm, vision_ctrl)
        elif args.watch: run_watch_mode(llm, vision_ctrl)
        elif args.text: run_text_mode(llm, executor, vision_ctrl if args.vision else None)
        else: run_voice_mode(llm, executor, vision_ctrl if args.vision else None)
    except KeyboardInterrupt: pass
    finally:
        servo.release()
        if vision_ctrl: vision_ctrl.release()
        print("\n  🔌 Offline. Goodbye!")

if __name__ == "__main__": main()
