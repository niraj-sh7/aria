"""ARIA Hardware Simulator.

Runs the full LLM pipeline (OpenRouter → Gemma 4) and broadcasts servo
state to a browser visualization over WebSocket. Zero hardware required.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

load_dotenv()

logger = logging.getLogger(__name__)

# ── Servo state ──────────────────────────────────────────────────────
PULSE_MIN = 150
PULSE_MAX = 600
PULSE_NEUTRAL = 375

INITIAL_STATE: dict[str, dict[str, Any]] = {
    "thumb":  {"channel": 0, "pulse": 150, "percent": 0},
    "index":  {"channel": 1, "pulse": 150, "percent": 0},
    "middle": {"channel": 2, "pulse": 150, "percent": 0},
    "ring":   {"channel": 3, "pulse": 150, "percent": 0},
    "pinky":  {"channel": 4, "pulse": 150, "percent": 0},
    "wrist":  {"channel": 5, "pulse": 375, "percent": 50},
}

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

def pulse_to_percent(pulse: int) -> int:
    return int((pulse - PULSE_MIN) / (PULSE_MAX - PULSE_MIN) * 100)

def pos_to_pulse(position: float) -> int:
    return int(PULSE_MIN + max(0.0, min(1.0, position)) * (PULSE_MAX - PULSE_MIN))

def angle_to_pulse(angle: float) -> int:
    p = int(PULSE_NEUTRAL + (angle / 90.0) * 225)
    return max(PULSE_MIN, min(PULSE_MAX, p))


# ── Simulator state shared across the app ────────────────────────────
class SimState:
    def __init__(self) -> None:
        self.servos: dict[str, dict[str, Any]] = copy.deepcopy(INITIAL_STATE)
        self.log_entries: list[dict[str, str]] = []
        self.state_buffer: deque[dict] = deque(maxlen=5)
        self.connections: list[WebSocket] = []

    def set_finger(self, name: str, pulse: int) -> None:
        pulse = max(PULSE_MIN, min(PULSE_MAX, pulse))
        self.servos[name]["pulse"] = pulse
        self.servos[name]["percent"] = pulse_to_percent(pulse)

    def set_all_fingers(self, pulse: int) -> None:
        for name in FINGER_NAMES:
            self.set_finger(name, pulse)

    def add_log(self, level: str, msg: str) -> dict[str, str]:
        entry = {"time": datetime.now().strftime("%H:%M:%S"), "level": level, "msg": msg}
        self.log_entries.append(entry)
        if len(self.log_entries) > 200:
            self.log_entries = self.log_entries[-200:]
        return entry

    def build_message(self, log_entries: list[dict]) -> str:
        return json.dumps({
            "type": "state_update",
            "state": self.servos,
            "log": log_entries,
        })

sim = SimState()


# ── Simulated Executor (same interface as CommandExecutor) ────────────
class SimulatedExecutor:
    """Dispatches LLM tool calls to sim state + WebSocket broadcast.

    Identical interface to CommandExecutor — swap the import for real HW.
    """

    def __init__(self, state: SimState) -> None:
        self.state = state

    def execute(self, tool_calls: list[dict[str, Any]]) -> None:
        loop = asyncio.get_event_loop()
        loop.create_task(self._execute_async(tool_calls))

    async def execute_async(self, tool_calls: list[dict[str, Any]]) -> None:
        await self._execute_async(tool_calls)

    async def _execute_async(self, tool_calls: list[dict[str, Any]]) -> None:
        logs: list[dict] = []
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            log_line = self.state.add_log("LLM", f"tool_call: {name}({_fmt_args(args)})")
            logs.append(log_line)
            print(f"  [TOOL] {name}({_fmt_args(args)})")
            new_logs = await self._dispatch(name, args)
            logs.extend(new_logs)

        sync_log = self.state.add_log("SYS", "✓ browser synced")
        logs.append(sync_log)
        await self._broadcast(logs)
        print("  [OK] Hand state updated → browser synced")

    async def _dispatch(self, name: str, args: dict) -> list[dict]:
        logs: list[dict] = []

        def record(finger: str) -> dict:
            p = self.state.servos[finger]["pulse"]
            pct = self.state.servos[finger]["percent"]
            msg = f"{finger} → {p} PWM ({pct}%)"
            print(f"  [EXEC] {msg}")
            return self.state.add_log("EXEC", msg)

        if name == "move_finger":
            finger = args.get("finger", "index")
            position = float(args.get("position", 0.0))
            self.state.set_finger(finger, pos_to_pulse(position))
            logs.append(record(finger))

        elif name == "open_hand":
            self.state.set_all_fingers(PULSE_MIN)
            for f in FINGER_NAMES:
                logs.append(record(f))

        elif name == "close_hand":
            self.state.set_all_fingers(PULSE_MAX)
            for f in FINGER_NAMES:
                logs.append(record(f))

        elif name == "pinch":
            strength = float(args.get("strength", 0.8))
            pulse = pos_to_pulse(strength)
            for f in ["thumb", "index"]:
                self.state.set_finger(f, pulse)
                logs.append(record(f))
            for f in ["middle", "ring", "pinky"]:
                self.state.set_finger(f, PULSE_MIN)
                logs.append(record(f))

        elif name == "point":
            self.state.set_finger("index", PULSE_MIN)
            for f in ["thumb", "middle", "ring", "pinky"]:
                self.state.set_finger(f, PULSE_MAX)
            for f in FINGER_NAMES:
                logs.append(record(f))

        elif name == "peace_sign":
            for f in ["index", "middle"]:
                self.state.set_finger(f, PULSE_MIN)
            for f in ["thumb", "ring", "pinky"]:
                self.state.set_finger(f, PULSE_MAX)
            for f in FINGER_NAMES:
                logs.append(record(f))

        elif name == "wave":
            reps = int(args.get("repetitions", 3))
            for i in range(reps):
                self.state.set_all_fingers(PULSE_MIN)
                await self._broadcast([self.state.add_log("EXEC", f"wave open {i+1}/{reps}")])
                await asyncio.sleep(0.4)
                self.state.set_all_fingers(PULSE_MAX)
                await self._broadcast([self.state.add_log("EXEC", f"wave close {i+1}/{reps}")])
                await asyncio.sleep(0.4)
            self.state.set_all_fingers(PULSE_MIN)
            logs.append(self.state.add_log("EXEC", "wave complete"))

        elif name == "rotate_wrist":
            angle = float(args.get("angle", 0.0))
            self.state.set_finger("wrist", angle_to_pulse(angle))
            logs.append(record("wrist"))

        elif name in ("grab_object", "track_and_grab"):
            self.state.set_all_fingers(PULSE_MAX)
            obj = args.get("object_name", "object")
            for f in FINGER_NAMES:
                logs.append(record(f))
            logs.append(self.state.add_log("SYS", f"grabbed: {obj}"))

        elif name == "reach_toward":
            direction = args.get("direction", "center")
            angles = {"left": -45, "right": 45, "center": 0, "up": 0, "down": 0}
            angle = angles.get(direction, 0)
            self.state.set_finger("wrist", angle_to_pulse(angle))
            self.state.set_all_fingers(PULSE_MIN)
            logs.append(record("wrist"))
            for f in FINGER_NAMES:
                logs.append(record(f))

        elif name == "describe_scene":
            logs.append(self.state.add_log("SYS", "describe_scene (no camera in sim)"))

        else:
            logs.append(self.state.add_log("ERR", f"Unknown tool: {name}"))

        return logs

    async def _broadcast(self, logs: list[dict]) -> None:
        msg = self.state.build_message(logs)
        self.state.state_buffer.append(msg)
        dead = []
        for ws in self.state.connections:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.state.connections.remove(ws)


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(title="ARIA Simulator")

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "simulator.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    sim.connections.append(ws)
    logger.info("Browser connected. %d client(s) active.", len(sim.connections))
    # Replay buffered state
    for msg in sim.state_buffer:
        try:
            await ws.send_text(msg)
        except Exception:
            break
    try:
        while True:
            await ws.receive_text()  # handle pings / keep-alive
    except WebSocketDisconnect:
        if ws in sim.connections:
            sim.connections.remove(ws)
        logger.info("Browser disconnected.")


# ── Built-in commands ─────────────────────────────────────────────────
DEMO_STEPS = [
    ("open_hand", {}),
    ("point", {}),
    ("peace_sign", {}),
    ("pinch", {"strength": 0.8}),
    ("close_hand", {}),
    ("wave", {"repetitions": 3}),
    ("open_hand", {}),
]

async def run_builtin(cmd: str, executor: SimulatedExecutor) -> bool:
    """Handle built-in simulator commands. Returns True if handled."""
    if cmd == "demo":
        print("  [SIM] Running demo sequence …")
        for name, args in DEMO_STEPS:
            await executor.execute_async([{"name": name, "arguments": args}])
            await asyncio.sleep(0.6)
        return True
    if cmd == "reset":
        for f in FINGER_NAMES:
            sim.set_finger(f, PULSE_MIN)
        sim.set_finger("wrist", PULSE_NEUTRAL)
        log = sim.add_log("SYS", "reset to neutral")
        await executor._broadcast([log])
        print("  [SIM] Reset to neutral.")
        return True
    if cmd == "state":
        print("\n  ┌─────────────────────────────────────────────┐")
        print("  │  Servo State                                │")
        print("  ├──────────┬──────────┬───────────┬──────────┤")
        print("  │  Finger  │  Channel │  Pulse    │  Pct     │")
        print("  ├──────────┼──────────┼───────────┼──────────┤")
        for name, data in sim.servos.items():
            print(f"  │ {name:<8s} │  CH{data['channel']}     │  {data['pulse']:<7d}  │  {data['percent']:>3d}%    │")
        print("  └──────────┴──────────┴───────────┴──────────┘\n")
        return True
    return False


# ── Terminal input loop ───────────────────────────────────────────────
async def terminal_loop(executor: SimulatedExecutor) -> None:
    from aria.llm_controller import LLMController, ARIABackendError

    backend = os.getenv("ARIA_BACKEND", "openrouter")
    try:
        llm = LLMController(backend=backend)
    except ARIABackendError as exc:
        print(f"\n  [ERR] LLM init failed: {exc}\n")
        llm = None

    print(f"\n  🤖 ARIA Simulator — LLM: {backend} / {llm.model if llm else 'NONE'}")
    print("  💡 Commands: natural language, 'demo', 'reset', 'state', 'quit'\n")

    loop = asyncio.get_event_loop()
    while True:
        try:
            user_input = await loop.run_in_executor(None, lambda: input("  > ").strip())
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        if await run_builtin(user_input.lower(), executor):
            continue

        print(f'  [ARIA] Parsing: "{user_input}"')
        sim.add_log("USER", user_input)

        if llm is None:
            print("  [ERR] No LLM connected.")
            continue

        try:
            tool_calls = llm.parse_command(user_input)
            if not tool_calls:
                err_log = sim.add_log("ERR", "LLM returned no tool calls — try rephrasing")
                await executor._broadcast([err_log])
                print("  [?] No tool calls returned. Try rephrasing.")
            else:
                await executor.execute_async(tool_calls)
        except Exception as exc:
            err_log = sim.add_log("ERR", str(exc))
            await executor._broadcast([err_log])
            print(f"  [ERR] {exc}")


# ── Launch ────────────────────────────────────────────────────────────
def launch() -> None:
    """Entry point called from main.py --sim."""
    print("  🖥  ARIA Simulator starting...")
    print("  🌐 Opening browser at http://localhost:7437")
    print("  ⚡ WebSocket server ready")

    executor = SimulatedExecutor(sim)

    config = uvicorn.Config(app, host="127.0.0.1", port=7437, log_level="warning")
    server = uvicorn.Server(config)

    async def _run() -> None:
        # Give uvicorn a moment to bind before opening browser
        asyncio.get_event_loop().call_later(1.2, lambda: webbrowser.open("http://localhost:7437"))
        await asyncio.gather(
            server.serve(),
            terminal_loop(executor),
        )
        server.should_exit = True

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n  🛑 Simulator stopped.")
