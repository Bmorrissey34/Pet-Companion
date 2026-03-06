#!/usr/bin/env python3
"""
PiCar-X Exploration Benchmark (rewritten)

- Fixed camera view (no pan/tilt required)
- LLM decides the robot actions from visual frames
- Uses OpenAI-compatible endpoint (LM Studio)
- Strict JSON action output with safety clamps
"""

import argparse
import base64
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from picamera2 import Picamera2
from picarx import Picarx


# -----------------------------
# Utilities
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def now_ts() -> str:
    # Human readable timestamp for filenames/logging
    return time.strftime("%Y%m%d_%H%M%S")


def b64_jpeg_from_path(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from a model response that might contain extra text.
    """
    if not text:
        return None

    # Fast path: response itself is JSON
    text_stripped = text.strip()
    if text_stripped.startswith("{") and text_stripped.endswith("}"):
        try:
            return json.loads(text_stripped)
        except Exception:
            pass

    # Try to find a JSON object inside the text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# -----------------------------
# Robot adapter (handles API differences)
# -----------------------------

class RobotAdapter:
    """
    Wraps Picarx and exposes a minimal, safe motion API.
    We avoid relying on version-specific methods beyond what most PiCar-X builds include.
    """

    def __init__(self, px: Picarx, max_power: int = 30, max_steer_deg: int = 25):
        self.px = px
        self.max_power = int(clamp(max_power, 10, 60))
        self.max_steer_deg = int(clamp(max_steer_deg, 10, 35))

    def stop(self) -> None:
        try:
            self.px.stop()
        except Exception:
            # Best effort
            pass
        try:
            if hasattr(self.px, "set_dir_servo_angle"):
                self.px.set_dir_servo_angle(0)
        except Exception:
            pass

    def steer(self, deg: float) -> None:
        deg = clamp(float(deg), -self.max_steer_deg, self.max_steer_deg)
        if hasattr(self.px, "set_dir_servo_angle"):
            self.px.set_dir_servo_angle(deg)

    def drive_for_ms(self, power: int, steer_deg: float, ms: int) -> None:
        """
        Drive for a short duration:
        - power > 0 forward
        - power < 0 backward
        """
        power = int(clamp(power, -self.max_power, self.max_power))
        ms = int(clamp(ms, 100, 1200))
        self.steer(steer_deg)

        # Try common method names.
        # Many PiCar-X builds have forward/backward/stop.
        try:
            if power > 0 and hasattr(self.px, "forward"):
                self.px.forward(power)
            elif power < 0 and hasattr(self.px, "backward"):
                self.px.backward(abs(power))
            else:
                # Fallback: if only set_power exists (rare)
                if hasattr(self.px, "set_power"):
                    self.px.set_power(power)
                else:
                    self.stop()
                    return

            time.sleep(ms / 1000.0)
        finally:
            # Always stop after the slice to keep control tight
            self.stop()


# -----------------------------
# Data models
# -----------------------------

@dataclass
class Observation:
    step: int
    label: str
    description: str
    image_path: str
    timestamp: float


@dataclass
class StepLog:
    step: int
    latency_s: float
    model_thinking: str
    action: Dict[str, Any]
    observation_made: bool
    observation_text: Optional[str]
    image_path: str
    timestamp: float


# -----------------------------
# LLM prompting
# -----------------------------

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["DRIVE", "STOP", "OBSERVE"]
        },
        "drive": {
            "type": "object",
            "properties": {
                "power": {"type": "integer"},     # -max_power..max_power
                "steer_deg": {"type": "number"},  # -max_steer..max_steer
                "ms": {"type": "integer"}         # 100..1200
            },
            "required": ["power", "steer_deg", "ms"]
        },
        "observe": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},         # short name of location
                "description": {"type": "string"}    # 1-2 sentences
            },
            "required": ["label", "description"]
        },
        "thinking": {"type": "string"}
    },
    "required": ["action", "thinking"]
}


SYSTEM_PROMPT = f"""
You control a PiCar-X robot exploring an indoor environment.
You receive a camera image each step. The camera is fixed (no pan/tilt).
You must decide the next action to explore new locations and avoid getting stuck.

You must output ONLY valid JSON matching this schema (no extra text):
{json.dumps(ACTION_SCHEMA, indent=2)}

Rules:
- Use DRIVE for short, safe movements. Prefer small moves and gentle steering.
- Use OBSERVE when you believe you have reached a distinct location/viewpoint.
- If unsure, rotate by steering and short drive pulses.
- Never output anything except JSON.
"""


def build_user_prompt(step: int, goal_locations: int, observations: List[Observation]) -> str:
    obs_lines = []
    for o in observations[-5:]:
        obs_lines.append(f"- {o.label}: {o.description}")

    obs_summary = "\n".join(obs_lines) if obs_lines else "(none yet)"

    return f"""
Goal: Visit {goal_locations} distinct locations. At each location, OBSERVE with a short label and 1-2 sentence description.
Current step: {step}
Observations so far ({len(observations)}/{goal_locations}):
{obs_summary}

Decide the next action to make progress toward new, distinct viewpoints.
Remember: output JSON only.
""".strip()


# -----------------------------
# Model / endpoint helpers
# -----------------------------

def autodetect_model(client: OpenAI) -> str:
    """
    Queries /v1/models and picks the first model id.
    """
    models = client.models.list()
    # LM Studio often returns a list with a single loaded model
    if hasattr(models, "data") and models.data:
        return models.data[0].id
    # fallback
    return "unknown-model"


# -----------------------------
# Camera
# -----------------------------

def init_camera(width: int = 640, height: int = 480) -> Picamera2:
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"format": "BGR888", "size": (width, height)})
    cam.configure(config)
    cam.start()
    time.sleep(0.4)
    return cam


# -----------------------------
# Main exploration loop
# -----------------------------

def llm_step(
    client: OpenAI,
    model: str,
    user_prompt: str,
    image_b64: str,
    temperature: float = 0.3,
    max_tokens: int = 300
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Sends a multimodal chat request to OpenAI-compatible server.
    Returns (latency_s, thinking, action_json).
    """
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ],
            },
        ],
    )
    latency = time.time() - t0

    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        text = ""

    action_obj = extract_first_json_obj(text)
    if action_obj is None:
        # Hard fail to STOP
        action_obj = {"action": "STOP", "thinking": "Invalid JSON output; stopping for safety."}

    thinking = str(action_obj.get("thinking", ""))[:500]
    return latency, thinking, action_obj


def normalize_action(action_obj: Dict[str, Any], max_power: int, max_steer_deg: int) -> Dict[str, Any]:
    """
    Enforce schema-ish shape and safety clamps.
    """
    action = str(action_obj.get("action", "STOP")).upper().strip()
    thinking = str(action_obj.get("thinking", ""))

    safe: Dict[str, Any] = {"action": "STOP", "thinking": thinking}

    if action == "DRIVE":
        drive = action_obj.get("drive", {}) if isinstance(action_obj.get("drive", {}), dict) else {}
        power = int(drive.get("power", 0))
        steer_deg = float(drive.get("steer_deg", 0))
        ms = int(drive.get("ms", 300))

        power = int(clamp(power, -max_power, max_power))
        steer_deg = float(clamp(steer_deg, -max_steer_deg, max_steer_deg))
        ms = int(clamp(ms, 100, 1200))

        safe["action"] = "DRIVE"
        safe["drive"] = {"power": power, "steer_deg": steer_deg, "ms": ms}
        return safe

    if action == "OBSERVE":
        obs = action_obj.get("observe", {}) if isinstance(action_obj.get("observe", {}), dict) else {}
        label = str(obs.get("label", "Location")).strip()[:40]
        desc = str(obs.get("description", "")).strip()[:240]
        if not desc:
            desc = "No description provided."
        safe["action"] = "OBSERVE"
        safe["observe"] = {"label": label, "description": desc}
        return safe

    # STOP or anything else
    safe["action"] = "STOP"
    return safe


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    # Client
    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key=args.api_key)

    model = args.model or autodetect_model(client)
    print(f"Using endpoint: {base_url}")
    print(f"Model: {model}")

    # Robot + camera
    px = Picarx()
    robot = RobotAdapter(px, max_power=args.max_power, max_steer_deg=args.max_steer_deg)
    cam = init_camera(args.width, args.height)
    print("[CAMERA] ready")

    ensure_dir(args.save_images)

    observations: List[Observation] = []
    steps: List[StepLog] = []

    try:
        print("\n" + "=" * 60)
        print("  EXPLORATION BENCHMARK (rewritten)")
        print(f"  Goal: Visit {args.goal_locations} locations, observe and describe each")
        print(f"  Max steps: {args.max_steps}")
        print("=" * 60 + "\n")

        for step in range(1, args.max_steps + 1):
            if len(observations) >= args.goal_locations:
                print(f"[DONE] Reached goal locations: {len(observations)}/{args.goal_locations}")
                break

            # Capture image
            img_name = f"step_{step:03d}.jpg"
            img_path = os.path.join(args.save_images, img_name) if args.save_images else f"/tmp/{img_name}"
            cam.capture_file(img_path)
            img_b64 = b64_jpeg_from_path(img_path)

            user_prompt = build_user_prompt(step, args.goal_locations, observations)

            latency, thinking, action_obj = llm_step(
                client=client,
                model=model,
                user_prompt=user_prompt,
                image_b64=img_b64,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            action_safe = normalize_action(action_obj, args.max_power, args.max_steer_deg)

            print(f"[STEP {step}/{args.max_steps}] obs {len(observations)}/{args.goal_locations} | latency {latency:.2f}s")
            if thinking:
                print(f"  thinking: {thinking[:140]}")

            observation_made = False
            observation_text = None

            # Execute
            try:
                if action_safe["action"] == "DRIVE":
                    d = action_safe["drive"]
                    print(f"  action: DRIVE power={d['power']} steer={d['steer_deg']:.1f} ms={d['ms']}")
                    robot.drive_for_ms(power=d["power"], steer_deg=d["steer_deg"], ms=d["ms"])

                elif action_safe["action"] == "OBSERVE":
                    o = action_safe["observe"]
                    observation_made = True
                    observation_text = f"{o['label']}: {o['description']}"
                    observations.append(
                        Observation(
                            step=step,
                            label=o["label"],
                            description=o["description"],
                            image_path=img_path if args.save_images else "",
                            timestamp=time.time(),
                        )
                    )
                    print(f"  OBSERVATION {len(observations)}: {observation_text}")

                    # small pause to avoid spamming
                    time.sleep(0.2)

                else:
                    print("  action: STOP")
                    robot.stop()
                    time.sleep(0.2)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  [WARN] execution error: {e} -> STOP")
                robot.stop()

            steps.append(
                StepLog(
                    step=step,
                    latency_s=latency,
                    model_thinking=thinking,
                    action=action_safe,
                    observation_made=observation_made,
                    observation_text=observation_text,
                    image_path=img_path if args.save_images else "",
                    timestamp=time.time(),
                )
            )

        # Final summary
        result = {
            "model": model,
            "endpoint": base_url,
            "goal_locations": args.goal_locations,
            "max_steps": args.max_steps,
            "observations": [asdict(o) for o in observations],
            "steps": [asdict(s) for s in steps],
        }
        return result

    finally:
        try:
            robot.stop()
        except Exception:
            pass
        try:
            cam.stop()
        except Exception:
            pass
        print("Done.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="LM Studio host (use Tailscale IP like 100.x.x.x)")
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--api-key", default="lm-studio", help="Ignored unless server requires auth")
    ap.add_argument("--model", default="", help="Model id. If blank, auto-detects from /v1/models")

    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--goal-locations", type=int, default=3)

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)

    ap.add_argument("--max-power", type=int, default=25, help="Clamp motor power")
    ap.add_argument("--max-steer-deg", type=int, default=25, help="Clamp steering servo degrees")

    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=300)

    ap.add_argument("--save-images", default="run_images", help="Dir to save step images, set empty to disable")
    ap.add_argument("--save", default="run_log.json", help="Save results JSON to this file, set empty to disable")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Normalize save_images flag
    if args.save_images and args.save_images.strip():
        ensure_dir(args.save_images)
    else:
        args.save_images = ""

    result = run_benchmark(args)

    if args.save and args.save.strip():
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[SAVED] {args.save}")


if __name__ == "__main__":
    main()