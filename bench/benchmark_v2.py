#!/usr/bin/env python3
"""
benchmark_v2.py (SCAN-THEN-COMMIT + DYNAMIC STEERING)

What this version adds vs your last run:

1) SCAN then COMMIT driving:
   - The script (controller) pans the camera to LEFT/CENTER/RIGHT, captures 3 images,
     asks the model to choose the best direction, then commits to a longer DRIVE.
   - This produces dynamic steering (steer changes based on scan).

2) Guaranteed observations:
   - Once travel + novelty + spacing gates are met, the controller forces OBSERVE
     (using a dedicated observe-only call) so you don't get stuck in endless driving.

3) Ultrasonic stability:
   - If ultrasonic returns None while moving, the driver reuses the last valid distance
     instead of treating it as unknown every slice.

4) Still autonomy-first obstacle avoidance:
   - LLM desired steer is respected when safe.
   - Caution needs consecutive hits before nudges.
   - Danger does a brief backup and alternates nudge direction.

Run example:
  cd ~/pet-companion
  source venv/bin/activate
  python bench/benchmark_v2.py --host 100.82.181.13 --port 1234 --best-of-two --settle-ms 250 --head-scan --debug-avoid --debug-scan

Dependencies in venv:
  pip install openai pillow
"""

import argparse
import base64
import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from picamera2 import Picamera2
from picarx import Picarx
from PIL import Image, ImageFilter


# -----------------------------
# Utility
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def b64_jpeg_from_path(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# -----------------------------
# Image metrics
# -----------------------------

def image_novelty(a_path: str, b_path: str, size: Tuple[int, int] = (96, 72)) -> float:
    """
    0..1 novelty score based on mean absolute pixel diff of downsampled grayscale images.
    Lower = more similar.
    """
    try:
        a = Image.open(a_path).convert("L").resize(size)
        b = Image.open(b_path).convert("L").resize(size)
        ap = a.tobytes()
        bp = b.tobytes()
        if len(ap) != len(bp) or not ap:
            return 1.0

        diff = 0
        for i in range(len(ap)):
            diff += abs(ap[i] - bp[i])

        return float(diff / (len(ap) * 255.0))
    except Exception:
        return 1.0


def sharpness_score(img_path: str, size: Tuple[int, int] = (320, 240)) -> float:
    """
    Cheap sharpness proxy (edge energy).
    """
    try:
        im = Image.open(img_path).convert("L").resize(size)
        edges = im.filter(ImageFilter.FIND_EDGES)
        px = edges.tobytes()
        if not px:
            return 0.0
        return float(sum(px) / len(px))
    except Exception:
        return 0.0


# -----------------------------
# Ultrasonic reading (official)
# -----------------------------

def read_distance_cm(px: Picarx) -> Optional[float]:
    """
    Official tutorial approach: px.ultrasonic.read()
    """
    try:
        if hasattr(px, "ultrasonic") and hasattr(px.ultrasonic, "read"):
            d = float(px.ultrasonic.read())
            if 0.5 < d < 600.0:
                return d
    except Exception:
        return None
    return None


def read_distance_cm_median(px: Picarx, samples: int = 7, delay_s: float = 0.010) -> Optional[float]:
    vals: List[float] = []
    for _ in range(samples):
        d = read_distance_cm(px)
        if d is not None:
            vals.append(d)
        time.sleep(delay_s)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


# -----------------------------
# Camera
# -----------------------------

def init_camera(width: int, height: int, lock_exposure: bool, exposure_us: int, gain: float) -> Picamera2:
    cam = Picamera2()
    cfg = cam.create_preview_configuration(main={"format": "BGR888", "size": (width, height)})
    cam.configure(cfg)
    cam.start()
    time.sleep(0.35)

    if lock_exposure:
        try:
            cam.set_controls(
                {
                    "AeEnable": True,
                    "ExposureTime": int(clamp(exposure_us, 1000, 20000)),
                    "AnalogueGain": float(clamp(gain, 1.0, 16.0)),
                }
            )
        except Exception:
            pass

    return cam


def capture_image(cam: Picamera2, out_path: str, best_of_two: bool) -> Tuple[bool, float, str]:
    """
    Returns (ok, sharpness, note). Also verifies file existence.
    """
    try:
        tmp_path = out_path.replace(".jpg", "_tmp.jpg")

        if best_of_two:
            cam.capture_file(out_path)
            time.sleep(0.05)
            cam.capture_file(tmp_path)

            if not os.path.exists(out_path):
                return False, 0.0, "capture_missing_primary"
            if not os.path.exists(tmp_path):
                s1 = sharpness_score(out_path)
                return True, s1, "best_of_two_tmp_missing"

            s1 = sharpness_score(out_path)
            s2 = sharpness_score(tmp_path)

            if s2 > s1:
                os.replace(tmp_path, out_path)
                return True, s2, "best_of_two_used_tmp"
            else:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return True, s1, "best_of_two_kept_primary"
        else:
            cam.capture_file(out_path)
            if not os.path.exists(out_path):
                return False, 0.0, "capture_missing_primary"
            s = sharpness_score(out_path)
            return True, s, "single"

    except Exception as e:
        return False, 0.0, f"capture_error: {e}"


# -----------------------------
# Robot Rig (movement + head)
# -----------------------------

class RobotRig:
    def __init__(self, px: Picarx, max_power: int, max_steer_deg: int):
        self.px = px
        self.max_power = clamp_int(max_power, 10, 90)
        self.max_steer_deg = clamp_int(max_steer_deg, 10, 35)

        self.pan = 0
        self.tilt = 0
        self.head_limit = 35
        self.look_settle_s = 0.18

        self.avoid_left_next = True

    def stop(self) -> None:
        try:
            self.px.stop()
        except Exception:
            pass
        try:
            self.px.set_dir_servo_angle(0)
        except Exception:
            pass

    def set_steer(self, deg: float) -> None:
        deg = float(clamp(deg, -self.max_steer_deg, self.max_steer_deg))
        try:
            self.px.set_dir_servo_angle(deg)
        except Exception:
            pass

    def apply_head(self, pan: int, tilt: int) -> None:
        pan = clamp_int(pan, -self.head_limit, self.head_limit)
        tilt = clamp_int(tilt, -self.head_limit, self.head_limit)

        try:
            if hasattr(self.px, "set_cam_pan_angle") and hasattr(self.px, "set_cam_tilt_angle"):
                self.px.set_cam_pan_angle(pan)
                self.px.set_cam_tilt_angle(tilt)
            else:
                if hasattr(self.px, "set_camera_servo1_angle"):
                    self.px.set_camera_servo1_angle(pan)
                if hasattr(self.px, "set_camera_servo2_angle"):
                    self.px.set_camera_servo2_angle(tilt)
            self.pan = pan
            self.tilt = tilt
        except Exception:
            self.pan = pan
            self.tilt = tilt

    def look_center(self) -> None:
        self.apply_head(0, 0)
        time.sleep(self.look_settle_s)

    def head_scan_tick(self, step: int, enabled: bool) -> None:
        if not enabled:
            return
        if step % 4 != 0:
            return
        for pan in (-22, 0, 22, 0):
            self.apply_head(pan, self.tilt)
            time.sleep(self.look_settle_s)

    def drive_with_autonomy_first_avoidance(
        self,
        power: int,
        requested_steer: float,
        total_ms: int,
        safe_cm: float,
        danger_cm: float,
        avoid_turn_deg: float,
        slice_ms: int = 90,
        median_samples: int = 7,
        confirm_slices: int = 3,
        exit_margin_cm: float = 10.0,
        debug: bool = False,
    ) -> Tuple[bool, Optional[float], str]:
        power = clamp_int(power, -self.max_power, self.max_power)
        total_ms = clamp_int(total_ms, 100, 20000)
        slice_ms = clamp_int(slice_ms, 60, 160)
        avoid_turn_deg = float(clamp(avoid_turn_deg, 10, self.max_steer_deg))
        requested_steer = float(clamp(requested_steer, -self.max_steer_deg, self.max_steer_deg))
        confirm_slices = clamp_int(confirm_slices, 1, 6)
        exit_margin_cm = float(clamp(exit_margin_cm, 0.0, 40.0))

        t_end = time.time() + (total_ms / 1000.0)
        last_cm: Optional[float] = None
        last_valid_d: Optional[float] = None

        completed = True
        note = "ok"

        too_close_streak = 0
        in_avoid_mode = False

        try:
            while time.time() < t_end:
                d = read_distance_cm_median(self.px, samples=median_samples, delay_s=0.010)

                if d is not None:
                    last_valid_d = d
                else:
                    d = last_valid_d

                last_cm = d

                # If we still never got a valid reading, keep going cautiously
                if d is None:
                    note = "dist_none"
                    self.set_steer(requested_steer)
                    if power > 0:
                        self.px.forward(max(12, int(power * 0.6)))
                    elif power < 0:
                        self.px.backward(max(12, int(abs(power) * 0.6)))
                    else:
                        self.stop()
                        return True, last_cm, note
                    time.sleep(slice_ms / 1000.0)
                    if debug:
                        print(f"[AVOID] d=None (no valid yet) steer={requested_steer:.1f} mode=dist_none")
                    continue

                # Reverse requested
                if power < 0:
                    note = "reverse"
                    self.set_steer(requested_steer)
                    self.px.backward(abs(power))
                    time.sleep(slice_ms / 1000.0)
                    if debug:
                        print(f"[AVOID] d={d:6.1f} steer={requested_steer:.1f} mode={note}")
                    continue

                # Update streak / avoid mode
                if d < safe_cm:
                    too_close_streak += 1
                else:
                    too_close_streak = 0

                if not in_avoid_mode and too_close_streak >= confirm_slices:
                    in_avoid_mode = True

                if in_avoid_mode and d >= (safe_cm + exit_margin_cm):
                    in_avoid_mode = False
                    too_close_streak = 0

                # HARD override: danger
                if d < danger_cm:
                    completed = False
                    note = "danger_backup"
                    turn = -avoid_turn_deg if self.avoid_left_next else +avoid_turn_deg
                    self.avoid_left_next = not self.avoid_left_next
                    self.set_steer(turn)
                    self.px.backward(power)
                    time.sleep(0.25)
                    if debug:
                        print(f"[AVOID] d={d:6.1f} streak={too_close_streak} avoid={in_avoid_mode} mode={note} turn={turn:+.1f}")
                    continue

                # SOFT zone: caution
                if d < safe_cm:
                    completed = False

                    if too_close_streak < confirm_slices:
                        note = "caution_llm_control"
                        self.set_steer(requested_steer)
                        self.px.forward(max(12, int(power * 0.65)))
                        time.sleep(slice_ms / 1000.0)
                        if debug:
                            print(f"[AVOID] d={d:6.1f} streak={too_close_streak} avoid={in_avoid_mode} mode={note} steer={requested_steer:+.1f}")
                        continue

                    note = "caution_nudge"
                    turn = +avoid_turn_deg if self.avoid_left_next else -avoid_turn_deg
                    self.avoid_left_next = not self.avoid_left_next
                    self.set_steer(turn)
                    self.px.forward(max(12, int(power * 0.70)))
                    time.sleep(slice_ms / 1000.0)
                    too_close_streak = max(0, confirm_slices - 1)

                    if debug:
                        print(f"[AVOID] d={d:6.1f} streak={too_close_streak} avoid={in_avoid_mode} mode={note} turn={turn:+.1f}")
                    continue

                # SAFE: model control
                note = "safe_forward"
                self.set_steer(requested_steer)
                self.px.forward(power)
                time.sleep(slice_ms / 1000.0)
                if debug:
                    print(f"[AVOID] d={d:6.1f} streak={too_close_streak} avoid={in_avoid_mode} mode={note} steer={requested_steer:+.1f}")

            return completed, last_cm, note

        finally:
            self.stop()


# -----------------------------
# LLM prompts
# -----------------------------

SYSTEM_PROMPT = """
You control a PiCar-X robot exploring an indoor environment using a forward camera.

Goal: make 3 observations from clearly different parts of the room.
You must TRAVEL FAR between observations so each is from a different side/area.

Output JSON only. No extra text.

Allowed actions:

DRIVE:
{
  "action": "DRIVE",
  "thinking": "short reasoning",
  "drive": { "power": <int 20..70>, "steer_deg": <float -18..18>, "ms": <int 1500..9000> }
}

LOOK:
{
  "action": "LOOK",
  "thinking": "short reasoning",
  "look": { "pan": <int -35..35>, "tilt": <int -35..35> }
}

OBSERVE:
{
  "action": "OBSERVE",
  "thinking": "short reasoning",
  "observe": { "label": "<short>", "description": "<1-2 sentences>" }
}

STOP:
{ "action": "STOP", "thinking": "short reasoning" }

Rules:
- Between observations, choose longer DRIVE segments to reach new viewpoints (ms 4000-9000) with small steering (abs(steer_deg) <= 12).
- The controller handles obstacle avoidance; still avoid obviously blocked paths.
- Do not OBSERVE repeatedly from the same area; only when the view is clearly new.
""".strip()

OBSERVE_ONLY_PROMPT = """
Return JSON ONLY:
{ "label": "<short location name>", "description": "<1-2 sentences describing what is distinct about this location>" }
No extra text.
""".strip()

SCAN_PROMPT = """
You are given 3 images taken with the robot's camera:
- LEFT: camera panned left
- CENTER: camera centered
- RIGHT: camera panned right

Choose the best direction to drive next to explore the room.
Criteria:
- Prefer open space (less clutter, farther visible floor).
- Prefer novelty (looks like it leads to a different area).
- Avoid tight areas with chair legs or obstacles close to the camera.

Return JSON ONLY, one of:
{
  "best_direction": "left" | "center" | "right",
  "confidence": <float 0..1>,
  "reason": "<short>",
  "drive_ms": <int 3500..9000>
}
No extra text.
""".strip()


def build_user_prompt(
    step: int,
    max_steps: int,
    goal_locations: int,
    observations: List[Dict[str, str]],
    front_dist_cm: Optional[float],
    novelty: float,
    pan: int,
    tilt: int,
    travel_ms_since_obs: int,
    drives_since_obs: int,
    req_travel_ms: int,
    req_drives: int,
    min_novelty: float,
    min_steps_between_observe: int,
    steps_since_obs: int,
) -> str:
    obs_lines = "\n".join([f"- {o['label']}: {o['description']}" for o in observations[-5:]]) or "(none yet)"
    dist_str = "unknown" if front_dist_cm is None else f"{front_dist_cm:.1f}"

    return f"""
Step {step}/{max_steps} | Observations {len(observations)}/{goal_locations}
Head: pan={pan}, tilt={tilt}
Front distance: {dist_str} cm
Novelty vs last accepted observation: {novelty:.3f}

Recent observations:
{obs_lines}

Controller gates for accepting OBSERVE:
- novelty >= {min_novelty:.2f}
- steps since last accepted OBSERVE >= {min_steps_between_observe} (currently {steps_since_obs})
- FAR TRAVEL REQUIRED between accepted observations:
    * drive time since last OBSERVE >= {req_travel_ms} ms (currently {travel_ms_since_obs})
    * DRIVE actions since last OBSERVE >= {req_drives} (currently {drives_since_obs})

Guidance:
- If travel requirement not met, choose a longer DRIVE (ms 4000-9000) with small steering.
- If you are in the same area, do not OBSERVE yet.

Decide next action now. Output JSON only.
""".strip()


# -----------------------------
# LLM calls
# -----------------------------

def autodetect_model(client: OpenAI) -> str:
    models = client.models.list()
    if hasattr(models, "data") and models.data:
        return models.data[0].id
    return "unknown-model"


def llm_chat_json(
    client: OpenAI,
    model: str,
    user_text: str,
    image_b64: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[float, str, Dict[str, Any]]:
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
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ],
    )
    latency = time.time() - t0

    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        pass

    obj = extract_first_json_obj(text)
    if obj is None:
        obj = {"action": "STOP", "thinking": "Invalid JSON output; stopping for safety."}

    thinking = str(obj.get("thinking", ""))[:400]
    return latency, thinking, obj


def llm_observe_json(
    client: OpenAI,
    model: str,
    image_b64: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[float, Dict[str, Any]]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "Output compact JSON only. No extra text."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OBSERVE_ONLY_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ],
    )
    latency = time.time() - t0

    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        pass

    obj = extract_first_json_obj(text) or {}
    label = str(obj.get("label", "Location")).strip()[:40] or "Location"
    desc = str(obj.get("description", "")).strip()[:260] or "No description provided."
    return latency, {"label": label, "description": desc}


def llm_scan_choose_direction(
    client: OpenAI,
    model: str,
    left_b64: str,
    center_b64: str,
    right_b64: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[float, Dict[str, Any], str]:
    """
    Returns (latency, decision_json, raw_text_snippet).
    decision_json keys:
      best_direction: left|center|right
      confidence: 0..1
      reason: str
      drive_ms: int
    """
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "Output JSON only. No extra text."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SCAN_PROMPT},
                    {"type": "text", "text": "LEFT image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{left_b64}"}},
                    {"type": "text", "text": "CENTER image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{center_b64}"}},
                    {"type": "text", "text": "RIGHT image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{right_b64}"}},
                ],
            },
        ],
    )
    latency = time.time() - t0

    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        pass

    obj = extract_first_json_obj(text) or {}
    snippet = (text or "")[:220]
    return latency, obj, snippet


# -----------------------------
# Action normalization
# -----------------------------

def normalize_action(raw: Dict[str, Any], max_power: int, max_steer_deg: int, drive_ms_min: int, drive_ms_max: int) -> Dict[str, Any]:
    action = str(raw.get("action", "STOP")).upper().strip()
    thinking = str(raw.get("thinking", ""))

    if action == "LOOK":
        look = raw.get("look", {}) if isinstance(raw.get("look", {}), dict) else {}
        pan = clamp_int(int(look.get("pan", 0)), -35, 35)
        tilt = clamp_int(int(look.get("tilt", 0)), -35, 35)
        return {"action": "LOOK", "thinking": thinking, "look": {"pan": pan, "tilt": tilt}}

    if action == "DRIVE":
        drive = raw.get("drive", {}) if isinstance(raw.get("drive", {}), dict) else {}
        power = clamp_int(int(drive.get("power", int(max_power * 0.75))), 20, max_power)
        steer_deg = float(clamp(float(drive.get("steer_deg", 0.0)), -max_steer_deg, max_steer_deg))
        ms = clamp_int(int(drive.get("ms", drive_ms_max)), drive_ms_min, drive_ms_max)
        return {"action": "DRIVE", "thinking": thinking, "drive": {"power": power, "steer_deg": steer_deg, "ms": ms}}

    if action == "OBSERVE":
        obs = raw.get("observe", {}) if isinstance(raw.get("observe", {}), dict) else {}
        label = str(obs.get("label", "Location")).strip()[:40] or "Location"
        desc = str(obs.get("description", "")).strip()[:260] or "No description provided."
        return {"action": "OBSERVE", "thinking": thinking, "observe": {"label": label, "description": desc}}

    return {"action": "STOP", "thinking": thinking}


# -----------------------------
# SCAN-THEN-COMMIT decision mapping
# -----------------------------

def direction_to_steer(best_direction: str, confidence: float, max_steer_deg: int) -> float:
    """
    Map direction + confidence to a steer angle.
    Produces dynamic steering, not a constant -10.

    center: 0
    left/right: angle increases with confidence (but capped).
    """
    best_direction = (best_direction or "").strip().lower()
    confidence = float(clamp(confidence, 0.0, 1.0))

    if best_direction == "center":
        return 0.0
    if best_direction not in ("left", "right"):
        # unknown: slight random wander
        return float(random.choice([-3.0, 0.0, 3.0]))

    # base + variable component
    base = 3.0
    span = min(12.0, float(max_steer_deg))
    angle = base + (span - base) * confidence

    if best_direction == "left":
        return float(-clamp(angle, 2.0, float(max_steer_deg)))
    return float(clamp(angle, 2.0, float(max_steer_deg)))


def normalize_scan_decision(raw: Dict[str, Any], drive_ms_min: int, drive_ms_max: int) -> Dict[str, Any]:
    d = str(raw.get("best_direction", "center")).strip().lower()
    if d not in ("left", "center", "right"):
        d = "center"

    conf = raw.get("confidence", 0.55)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.55
    conf = float(clamp(conf, 0.0, 1.0))

    reason = str(raw.get("reason", "")).strip()[:200]
    ms = raw.get("drive_ms", 6000)
    try:
        ms = int(ms)
    except Exception:
        ms = 6000
    ms = clamp_int(ms, max(1500, drive_ms_min), drive_ms_max)

    return {"best_direction": d, "confidence": conf, "reason": reason, "drive_ms": ms}


# -----------------------------
# Logging
# -----------------------------

@dataclass
class StepLog:
    step: int
    latency_s: float
    model_thinking: str
    action: Dict[str, Any]
    front_dist_cm: Optional[float]
    novelty: float
    sharpness: float
    capture_note: str
    pan: int
    tilt: int
    travel_ms_since_obs: int
    drives_since_obs: int
    obs_accepted: bool
    obs_rejected_reason: str
    drive_note: str
    image_path: str
    timestamp: float

    scan_used: bool
    scan_latency_s: float
    scan_decision: Dict[str, Any]
    scan_steer_deg: float
    scan_drive_ms: int


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--host", required=True, help="LM Studio host (use Tailscale IP like 100.x.x.x)")
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--api-key", default="lm-studio")
    ap.add_argument("--model", default="", help="If blank, auto-detect from /v1/models")

    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--goal-locations", type=int, default=3)

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)

    ap.add_argument("--max-power", type=int, default=45)
    ap.add_argument("--max-steer-deg", type=int, default=18)

    ap.add_argument("--drive-ms-min", type=int, default=2500)
    ap.add_argument("--drive-ms-max", type=int, default=9000)

    # Avoidance thresholds (tutorial defaults)
    ap.add_argument("--safe-cm", type=float, default=40.0)
    ap.add_argument("--danger-cm", type=float, default=20.0)

    # Avoidance behavior (tuned)
    ap.add_argument("--avoid-turn-deg", type=float, default=18.0)
    ap.add_argument("--avoid-slice-ms", type=int, default=90)
    ap.add_argument("--avoid-median-samples", type=int, default=7)
    ap.add_argument("--avoid-confirm-slices", type=int, default=3)
    ap.add_argument("--avoid-exit-margin-cm", type=float, default=10.0)

    # FAR TRAVEL gating
    ap.add_argument("--req-travel-ms-between-observes", type=int, default=14000)
    ap.add_argument("--req-drives-between-observes", type=int, default=2)

    # Observation gating
    ap.add_argument("--min-steps-between-observe", type=int, default=5)
    ap.add_argument("--min-novelty", type=float, default=0.12)

    # Capture stability
    ap.add_argument("--settle-ms", type=int, default=250)
    ap.add_argument("--best-of-two", action="store_true")
    ap.add_argument("--lock-exposure", action="store_true")
    ap.add_argument("--exposure-us", type=int, default=6000)
    ap.add_argument("--gain", type=float, default=6.0)

    # Head scan
    ap.add_argument("--head-scan", action="store_true", help="Periodic head scan for proof of pan/tilt activity")

    # SCAN-THEN-COMMIT
    ap.add_argument("--scan-enabled", action="store_true", help="Use scan then commit for dynamic steering")
    ap.add_argument("--scan-pan-deg", type=int, default=25, help="Pan degrees for left/right scan (suggest 20-30)")
    ap.add_argument("--scan-tilt-deg", type=int, default=0, help="Tilt degrees for scan (usually 0)")
    ap.add_argument("--scan-settle-ms", type=int, default=220, help="Settle time after pan move before capture")
    ap.add_argument("--scan-temperature", type=float, default=0.2)
    ap.add_argument("--scan-max-tokens", type=int, default=220)

    # LLM params
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=420)

    # Debug
    ap.add_argument("--debug-avoid", action="store_true")
    ap.add_argument("--debug-scan", action="store_true")

    # Outputs
    ap.add_argument("--save-images", default="run_images_benchmark_v2")
    ap.add_argument("--save", default="run_log_benchmark_v2.json")

    return ap.parse_args()


# -----------------------------
# Scan helper
# -----------------------------

def scan_then_choose(
    rig: RobotRig,
    cam: Picamera2,
    client: OpenAI,
    model: str,
    base_dir: str,
    step: int,
    best_of_two: bool,
    scan_pan_deg: int,
    scan_tilt_deg: int,
    settle_ms: int,
    drive_ms_min: int,
    drive_ms_max: int,
    max_steer_deg: int,
    temperature: float,
    max_tokens: int,
    debug: bool,
) -> Tuple[bool, float, Dict[str, Any], float, int, Dict[str, str]]:
    """
    Returns:
      ok, scan_latency, scan_decision_norm, steer_deg, drive_ms, paths dict
    """
    scan_pan_deg = clamp_int(scan_pan_deg, 10, 35)
    scan_tilt_deg = clamp_int(scan_tilt_deg, -20, 20)
    settle_s = clamp_int(settle_ms, 80, 900) / 1000.0

    paths = {
        "left": os.path.join(base_dir, f"step_{step:03d}_scan_left.jpg"),
        "center": os.path.join(base_dir, f"step_{step:03d}_scan_center.jpg"),
        "right": os.path.join(base_dir, f"step_{step:03d}_scan_right.jpg"),
    }

    rig.stop()

    # LEFT
    rig.apply_head(-scan_pan_deg, scan_tilt_deg)
    time.sleep(settle_s)
    ok_l, _, note_l = capture_image(cam, paths["left"], best_of_two)

    # CENTER
    rig.apply_head(0, scan_tilt_deg)
    time.sleep(settle_s)
    ok_c, _, note_c = capture_image(cam, paths["center"], best_of_two)

    # RIGHT
    rig.apply_head(+scan_pan_deg, scan_tilt_deg)
    time.sleep(settle_s)
    ok_r, _, note_r = capture_image(cam, paths["right"], best_of_two)

    # Return head to center for driving
    rig.apply_head(0, 0)
    time.sleep(rig.look_settle_s)

    if debug:
        print(f"[SCAN] captures: left={ok_l}({note_l}) center={ok_c}({note_c}) right={ok_r}({note_r})")

    if not (ok_l and ok_c and ok_r):
        # If scan capture fails, fall back to no-scan
        return False, 0.0, {}, 0.0, 0, paths

    left_b64 = b64_jpeg_from_path(paths["left"])
    center_b64 = b64_jpeg_from_path(paths["center"])
    right_b64 = b64_jpeg_from_path(paths["right"])

    scan_latency, raw_decision, raw_snip = llm_scan_choose_direction(
        client=client,
        model=model,
        left_b64=left_b64,
        center_b64=center_b64,
        right_b64=right_b64,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    decision = normalize_scan_decision(raw_decision, drive_ms_min, drive_ms_max)
    steer = direction_to_steer(decision["best_direction"], decision["confidence"], max_steer_deg=max_steer_deg)
    ms = int(decision["drive_ms"])

    if debug:
        print(f"[SCAN] latency={scan_latency:.2f}s decision={decision} steer={steer:+.1f}ms={ms} raw={raw_snip!r}")

    return True, scan_latency, decision, steer, ms, paths


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    random.seed()

    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key=args.api_key)
    model = args.model.strip() or autodetect_model(client)

    ensure_dir(args.save_images)

    px = Picarx()
    rig = RobotRig(px, max_power=args.max_power, max_steer_deg=args.max_steer_deg)
    rig.look_center()

    cam = init_camera(args.width, args.height, args.lock_exposure, args.exposure_us, args.gain)
    print("[CAMERA] ready")
    print(f"Endpoint: {base_url}")
    print(f"Model: {model}")

    observations: List[Dict[str, str]] = []
    step_logs: List[StepLog] = []

    last_observe_step = 0
    last_observe_image_path: Optional[str] = None

    # Travel counters since last accepted observation
    travel_ms_since_obs = 10**9
    drives_since_obs = 10**9

    settle_s = clamp_int(args.settle_ms, 0, 2000) / 1000.0

    try:
        for step in range(1, args.max_steps + 1):
            if len(observations) >= args.goal_locations:
                print(f"[DONE] Reached goal {len(observations)}/{args.goal_locations}")
                break

            # Ensure motors stopped before capture
            rig.stop()
            rig.head_scan_tick(step, enabled=args.head_scan)
            time.sleep(settle_s)

            # Main step capture (center, neutral head)
            rig.apply_head(0, 0)
            time.sleep(rig.look_settle_s)

            img_path = os.path.join(args.save_images, f"step_{step:03d}.jpg")
            ok_cap, sharp, cap_note = capture_image(cam, img_path, best_of_two=args.best_of_two)
            if not ok_cap:
                print(f"[CAPTURE ERROR] step={step} note={cap_note}")
                break

            img_b64 = b64_jpeg_from_path(img_path)

            front_cm = read_distance_cm_median(px, samples=7, delay_s=0.010)

            novelty = 1.0 if last_observe_image_path is None else image_novelty(img_path, last_observe_image_path)

            steps_since_obs = step - last_observe_step

            req_ms = args.req_travel_ms_between_observes
            req_drives = args.req_drives_between_observes

            travel_ms_display = 0 if travel_ms_since_obs > 10**8 else travel_ms_since_obs
            drives_display = 0 if drives_since_obs > 10**8 else drives_since_obs

            need_travel = (travel_ms_display < req_ms) or (drives_display < req_drives)

            # OBSERVE gates
            gates_met_for_observe = (
                (not need_travel) and
                (steps_since_obs >= args.min_steps_between_observe) and
                (novelty >= args.min_novelty)
            )

            # If observe gates are met, force OBSERVE (guarantees observations happen)
            latency = 0.0
            thinking = ""
            action: Dict[str, Any] = {"action": "STOP", "thinking": ""}

            scan_used = False
            scan_latency = 0.0
            scan_decision: Dict[str, Any] = {}
            scan_steer_deg = 0.0
            scan_drive_ms = 0

            obs_accepted = False
            obs_rejected_reason = ""
            drive_note = ""

            if gates_met_for_observe:
                obs_latency, obs = llm_observe_json(
                    client=client,
                    model=model,
                    image_b64=img_b64,
                    temperature=0.2,
                    max_tokens=220,
                )
                latency = obs_latency
                thinking = "Controller: gates met, forcing OBSERVE"
                action = {"action": "OBSERVE", "thinking": thinking, "observe": obs}
            else:
                # If travel needed and scan enabled, use scan then commit
                if args.scan_enabled and need_travel:
                    scan_used, scan_latency, scan_decision, scan_steer_deg, scan_drive_ms, _scan_paths = scan_then_choose(
                        rig=rig,
                        cam=cam,
                        client=client,
                        model=model,
                        base_dir=args.save_images,
                        step=step,
                        best_of_two=args.best_of_two,
                        scan_pan_deg=args.scan_pan_deg,
                        scan_tilt_deg=args.scan_tilt_deg,
                        settle_ms=args.scan_settle_ms,
                        drive_ms_min=args.drive_ms_min,
                        drive_ms_max=args.drive_ms_max,
                        max_steer_deg=args.max_steer_deg,
                        temperature=args.scan_temperature,
                        max_tokens=args.scan_max_tokens,
                        debug=args.debug_scan,
                    )

                    if scan_used:
                        # Create a DRIVE action from scan decision
                        action = {
                            "action": "DRIVE",
                            "thinking": f"SCAN then COMMIT: {scan_decision.get('best_direction')} (conf {scan_decision.get('confidence')})",
                            "drive": {
                                "power": int(args.max_power * 0.85),
                                "steer_deg": float(clamp(scan_steer_deg, -args.max_steer_deg, args.max_steer_deg)),
                                "ms": int(scan_drive_ms),
                            },
                        }
                        latency = scan_latency
                        thinking = action["thinking"]
                    else:
                        # Scan failed, fall back to normal LLM control
                        user_prompt = build_user_prompt(
                            step=step,
                            max_steps=args.max_steps,
                            goal_locations=args.goal_locations,
                            observations=observations,
                            front_dist_cm=front_cm,
                            novelty=novelty,
                            pan=rig.pan,
                            tilt=rig.tilt,
                            travel_ms_since_obs=travel_ms_display,
                            drives_since_obs=drives_display,
                            req_travel_ms=req_ms,
                            req_drives=req_drives,
                            min_novelty=args.min_novelty,
                            min_steps_between_observe=args.min_steps_between_observe,
                            steps_since_obs=steps_since_obs,
                        )
                        latency, thinking, raw = llm_chat_json(
                            client=client,
                            model=model,
                            user_text=user_prompt,
                            image_b64=img_b64,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                        action = normalize_action(raw, args.max_power, args.max_steer_deg, args.drive_ms_min, args.drive_ms_max)

                else:
                    # Normal LLM control
                    user_prompt = build_user_prompt(
                        step=step,
                        max_steps=args.max_steps,
                        goal_locations=args.goal_locations,
                        observations=observations,
                        front_dist_cm=front_cm,
                        novelty=novelty,
                        pan=rig.pan,
                        tilt=rig.tilt,
                        travel_ms_since_obs=travel_ms_display,
                        drives_since_obs=drives_display,
                        req_travel_ms=req_ms,
                        req_drives=req_drives,
                        min_novelty=args.min_novelty,
                        min_steps_between_observe=args.min_steps_between_observe,
                        steps_since_obs=steps_since_obs,
                    )
                    latency, thinking, raw = llm_chat_json(
                        client=client,
                        model=model,
                        user_text=user_prompt,
                        image_b64=img_b64,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    action = normalize_action(raw, args.max_power, args.max_steer_deg, args.drive_ms_min, args.drive_ms_max)

                # Gate OBSERVE if model attempts it too early
                if action["action"] == "OBSERVE":
                    too_soon = steps_since_obs < args.min_steps_between_observe
                    not_novel = novelty < args.min_novelty
                    not_traveled = need_travel

                    if too_soon or not_novel or not_traveled:
                        reasons = []
                        if too_soon:
                            reasons.append(f"too soon ({steps_since_obs}<{args.min_steps_between_observe})")
                        if not_novel:
                            reasons.append(f"not novel ({novelty:.3f}<{args.min_novelty})")
                        if not_traveled:
                            reasons.append(f"not traveled (ms {travel_ms_display}<{req_ms} or drives {drives_display}<{req_drives})")
                        obs_rejected_reason = ", ".join(reasons)

                        # Force travel. Use scan if enabled, otherwise mild wander.
                        if args.scan_enabled:
                            # quick scan-based drive
                            scan_used, scan_latency, scan_decision, scan_steer_deg, scan_drive_ms, _ = scan_then_choose(
                                rig=rig,
                                cam=cam,
                                client=client,
                                model=model,
                                base_dir=args.save_images,
                                step=step,
                                best_of_two=args.best_of_two,
                                scan_pan_deg=args.scan_pan_deg,
                                scan_tilt_deg=args.scan_tilt_deg,
                                settle_ms=args.scan_settle_ms,
                                drive_ms_min=args.drive_ms_min,
                                drive_ms_max=args.drive_ms_max,
                                max_steer_deg=args.max_steer_deg,
                                temperature=args.scan_temperature,
                                max_tokens=args.scan_max_tokens,
                                debug=args.debug_scan,
                            )
                            if scan_used:
                                action = {
                                    "action": "DRIVE",
                                    "thinking": "Controller: OBSERVE gated, scan then commit travel",
                                    "drive": {"power": int(args.max_power * 0.85), "steer_deg": scan_steer_deg, "ms": scan_drive_ms},
                                }
                            else:
                                action = {
                                    "action": "DRIVE",
                                    "thinking": "Controller: OBSERVE gated, forcing travel",
                                    "drive": {
                                        "power": int(args.max_power * 0.85),
                                        "steer_deg": float(random.choice([-6.0, -3.0, 0.0, 3.0, 6.0])),
                                        "ms": int(args.drive_ms_max),
                                    },
                                }
                        else:
                            action = {
                                "action": "DRIVE",
                                "thinking": "Controller: OBSERVE gated, forcing travel",
                                "drive": {
                                    "power": int(args.max_power * 0.85),
                                    "steer_deg": float(random.choice([-6.0, -3.0, 0.0, 3.0, 6.0])),
                                    "ms": int(args.drive_ms_max),
                                },
                            }

                # If travel still needed and action is not DRIVE, force a travel DRIVE
                if need_travel and action["action"] != "DRIVE":
                    if args.scan_enabled:
                        scan_used, scan_latency, scan_decision, scan_steer_deg, scan_drive_ms, _ = scan_then_choose(
                            rig=rig,
                            cam=cam,
                            client=client,
                            model=model,
                            base_dir=args.save_images,
                            step=step,
                            best_of_two=args.best_of_two,
                            scan_pan_deg=args.scan_pan_deg,
                            scan_tilt_deg=args.scan_tilt_deg,
                            settle_ms=args.scan_settle_ms,
                            drive_ms_min=args.drive_ms_min,
                            drive_ms_max=args.drive_ms_max,
                            max_steer_deg=args.max_steer_deg,
                            temperature=args.scan_temperature,
                            max_tokens=args.scan_max_tokens,
                            debug=args.debug_scan,
                        )
                        if scan_used:
                            action = {
                                "action": "DRIVE",
                                "thinking": "Controller: travel required, scan then commit",
                                "drive": {"power": int(args.max_power * 0.85), "steer_deg": scan_steer_deg, "ms": scan_drive_ms},
                            }
                        else:
                            action = {
                                "action": "DRIVE",
                                "thinking": "Controller: travel required, forcing drive",
                                "drive": {
                                    "power": int(args.max_power * 0.85),
                                    "steer_deg": float(random.choice([-6.0, -3.0, 0.0, 3.0, 6.0])),
                                    "ms": int(args.drive_ms_max),
                                },
                            }
                    else:
                        action = {
                            "action": "DRIVE",
                            "thinking": "Controller: travel required, forcing drive",
                            "drive": {
                                "power": int(args.max_power * 0.85),
                                "steer_deg": float(random.choice([-6.0, -3.0, 0.0, 3.0, 6.0])),
                                "ms": int(args.drive_ms_max),
                            },
                        }

            dist_str = "?" if front_cm is None else f"{front_cm:.1f}"
            print(
                f"[STEP {step}/{args.max_steps}] obs {len(observations)}/{args.goal_locations} "
                f"lat {latency:.2f}s front={dist_str}cm nov={novelty:.3f} sharp={sharp:.1f} "
                f"travel_ms={travel_ms_display} drives={drives_display} head(pan={rig.pan},tilt={rig.tilt}) "
                f"scan={'Y' if scan_used else 'N'}"
            )
            if thinking:
                print(f"  thinking: {thinking[:180]}")
            if obs_rejected_reason:
                print(f"  observe_gate: {obs_rejected_reason}")
            print(f"  capture: {cap_note}")
            if scan_used and args.debug_scan:
                print(f"  scan_decision: {scan_decision} steer={scan_steer_deg:+.1f} drive_ms={scan_drive_ms}")

            # Execute action
            try:
                if action["action"] == "LOOK":
                    pan = int(action["look"]["pan"])
                    tilt = int(action["look"]["tilt"])
                    rig.apply_head(pan, tilt)
                    time.sleep(rig.look_settle_s)

                elif action["action"] == "DRIVE":
                    d = action["drive"]
                    power = clamp_int(int(d["power"]), 20, args.max_power)
                    steer = float(clamp(float(d["steer_deg"]), -args.max_steer_deg, args.max_steer_deg))
                    ms = clamp_int(int(d["ms"]), args.drive_ms_min, args.drive_ms_max)

                    completed, last_d, drive_note = rig.drive_with_autonomy_first_avoidance(
                        power=power,
                        requested_steer=steer,
                        total_ms=ms,
                        safe_cm=args.safe_cm,
                        danger_cm=args.danger_cm,
                        avoid_turn_deg=args.avoid_turn_deg,
                        slice_ms=args.avoid_slice_ms,
                        median_samples=args.avoid_median_samples,
                        confirm_slices=args.avoid_confirm_slices,
                        exit_margin_cm=args.avoid_exit_margin_cm,
                        debug=args.debug_avoid,
                    )

                    if travel_ms_since_obs > 10**8:
                        travel_ms_since_obs = ms
                        drives_since_obs = 1
                    else:
                        travel_ms_since_obs += ms
                        drives_since_obs += 1

                elif action["action"] == "OBSERVE":
                    o = action["observe"]
                    observations.append({"label": o["label"], "description": o["description"]})
                    last_observe_step = step
                    last_observe_image_path = img_path
                    obs_accepted = True

                    travel_ms_since_obs = 0
                    drives_since_obs = 0

                    print(f"  OBSERVATION {len(observations)}: {o['label']}: {o['description']}")
                    time.sleep(0.25)

                else:
                    rig.stop()
                    time.sleep(0.2)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[WARN] execution error: {e}")
                rig.stop()

            # Log step
            step_logs.append(
                StepLog(
                    step=step,
                    latency_s=latency,
                    model_thinking=thinking,
                    action=action,
                    front_dist_cm=front_cm,
                    novelty=novelty,
                    sharpness=sharp,
                    capture_note=cap_note,
                    pan=rig.pan,
                    tilt=rig.tilt,
                    travel_ms_since_obs=0 if travel_ms_since_obs > 10**8 else travel_ms_since_obs,
                    drives_since_obs=0 if drives_since_obs > 10**8 else drives_since_obs,
                    obs_accepted=obs_accepted,
                    obs_rejected_reason=obs_rejected_reason,
                    drive_note=drive_note,
                    image_path=img_path,
                    timestamp=time.time(),
                    scan_used=scan_used,
                    scan_latency_s=scan_latency,
                    scan_decision=scan_decision,
                    scan_steer_deg=scan_steer_deg,
                    scan_drive_ms=scan_drive_ms,
                )
            )

        result = {
            "model": model,
            "endpoint": base_url,
            "args": vars(args),
            "observations": observations,
            "steps": [asdict(s) for s in step_logs],
        }

        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[SAVED] {args.save}")

    finally:
        try:
            rig.apply_head(0, 0)
            rig.stop()
        except Exception:
            pass
        try:
            cam.stop()
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    main()
