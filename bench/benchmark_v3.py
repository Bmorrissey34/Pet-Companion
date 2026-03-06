#!/usr/bin/env python3
"""
PiCar-X Pet Monitor benchmark_v3: scan then commit

- Scans camera right/center/left (or configurable order)
- Pauses after each servo move (pan settle) to avoid motion blur
- Captures images at each pan angle (best-of-two option)
- Asks VL model (LM Studio OpenAI-compatible API) which direction is best
- Commits to that direction for a longer drive segment
- Adds practical ultrasonic avoidance tuning
- Gates observations by steps, travel distance, and novelty

Assumptions:
- PiCar-X v20 with picarx module available
- Camera via Picamera2
- LM Studio OpenAI compatible endpoint at http://<host>:<port>/v1
- openai python SDK installed in venv
"""

import argparse
import base64
import io
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image
from picamera2 import Picamera2
from openai import OpenAI
from picarx import Picarx


# ----------------------------
# Utility: image helpers
# ----------------------------

def pil_to_jpeg_bytes(img: Image.Image, quality: int = 75) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def b64_data_url(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def image_hash_simple(img: Image.Image, size: int = 16) -> int:
    """
    Simple average-hash style fingerprint.
    Returns an int bitmask with size*size bits.
    """
    g = img.convert("L").resize((size, size))
    pixels = list(g.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p > avg:
            bits |= (1 << i)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------
# Camera capture
# ----------------------------

@dataclass
class CaptureConfig:
    width: int = 640
    height: int = 360
    jpeg_quality: int = 75
    best_of_two: bool = True
    settle_ms: int = 250  # camera pipeline settle (not servo settle)

class Camera:
    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self.picam2 = Picamera2()

        # Fast preview config
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (cfg.width, cfg.height)}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.2)

    def close(self):
        try:
            self.picam2.stop()
        except Exception:
            pass

    def capture_pil(self) -> Image.Image:
        time.sleep(self.cfg.settle_ms / 1000.0)
        arr = self.picam2.capture_array()
        return Image.fromarray(arr)

    def capture_best(self) -> Image.Image:
        if not self.cfg.best_of_two:
            return self.capture_pil()

        img1 = self.capture_pil()
        img2 = self.capture_pil()

        # Pick sharper using a cheap edge metric
        def sharpness(im: Image.Image) -> float:
            g = im.convert("L").resize((160, 90))
            px = list(g.getdata())
            w, h = g.size
            s = 0
            for y in range(h):
                row = px[y*w:(y+1)*w]
                for x in range(w - 1):
                    s += abs(row[x + 1] - row[x])
            return float(s)

        return img1 if sharpness(img1) >= sharpness(img2) else img2


# ----------------------------
# Robot control + avoidance
# ----------------------------

@dataclass
class DriveConfig:
    speed: int = 30
    commit_steps: int = 12
    step_duration_s: float = 0.18
    steer_center: int = 0
    steer_left: int = -18
    steer_right: int = 18
    steer_soft: int = 10

    # avoidance tuning
    avoid_stop_cm: float = 25.0
    avoid_slow_cm: float = 45.0
    backup_time_s: float = 0.45
    turn_time_s: float = 0.55
    turn_steer_angle: int = 28
    random_turn: bool = True

@dataclass
class ScanConfig:
    tilt_angle: int = 0
    pan_angles: Tuple[int, int, int] = (-45, 0, 45)  # (LEFT, CENTER, RIGHT)
    pan_settle_ms: int = 500  # critical: servo settle to prevent blurry scans
    scan_order: Tuple[str, str, str] = ("RIGHT", "CENTER", "LEFT")

@dataclass
class ObserveConfig:
    max_observations: int = 3
    min_steps_between: int = 10
    min_travel_cm: float = 60.0
    novelty_hamming_min: int = 30
    est_cm_per_step: float = 8.0


class Robot:
    def __init__(self, drive_cfg: DriveConfig):
        self.drive_cfg = drive_cfg
        self.px = Picarx()
        self.last_ultra_cm: Optional[float] = None

    def stop(self):
        try:
            self.px.forward(0)
        except Exception:
            pass

    def set_steer(self, angle: int):
        self.px.set_dir_servo_angle(int(angle))

    def set_pan_tilt(self, pan: Optional[int] = None, tilt: Optional[int] = None):
        # official calls confirmed working
        if tilt is not None:
            self.px.set_cam_tilt_angle(int(tilt))
        if pan is not None:
            self.px.set_cam_pan_angle(int(pan))

    def read_ultrasonic_cm(self) -> float:
        try:
            d = float(self.px.get_distance())
            if math.isnan(d) or d <= 0:
                return 999.0
            self.last_ultra_cm = d
            return d
        except Exception:
            return 999.0

    def forward_for(self, speed: int, steer_angle: int, duration_s: float):
        self.set_steer(steer_angle)
        self.px.forward(int(speed))
        time.sleep(duration_s)

    def backward_for(self, speed: int, steer_angle: int, duration_s: float):
        self.set_steer(steer_angle)
        self.px.backward(int(speed))
        time.sleep(duration_s)

    def avoid_if_needed(self, debug: bool = False) -> bool:
        d = self.read_ultrasonic_cm()

        if d <= self.drive_cfg.avoid_stop_cm:
            if debug:
                print(f"[avoid] STOP: ultrasonic={d:.1f}cm <= {self.drive_cfg.avoid_stop_cm}")
            self.stop()

            # backup straight a bit
            self.backward_for(speed=25, steer_angle=0, duration_s=self.drive_cfg.backup_time_s)
            self.stop()
            time.sleep(0.05)

            # turn maneuver
            turn_dir = random.choice([-1, 1]) if self.drive_cfg.random_turn else 1
            steer = turn_dir * self.drive_cfg.turn_steer_angle
            if debug:
                side = "LEFT" if turn_dir < 0 else "RIGHT"
                print(f"[avoid] TURN {side}: steer={steer}, turn_time={self.drive_cfg.turn_time_s}s")

            self.forward_for(speed=24, steer_angle=steer, duration_s=self.drive_cfg.turn_time_s)
            self.stop()
            time.sleep(0.05)
            return True

        return False

    def speed_for_distance(self, cm: float, base_speed: int) -> int:
        if cm >= self.drive_cfg.avoid_slow_cm:
            return base_speed
        t = (cm - self.drive_cfg.avoid_stop_cm) / max(1e-6, (self.drive_cfg.avoid_slow_cm - self.drive_cfg.avoid_stop_cm))
        t = clamp(t, 0.0, 1.0)
        return int(15 + t * (base_speed - 15))


# ----------------------------
# Model: scan decision + observation
# ----------------------------

def call_vl_direction(
    client: OpenAI,
    model_name: str,
    images: Dict[str, Image.Image],
    debug: bool = False
) -> Tuple[str, str]:
    """
    Ask model to choose LEFT/CENTER/RIGHT.
    Returns (choice, raw_text).
    """
    user_content = [
        {"type": "text", "text": (
            "You are controlling a small robot exploring a room.\n"
            "Choose the best direction to drive next based on the images.\n"
            "Prefer the direction that looks most open and leads somewhere new.\n"
            "Avoid directions that look blocked by chair legs or close obstacles.\n\n"
            "Reply with ONLY one token: LEFT or CENTER or RIGHT.\n"
        )}
    ]

    for key in ["LEFT", "CENTER", "RIGHT"]:
        jpeg = pil_to_jpeg_bytes(images[key], quality=75)
        user_content.append({"type": "text", "text": f"{key} view:"})
        user_content.append({"type": "image_url", "image_url": {"url": b64_data_url(jpeg)}})

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a careful navigation assistant for a robot."},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=8,
    )

    text = (resp.choices[0].message.content or "").strip().upper()
    if debug:
        print(f"[scan] model raw: {text!r}")

    if "LEFT" in text:
        return "LEFT", text
    if "RIGHT" in text:
        return "RIGHT", text
    if "CENTER" in text or "CENTRE" in text:
        return "CENTER", text

    return "CENTER", text


def call_vl_observation(
    client: OpenAI,
    model_name: str,
    img: Image.Image,
    debug: bool = False
) -> str:
    jpeg = pil_to_jpeg_bytes(img, quality=70)
    content = [
        {"type": "text", "text": (
            "Give a short observation (1-2 sentences) describing what you see.\n"
            "Mention obstacles and notable objects.\n"
        )},
        {"type": "image_url", "image_url": {"url": b64_data_url(jpeg)}},
    ]

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You describe images for a robot exploration log."},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
        max_tokens=120,
    )

    text = (resp.choices[0].message.content or "").strip()
    if debug:
        print(f"[obs] {text}")
    return text


# ----------------------------
# Main loop
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--model", default="", help="optional override, else use qwen/qwen3-vl-4b")
    ap.add_argument("--best-of-two", action="store_true")
    ap.add_argument("--settle-ms", type=int, default=250, help="camera settle per capture (ms)")
    ap.add_argument("--pan-settle-ms", type=int, default=500, help="servo settle after pan move (ms)")
    ap.add_argument("--head-scan", action="store_true", help="enable pan scan (recommended)")
    ap.add_argument("--debug-avoid", action="store_true")
    ap.add_argument("--debug-scan", action="store_true")
    ap.add_argument("--debug-obs", action="store_true")
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key="lm-studio")

    model_name = args.model.strip() if args.model.strip() else "qwen/qwen3-vl-4b"

    cap_cfg = CaptureConfig(
        width=640,
        height=360,
        jpeg_quality=75,
        best_of_two=bool(args.best_of_two),
        settle_ms=int(args.settle_ms),
    )
    cam = Camera(cap_cfg)

    drive_cfg = DriveConfig()
    scan_cfg = ScanConfig(pan_settle_ms=int(args.pan_settle_ms))
    obs_cfg = ObserveConfig()

    robot = Robot(drive_cfg)

    # Start centered
    robot.set_pan_tilt(pan=0, tilt=scan_cfg.tilt_angle)
    robot.set_steer(drive_cfg.steer_center)
    robot.stop()
    time.sleep(0.25)

    # logging state
    step = 0
    last_obs_step = -999
    travel_cm = 0.0
    observations: List[Dict] = []
    last_hashes: List[int] = []

    commit_remaining = 0
    current_choice = "CENTER"

    def should_observe() -> bool:
        if len(observations) >= obs_cfg.max_observations:
            return False
        if step - last_obs_step < obs_cfg.min_steps_between:
            return False
        if travel_cm < obs_cfg.min_travel_cm * (len(observations) + 1):
            return False
        return True

    def is_novel(img: Image.Image) -> bool:
        h = image_hash_simple(img, size=16)
        if not last_hashes:
            last_hashes.append(h)
            return True
        best = min(hamming_distance(h, prev) for prev in last_hashes)
        if best >= obs_cfg.novelty_hamming_min:
            last_hashes.append(h)
            if len(last_hashes) > 10:
                last_hashes.pop(0)
            return True
        return False

    print(f"[start] base_url={base_url} model={model_name}")
    print(f"[start] steps={args.steps} best_of_two={cap_cfg.best_of_two} head_scan={args.head_scan}")
    print(f"[start] settle_ms(camera)={cap_cfg.settle_ms} pan_settle_ms(servo)={scan_cfg.pan_settle_ms}")

    try:
        while step < args.steps:
            # Avoidance first
            if robot.avoid_if_needed(debug=args.debug_avoid):
                travel_cm += obs_cfg.est_cm_per_step * 0.8
                step += 1
                continue

            # Decide direction if not currently committed
            if commit_remaining <= 0:
                if args.head_scan:
                    views: Dict[str, Image.Image] = {}
                    pan_map = {
                        "LEFT": scan_cfg.pan_angles[0],
                        "CENTER": scan_cfg.pan_angles[1],
                        "RIGHT": scan_cfg.pan_angles[2],
                    }

                    # Scan in the order you asked for: right -> pause -> photo -> center -> pause -> photo -> left -> pause -> photo
                    for key in scan_cfg.scan_order:
                        robot.set_pan_tilt(pan=pan_map[key], tilt=scan_cfg.tilt_angle)

                        # Critical: let the servo finish moving before we capture
                        time.sleep(scan_cfg.pan_settle_ms / 1000.0)

                        views[key] = cam.capture_best()

                    # Ensure all keys exist in expected dict positions
                    # (scan_order may change, but model prompt expects LEFT/CENTER/RIGHT)
                    # If scan_order omitted something (shouldn't), fill with center.
                    for k in ["LEFT", "CENTER", "RIGHT"]:
                        if k not in views:
                            views[k] = views.get("CENTER") or cam.capture_best()

                    # Return head to center
                    robot.set_pan_tilt(pan=0, tilt=scan_cfg.tilt_angle)

                    choice, raw = call_vl_direction(
                        client=client,
                        model_name=model_name,
                        images=views,
                        debug=args.debug_scan,
                    )
                    current_choice = choice
                    commit_remaining = drive_cfg.commit_steps

                    if args.debug_scan:
                        print(f"[scan] choice={choice} commit_steps={commit_remaining}")
                else:
                    current_choice = "CENTER"
                    commit_remaining = drive_cfg.commit_steps

            # Drive one step based on choice
            dcm = robot.read_ultrasonic_cm()
            speed = robot.speed_for_distance(dcm, drive_cfg.speed)

            if current_choice == "LEFT":
                steer = drive_cfg.steer_left
            elif current_choice == "RIGHT":
                steer = drive_cfg.steer_right
            else:
                steer = random.choice([-drive_cfg.steer_soft, 0, drive_cfg.steer_soft])

            robot.forward_for(speed=speed, steer_angle=steer, duration_s=drive_cfg.step_duration_s)
            robot.stop()
            time.sleep(0.03)

            commit_remaining -= 1
            travel_cm += obs_cfg.est_cm_per_step
            step += 1

            # Observations
            if should_observe():
                robot.set_pan_tilt(pan=0, tilt=scan_cfg.tilt_angle)
                # small pause to avoid capturing during micro-adjustments
                time.sleep(0.15)
                img = cam.capture_best()

                if is_novel(img):
                    obs_text = call_vl_observation(client, model_name, img, debug=args.debug_obs)
                    entry = {
                        "step": step,
                        "travel_cm_est": round(travel_cm, 1),
                        "ultrasonic_cm": round(dcm, 1),
                        "observation": obs_text,
                    }
                    observations.append(entry)
                    last_obs_step = step
                    print(f"[obs] logged {len(observations)}/{obs_cfg.max_observations} at step={step} travel~{travel_cm:.1f}cm")
                else:
                    if args.debug_obs:
                        print("[obs] skipped, not novel enough")

        print("\n[done] run complete")
        print(json.dumps({
            "steps": step,
            "travel_cm_est": round(travel_cm, 1),
            "observations": observations,
        }, indent=2))

    finally:
        robot.stop()
        try:
            robot.set_steer(0)
            robot.set_pan_tilt(pan=0, tilt=0)
        except Exception:
            pass
        cam.close()


if __name__ == "__main__":
    main()
