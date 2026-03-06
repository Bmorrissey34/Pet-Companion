#!/usr/bin/env python3
"""
PiCar-X Pet Monitor Benchmark v4
- Strategic scan decision via remote OpenAI-compatible endpoint (LM Studio)
- Local Vilib object detection used as steering + speed bias during commit drive
- Ultrasonic emergency maneuver: STOP -> backup straight -> ~45-degree peel turn

Designed for SunFounder PiCar-X v20 + Raspberry Pi 5 (Raspberry Pi OS).
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import importlib.util
import os
import sys
import time
import random
import json
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

# PiCar-X driving
from picarx import Picarx

# Vilib (camera + vision)
from vilib import Vilib

# OpenAI-compatible client (LM Studio)
from openai import OpenAI


_VILIB_DETECTION_ATTR_CANDIDATES: Tuple[str, ...] = (
    "object_detection_list_parameter",
    "object_detection_list",
    "detect_obj_parameter",
    "detect_obj_list",
    "detect_obj",
)
_VILIB_DETECTION_ATTR_NAME: Optional[str] = None
_VILIB_DETECTION_LOGGED: bool = False
_VILIB_OBJECT_DETECTION_ACTIVE: bool = False


# --------------------------- Config ---------------------------

@dataclasses.dataclass
class DriveConfig:
    # General motion
    speed_fwd: int = 28
    speed_turn: int = 24
    steer_scan_left: int = -18
    steer_scan_center: int = 0
    steer_scan_right: int = 18
    max_steer_deg: int = 28

    # Smoothing
    steer_slew_step: int = 3
    steer_slew_delay_s: float = 0.02
    speed_slew_step: int = 3
    speed_slew_delay_s: float = 0.02

    # Ultrasonic emergency
    avoid_stop_cm: float = 25.0
    avoid_backup_speed: int = 25
    avoid_backup_time_s: float = 1.2
    avoid_peel_speed: int = 22
    avoid_peel_steer_angle: int = 22
    avoid_peel_time_s: float = 0.55
    random_turn: bool = True

    # Commit drive
    commit_time_s: float = 1.0  # how long to drive after choosing a direction
    adaptive_commit_enable: bool = True
    commit_time_clear_s: float = 1.8
    adaptive_front_clear_cm: float = 55.0
    adaptive_center_band_frac: float = 0.35
    loop_hz: float = 6.0        # control loop frequency during commit

    # Scan head
    pan_angles: Tuple[int, int, int] = (-45, 0, 45)
    tilt_angle: int = 0
    pan_settle_ms: int = 500
    scan_pause_ms: int = 120

    # Photo capture
    best_of_two: bool = True
    best_of_two_gap_ms: int = 120
    photo_dir: str = "/tmp/picarx_bench_photos"
    photo_keep_latest: int = 300

    # Status logging
    heartbeat_every_loops: int = 10

    # Vilib object avoidance bias
    vilib_enable_object_bias: bool = True
    vilib_min_conf: float = 0.50
    vilib_min_area: float = 0.010  # normalized area threshold
    vilib_bias_gain: float = 45.0  # bigger = stronger push-away
    vilib_bias_max: float = 16.0   # clamp added steer bias
    vilib_speed_slow_min: float = 0.45  # min multiplier when cluttered
    vilib_speed_slow_gain: float = 1.8  # bigger = slower for same clutter
    vilib_relevant_classes: Tuple[str, ...] = (
        "chair", "couch", "dining table", "bed",
        "person", "dog", "cat",
        "toilet", "refrigerator", "sink",
    )

    # Debug
    debug_avoid: bool = False
    debug_scan: bool = False
    debug_vilib: bool = False


@dataclasses.dataclass
class LLMConfig:
    host: str
    port: int
    model: str
    temperature: float = 0.2
    max_tokens: int = 250
    timeout_s: float = 8.0
    retries: int = 1
    retry_backoff_s: float = 0.35


@dataclasses.dataclass
class ObservationConfig:
    enable_observations: bool = True
    max_observations: int = 3
    min_seconds_between_obs: float = 6.0
    min_commit_loops_between_obs: int = 3
    novelty_hamming_threshold: int = 8
    keep_last_hashes: int = 12
    observation_photo_dir: str = "/tmp/picarx_observations"
    run_log_path: str = "/tmp/picarx_run_log.jsonl"
    obs_tilt_angle: int = 0
    obs_pan_angle: int = 0
    obs_settle_ms: int = 600
    obs_best_of_two: bool = True
    obs_max_tokens: int = 160
    obs_temperature: float = 0.1
    run_log_path: str = "/tmp/picarx_run_log.jsonl"
    observation_photo_dir: str = "/tmp/picarx_observations"


# --------------------------- Helpers ---------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def cleanup_photo_dir(path_dir: str, keep_latest: int, debug: bool = False) -> None:
    if keep_latest <= 0:
        return
    try:
        entries = [
            e for e in os.scandir(path_dir)
            if e.is_file() and e.name.lower().endswith(".jpg")
        ]
    except Exception:
        return

    if len(entries) <= keep_latest:
        return

    entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
    stale = entries[keep_latest:]
    removed = 0
    for e in stale:
        try:
            os.remove(e.path)
            removed += 1
        except Exception:
            continue
    if debug and removed:
        print(f"[scan] cleaned {removed} old photos (keep_latest={keep_latest})")

def img_to_data_url_jpg(path: str, max_side: int = 640, quality: int = 80) -> str:
    """Load image, downscale, encode as base64 JPEG data URL."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def sharpness_score(path: str) -> float:
    """Simple sharpness metric: mean absolute gradient magnitude."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
    gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
    return float(gx + gy)

def pick_best_of_two(path_a: str, path_b: str) -> str:
    sa = sharpness_score(path_a)
    sb = sharpness_score(path_b)
    return path_a if sa >= sb else path_b

def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def dhash_64(path: str, hash_size: int = 8) -> int:
    """Return dHash as integer using grayscale adjacent-pixel comparisons."""
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    bits = arr[:, 1:] > arr[:, :-1]
    h = 0
    for bit in bits.flatten():
        h = (h << 1) | int(bool(bit))
    return int(h)


def hamming_distance(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


@dataclasses.dataclass
class ObservationManager:
    cfg: ObservationConfig
    observation_count: int = 0
    last_observation_time: float = dataclasses.field(default_factory=time.time)
    loops_since_last_observation: int = 0
    recent_hashes: List[int] = dataclasses.field(default_factory=list)

    def tick_loop(self) -> None:
        self.loops_since_last_observation += 1

    def can_attempt(self, now_s: float) -> bool:
        if not self.cfg.enable_observations:
            return False
        if self.observation_count >= self.cfg.max_observations:
            return False
        if (now_s - self.last_observation_time) < self.cfg.min_seconds_between_obs:
            return False
        if self.loops_since_last_observation < self.cfg.min_commit_loops_between_obs:
            return False
        return True

    def novelty_distance(self, path: str) -> Tuple[bool, Optional[int], int]:
        h = dhash_64(path)
        if not self.recent_hashes:
            return True, None, h
        min_dist = min(hamming_distance(h, old_h) for old_h in self.recent_hashes)
        return min_dist >= self.cfg.novelty_hamming_threshold, int(min_dist), h

    def add_hash(self, image_hash: int) -> None:
        self.recent_hashes.append(int(image_hash))
        keep_n = max(1, int(self.cfg.keep_last_hashes))
        if len(self.recent_hashes) > keep_n:
            self.recent_hashes = self.recent_hashes[-keep_n:]

    def register_observation(self, now_s: float, image_hash: int) -> None:
        self.observation_count += 1
        self.last_observation_time = float(now_s)
        self.loops_since_last_observation = 0
        self.add_hash(image_hash)


# --------------------------- Vision (Vilib) ---------------------------

def vilib_start(enable_object_detection: bool, debug: bool = False) -> None:
    """
    Start Vilib camera and enable object detection if requested.
    Vilib owns the camera pipeline in this script.
    """
    # vflip/hflip can be adjusted if your camera is mounted differently.
    Vilib.camera_start(vflip=False, hflip=False)
    # No display by default (we're headless most of the time)
    # If you want the stream, set web=True and/or local=True
    # Vilib.display(local=False, web=True)

    # Optional FPS overlay
    try:
        Vilib.show_fps()
    except Exception:
        pass

    global _VILIB_OBJECT_DETECTION_ACTIVE
    _VILIB_OBJECT_DETECTION_ACTIVE = False

    def has_tflite_runtime() -> bool:
        try:
            return importlib.util.find_spec("tflite_runtime.interpreter") is not None
        except ModuleNotFoundError:
            return False
        except Exception:
            return False

    if enable_object_detection:
        has_tflite = has_tflite_runtime()
        if not has_tflite:
            print("[vilib] object detection unavailable (missing tflite_runtime); continuing without OD")
            try:
                Vilib.object_detect_switch(False)
            except Exception:
                pass
            return

        # Optional: load custom model/labels if needed:
        # Vilib.object_detect_set_model(path='/opt/vilib/detect.tflite')
        # Vilib.object_detect_set_labels(path='/opt/vilib/coco_labels.txt')
        try:
            Vilib.object_detect_switch(True)
            _VILIB_OBJECT_DETECTION_ACTIVE = True
            if debug:
                print("[vilib] object_detect_switch(True)")
        except Exception as exc:
            print(f"[vilib] object detection enable failed ({exc}); continuing without OD")
            try:
                Vilib.object_detect_switch(False)
            except Exception:
                pass
    else:
        try:
            Vilib.object_detect_switch(False)
        except Exception:
            pass


def vilib_object_detection_active() -> bool:
    return _VILIB_OBJECT_DETECTION_ACTIVE

def vilib_stop() -> None:
    try:
        Vilib.camera_close()
    except Exception:
        pass

def vilib_take_photo(photo_name: str, path_dir: str) -> str:
    ensure_dir(path_dir)
    Vilib.take_photo(photo_name, path_dir)
    return os.path.join(path_dir, f"{photo_name}.jpg")

def vilib_get_detections(debug: bool = False) -> Any:
    """
    Return Vilib's object detections if available.
    Based on SunFounder examples, detections may be in:
      - Vilib.object_detection_list_parameter
    """
    global _VILIB_DETECTION_ATTR_NAME, _VILIB_DETECTION_LOGGED

    def is_non_empty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, bytes)):
            return len(value) > 0
        try:
            return len(value) > 0  # type: ignore[arg-type]
        except Exception:
            return bool(value)

    first_present_attr: Optional[str] = None

    if _VILIB_DETECTION_ATTR_NAME:
        value = safe_getattr(Vilib, _VILIB_DETECTION_ATTR_NAME, None)
        if first_present_attr is None and value is not None:
            first_present_attr = _VILIB_DETECTION_ATTR_NAME
        if is_non_empty(value):
            if debug and not _VILIB_DETECTION_LOGGED:
                print(f"[vilib] detections attribute={_VILIB_DETECTION_ATTR_NAME}")
                _VILIB_DETECTION_LOGGED = True
            return value

    for attr in _VILIB_DETECTION_ATTR_CANDIDATES:
        value = safe_getattr(Vilib, attr, None)
        if first_present_attr is None and value is not None:
            first_present_attr = attr
        if is_non_empty(value):
            _VILIB_DETECTION_ATTR_NAME = attr
            if debug and not _VILIB_DETECTION_LOGGED:
                print(f"[vilib] detections attribute={attr}")
                _VILIB_DETECTION_LOGGED = True
            return value

    if _VILIB_DETECTION_ATTR_NAME is None and first_present_attr is not None:
        _VILIB_DETECTION_ATTR_NAME = first_present_attr

    if debug and not _VILIB_DETECTION_LOGGED:
        if _VILIB_DETECTION_ATTR_NAME is not None:
            print(f"[vilib] detections attribute={_VILIB_DETECTION_ATTR_NAME} (currently empty)")
        else:
            print("[vilib] detections attribute=none")
        _VILIB_DETECTION_LOGGED = True

    return None


def vilib_get_detection_attr_name() -> Optional[str]:
    return _VILIB_DETECTION_ATTR_NAME


# --------------------------- Robot Control ---------------------------

class Robot:
    def __init__(self, drive_cfg: DriveConfig):
        self.cfg = drive_cfg
        self.px = Picarx()
        self._last_steer = 0
        self._last_speed = 0
        self._ultrasonic_source_name: Optional[str] = None
        self._ultrasonic_source_logged: bool = False

        # Start with camera centered
        self.set_cam(pan=0, tilt=self.cfg.tilt_angle)

    # ----- low-level -----
    def set_steer(self, angle: int) -> None:
        angle = int(clamp(angle, -self.cfg.max_steer_deg, self.cfg.max_steer_deg))
        self.px.set_dir_servo_angle(angle)
        self._last_steer = angle

    def set_speed(self, speed: int) -> None:
        if speed == 0:
            self.stop()
        elif speed > 0:
            self.forward(speed)
        else:
            self.backward(abs(speed))

    def _apply_signed_speed(self, speed: int) -> None:
        if speed == 0:
            self.px.stop()
        elif speed > 0:
            self.px.forward(abs(speed))
        else:
            self.px.backward(abs(speed))
        self._last_speed = int(speed)

    def stop(self) -> None:
        self.px.stop()
        self._last_speed = 0

    def forward(self, speed: int) -> None:
        self._apply_signed_speed(int(abs(speed)))

    def backward(self, speed: int) -> None:
        self._apply_signed_speed(-int(abs(speed)))

    def set_cam(self, pan: int, tilt: int) -> None:
        # These are the confirmed working methods on PiCar-X v20
        self.px.set_cam_pan_angle(pan)
        self.px.set_cam_tilt_angle(tilt)

    def read_ultrasonic_cm(self) -> float:
        def sanitize(value: Any) -> Optional[float]:
            try:
                f = float(value)
            except Exception:
                return None
            if not np.isfinite(f):
                return None
            if f <= 0.0:
                return None
            return f

        method_used = "none"

        get_distance = safe_getattr(self.px, "get_distance", None)
        if callable(get_distance):
            method_used = "get_distance"
            try:
                d = sanitize(get_distance())
                if d is not None:
                    self._ultrasonic_source_name = method_used
                    if self.cfg.debug_avoid and not self._ultrasonic_source_logged:
                        print(f"[avoid] ultrasonic source={method_used}")
                        self._ultrasonic_source_logged = True
                    return d
            except Exception:
                pass

        ultrasonic = safe_getattr(self.px, "ultrasonic", None)
        read_fn = safe_getattr(ultrasonic, "read", None) if ultrasonic is not None else None
        if callable(read_fn):
            method_used = "ultrasonic.read"
            try:
                d = sanitize(read_fn())
                if d is not None:
                    self._ultrasonic_source_name = method_used
                    if self.cfg.debug_avoid and not self._ultrasonic_source_logged:
                        print(f"[avoid] ultrasonic source={method_used}")
                        self._ultrasonic_source_logged = True
                    return d
            except Exception:
                pass

        if self._ultrasonic_source_name is None:
            self._ultrasonic_source_name = method_used
        if self.cfg.debug_avoid and not self._ultrasonic_source_logged:
            print(f"[avoid] ultrasonic source={self._ultrasonic_source_name}")
            self._ultrasonic_source_logged = True
        return 999.0

    def ultrasonic_source_name(self) -> str:
        if self._ultrasonic_source_name:
            return self._ultrasonic_source_name
        return "unknown"

    # ----- smoothing -----
    def smooth_steer_to(self, target: int) -> None:
        cur = self._last_steer
        if cur == target:
            return
        step = self.cfg.steer_slew_step if target > cur else -self.cfg.steer_slew_step
        a = cur
        while (a < target and step > 0) or (a > target and step < 0):
            a = int(clamp(a + step, -self.cfg.max_steer_deg, self.cfg.max_steer_deg))
            # avoid overshoot
            if (step > 0 and a > target) or (step < 0 and a < target):
                a = target
            self.set_steer(a)
            time.sleep(self.cfg.steer_slew_delay_s)

    def smooth_speed_to(self, target: int) -> None:
        cur = int(self._last_speed)
        target = int(target)
        if cur == target:
            return
        step_mag = max(1, int(self.cfg.speed_slew_step))
        step = step_mag if target > cur else -step_mag
        s = cur
        while (s < target and step > 0) or (s > target and step < 0):
            s += step
            if (step > 0 and s > target) or (step < 0 and s < target):
                s = target
            self._apply_signed_speed(int(s))
            time.sleep(self.cfg.speed_slew_delay_s)

    # ----- timed primitives -----
    def forward_for(self, speed: int, steer_angle: int, duration_s: float) -> None:
        self.smooth_steer_to(steer_angle)
        self.forward(speed)
        time.sleep(duration_s)
        self.stop()

    def backward_for(self, speed: int, steer_angle: int, duration_s: float) -> None:
        self.smooth_steer_to(steer_angle)
        self.backward(speed)
        time.sleep(duration_s)
        self.stop()

    # ----- avoidance -----
    def avoid_if_needed(self, debug: bool = False) -> bool:
        d = self.read_ultrasonic_cm()
        if d <= self.cfg.avoid_stop_cm:
            if debug:
                print(f"[avoid] STOP ultrasonic={d:.1f}cm <= {self.cfg.avoid_stop_cm}")
            self.stop()
            time.sleep(0.06)

            # 1) back up straight
            if debug:
                print(f"[avoid] BACKUP speed={self.cfg.avoid_backup_speed} "
                      f"time={self.cfg.avoid_backup_time_s:.2f}s")
            self.backward_for(self.cfg.avoid_backup_speed, 0, self.cfg.avoid_backup_time_s)
            time.sleep(0.08)

            # 2) peel turn about ~45 degrees
            turn_dir = random.choice([-1, 1]) if self.cfg.random_turn else 1
            steer = int(turn_dir * self.cfg.avoid_peel_steer_angle)
            side = "LEFT" if turn_dir < 0 else "RIGHT"
            if debug:
                print(f"[avoid] PEEL {side} speed={self.cfg.avoid_peel_speed} "
                      f"steer={steer} time={self.cfg.avoid_peel_time_s:.2f}s")
            self.forward_for(self.cfg.avoid_peel_speed, steer, self.cfg.avoid_peel_time_s)
            time.sleep(0.05)

            self.set_steer(0)
            return True
        return False


# --------------------------- Object Bias from Vilib ---------------------------

def parse_vilib_detections(raw: Any) -> List[Dict[str, Any]]:
    """
    Normalize Vilib detections into a list of dicts:
      {name, conf, x, y, w, h}
    If raw is unknown, returns empty list.
    """
    if raw is None:
        return []

    # Raw may already be a list of dicts, or list of lists, or a stringified json.
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []

    dets: List[Dict[str, Any]] = []

    if isinstance(raw, dict):
        # Some versions might store under a key
        for key in ("detections", "objects", "results", "data"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name") or item.get("label") or item.get("class") or item.get("class_name")
                conf = item.get("conf") or item.get("confidence") or item.get("score") or item.get("prob")
                x = item.get("x") or item.get("xmin") or item.get("left")
                y = item.get("y") or item.get("ymin") or item.get("top")
                w = item.get("w") or item.get("width")
                h = item.get("h") or item.get("height")
                # Some formats use x1,y1,x2,y2
                x2 = item.get("x2") or item.get("xmax") or item.get("right")
                y2 = item.get("y2") or item.get("ymax") or item.get("bottom")
                if (w is None or h is None) and (x is not None and y is not None and x2 is not None and y2 is not None):
                    w = abs(float(x2) - float(x))
                    h = abs(float(y2) - float(y))
                try:
                    dets.append({
                        "name": str(name) if name is not None else "",
                        "conf": float(conf) if conf is not None else 0.0,
                        "x": float(x) if x is not None else 0.0,
                        "y": float(y) if y is not None else 0.0,
                        "w": float(w) if w is not None else 0.0,
                        "h": float(h) if h is not None else 0.0,
                    })
                except Exception:
                    continue
            elif isinstance(item, (list, tuple)) and len(item) >= 6:
                # e.g. [name, conf, x, y, w, h] or [id, name, conf, x, y, w, h]
                try:
                    if isinstance(item[0], str):
                        name = item[0]; conf, x, y, w, h = item[1:6]
                    else:
                        name = item[1]; conf, x, y, w, h = item[2:7]
                    dets.append({
                        "name": str(name),
                        "conf": float(conf),
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h),
                    })
                except Exception:
                    continue

    return dets

def compute_object_bias(
    dets: List[Dict[str, Any]],
    relevant_names: Tuple[str, ...],
    min_conf: float,
    min_area: float,
    gain: float,
    max_bias: float,
    debug: bool = False,
) -> Tuple[float, float]:
    """
    Return (steer_bias_deg, clutter_score) where clutter_score is 0..~1+.
    Assumes Vilib uses 640x480 or 640x360-ish. We normalize against 640x480.
    """
    if not dets:
        return 0.0, 0.0

    W, H = 640.0, 480.0
    cx0 = W / 2.0

    steer = 0.0
    clutter = 0.0

    for d in dets:
        name = (d.get("name") or "").lower().strip()
        if not name:
            continue
        if name not in relevant_names:
            continue

        conf = float(d.get("conf", 0.0))
        if conf < min_conf:
            continue

        x = float(d.get("x", 0.0))
        y = float(d.get("y", 0.0))
        w = float(d.get("w", 0.0))
        h = float(d.get("h", 0.0))

        area = (w * h) / (W * H) if W > 0 and H > 0 else 0.0
        if area < min_area:
            continue

        cx = x + w / 2.0
        offset = (cx - cx0) / cx0  # -1..+1 approx
        risk = conf * area

        # Push away from obstacle: obstacle on left (offset<0) pushes steer right (positive)
        steer += (-offset) * risk
        clutter += risk

        if debug:
            print(f"[vilib] {name} conf={conf:.2f} area={area:.3f} offset={offset:+.2f} risk={risk:.3f}")

    steer_bias = float(clamp(steer * gain, -max_bias, max_bias))
    return steer_bias, float(clutter)


def has_front_object(
    dets: List[Dict[str, Any]],
    relevant_names: Tuple[str, ...],
    min_conf: float,
    min_area: float,
    center_band_frac: float,
) -> bool:
    """
    True if a relevant detection passes confidence/area thresholds and lies in
    the center horizontal band of the image.
    """
    if not dets:
        return False

    W, H = 640.0, 480.0
    frac = float(clamp(float(center_band_frac), 0.05, 0.95))
    band_half = (W * frac) / 2.0
    cx0 = W / 2.0
    band_left = cx0 - band_half
    band_right = cx0 + band_half

    for d in dets:
        if not isinstance(d, dict):
            continue

        name = str(d.get("name") or "").lower().strip()
        if not name or name not in relevant_names:
            continue

        try:
            conf = float(d.get("conf", 0.0) or 0.0)
            x = float(d.get("x", 0.0) or 0.0)
            w = float(d.get("w", 0.0) or 0.0)
            h = float(d.get("h", 0.0) or 0.0)
        except Exception:
            continue

        if conf < min_conf:
            continue

        area = (w * h) / (W * H) if W > 0 and H > 0 else 0.0
        if area < min_area:
            continue

        cx = x + (w / 2.0)
        if band_left <= cx <= band_right:
            return True

    return False


def speed_scale_from_clutter(clutter: float, slow_gain: float, slow_min: float) -> float:
    # clutter ~0 -> 1.0, bigger clutter -> slower
    scale = 1.0 / (1.0 + slow_gain * clutter)
    return float(clamp(scale, slow_min, 1.0))


# --------------------------- LLM Scan Decision ---------------------------

def llm_choose_direction(
    client: OpenAI,
    llm_cfg: LLMConfig,
    img_left_path: str,
    img_center_path: str,
    img_right_path: str,
    debug: bool = False,
) -> str:
    """
    Ask the vision model which direction is most open / leads somewhere new.
    Returns: "LEFT" | "CENTER" | "RIGHT"
    """
    left_url = img_to_data_url_jpg(img_left_path)
    center_url = img_to_data_url_jpg(img_center_path)
    right_url = img_to_data_url_jpg(img_right_path)

    prompt = (
        "You are the navigation brain for a small indoor robot car. "
        "Given three camera views (LEFT, CENTER, RIGHT), choose the best direction to drive next. "
        "Pick the direction that looks most open and likely leads somewhere new (not blocked by furniture). "
        "Only answer with one token: LEFT or CENTER or RIGHT."
    )

    # OpenAI-compatible multimodal message
    msg = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "LEFT view:"},
        {"type": "image_url", "image_url": {"url": left_url}},
        {"type": "text", "text": "CENTER view:"},
        {"type": "image_url", "image_url": {"url": center_url}},
        {"type": "text", "text": "RIGHT view:"},
        {"type": "image_url", "image_url": {"url": right_url}},
    ]

    def parse_choice(text: str) -> Optional[str]:
        cleaned = text.strip().upper().strip(" \t\r\n\"'`.,;:!?()[]{}")
        if cleaned in ("LEFT", "CENTER", "RIGHT"):
            return cleaned
        return None

    attempts = max(0, int(llm_cfg.retries)) + 1
    for attempt in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=llm_cfg.model,
                temperature=llm_cfg.temperature,
                max_tokens=llm_cfg.max_tokens,
                timeout=llm_cfg.timeout_s,
                messages=[{"role": "user", "content": msg}],
            )
            raw_text = (resp.choices[0].message.content or "")
            choice = parse_choice(raw_text)
            if choice is not None:
                if debug:
                    print(f"[scan] llm={raw_text!r} parsed={choice}")
                return choice
            if debug:
                print(f"[scan] llm invalid token response={raw_text!r}")
        except Exception as exc:
            if debug:
                print(f"[scan] llm request failed attempt={attempt + 1}/{attempts}: {exc}")

        if attempt < attempts - 1:
            time.sleep(llm_cfg.retry_backoff_s * (2 ** attempt))

    if debug:
        print("[scan] llm fallback=CENTER")
    return "CENTER"


def take_observation(robot: Robot, drive_cfg: DriveConfig, obs_cfg: ObservationConfig) -> str:
    ensure_dir(obs_cfg.observation_photo_dir)
    robot.set_cam(pan=obs_cfg.obs_pan_angle, tilt=obs_cfg.obs_tilt_angle)
    time.sleep(obs_cfg.obs_settle_ms / 1000.0)

    base = f"obs_{now_ts()}"
    p1 = vilib_take_photo(base + "_a", obs_cfg.observation_photo_dir)
    if obs_cfg.obs_best_of_two:
        time.sleep(drive_cfg.best_of_two_gap_ms / 1000.0)
        p2 = vilib_take_photo(base + "_b", obs_cfg.observation_photo_dir)
        return pick_best_of_two(p1, p2)
    return p1


def llm_describe_observation(
    client: OpenAI,
    llm_cfg: LLMConfig,
    photo_path: str,
    obs_cfg: ObservationConfig,
) -> Dict[str, Any]:
    photo_url = img_to_data_url_jpg(photo_path)
    prompt = (
        "Describe what you see for a robotics research log. "
        "Respond ONLY as JSON with keys: {time, summary, hazards, open_paths, notable_objects}. "
        "hazards: list of strings. "
        "open_paths: list of strings (e.g., 'left corridor', 'open center', 'right doorway'). "
        "notable_objects: list of strings."
    )

    msg = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": photo_url}},
    ]

    fallback = {
        "time": now_ts(),
        "summary": "",
        "hazards": [],
        "open_paths": [],
        "notable_objects": [],
        "raw_response": "",
        "parse_ok": False,
    }

    try:
        resp = client.chat.completions.create(
            model=llm_cfg.model,
            temperature=obs_cfg.obs_temperature,
            max_tokens=obs_cfg.obs_max_tokens,
            timeout=llm_cfg.timeout_s,
            messages=[{"role": "user", "content": msg}],
        )
        raw_text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        fallback["summary"] = f"LLM observation failed: {exc}"
        return fallback

    payload: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        pass

    if payload is None:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw_text[start:end + 1])
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                payload = None

    if payload is None:
        fallback["summary"] = raw_text
        fallback["raw_response"] = raw_text
        return fallback

    result = {
        "time": str(payload.get("time") or now_ts()),
        "summary": str(payload.get("summary") or ""),
        "hazards": payload.get("hazards") if isinstance(payload.get("hazards"), list) else [],
        "open_paths": payload.get("open_paths") if isinstance(payload.get("open_paths"), list) else [],
        "notable_objects": payload.get("notable_objects") if isinstance(payload.get("notable_objects"), list) else [],
        "raw_response": raw_text,
        "parse_ok": True,
    }
    return result


def append_jsonl(path: str, data: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# --------------------------- Main loop ---------------------------

def capture_scan_images(robot: Robot, cfg: DriveConfig) -> Tuple[str, str, str]:
    """
    Pan LEFT/CENTER/RIGHT, settle, capture image(s).
    Returns paths in order (left, center, right).
    """
    ensure_dir(cfg.photo_dir)

    # ensure tilt is set
    robot.set_cam(pan=0, tilt=cfg.tilt_angle)
    time.sleep(cfg.pan_settle_ms / 1000.0)

    paths: List[str] = []

    labels = ["L", "C", "R"]
    for i, pan in enumerate(cfg.pan_angles):
        robot.set_cam(pan=pan, tilt=cfg.tilt_angle)
        time.sleep(cfg.pan_settle_ms / 1000.0)

        # Additional settle pause to reduce motion blur
        time.sleep(cfg.scan_pause_ms / 1000.0)

        base = f"scan_{now_ts()}_{labels[i]}"
        p1 = vilib_take_photo(base + "_a", cfg.photo_dir)

        if cfg.best_of_two:
            time.sleep(cfg.best_of_two_gap_ms / 1000.0)
            p2 = vilib_take_photo(base + "_b", cfg.photo_dir)
            best = pick_best_of_two(p1, p2)
            paths.append(best)
        else:
            paths.append(p1)

    # return camera to center
    robot.set_cam(pan=0, tilt=cfg.tilt_angle)
    time.sleep(cfg.pan_settle_ms / 1000.0)

    cleanup_photo_dir(cfg.photo_dir, keep_latest=cfg.photo_keep_latest, debug=cfg.debug_scan)

    return paths[0], paths[1], paths[2]

def steer_from_choice(cfg: DriveConfig, choice: str) -> int:
    if choice == "LEFT":
        return cfg.steer_scan_left
    if choice == "RIGHT":
        return cfg.steer_scan_right
    return cfg.steer_scan_center

def run_benchmark(drive_cfg: DriveConfig, llm_cfg: LLMConfig, obs_cfg: ObservationConfig) -> None:
    # Create log immediately so startup failures are still recorded.
    append_jsonl(
        obs_cfg.run_log_path,
        {
            "event": "session_start",
            "ts": {
                "unix": time.time(),
                "human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            },
            "log_path": obs_cfg.run_log_path,
            "observation_photo_dir": obs_cfg.observation_photo_dir,
            "max_observations": obs_cfg.max_observations,
            "min_seconds_between_obs": obs_cfg.min_seconds_between_obs,
            "min_commit_loops_between_obs": obs_cfg.min_commit_loops_between_obs,
            "novelty_hamming_threshold": obs_cfg.novelty_hamming_threshold,
        },
    )
    print(f"[obs] run log: {obs_cfg.run_log_path}")

    # Start vision after log creation
    vilib_start(
        enable_object_detection=(drive_cfg.vilib_enable_object_bias or drive_cfg.adaptive_commit_enable),
        debug=drive_cfg.debug_vilib,
    )

    client = OpenAI(base_url=f"http://{llm_cfg.host}:{llm_cfg.port}/v1", api_key="lm-studio")

    robot = Robot(drive_cfg)
    relevant_names_lower = tuple(n.lower() for n in drive_cfg.vilib_relevant_classes)
    object_detection_ready = vilib_object_detection_active()
    obs_mgr = ObservationManager(cfg=obs_cfg)
    loop_count = 0

    # Ensure run log exists immediately and record session start.
    append_jsonl(
        obs_cfg.run_log_path,
        {
            "event": "session_start",
            "ts": {
                "unix": time.time(),
                "human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            },
            "log_path": obs_cfg.run_log_path,
            "observation_photo_dir": obs_cfg.observation_photo_dir,
            "max_observations": obs_cfg.max_observations,
            "min_seconds_between_obs": obs_cfg.min_seconds_between_obs,
            "min_commit_loops_between_obs": obs_cfg.min_commit_loops_between_obs,
            "novelty_hamming_threshold": obs_cfg.novelty_hamming_threshold,
        },
    )
    print(f"[obs] run log: {obs_cfg.run_log_path}")

    try:
        print("[bench] starting loop (Ctrl+C to stop)")
        while True:
            # 1) emergency avoidance
            if robot.avoid_if_needed(debug=drive_cfg.debug_avoid):
                continue

            # 2) scan decision
            if drive_cfg.debug_scan:
                print("[scan] capturing L/C/R")
            left_img, center_img, right_img = capture_scan_images(robot, drive_cfg)

            choice = llm_choose_direction(
                client, llm_cfg,
                img_left_path=left_img,
                img_center_path=center_img,
                img_right_path=right_img,
                debug=drive_cfg.debug_scan,
            )
            base_steer = steer_from_choice(drive_cfg, choice)
            obs_mgr.tick_loop()

            if drive_cfg.debug_scan:
                print(f"[scan] choice={choice} base_steer={base_steer}")

            now_s = time.time()
            if obs_mgr.can_attempt(now_s):
                robot.stop()
                obs_photo = take_observation(robot, drive_cfg, obs_cfg)
                is_novel, min_dist, obs_hash = obs_mgr.novelty_distance(obs_photo)
                ultrasonic_now = robot.read_ultrasonic_cm()

                if not is_novel:
                    append_jsonl(
                        obs_cfg.run_log_path,
                        {
                            "event": "observation_attempt",
                            "ts": {
                                "unix": now_s,
                                "human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_s)),
                            },
                            "loop_count": loop_count,
                            "choice": choice,
                            "ultrasonic_cm": ultrasonic_now,
                            "photo_path": obs_photo,
                            "novel": False,
                            "novelty_distance": min_dist,
                            "image_hash": obs_hash,
                            "observation_count": obs_mgr.observation_count,
                            "max_observations": obs_cfg.max_observations,
                        },
                    )
                    if drive_cfg.debug_scan or drive_cfg.debug_vilib:
                        print(f"[obs] skipped non-novel dist={min_dist}")
                    print(f"[obs] captured {obs_mgr.observation_count}/{obs_cfg.max_observations} novel=False")
                else:
                    obs_data = llm_describe_observation(
                        client=client,
                        llm_cfg=llm_cfg,
                        photo_path=obs_photo,
                        obs_cfg=obs_cfg,
                    )
                    append_jsonl(
                        obs_cfg.run_log_path,
                        {
                            "event": "observation_attempt",
                            "ts": {
                                "unix": now_s,
                                "human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_s)),
                            },
                            "loop_count": loop_count,
                            "choice": choice,
                            "ultrasonic_cm": ultrasonic_now,
                            "photo_path": obs_photo,
                            "novel": True,
                            "novelty_distance": min_dist,
                            "image_hash": obs_hash,
                            "observation_count_next": obs_mgr.observation_count + 1,
                            "max_observations": obs_cfg.max_observations,
                            "llm_observation": obs_data,
                        },
                    )
                    obs_mgr.register_observation(now_s, obs_hash)

                    print(f"[obs] captured {obs_mgr.observation_count}/{obs_cfg.max_observations} novel=True")
                    print(f"[obs] photo={obs_photo}")
                    print(f"[obs] summary: {obs_data.get('summary', '')}")
                    print(f"[obs] hazards: {obs_data.get('hazards', [])}")
                    print(f"[obs] open_paths: {obs_data.get('open_paths', [])}")
                    print(f"[obs] notable_objects: {obs_data.get('notable_objects', [])}")

            # 3) commit drive with object bias blending
            commit_time_this_loop = drive_cfg.commit_time_s
            if drive_cfg.adaptive_commit_enable and object_detection_ready:
                dets = parse_vilib_detections(vilib_get_detections(debug=drive_cfg.debug_vilib))
                front_obj = has_front_object(
                    dets=dets,
                    relevant_names=relevant_names_lower,
                    min_conf=drive_cfg.vilib_min_conf,
                    min_area=drive_cfg.vilib_min_area,
                    center_band_frac=drive_cfg.adaptive_center_band_frac,
                )
                d_front = robot.read_ultrasonic_cm()
                ultrasonic_clear = d_front >= drive_cfg.adaptive_front_clear_cm
                if ultrasonic_clear and not front_obj:
                    commit_time_this_loop = max(drive_cfg.commit_time_s, drive_cfg.commit_time_clear_s)

                if drive_cfg.debug_scan or drive_cfg.debug_vilib:
                    print(
                        f"[bench] adaptive_commit ultra={d_front:.1f} "
                        f"clear={ultrasonic_clear} front_obj={front_obj} "
                        f"commit_s={commit_time_this_loop:.2f}"
                    )
            elif drive_cfg.adaptive_commit_enable and (drive_cfg.debug_scan or drive_cfg.debug_vilib):
                print("[bench] adaptive_commit disabled (object detection unavailable)")

            dt = 1.0 / max(1e-6, drive_cfg.loop_hz)
            steps = max(1, int(commit_time_this_loop / dt))
            last_speed_scale = 1.0

            for _ in range(steps):
                # emergency check each step
                if robot.avoid_if_needed(debug=drive_cfg.debug_avoid):
                    break

                steer_cmd = float(base_steer)
                speed_cmd = float(drive_cfg.speed_fwd)

                if drive_cfg.vilib_enable_object_bias and object_detection_ready:
                    raw = vilib_get_detections(debug=drive_cfg.debug_vilib)
                    dets = parse_vilib_detections(raw)
                    steer_bias, clutter = compute_object_bias(
                        dets=dets,
                        relevant_names=relevant_names_lower,
                        min_conf=drive_cfg.vilib_min_conf,
                        min_area=drive_cfg.vilib_min_area,
                        gain=drive_cfg.vilib_bias_gain,
                        max_bias=drive_cfg.vilib_bias_max,
                        debug=drive_cfg.debug_vilib,
                    )
                    if clutter > 0:
                        steer_cmd += steer_bias
                        speed_cmd *= speed_scale_from_clutter(
                            clutter=clutter,
                            slow_gain=drive_cfg.vilib_speed_slow_gain,
                            slow_min=drive_cfg.vilib_speed_slow_min,
                        )
                        last_speed_scale = speed_cmd / drive_cfg.speed_fwd if drive_cfg.speed_fwd else 1.0
                        if drive_cfg.debug_vilib:
                            print(f"[vilib] steer_bias={steer_bias:+.1f} clutter={clutter:.3f} speed_scale={speed_cmd/drive_cfg.speed_fwd:.2f}")

                steer_cmd = clamp(steer_cmd, -drive_cfg.max_steer_deg, drive_cfg.max_steer_deg)
                speed_cmd = clamp(speed_cmd, 12, 40)

                robot.smooth_steer_to(int(round(steer_cmd)))
                robot.smooth_speed_to(int(round(speed_cmd)))
                time.sleep(dt)

            robot.stop()
            time.sleep(0.05)

            loop_count += 1
            if drive_cfg.heartbeat_every_loops > 0 and (loop_count % drive_cfg.heartbeat_every_loops) == 0:
                d = robot.read_ultrasonic_cm()
                print(
                    f"[bench] loop={loop_count} choice={choice} "
                    f"ultrasonic={d:.1f}cm speed_scale={last_speed_scale:.2f}"
                )

    finally:
        append_jsonl(
            obs_cfg.run_log_path,
            {
                "event": "session_end",
                "ts": {
                    "unix": time.time(),
                    "human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                },
                "loop_count": loop_count,
            },
        )
        robot.stop()
        vilib_stop()


def run_self_test(drive_cfg: DriveConfig) -> None:
    vilib_start(enable_object_detection=True, debug=True)
    robot = Robot(drive_cfg)
    try:
        d = robot.read_ultrasonic_cm()
        print(f"[self-test] ultrasonic source={robot.ultrasonic_source_name()} reading={d:.1f}cm")

        raw0 = vilib_get_detections(debug=True)
        det_attr = vilib_get_detection_attr_name() or "none"
        print(f"[self-test] vilib detections attribute={det_attr}")

        ensure_dir(drive_cfg.photo_dir)
        photo_base = f"test_{now_ts()}"
        photo_path = vilib_take_photo(photo_base, drive_cfg.photo_dir)
        print(f"[self-test] photo={photo_path}")

        for i in range(3):
            raw = raw0 if i == 0 else vilib_get_detections(debug=True)
            parsed = parse_vilib_detections(raw)
            print(f"[self-test] t+{i+1}s raw={raw!r} parsed_count={len(parsed)}")
            time.sleep(1.0)
    finally:
        robot.stop()
        vilib_stop()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PiCar-X Pet Monitor Benchmark v4")
    p.add_argument("--host", help="LM Studio host (Tailscale IP)")
    p.add_argument("--port", type=int, default=1234, help="LM Studio port (default 1234)")
    p.add_argument("--model", help="Model id from /v1/models (must be vision-capable)")

    p.add_argument("--best-of-two", action="store_true", help="Capture 2 photos per view and keep the sharper one")
    p.add_argument("--pan-settle-ms", type=int, default=500, help="Settle time after pan before capture (ms)")
    p.add_argument("--scan-pause-ms", type=int, default=120, help="Extra pause before capture to reduce blur (ms)")
    p.add_argument("--commit-time-s", type=float, default=2.5, help="How long to drive after choosing direction (seconds)")
    p.add_argument("--commit-time-clear-s", type=float, default=1.8, help="Adaptive commit time when front is clear (seconds)")
    p.add_argument("--adaptive-front-clear-cm", type=float, default=55.0, help="Ultrasonic distance considered clear for adaptive commit (cm)")
    p.add_argument("--adaptive-center-band-frac", type=float, default=0.35, help="Center-band width fraction for front object check (0.05..0.95)")
    p.add_argument("--no-adaptive-commit", action="store_true", help="Disable adaptive commit time expansion when front is clear")
    p.add_argument("--photo-keep-latest", type=int, default=300, help="Keep only latest N scan photos in /tmp to cap disk use")
    p.add_argument("--heartbeat-every-loops", type=int, default=10, help="Emit [bench] status log every N loops (0 disables)")

    p.add_argument("--avoid-stop-cm", type=float, default=25.0, help="Ultrasonic stop threshold (cm)")
    p.add_argument("--debug-avoid", action="store_true")
    p.add_argument("--debug-scan", action="store_true")
    p.add_argument("--debug-vilib", action="store_true")

    p.add_argument("--no-object-bias", action="store_true", help="Disable Vilib object bias steering")
    p.add_argument("--llm-timeout-s", type=float, default=8.0, help="Timeout per LLM request in seconds")
    p.add_argument("--llm-retries", type=int, default=1, help="Retry count for LLM request/parse failures")
    p.add_argument("--llm-retry-backoff-s", type=float, default=0.35, help="Base exponential backoff seconds between LLM retries")
    p.add_argument("--self-test", action="store_true", help="Run hardware compatibility self-test and exit")
    p.add_argument("--no-observations", action="store_true", help="Disable periodic observation capture/logging")
    p.add_argument("--max-observations", type=int, default=3, help="Maximum novel observations per run")
    p.add_argument("--min-seconds-between-obs", type=float, default=6.0, help="Minimum seconds between observation attempts")
    p.add_argument("--novelty-hamming-threshold", type=int, default=8, help="Minimum dHash distance to count as novel")
    p.add_argument("--run-log-path", default="/tmp/picarx_run_log.jsonl", help="JSONL run log output path")
    p.add_argument("--observation-photo-dir", default="/tmp/picarx_observations", help="Observation photo output directory")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    drive_cfg = DriveConfig(
        best_of_two=bool(args.best_of_two),
        pan_settle_ms=int(args.pan_settle_ms),
        scan_pause_ms=int(args.scan_pause_ms),
        commit_time_s=float(args.commit_time_s),
        adaptive_commit_enable=(not bool(args.no_adaptive_commit)),
        commit_time_clear_s=max(0.1, float(args.commit_time_clear_s)),
        adaptive_front_clear_cm=max(1.0, float(args.adaptive_front_clear_cm)),
        adaptive_center_band_frac=clamp(float(args.adaptive_center_band_frac), 0.05, 0.95),
        photo_keep_latest=max(0, int(args.photo_keep_latest)),
        heartbeat_every_loops=max(0, int(args.heartbeat_every_loops)),
        avoid_stop_cm=float(args.avoid_stop_cm),
        debug_avoid=bool(args.debug_avoid),
        debug_scan=bool(args.debug_scan),
        debug_vilib=bool(args.debug_vilib),
        vilib_enable_object_bias=(not bool(args.no_object_bias)),
    )

    if bool(args.self_test):
        print("[self-test] starting (no driving)")
        run_self_test(drive_cfg)
        return

    if not args.host or not args.model:
        parser.error("--host and --model are required unless --self-test is set")

    obs_cfg = ObservationConfig(
        enable_observations=(not bool(args.no_observations)),
        max_observations=max(0, int(args.max_observations)),
        min_seconds_between_obs=max(0.0, float(args.min_seconds_between_obs)),
        novelty_hamming_threshold=max(0, int(args.novelty_hamming_threshold)),
        run_log_path=str(args.run_log_path),
        observation_photo_dir=str(args.observation_photo_dir),
    )

    llm_cfg = LLMConfig(
        host=str(args.host),
        port=int(args.port),
        model=str(args.model),
        timeout_s=max(0.5, float(args.llm_timeout_s)),
        retries=max(0, int(args.llm_retries)),
        retry_backoff_s=max(0.05, float(args.llm_retry_backoff_s)),
    )

    # Quick info
    print(f"[bench] LM Studio endpoint: http://{llm_cfg.host}:{llm_cfg.port}/v1  model={llm_cfg.model}")
    if drive_cfg.vilib_enable_object_bias:
        print("[bench] Vilib object bias: ON")
    else:
        print("[bench] Vilib object bias: OFF")

    run_benchmark(drive_cfg, llm_cfg, obs_cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
