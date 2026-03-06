#!/usr/bin/env python3
"""
Simple single-task benchmark: can a model guide the PiCar-X to 3 different
locations in a room, take a picture at each, and describe what is happening
in the environment?

This runs on the actual PiCar-X with a camera. The LLM navigates the robot,
decides when it has reached a new location, captures an observation, then
moves on to the next. After 3 observations, a final prompt asks the LLM
to summarize the overall environment.

Usage:
    sudo python3 benchmark_simple.py --host 192.168.1.100
    sudo python3 benchmark_simple.py --host 192.168.1.100 --model qwen2.5-vl-7b
    sudo python3 benchmark_simple.py --host 192.168.1.100 --max-steps 30
    sudo python3 benchmark_simple.py --host 192.168.1.100 --save-images ./run1
"""

import argparse
import base64
import json
import os
import re
import time

from openai import OpenAI
from picarx import Picarx
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Actions (same as lmstudio_picarx.py)
# ---------------------------------------------------------------------------

SPEED = 30
TURN_ANGLE = 30


def do_forward(px, duration=1.0):
    px.set_dir_servo_angle(0)
    px.forward(SPEED)
    time.sleep(duration)
    px.stop()


def do_backward(px, duration=1.0):
    px.set_dir_servo_angle(0)
    px.backward(SPEED)
    time.sleep(duration)
    px.stop()


def do_turn_left(px, duration=1.0):
    px.set_dir_servo_angle(-TURN_ANGLE)
    px.forward(SPEED)
    time.sleep(duration)
    px.stop()
    px.set_dir_servo_angle(0)


def do_turn_right(px, duration=1.0):
    px.set_dir_servo_angle(TURN_ANGLE)
    px.forward(SPEED)
    time.sleep(duration)
    px.stop()
    px.set_dir_servo_angle(0)


def do_stop(px, **_):
    px.stop()
    px.set_dir_servo_angle(0)


def do_look_left(px, **_):
    px.set_camera_servo1_angle(30)
    time.sleep(0.5)
    px.set_camera_servo1_angle(0)


def do_look_right(px, **_):
    px.set_camera_servo1_angle(-30)
    time.sleep(0.5)
    px.set_camera_servo1_angle(0)


def do_look_up(px, **_):
    px.set_camera_servo2_angle(30)
    time.sleep(0.5)
    px.set_camera_servo2_angle(0)


def do_look_down(px, **_):
    px.set_camera_servo2_angle(-20)
    time.sleep(0.5)
    px.set_camera_servo2_angle(0)


ACTIONS = {
    "forward": do_forward,
    "backward": do_backward,
    "turn_left": do_turn_left,
    "turn_right": do_turn_right,
    "stop": do_stop,
    "look_left": do_look_left,
    "look_right": do_look_right,
    "look_up": do_look_up,
    "look_down": do_look_down,
}

ACTION_NAMES = ", ".join(ACTIONS.keys())

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You are the brain of a small robot car called PiCar-X exploring a room.
Each message includes a photo from the robot's camera.

YOUR TASK:
Visit 3 different locations in the room. At each new location, take an
observation by setting "observe" to true and writing a detailed description
of what you see in the "observation" field.

A "new location" means the camera sees meaningfully different surroundings
compared to previous observations. Do NOT observe the same spot twice.
Navigate until the scene changes enough, then observe.

Available actions (use these exact names):
  {ACTION_NAMES}

Movement actions accept an optional "duration" in seconds (default 1.0):
  forward, backward, turn_left, turn_right

Reply ONLY with a JSON object in this format — no extra text:
{{
  "thinking": "<brief reasoning about what you see and your plan>",
  "observe": <true if you want to record this location, false to keep moving>,
  "observation": "<detailed description of the scene — only when observe is true>",
  "actions": [
    {{"name": "<action_name>", "args": {{"duration": <seconds>}}}},
    ...
  ]
}}

Rules:
- Always include at least one action to keep the robot moving.
- Set "observe" to true only when you have reached a new distinct location.
- "observation" should describe objects, furniture, colors, layout — be specific.
- Do NOT wrap the JSON in markdown code fences.
"""

SUMMARY_PROMPT = """\
You visited 3 locations in a room and recorded the following observations:

{observations}

Based on these 3 observations, provide a JSON response:
{{
  "summary": "<2-3 sentence summary of the overall environment/room>",
  "locations_distinct": <true if the 3 observations describe clearly different locations, false otherwise>,
  "objects_seen": ["<list of distinct objects/features you identified across all observations>"],
  "confidence": <0.0 to 1.0 — how confident are you that you explored different parts of the room?>
}}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in response")


def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_vision_message(text, image_path):
    b64 = encode_image_base64(image_path)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
        ],
    }


def execute_actions(px, actions):
    for action in actions:
        name = action.get("name", "")
        args = action.get("args", {})
        fn = ACTIONS.get(name)
        if fn:
            print(f"    -> {name} {args}")
            fn(px, **args)
        else:
            print(f"    -> unknown action: {name}")


def capture_image(camera, path="/tmp/picarx-bench.jpg"):
    camera.capture_file(path)
    return path


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_exploration(client, model, px, camera, max_steps, save_dir):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    observations = []
    step_log = []
    total_start = time.time()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  EXPLORATION BENCHMARK")
    print(f"  Model: {model}")
    print(f"  Goal: Visit 3 locations, observe and describe each")
    print(f"  Max steps: {max_steps}")
    print(f"{'=' * 60}\n")

    for step in range(1, max_steps + 1):
        remaining = 3 - len(observations)
        print(f"[STEP {step}/{max_steps}] Observations so far: {len(observations)}/3")

        img_path = capture_image(camera)

        if save_dir:
            save_path = os.path.join(save_dir, f"step_{step:03d}.jpg")
            os.system(f"cp {img_path} {save_path}")

        prompt = (
            f"You have recorded {len(observations)} of 3 observations so far. "
            f"{remaining} remaining.\n"
        )
        if observations:
            prompt += "Previous observations:\n"
            for i, obs in enumerate(observations, 1):
                prompt += f"  Location {i}: {obs['observation'][:100]}...\n"
            prompt += "\n"
        prompt += "Look at the current camera image. Decide: observe here or keep moving?"

        msg = build_vision_message(prompt, img_path)
        messages.append(msg)

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
            )
            latency = time.time() - t0
            reply = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"  ERROR: {e}")
            step_log.append({"step": step, "error": str(e)})
            time.sleep(1)
            continue

        print(f"  Latency: {latency:.2f}s")

        try:
            parsed = extract_json(reply)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ERROR parsing JSON: {e}")
            print(f"  Raw: {reply[:200]}")
            step_log.append({"step": step, "error": f"JSON parse: {e}", "latency": latency})
            do_forward(px, 0.5)
            continue

        thinking = parsed.get("thinking", "")
        observe = parsed.get("observe", False)
        observation = parsed.get("observation", "")
        actions = parsed.get("actions", [])

        if thinking:
            print(f"  Thinking: {thinking[:100]}")

        step_entry = {
            "step": step,
            "latency": round(latency, 2),
            "observe": observe,
            "actions": [a.get("name", "") for a in actions],
        }

        if observe and observation:
            obs_num = len(observations) + 1
            print(f"  ** OBSERVATION {obs_num}: {observation[:120]}")

            if save_dir:
                obs_path = os.path.join(save_dir, f"observation_{obs_num}.jpg")
                os.system(f"cp {img_path} {obs_path}")

            observations.append({
                "number": obs_num,
                "step": step,
                "observation": observation,
                "image": img_path,
                "latency": latency,
            })
            step_entry["observation"] = observation

        if actions:
            execute_actions(px, actions)
        else:
            print("    (no actions returned, nudging forward)")
            do_forward(px, 0.5)

        step_log.append(step_entry)

        if len(messages) > 11:
            messages = [messages[0]] + messages[-10:]

        if len(observations) >= 3:
            print(f"\n  All 3 observations collected!")
            break

        time.sleep(0.5)

    total_time = time.time() - total_start
    return observations, step_log, total_time


def run_summary(client, model, observations):
    obs_text = ""
    for obs in observations:
        obs_text += f"Location {obs['number']} (step {obs['step']}): {obs['observation']}\n\n"

    prompt = SUMMARY_PROMPT.format(observations=obs_text)
    messages = [
        {"role": "system", "content": "You are an analytical assistant. Respond only with JSON."},
        {"role": "user", "content": prompt},
    ]

    print("\n  Asking LLM to judge its own exploration...")
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    latency = time.time() - t0
    reply = response.choices[0].message.content or ""

    try:
        parsed = extract_json(reply)
        return parsed, latency
    except (json.JSONDecodeError, ValueError):
        return {"raw": reply}, latency


# ---------------------------------------------------------------------------
# Scoring and reporting
# ---------------------------------------------------------------------------


def score_run(observations, step_log, summary, total_time):
    scores = {}

    # 1. Did it collect all 3 observations?
    scores["observations_collected"] = len(observations)
    scores["all_3_collected"] = len(observations) >= 3

    # 2. Did observations include actual descriptions (not empty)?
    non_empty = sum(1 for o in observations if len(o["observation"].strip()) > 20)
    scores["descriptive_observations"] = non_empty

    # 3. Self-assessed distinctness
    scores["self_judged_distinct"] = summary.get("locations_distinct", False)
    scores["self_judged_confidence"] = summary.get("confidence", 0.0)
    scores["objects_seen"] = summary.get("objects_seen", [])

    # 4. Efficiency
    total_steps = len(step_log)
    error_steps = sum(1 for s in step_log if "error" in s)
    scores["total_steps"] = total_steps
    scores["error_steps"] = error_steps
    scores["total_time_s"] = round(total_time, 1)
    latencies = [s["latency"] for s in step_log if "latency" in s]
    scores["avg_latency_s"] = round(sum(latencies) / len(latencies), 2) if latencies else 0

    # 5. Movement variety (not just repeating the same action)
    all_actions = []
    for s in step_log:
        all_actions.extend(s.get("actions", []))
    unique_actions = set(all_actions)
    scores["unique_actions_used"] = len(unique_actions)
    scores["total_actions"] = len(all_actions)

    return scores


def print_report(model, observations, step_log, summary, scores):
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}\n")

    print(f"  Model:               {model}")
    print(f"  Observations:        {scores['observations_collected']}/3")
    print(f"  Descriptive:         {scores['descriptive_observations']}/3")
    print(f"  Distinct locations:  {'Yes' if scores['self_judged_distinct'] else 'No'}")
    print(f"  Confidence:          {scores['self_judged_confidence']}")
    print(f"  Objects seen:        {', '.join(scores['objects_seen']) if scores['objects_seen'] else 'none'}")
    print(f"  Total steps:         {scores['total_steps']}")
    print(f"  Errors:              {scores['error_steps']}")
    print(f"  Unique actions used: {scores['unique_actions_used']}")
    print(f"  Total actions:       {scores['total_actions']}")
    print(f"  Avg LLM latency:     {scores['avg_latency_s']}s")
    print(f"  Total time:          {scores['total_time_s']}s")

    if summary.get("summary"):
        print(f"\n  Environment summary (LLM self-assessment):")
        print(f"    {summary['summary']}")

    print(f"\n  Observations:")
    for obs in observations:
        print(f"    Location {obs['number']} (step {obs['step']}):")
        print(f"      {obs['observation'][:200]}")

    # Overall pass/fail
    passed = (
        scores["all_3_collected"]
        and scores["descriptive_observations"] >= 3
        and scores["self_judged_distinct"]
        and scores["error_steps"] <= scores["total_steps"] * 0.3
    )
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'}")

    if not scores["all_3_collected"]:
        print(f"    - Did not collect all 3 observations")
    if scores["descriptive_observations"] < 3:
        print(f"    - Some observations lacked detail")
    if not scores["self_judged_distinct"]:
        print(f"    - Locations were not judged as distinct")
    if scores["error_steps"] > scores["total_steps"] * 0.3:
        print(f"    - Too many errors ({scores['error_steps']}/{scores['total_steps']})")

    print()
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: navigate to 3 locations and describe the environment"
    )
    parser.add_argument(
        "--host", default="localhost",
        help="LM Studio host IP (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=1234,
        help="LM Studio port (default: 1234)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model identifier (default: auto-detect)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=20,
        help="Max navigation steps before giving up (default: 20)",
    )
    parser.add_argument(
        "--save-images", default=None,
        help="Directory to save captured images from the run",
    )
    parser.add_argument(
        "--save", default=None,
        help="Save results to a JSON file",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key="lm-studio")

    model = args.model
    if not model:
        models = client.models.list()
        if models.data:
            model = models.data[0].id
            print(f"Auto-detected model: {model}")
        else:
            print("No models loaded in LM Studio.")
            return

    px = Picarx()

    camera = Picamera2()
    config = camera.create_still_configuration(main={"size": (640, 480)})
    camera.configure(config)
    camera.start()
    time.sleep(2)
    print("[CAMERA] ready")

    try:
        observations, step_log, total_time = run_exploration(
            client, model, px, camera, args.max_steps, args.save_images
        )

        if observations:
            summary, summary_latency = run_summary(client, model, observations)
        else:
            summary = {}
            summary_latency = 0

        scores = score_run(observations, step_log, summary, total_time)
        passed = print_report(model, observations, step_log, summary, scores)

        if args.save:
            results = {
                "model": model,
                "passed": passed,
                "scores": scores,
                "summary": summary,
                "summary_latency": round(summary_latency, 2),
                "observations": [
                    {
                        "number": o["number"],
                        "step": o["step"],
                        "observation": o["observation"],
                        "latency": o["latency"],
                    }
                    for o in observations
                ],
                "step_log": step_log,
            }
            with open(args.save, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.save}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        px.stop()
        px.set_dir_servo_angle(0)
        px.set_camera_servo1_angle(0)
        px.set_camera_servo2_angle(0)
        camera.stop()
        print("Done.")


if __name__ == "__main__":
    main()
