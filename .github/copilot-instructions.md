# PiCar-X Vision-Based Autonomous Navigation

## Project Overview

This codebase implements **vision-language model (VLM) driven autonomous navigation** for the SunFounder PiCar-X robot (Raspberry Pi-based). The robot uses a remote OpenAI-compatible LLM endpoint (LM Studio) to make navigation decisions from camera images, enabling exploration and obstacle avoidance without pre-programmed paths.

**Core Components:**
- **Hardware:** SunFounder PiCar-X v20, Raspberry Pi 5, Picamera2, Ultrasonic sensor
- **Vision:** Vilib library (object detection via TensorFlow Lite), camera pan/tilt servos
- **LLM Integration:** OpenAI client targeting LM Studio (local inference server)
- **Navigation:** Scan-then-commit driving with vision-based steering corrections

## Architecture & Data Flow

### Benchmark Variants (Evolutionary Development)

The `bench/` directory contains **iterative versions** of the navigation system:

1. **`benchmark_simple.py`** - Basic 3-location exploration with LLM navigation decisions
2. **`benchmark_v2.py`** - Adds scan-then-commit driving with dynamic steering
3. **`benchmark_v3.py`** - Enhanced scan logic (details in file header)
4. **`benchmark_v4.py`** ⭐ **CURRENT** - Full system with Vilib object detection steering bias
5. **`benchmark_rewrite.py`** - Simplified single-file variant (no Vilib dependencies)

**Use `benchmark_v4.py` as the reference implementation.** Earlier versions are preserved for rollback/comparison.

### Control Loop (benchmark_v4.py)

```
1. SCAN PHASE:
   - Pan camera LEFT/CENTER/RIGHT (pan_angles: -45°, 0°, 45°)
   - Capture 3 images (optional best-of-two sharpness selection)
   - LLM chooses best direction via llm_choose_direction()

2. COMMIT PHASE:
   - Drive in chosen direction for commit_time_s (1.0-2.5s typical)
   - Loop at loop_hz (6Hz default), checking:
     * Ultrasonic sensor (emergency avoidance at <25cm)
     * Vilib object detections (steer bias away from obstacles)
   - Adaptive commit time: extends if front is clear (no objects in center band)

3. OBSERVATION CAPTURE:
   - Triggered when novel + time/distance gates pass
   - Uses dHash novelty detection (hamming_distance >= novelty_hamming_threshold)
   - LLM describes scene via llm_observe()
   - Logged to picarx_run_log.jsonl (JSONL format)
```

### Key Abstractions

**Robot Adapter Pattern:**
- `RobotAdapter` (benchmark_rewrite.py) wraps `Picarx` to handle API version differences
- `Robot` class (benchmark_v4.py) combines Picarx + Vilib with smooth slewing for steer/speed

**Vilib Object Detection Integration:**
- `vilib_get_detections()` - Extracts detections from multiple attribute candidates (handles Vilib API variations)
- `parse_vilib_detections()` - Normalizes to `{"name", "conf", "x", "y", "w", "h"}` dicts
- `compute_object_bias()` - Converts detections to steering correction (pushes away from obstacles)
- Relevant classes: `chair, couch, dining table, bed, person, dog, cat, toilet, refrigerator, sink`

**Observation Manager:**
- Tracks observation count, timing, and novelty via dHash comparison
- Gates: `enable_observations`, `max_observations`, `min_seconds_between_obs`, `min_commit_loops_between_obs`
- Novelty: dHash hamming distance (8-bit default threshold)

## Critical Developer Workflows

### Running Benchmarks

**All benchmarks require `sudo` for GPIO/camera access on Raspberry Pi:**

```bash
# Navigate to project
cd ~/pet-companion  # Or wherever picar/pet-companion is located
source venv/bin/activate  # If using venv

# Benchmark v4 (full system)
sudo python3 bench/benchmark_v4.py \
  --host 100.82.181.13 \  # Tailscale IP of LM Studio server
  --port 1234 \
  --model "qwen2.5-vl-7b-instruct" \
  --best-of-two \
  --debug-vilib

# Rewrite variant (no Vilib)
sudo python3 bench/benchmark_rewrite.py \
  --host 192.168.1.100 \
  --max-steps 20 \
  --goal-locations 3
```

**Key Flags:**
- `--host` - **Required** (LM Studio host IP, often Tailscale 100.x)
- `--model` - **Required** for v4 (vision-capable model like qwen2.5-vl, llava)
- `--best-of-two` - Capture 2 photos per view, keep sharper (reduces motion blur)
- `--debug-avoid`, `--debug-scan`, `--debug-vilib` - Enable verbose logging
- `--self-test` - Test hardware without driving (v4 only)
- `--no-observations` - Disable observation logging

### Hardware Self-Test

```bash
sudo python3 bench/benchmark_v4.py --self-test --debug-vilib
```
Verifies camera, Vilib object detection, servos without driving.

### Dependencies

**System packages** (Raspberry Pi OS):
```bash
sudo apt-get install python3-picamera2 python3-lgpio
```

**Python packages** (in venv):
```bash
pip install openai pillow numpy
pip install picarx  # SunFounder library (see note below)
pip install vilib    # SunFounder vision library
```

**Note:** `picarx` and `vilib` are SunFounder-provided libraries installed via their setup scripts. Refer to SunFounder documentation for initial robot setup.

## Project-Specific Conventions

### Configuration via Dataclasses

All configuration uses **immutable dataclasses** (see `DriveConfig`, `LLMConfig`, `ObservationConfig` in benchmark_v4.py):
- **Do NOT use globals** for config values
- Pass config objects through function signatures
- Use `dataclasses.replace()` for variants in tests

### Safety Clamping Pattern

Motor power and steering **ALWAYS** use `clamp()`:
```python
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

# Usage:
power = clamp(llm_power, -max_power, max_power)
steer = clamp(llm_steer, -max_steer_deg, max_steer_deg)
```
**Never trust LLM outputs directly** - safety limits are critical for physical robots.

### LLM Prompt Structure

All prompts follow this pattern:
1. **System context** - Robot capabilities, output schema, safety rules
2. **Current state** - Observation history, step count, sensor readings
3. **Decision request** - Explicit output format (JSON only)
4. **Thinking field** - LLM outputs reasoning before action (aids debugging)

Example (benchmark_v4.py `llm_choose_direction()`):
```python
prompt = (
    "You are the navigation brain for a small indoor robot car. "
    "Given three camera views (LEFT, CENTER, RIGHT), choose the best direction..."
    "Output ONLY valid JSON: {\"choice\": \"LEFT|CENTER|RIGHT\", \"thinking\": \"...\"}"
)
```

### JSON Parsing Robustness

LLMs occasionally wrap JSON in markdown. Use `extract_first_json_obj()` pattern:
```python
# Fast path: pure JSON response
if text.strip().startswith("{"):
    return json.loads(text)

# Fallback: extract first {...} block
m = re.search(r"\{.*\}", text, flags=re.DOTALL)
return json.loads(m.group(0))
```

### Logging Format

**JSONL for run logs** (`picarx_run_log.jsonl`):
- One JSON object per line (newline-delimited)
- Required fields: `{"event": "...", "ts": {"unix": float, "human": str}}`
- Use `append_jsonl(path, data)` helper
- Events: `session_start`, `observation_attempt`, `ultrasonic_avoid`, etc.

### Photo Management

Benchmarks capture **many photos** - implement cleanup:
```python
cleanup_photo_dir(photo_dir, keep_latest=300)
```
- Photos named with timestamp: `scan_20260227_134647_L_a.jpg`
- `best_of_two` mode: captures `_a.jpg` and `_b.jpg`, keeps sharper

### Ultrasonic Emergency Avoidance

Standard 3-step maneuver (benchmark_v4.py):
```python
1. STOP immediately
2. Backup straight (avoid_backup_time_s ~1.2s)
3. Peel turn (avoid_peel_steer_angle ~22°, random_turn selects L/R)
```
Triggers at `avoid_stop_cm` (25cm default). **Never disable** - prevents wall collisions.

## Integration Points

### LM Studio Setup (Remote Server)

1. Install LM Studio on desktop/server
2. Load vision-language model (qwen2.5-vl-7b-instruct recommended)
3. Start server: `http://0.0.0.0:1234`
4. Use Tailscale for secure Pi → server connection

**Model Requirements:**
- Must support vision inputs (multimodal)
- Recommended: qwen2.5-vl, llava, minicpm-v
- Minimum 7B parameters for navigation decisions

### Vilib API Quirks

Vilib's object detection attribute **varies by version**:
```python
# Check multiple candidates in order:
_VILIB_DETECTION_ATTR_CANDIDATES = (
    "object_detection_list_parameter",  # Most common
    "object_detection_list",
    "detect_obj_parameter",
    "detect_obj_list",
    "detect_obj",
)
```
Use `vilib_get_detections()` wrapper - handles all variants.

### Picamera2 vs Vilib

- **benchmark_v4**: Uses **Vilib** (owns camera, provides object detection)
- **benchmark_rewrite**: Uses **Picamera2** directly (no Vilib, no object detection)

**Cannot use both simultaneously** - they compete for camera resources.

## Testing Strategy

**No unit tests exist.** Testing is hardware-in-the-loop:
1. `--self-test` flag (v4) - Verifies hardware without driving
2. Short runs with `--debug-*` flags for subsystem validation
3. Save runs with `--save run_log.json` for post-analysis

**When modifying navigation:**
- Test with `--max-observations 1` for quick iteration
- Use `--debug-scan --debug-vilib` to verify sensor integration
- Always test emergency avoidance near walls

## Common Pitfalls

1. **Forgetting `sudo`** - GPIO/camera require root on Pi
2. **Wrong model type** - LLM must support vision inputs
3. **Network firewall** - Ensure LM Studio port (1234) reachable from Pi
4. **Vilib initialization** - Call `Vilib.camera_start()` before any photo operations
5. **Stale observations** - dHash novelty threshold too low causes duplicate captures
6. **Over-tuned safety limits** - Too-conservative ultrasonic thresholds prevent exploration

## File Structure Reference

```
bench/
  benchmark_v4.py          ⭐ Primary implementation
  benchmark_rewrite.py     Vilib-free variant
  benchmark_v2.py          Scan-commit driving baseline
  benchmark_simple.py      Minimal 3-location exploration
picarx_observations/       Observation photos (auto-managed)
picarx_run_log.jsonl       JSONL event log
venv/                      Python virtual environment
```

## Getting Help

- **Hardware issues:** SunFounder PiCar-X documentation
- **Vilib errors:** Check `tflite_runtime` installation (required for object detection)
- **LLM timeouts:** Increase `--llm-timeout-s` or check network connectivity
- **Observation spam:** Increase `--novelty-hamming-threshold` (default 8)

For agent iterations: Always preserve working benchmarks as new versioned files (v5, v6, etc.) rather than overwriting v4.
