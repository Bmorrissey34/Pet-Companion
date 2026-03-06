# PiCar-X Vision-Language Navigation

## Overview

This project implements autonomous indoor navigation for a Raspberry Pi 5 robot built on the SunFounder PiCar-X platform. The robot captures camera images, sends them to a Vision-Language Model (VLM) through LM Studio’s OpenAI-compatible API, receives navigation decisions, and executes movement commands in real time.

During development and testing, the primary model was **Qwen3-VL-8B**.

## System Overview

### Hardware

- Raspberry Pi 5
- SunFounder PiCar-X
- Camera
- Ultrasonic sensor
- Pan-tilt mount

### Software

- Python
- LM Studio
- Qwen3-VL-8B model
- Tailscale networking

## Repository Structure

```text
pet-companion/
├── bench/
│   ├── benchmark_simple.py
│   ├── benchmark_v2.py
│   ├── benchmark_v3.py
│   ├── benchmark_v4.py
│   └── benchmark_rewrite.py
└── picarx_run_log.jsonl
```

## Benchmark Experiments

All experiments are implemented in the `bench/` directory:

- `benchmark_simple.py`: initial proof-of-concept navigation.
- `benchmark_v2.py`: introduced camera head scanning and multiple observations.
- `benchmark_v3.py`: added structured JSON outputs and logging.
- `benchmark_v4.py`: hybrid navigation combining LLM reasoning with onboard sensors.
- `benchmark_rewrite.py`: rewrite of the original `benchmark_simple.py` with updated commands matching actual PiCar function calls.

## Deployment

For a full reproducible setup on Raspberry Pi and LM Studio host, see [deployment.md](deployment.md).

## Additional Documentation

- [architecture.md](architecture.md)
- [benchmarks.md](benchmarks.md)
