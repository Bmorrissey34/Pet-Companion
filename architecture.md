# Architecture

## Overview

This project implements a vision-language navigation loop for a SunFounder PiCar-X robot controlled by a Raspberry Pi 5. The robot captures camera observations, sends them to an LM Studio-hosted model through an OpenAI-compatible API, and executes returned navigation decisions.

## Robot Hardware Components

- Raspberry Pi 5 (main compute)
- SunFounder PiCar-X chassis and motor control
- Camera on pan-tilt mount
- Ultrasonic sensor for close-range safety checks

## Camera Observation Pipeline

1. Camera captures scene frames.
2. Frames are packaged by Python benchmark scripts.
3. Image data is sent to the remote model endpoint in LM Studio.

The pan-tilt mount supports multi-angle observation during benchmark runs.

## LM Studio API Interface

- LM Studio runs on a host computer.
- OpenAI-compatible API server is enabled (host `0.0.0.0`, port `1234`).
- Raspberry Pi connects over Tailscale to reach the API endpoint.

## Vision-Language Model Reasoning

- Tested model: **Qwen3-VL-8B**.
- The model receives image-based context and navigation prompts.
- The model returns direction or movement decisions used by control scripts.

## Robot Motion Control

- Benchmark scripts convert model decisions into PiCar-X motion commands.
- Steering and movement are issued through SunFounder libraries.
- Different benchmark versions evaluate incremental control strategies.

## Sensor Safety Overrides

- Ultrasonic sensing provides near-obstacle awareness.
- Safety logic can override planned movement to avoid collisions.
- Hybrid behavior in later benchmarks combines model reasoning with onboard sensor constraints.

## End-to-End Pipeline

Camera → Image Capture → LM Studio API → Vision Language Model → Navigation Decision → Robot Motion
