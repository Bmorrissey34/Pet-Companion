# Deployment Guide

## System Overview

This deployment configures a Raspberry Pi 5 running a SunFounder PiCar-X robot to use a Vision-Language Model served from LM Studio. The robot captures camera images and sends them to LM Studio via the OpenAI-compatible API. Networking between the Raspberry Pi and host machine is handled with Tailscale.

Model used during testing: **Qwen3-VL-8B**.

## Project Structure

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

## Raspberry Pi Setup

### 1) Install Raspberry Pi OS and update packages

```bash
sudo apt update
sudo apt upgrade -y
```

### 2) Install required system packages

```bash
sudo apt install python3-pip python3-venv git curl -y
sudo apt install python3-picamera2 python3-lgpio -y
```

### 3) Enable camera

Run the configuration utility:

```bash
sudo raspi-config
```

Then enable the camera interface and reboot if prompted.

## Install Tailscale

### 1) Install

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

### 2) Connect node

```bash
sudo tailscale up
```

Log in from the browser flow when prompted.

### 3) Retrieve Raspberry Pi Tailscale IP

```bash
tailscale ip -4
```

Use this IPv4 address for LM Studio host communication.

## Clone the Repository

```bash
cd ~
git clone <repository-url>
cd pet-companion
```

## Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Python Dependencies

```bash
pip install pillow numpy requests openai
```

## Install PiCar-X Library

```bash
cd ~
git clone https://github.com/sunfounder/picar-x.git
cd picar-x
pip install .
```

## Install Vilib

```bash
cd ~
git clone https://github.com/sunfounder/vilib.git
cd vilib
pip install .
```

## LM Studio Setup

1. Install LM Studio on the host computer.
2. Load the model: **Qwen3-VL-8B**.
3. Enable the OpenAI-compatible API server.
4. Configure:
   - Host: `0.0.0.0`
   - Port: `1234`

## Running the Benchmark

On Raspberry Pi:

```bash
cd ~/pet-companion/bench
python3 benchmark_v4.py \
--host <tailscale-ip> \
--model qwen3-vl-8b \
--run-log-path ../picarx_run_log.jsonl
```

If your hardware permissions require elevated access, run with `sudo`.

## Output Files

- `picarx_run_log.jsonl`: newline-delimited JSON run log containing benchmark events and observations.

## Troubleshooting

### Camera not detected

- Confirm camera cable seating and orientation.
- Re-run `sudo raspi-config` and verify camera is enabled.
- Ensure `python3-picamera2` is installed.

### GPIO errors

- Ensure `python3-lgpio` is installed.
- Run from Raspberry Pi hardware (not a non-Pi environment).
- If required by your setup, run benchmark commands with `sudo`.

### LM Studio connection issues

- Confirm LM Studio API server is enabled.
- Verify host is `0.0.0.0` and port is `1234`.
- Confirm both devices are online in Tailscale.
- Test that `--host <tailscale-ip>` matches the LM Studio host’s Tailscale IP.

### Python module errors

- Activate virtual environment: `source venv/bin/activate`.
- Reinstall missing dependencies with `pip install ...`.
- Ensure `picar-x` and `vilib` were installed with `pip install .`.

### Memory errors

- Close other heavy applications on Pi and host.
- Use a single benchmark process at a time.
- Check that the LM Studio host has enough memory for Qwen3-VL-8B.

### Robot movement issues

- Verify PiCar-X power supply and battery state.
- Check servo/motor connections.
- Confirm ultrasonic and camera modules are connected and initialized.
- Re-run benchmark with shorter sessions to isolate behavior.
