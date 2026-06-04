# Hardware Requirements Guide

> VRAM, compute, and platform requirements for inference, fine-tuning, and edge deployment across all Tether-supported verticals.

---

## Inference Requirements

### GPU VRAM by Model

| Model | Precision | Min VRAM | Recommended GPU | Latency (p50) |
|---|---|---|---|---|
| SmolVLA | FP16 | 4 GB | Jetson Orin Nano, RTX 3060 | ~25 ms |
| SmolVLA | INT8 | 2 GB | Jetson Orin Nano | ~20 ms |
| Pi0 | FP16 | 16 GB | RTX 4090, A10G | ~50 ms |
| Pi0 | FP8 | 10 GB | RTX 4090 | ~35 ms |
| Pi0.5 | FP16 | 32 GB | A100-40, AGX Orin 64GB | ~80 ms |
| Pi0.5 (distilled) | FP16 | 24 GB | RTX 4090, A100-40 | ~45 ms |
| GR00T N1 | FP16 | 40 GB | A100-40, H100 | ~100 ms |

### Jetson Platform Matrix

| Jetson Module | VRAM (shared) | Max Model | Ideal Vertical |
|---|---|---|---|
| **Orin Nano 8GB** | 8 GB | SmolVLA | Farm robots, hobby arms (SO-100) |
| **Orin NX 16GB** | 16 GB | SmolVLA, Pi0 (INT8) | Warehouse arms, retail cameras |
| **AGX Orin 32GB** | 32 GB | Pi0, Pi0.5 (distilled) | Industrial arms, drones |
| **AGX Orin 64GB** | 64 GB | Pi0.5, GR00T N1 | Multi-camera rigs, traffic AI |
| **Thor** | 128 GB | All models | Data centers, cloud inference |

> **Drone deployments:** Jetson Orin Nano or NX is the sweet spot — SmolVLA fits comfortably in 8 GB with enough headroom for MAVROS, the camera pipeline, and the flight controller bridge. The AGX Orin is preferred for multi-camera (front + downward) setups.

### Desktop / Cloud GPU Matrix

| GPU | VRAM | Max Model | Typical Use |
|---|---|---|---|
| **RTX 3060** | 12 GB | SmolVLA | Development, testing |
| **RTX 4090** | 24 GB | Pi0, Pi0.5 (distilled) | Local development, single-arm |
| **A10G** | 24 GB | Pi0 | Cloud inference (AWS g5) |
| **A100-40** | 40 GB | Pi0.5, GR00T N1 | Cloud training + inference |
| **A100-80** | 80 GB | All models | Full fine-tuning |
| **H100** | 80 GB | All models | Fastest inference + training |
| **H200** | 141 GB | All models + large batch | Multi-instance serving |

---

## Fine-Tuning Requirements

### LoRA Fine-Tuning

| Model | Method | Min VRAM | Recommended GPU | Training Time (1k episodes) |
|---|---|---|---|---|
| SmolVLA | LoRA (r=16) | 12 GB | RTX 3060 / RTX 4090 | ~2 hours |
| Pi0 | LoRA (r=16) | 24 GB | RTX 4090 / A10G | ~6 hours |
| Pi0 | LoRA (r=32) | 32 GB | A100-40 | ~8 hours |
| Pi0.5 | LoRA (r=16) | 40 GB | A100-40 | ~12 hours |

### Full Fine-Tuning

| Model | Method | Min VRAM | Recommended GPU | Training Time (1k episodes) |
|---|---|---|---|---|
| SmolVLA | Full | 24 GB | RTX 4090 | ~4 hours |
| Pi0 | Full | 80 GB | A100-80 | ~24 hours |
| Pi0.5 | Full + distill | 80 GB | A100-80 / H100 | ~48 hours |

### Distillation

| Task | Min VRAM | Recommended Setup |
|---|---|---|
| Pi0.5 → Pi0 distill | 80 GB | A100-80 or 2× A100-40 (DDP) |
| SmolVLA self-distill | 24 GB | RTX 4090 |

---

## CPU & Memory Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **CPU** | 4 cores | 8+ cores (ARM or x86_64) |
| **System RAM** | 8 GB | 16+ GB |
| **Disk (model cache)** | 5 GB | 20+ GB SSD |
| **Disk (datasets)** | 10 GB | 100+ GB NVMe |

> **Jetson note:** System RAM is shared with GPU VRAM. An 8 GB Orin Nano has ~5.5 GB effective for the model after OS + ROS2 overhead.

---

## Network Requirements

| Operation | Bandwidth | Notes |
|---|---|---|
| `tether models pull` | 50+ Mbps recommended | SmolVLA ~500 MB, Pi0.5 ~4 GB |
| `tether serve` (local) | N/A | All inference is local |
| `tether serve` (cloud fallback) | 10+ Mbps | For `--cloud-fallback` mode |
| ROS2 image topics | LAN only | ~30 MB/s for 640×480 RGB at 30 fps |

---

## Vertical-Specific Hardware Recommendations

### Warehouse Robotics (Pick-and-Place Arms)

| Component | Recommendation |
|---|---|
| **Compute** | Jetson AGX Orin 32 GB or desktop RTX 4090 |
| **Model** | Pi0 for precision, SmolVLA for speed |
| **Camera** | Intel RealSense D435i (wrist) + overhead USB cam |
| **Control rate** | 20–30 Hz |
| **Notes** | Mount Jetson on the robot base; use PoE for cameras |

### Farm Robotics (SO-100, Outdoor Arms)

| Component | Recommendation |
|---|---|
| **Compute** | Jetson Orin Nano 8 GB (low power, weatherproof enclosure) |
| **Model** | SmolVLA (fits in 8 GB with room for ROS2) |
| **Camera** | USB webcam with UV filter for outdoor lighting |
| **Control rate** | 20 Hz |
| **Notes** | Power from 12V battery; consider IP67 enclosure |

### Aerial Drones (Quadcopters, Fixed-Wing)

| Component | Recommendation |
|---|---|
| **Compute** | Jetson Orin Nano (weight-constrained) or Orin NX (multi-cam) |
| **Model** | SmolVLA (low latency, low VRAM) |
| **Camera** | Front-facing + downward-facing RGB |
| **Control rate** | 50 Hz (PX4/ArduPilot outer loop) |
| **Flight controller** | Pixhawk 6C/X with MAVROS bridge |
| **Notes** | Weight budget: Orin Nano ~50g, NX ~75g. Use `--deadline-ms` for thermal throttle resilience |

### Retail Loss Prevention (Camera-Only)

| Component | Recommendation |
|---|---|
| **Compute** | RTX 4090 or A10G (cloud/edge server) |
| **Model** | Pi0 or Pi0.5 for high-accuracy perception |
| **Camera** | PoE IP cameras (Hikvision, Axis) via RTSP |
| **Throughput** | `--max-batch-cost-ms 200` for multi-camera batching |
| **Notes** | No robot arm — pure visual inference. Use `--deadline-ms` for real-time alerting |

### Traffic Management AI

| Component | Recommendation |
|---|---|
| **Compute** | A100-40 or H100 (high throughput) |
| **Model** | Pi0.5 for complex scene understanding |
| **Camera** | Intersection-mounted fixed cameras via RTSP |
| **Throughput** | `--max-batch-cost-ms 300` for maximum batching efficiency |
| **Notes** | Multi-GPU with `CUDA_VISIBLE_DEVICES` for scaling |

---

## Quick Compatibility Check

```bash
# Run tether doctor to verify your setup
tether doctor

# Check GPU info
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
```

---

## Further Reading

- [Troubleshooting](./TROUBLESHOOTING.md) — CUDA and GPU error fixes
- [CLI Command Reference](./cli_reference.md) — all `tether` flags
- [Adding a Robot](./adding_a_robot.md) — embodiment setup guide
