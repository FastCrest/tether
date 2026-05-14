# Troubleshooting

Common errors and fixes when deploying Reflex VLA on edge devices, cloud GPUs, ROS2 robots, and drones.

---

## CUDA & GPU Errors

### `libcudnn version mismatch`

```
Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file
```

**Cause:** Your cuDNN version doesn't match what ONNX Runtime was compiled against.

**Fix:**
```bash
# Check your versions
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Install matching cuDNN (example for CUDA 12.x)
pip install onnxruntime-gpu==1.18.0  # matches CUDA 12.x + cuDNN 8.9
```

**Jetson fix:** On JetPack 6.0+, use the NVIDIA-provided ORT wheel:
```bash
pip install onnxruntime-gpu --extra-index-url https://download.onnxruntime.ai/onnxruntime_stable_jetpack60.html
```

---

### `NVIDIA driver too old for CUDA runtime`

```
CUDA driver version is insufficient for CUDA runtime version
```

**Cause:** Your GPU driver is older than the CUDA toolkit Reflex was built against.

**Fix:**
```bash
# Check driver version
nvidia-smi

# Minimum driver versions:
# CUDA 12.0 → driver 525.60+
# CUDA 12.2 → driver 535.54+
# CUDA 12.4 → driver 550.54+

# Update driver (Ubuntu)
sudo apt install nvidia-driver-550
sudo reboot
```

---

### `Out of memory during graph capture`

```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Cause:** The model + CUDA graph capture exceeds your GPU's VRAM.

**Fix:**
| GPU | Available VRAM | Max Model |
|---|---|---|
| Jetson Orin Nano | 8 GB | SmolVLA only |
| Jetson AGX Orin | 32–64 GB | Pi0, Pi0.5 |
| RTX 3060 | 12 GB | SmolVLA |
| RTX 4090 | 24 GB | SmolVLA, Pi0 (LoRA) |
| A100-40 | 40 GB | Pi0 full |
| A100-80 / H100 | 80 GB | Pi0.5 distill |

```bash
# Reduce memory usage
reflex export <model> --precision fp16     # use half precision
reflex serve ./export/ --device cuda       # ensure GPU, not CPU fallback

# Monitor VRAM
watch -n1 nvidia-smi
```

---

### `CUDAExecutionProvider not available`

```
[WARNING] CUDAExecutionProvider is not available. Falling back to CPUExecutionProvider.
```

**Cause:** ONNX Runtime was installed without CUDA support, or CUDA libraries aren't on `LD_LIBRARY_PATH`.

**Fix:**
```bash
# Install GPU-enabled ORT
pip install onnxruntime-gpu

# Verify
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']

# If CUDA libs aren't found:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## Jetson-Specific Issues

### `JetPack version not supported`

**Minimum:** JetPack 6.0 (L4T R36.x) with CUDA 12.2+.

```bash
# Check JetPack version
cat /etc/nv_tegra_release
# or
dpkg -l | grep nvidia-l4t-core
```

### `Thermal throttling during inference`

**Symptoms:** Latency spikes after 5-10 minutes of continuous inference.

**Fix:**
```bash
# Set max power mode
sudo nvpmodel -m 0        # MAXN on Orin
sudo jetson_clocks         # lock clocks to max

# Add heatsink or active cooling for sustained workloads
```

> **For drone deployments:** Jetson modules in UAV enclosures have limited airflow. Use `--deadline-ms` to gracefully handle thermal throttle spikes rather than dropping frames.

---

## ROS2 Bridge Issues

### `No state data received`

```
[ros2_bridge] unsupported image size/encoding
```

**Cause:** The bridge isn't receiving data on the expected topics.

**Fix:**
```bash
# Check topics are publishing
ros2 topic list
ros2 topic echo /joint_states --once    # arms
ros2 topic echo /mavros/imu/data --once # drones

# Ensure topic names match your launch
reflex ros2-serve ./export/ \
  --state-topic /joint_states            # arms
  # or
  --state-topic /mavros/imu/data         # drones (MAVROS)
```

### `Empty state vector for drone topics`

**Cause:** The bridge expects `msg.position` (from `JointState`) but receives `msg.orientation` (from `Imu`).

**Fix:** Update to the latest `reflex-vla` — PR #121 adds auto-detection for IMU quaternion orientation. Or specify the correct topic type.

### `ROS2 not found`

```
ModuleNotFoundError: No module named 'rclpy'
```

**Fix:**
```bash
# Source your ROS2 workspace
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Then run
reflex ros2-serve ./export/ ...
```

---

## Drone / MAVROS Issues

### `MAVROS not connecting to FCU`

```
[WARN] FCU: DeviceError: serial0: open: No such file or directory
```

**Fix:**
```bash
# Check serial port
ls /dev/ttyACM* /dev/ttyUSB*

# Launch MAVROS with correct port
ros2 launch mavros mavros.launch.py fcu_url:=serial:///dev/ttyACM0:921600
```

### `IMU data not arriving`

```bash
# Verify MAVROS is publishing
ros2 topic hz /mavros/imu/data
# Should show 50-200 Hz

# If 0 Hz, check FCU connection and MAVROS status
ros2 topic echo /mavros/state --once
```

### `Action output not reaching flight controller`

**Cause:** Reflex publishes to `/reflex/actions` as `Float32MultiArray`, but PX4/ArduPilot expects `mavros_msgs/AttitudeTarget`.

**Fix:** You need a bridge node to convert. Example:
```python
# Convert reflex actions → MAVROS attitude target
# action[0:3] = roll_rate, pitch_rate, yaw_rate
# action[3]   = thrust (0-1)
```

---

## Export & Validation Issues

### `Opset 19 not supported`

```
onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph: Unsupported opset 19
```

**Fix:**
```bash
# Downgrade to opset 17
reflex export <model> --opset 17

# Or upgrade ORT
pip install --upgrade onnxruntime-gpu
```

### `VERIFICATION.md says "Not yet verified"`

**Cause:** Export completed but validation hasn't been run yet.

**Fix:**
```bash
reflex validate ./reflex_export/
# This populates the parity table in VERIFICATION.md
```

See [Understanding VERIFICATION.md](./understanding_verification.md) for interpreting the results.

---

## Model Registry Issues

### `Unknown embodiment preset`

```
ValueError: Unknown embodiment preset 'myrobot'.
```

**Fix:**
```bash
# See available presets
python3 -c "from reflex.embodiments import list_presets; print(list_presets())"

# Use a custom config file instead
reflex go --model smolvla-base --custom-embodiment-config ./myrobot.json
```

### `Model pull timeout`

```bash
# Use a mirror or set longer timeout
export HF_HUB_DOWNLOAD_TIMEOUT=300
reflex models pull smolvla-base
```

---

## Quick Diagnostic Commands

```bash
# Full system check
reflex doctor

# Deploy-specific diagnostics
reflex doctor --model ./reflex_export/ --embodiment franka

# Drone diagnostics
reflex doctor --model ./reflex_export/ --embodiment quadcopter

# GPU info
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# ORT providers
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```
