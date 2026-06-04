# Troubleshooting

Common errors and fixes when deploying Tether on edge devices, cloud GPUs, ROS2 robots, and drones.

> **First step on any failure:** run [`tether doctor`](./doctor_check_list.md). As of v0.9.4 it surfaces the four most-common silent-failure modes (multi-GPU mixed architecture, Jetson R35 silent CPU fallback, cuDNN vs driver version skew, TensorRT EP loadchain breakage) with the specific remediation commands. Most of the errors below are also caught by `tether doctor` before they manifest at deploy time.

---

## CUDA & GPU errors

### `libcudnn version mismatch`

```
Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file
```

**Cause:** Your cuDNN version doesn't match what ONNX Runtime was compiled against. Tether requires cuDNN 9.5+ since v0.9.2 (Blackwell support bump).

**Fix:**
```bash
# Check your versions
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import onnxruntime; print(onnxruntime.__version__, onnxruntime.get_available_providers())"

# Install the supported ORT + cuDNN combo (CUDA 12.x)
pip install 'onnxruntime-gpu>=1.25.1' 'nvidia-cudnn-cu12>=9.5' 'nvidia-cublas-cu12>=12.6'
```

**Jetson fix:** On JetPack 6.0+, do not install the `[gpu]` extra from standard PyPI (since those wheels are `x86_64` only). Install `[serve,monolithic]` and then pull the Jetson-compatible `onnxruntime-gpu` wheel from the Jetson AI Lab index:
```bash
# Pin numpy<2 for Jetson Zoo ABI compatibility
pip install 'numpy<2'

# JetPack 6.0 / 6.1 (cu126)
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# Or JetPack 6.2+ (cu129)
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu129
```

JetPack 5.x (R35 / CUDA 11.4) is not supported — `tether doctor` will flag this loudly with the upgrade path.

---

### `cudaErrorNoKernelImageForDevice` (Blackwell / RTX 5090 / B200)

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Cause:** Blackwell GPUs (RTX 50-series, RTX PRO Blackwell, B200, GB200) require ONNX Runtime ≥ 1.25.0 — earlier versions don't ship sm_100 / sm_120 kernels.

**Fix:**
```bash
pip install -U 'onnxruntime-gpu>=1.25.1'
```

This is the v0.9.3 doctor guard. Caveat: there's an open ORT issue (#27621) about a silent threading deadlock on sm_120 with multi-threaded `InferenceSession.run()`. Tether's single-server, single-request path doesn't trigger it; customers running `--max-batch >1` should monitor.

---

### `NVIDIA driver too old for CUDA runtime`

```
CUDA driver version is insufficient for CUDA runtime version
```

**Cause:** Your GPU driver is older than the CUDA toolkit / cuDNN combo Tether was built against.

**Fix:**
```bash
nvidia-smi  # check current driver

# Tether floors (since v0.9.2):
#   cuDNN 9.0-9.4 → driver R550+
#   cuDNN 9.5+    → driver R555+  (current default — Blackwell support)

# Update (Ubuntu)
sudo apt install nvidia-driver-555
sudo reboot
```

The cuDNN-vs-driver skew is one of the silent-failure modes `tether doctor` checks for explicitly since v0.9.4 — it reads `nvidia-smi --query-gpu=driver_version` plus `importlib.metadata.version('nvidia-cudnn-cu12')` and surfaces the gap.

---

### `Out of memory during graph capture`

```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Cause:** The model + CUDA graph capture exceeds your GPU's VRAM.

| GPU | Available VRAM | Max model |
|---|---|---|
| Jetson Orin Nano | 8 GB | SmolVLA only |
| Jetson AGX Orin | 32–64 GB | Pi0, Pi0.5 |
| RTX 3060 | 12 GB | SmolVLA |
| RTX 4090 | 24 GB | SmolVLA, Pi0 (LoRA) |
| A10G | 24 GB | SmolVLA, Pi0 |
| A100-40 | 40 GB | Pi0 full |
| A100-80 / H100 | 80 GB | Pi0.5, GR00T |

**Fix:**
```bash
# Reduce memory usage
tether export <model> --precision fp16     # half precision (default)
tether serve ./export/ --device cuda       # ensure GPU, not CPU fallback

# Monitor VRAM
watch -n1 nvidia-smi
```

---

### `CUDAExecutionProvider not available`

```
[WARNING] CUDAExecutionProvider is not available. Falling back to CPUExecutionProvider.
```

**Cause:** ONNX Runtime was installed without CUDA support, OR CUDA libraries aren't on `LD_LIBRARY_PATH`, OR the TRT EP loadchain is broken (libnvinfer.so.10 missing).

**Fix:**
```bash
# Confirm what ORT actually sees
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# Reinstall GPU-enabled ORT
pip install 'onnxruntime-gpu>=1.25.1'

# Add CUDA libs to path if needed
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

The v0.9.4 `tether doctor` guard goes further than `get_available_providers()` — it creates a stub ONNX model + forces TRT EP load and checks `sess.get_providers()` for actual session-init success. The default `available_providers` reports the lib loaded; the empirical session test catches the case where TRT silently falls back to CUDA EP because of a `libnvinfer.so.10` dlopen failure.

---

## Jetson-specific issues

### `JetPack version not supported`

**Minimum:** JetPack 6.0 (L4T R36.x) with CUDA 12.2+.

```bash
cat /etc/nv_tegra_release
# or
dpkg -l | grep nvidia-l4t-core
```

JetPack 5.x is not supported. The v0.9.4 doctor guard parses `/etc/nv_tegra_release`, detects R35, and surfaces the upgrade path loudly — without it, ORT silently falls back to CPU and you get useless latency numbers.

### `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

```
ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**Cause:** The Jetson AI Lab `torch` and `onnxruntime-gpu` wheels are compiled against NumPy 1.x C ABI. If `numpy>=2.0` is installed (pip's default), both libraries crash on import.

**Fix:** Pin `numpy<2` **before** installing torch or onnxruntime-gpu:
```bash
pip install 'numpy<2'
# Then install torch / ort from the Jetson AI Lab index
```

If you already installed numpy 2.x, downgrade:
```bash
pip install 'numpy<2' --force-reinstall
```

### `No matching distribution found for lerobot==0.5.1` (Python 3.10)

```
ERROR: Could not find a version that satisfies the requirement lerobot==0.5.1; extra == "monolithic"
ERROR: Ignored the following versions that require a different python version: 0.5.0 Requires-Python >=3.12; 0.5.1 Requires-Python >=3.12
```

**Cause:** The `[monolithic]` (and `[native]`, `[rtc]`) extras depend on `lerobot==0.5.1`, which requires Python ≥ 3.12. JetPack 6 ships Python 3.10.

**Fix:** On Jetson, install `[serve]` only — **not** `[monolithic]`:
```bash
pip install 'tether[serve]'
```

The monolithic ONNX export (`tether export --monolithic`) requires lerobot and must run on a **Python 3.12+ host** (desktop, cloud GPU, or Docker). Export there, then copy the ONNX to the Jetson and serve it:
```bash
# On Jetson — serve a pre-exported model
tether serve /path/to/exported/model/
```

### `Thermal throttling during inference`

**Symptoms:** Latency spikes after 5–10 minutes of continuous inference.

**Fix:**
```bash
sudo nvpmodel -m 0     # MAXN on Orin
sudo jetson_clocks     # lock clocks to max

# Sustained workloads need active cooling — passive heatsinks aren't enough
```

> **For drone deployments:** Jetson modules in UAV enclosures have limited airflow. Use `tether serve --deadline-ms` to gracefully drop late frames rather than blocking the control loop when thermal throttling kicks in.

### `Multi-GPU mixed architecture warning`

If `nvidia-smi` reports 2+ GPUs of different generations (e.g. 1× H100 + 1× RTX 5090), `tether doctor` warns since v0.9.4: ORT only uses `CUDA_VISIBLE_DEVICES[0]`, and switching GPUs at runtime silently fails with arch-mismatched kernels. Set `CUDA_VISIBLE_DEVICES` explicitly to the GPU you actually want.

---

## ROS2 bridge issues

### `ModuleNotFoundError: No module named 'rclpy'`

`rclpy` is NOT pip-installable. You need a ROS2 install (humble / iron / jazzy) via apt or robostack.

```bash
source /opt/ros/humble/setup.bash   # or iron / jazzy
source ~/ros2_ws/install/setup.bash # if you have a workspace
tether ros2-serve ./export/ ...
```

### `No state data received`

```
[ros2_bridge] state callback never fired
```

**Cause:** The bridge is subscribed to the wrong topic, OR the publisher upstream isn't running, OR the message type doesn't match the configured extractor.

**Fix:** Verify the topic is publishing:

```bash
ros2 topic list
ros2 topic echo /joint_states --once               # arms
ros2 topic echo /mavros/local_position/odom --once # drones (full state)
ros2 topic echo /mavros/imu/data --once            # drones (orientation-only fallback)
```

Then match the bridge config to the topic type:

```bash
# Arms (default)
tether ros2-serve ./export/ \
  --state-topic /joint_states \
  --state-msg-type joint_state

# Drone with full 10-DOF state (pos + quat + linear velocity)
tether ros2-serve ./export/ \
  --state-topic /mavros/local_position/odom \
  --state-msg-type odom

# Drone with orientation-only fallback (4 DOF)
tether ros2-serve ./export/ \
  --state-topic /mavros/imu/data \
  --state-msg-type imu
```

### `State vector length doesn't match embodiment state_dim`

If you see this warning at startup:

```
[ros2_bridge] WARN: state vector length 4 from 'imu' extractor does NOT match embodiment state_dim 10
```

It means the extractor's output shape doesn't match what the policy was trained on — common when wiring a drone preset (10-DOF state expected) to the IMU extractor (4-DOF orientation only). Switch to `--state-msg-type odom` with `/mavros/local_position/odom`, or load an embodiment whose `state_dim` matches the 4-DOF IMU output.

---

## Drone / MAVROS issues

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
# Expected: 50–200 Hz

# If 0 Hz, FCU isn't talking — check state topic
ros2 topic echo /mavros/state --once
```

### `Action output not reaching flight controller`

**Cause:** Tether publishes actions to `/tether/actions` as `std_msgs/Float32MultiArray`, but PX4/ArduPilot expects `mavros_msgs/AttitudeTarget` (or `mavros_msgs/PositionTarget` depending on control mode). You need a bridge node to convert.

**Skeleton:**
```python
# Map tether 5-DOF action chunks → MAVROS attitude targets
# Per the shipped quadcopter preset:
#   action[0:3] = body rates (roll_rate, pitch_rate, yaw_rate) in rad/s
#   action[3]   = thrust normalized [0, 1]
#   action[4]   = payload_release trigger (≥ trigger_threshold = 0.5 → release)
```

See [`docs/adding_a_robot.md`](./adding_a_robot.md) for the full embodiment spec.

---

## Export & validation issues

### `Opset 19 not supported`

```
onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph: Unsupported opset 19
```

**Fix:**
```bash
# Downgrade the export's opset
tether export <model> --opset 17

# Or upgrade ORT (preferred — opset 19 is the current default)
pip install --upgrade 'onnxruntime-gpu>=1.25.1'
```

### `VERIFICATION.md says "Not yet verified"`

**Cause:** Export completed but `tether validate` hasn't been run yet.

**Fix:**
```bash
tether validate ./tether_export/
# Populates the parity table in VERIFICATION.md
```

See [`docs/verification.md`](./verification.md) for how to read the resulting report.

### `Parity fails with max_abs_diff > 1e-04`

Diagnostic sequence:

1. **Re-export with default precision** — some custom `--precision fp8` / `int8` settings widen the tolerance gap.
2. **Try a lower opset:** `--opset 17` falls back to more numerically stable kernels for attention-heavy models.
3. **Run `tether doctor --export-dir <dir>`** — catches the common silent-failure modes (cuDNN skew, TRT EP loadchain breaks, JetPack mismatch) that manifest as parity failures.
4. **File an issue** with `VERIFICATION.md` + `tether doctor` output attached.

---

## Model registry issues

### `Unknown embodiment preset`

```
ValueError: Unknown embodiment preset 'myrobot'.
```

**Fix:**
```bash
# See available presets
python3 -c "from tether.embodiments import list_presets; print(list_presets())"
# ['franka', 'quadcopter', 'so100', 'ur5'] as of v0.9.6

# Or use a custom config
tether go --model smolvla-base --custom-embodiment-config ./myrobot.json
```

See [`docs/adding_a_robot.md`](./adding_a_robot.md) for adding a new preset to the registry.

### `Model pull timeout`

```bash
# Increase timeout
export HF_HUB_DOWNLOAD_TIMEOUT=300
tether models pull smolvla-base

# Use a regional mirror if you're outside the US
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Quick diagnostic commands

The fastest way to localize a problem:

```bash
# Full system check — version skew, GPU detection, ORT providers, JetPack, cuDNN/driver
tether doctor

# Deploy-specific diagnostics (validates the actual export will run on this hardware)
tether doctor --export-dir ./tether_export/

# Embodiment-specific
tether doctor --export-dir ./tether_export/ --embodiment franka
tether doctor --export-dir ./tether_export/ --embodiment quadcopter

# Raw GPU + ORT introspection
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
python3 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

---

## See also

- [`docs/doctor_check_list.md`](./doctor_check_list.md) — what `tether doctor` checks, full list with remediation
- [`docs/cli_reference.md`](./cli_reference.md) — every tether command + flag
- [`docs/verification.md`](./verification.md) — interpreting parity output
- [`docs/adding_a_robot.md`](./adding_a_robot.md) — embodiment cookbook
- Discord: [discord.gg/wrPUdcxdPu](https://discord.gg/wrPUdcxdPu) — fastest path to a human if you're stuck
