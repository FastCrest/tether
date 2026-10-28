# Source after activating rocm-nightly, before importing HF or LeRobot.
export HF_HOME=/data/tether/hf
export HF_HUB_CACHE=/data/tether/hf/hub
export HF_LEROBOT_HOME=/data/tether/lerobot
export HF_DATASETS_CACHE=/data/tether/hf/datasets
export TOKENIZERS_PARALLELISM=false
# ROCm PyTorch uses the torch.cuda API. Do not install the project's CUDA extras.
if test -f /data/tether/rocm-venv/bin/activate; then
    source /data/tether/rocm-venv/bin/activate
fi
export PYTHONPATH=/home/haojiema/code/amd_robot/models/tether/src:${PYTHONPATH:-}
export TMPDIR=/data/tether/tmp
export LIBERO_CONFIG_PATH=/data/tether/libero-config
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TETHER_TEACHER_CHECKPOINT=/data/tether/checkpoints/pi05-libero90mix-torch
