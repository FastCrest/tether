#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup supports Linux only." >&2
  exit 2
fi
command -v nvidia-smi >/dev/null || { echo "An NVIDIA driver and nvidia-smi are required." >&2; exit 2; }
command -v uv >/dev/null || { echo "Install uv before running this setup." >&2; exit 2; }

if command -v apt-get >/dev/null; then
    if [[ "$(id -u)" -eq 0 ]]; then
        apt=(apt-get)
    elif command -v sudo >/dev/null; then
        apt=(sudo apt-get)
    else
        echo "apt-get needs root access. Install the packages listed in docs/eval_local_linux.md." >&2
        exit 2
    fi
    "${apt[@]}" update
    DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y \
        build-essential cmake ffmpeg git libegl1 libgl1 libglib2.0-0 \
        libglvnd0 libosmesa6 libosmesa6-dev
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${TETHER_EVAL_ENV:-$repo_dir/.venv-eval-linux}"
libero_dir="${TETHER_LIBERO_DIR:-$repo_dir/.deps/LIBERO}"
libero_commit="8f1084e3132a39270c3a13ebe37270a43ece2a01"

uv venv --python 3.12 "$env_dir"
uv pip install --python "$env_dir/bin/python" -e "$repo_dir[eval-local]"
if [[ ! -d "$libero_dir/.git" ]]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$libero_dir"
fi
git -C "$libero_dir" fetch origin "$libero_commit"
git -C "$libero_dir" checkout --detach "$libero_commit"
uv pip install --python "$env_dir/bin/python" --no-deps -e "$libero_dir"

MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa "$env_dir/bin/python" -c \
  'import torch, libero, lerobot, peft; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
echo "Evaluation environment ready: $env_dir"
