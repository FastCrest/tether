#!/usr/bin/env bash
# Reflex VLA bootstrap installer
# Runs hardware + Python checks BEFORE pip install so users on
# unsupported configs don't waste time debugging the wrong thing.
#
# Usage:
#   curl -sSf https://fastcrest.com/install | sh
#   curl -sSf https://fastcrest.com/install | sh -s -- --extras serve,gpu
#   curl -sSf https://fastcrest.com/install | sh -s -- --jetson --jetpack 6.2
#
# Source: https://github.com/FastCrest/reflex-vla
set -eu

# -- ANSI colors --------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  BOLD="\033[1m"; DIM="\033[2m"; RESET="\033[0m"
  RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"; BLUE="\033[34m"; CYAN="\033[36m"
else
  BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""
fi

info()  { printf "%b%s%b\n" "${BLUE}" "$*" "${RESET}"; }
ok()    { printf "%b✓%b %s\n" "${GREEN}" "${RESET}" "$*"; }
warn()  { printf "%b⚠%b %s\n" "${YELLOW}" "${RESET}" "$*"; }
fail()  { printf "%b✗%b %s\n" "${RED}" "${RESET}" "$*"; }
note()  { printf "%b%s%b\n" "${DIM}" "$*" "${RESET}"; }

printf "\n%b%s%b\n" "${BOLD}${CYAN}" "Reflex VLA installer" "${RESET}"
note   "  Bootstrap that checks your environment before installing."
echo

# -- Parse args ---------------------------------------------------------------
EXTRAS=""
FORCE_JETSON=0
JETPACK_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --extras) EXTRAS="$2"; shift 2 ;;
    --extras=*) EXTRAS="${1#--extras=}"; shift ;;
    --jetson) FORCE_JETSON=1; shift ;;
    --jetpack) JETPACK_ARG="$2"; shift 2 ;;
    --jetpack=*) JETPACK_ARG="${1#--jetpack=}"; shift ;;
    *) shift ;;
  esac
done

# -- Detect OS ----------------------------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"
info "Environment"
echo "  OS:   $OS"
echo "  Arch: $ARCH"

# -- Detect Python ------------------------------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"; break
  fi
done

if [ -z "$PYTHON" ]; then
  echo
  fail "No Python interpreter found."
  note "  Install Python 3.10+ first:"
  note "    macOS:  brew install python@3.11"
  note "    Linux:  pyenv install 3.11.9 && pyenv global 3.11.9"
  exit 1
fi

PY_VER="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
PY_MAJOR="$($PYTHON -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$($PYTHON -c 'import sys; print(sys.version_info.minor)')"

echo "  Python: $PY_VER  ($PYTHON)"
echo

# -- Hardware detection -------------------------------------------------------
JETSON_MODEL=""
if [ -r /proc/device-tree/model ]; then
  # Strip null bytes that Jetson DT model files often have
  JETSON_MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)"
fi

IS_OLD_NANO=0
IS_JETSON=0
JETPACK_VERSION=""
if [ -n "$JETSON_MODEL" ]; then
  case "$JETSON_MODEL" in
    *"Jetson Nano"*|*"jetson-nano"*|*"Tegra X1"*|*"p3450"*|*"p3448"*)
      IS_OLD_NANO=1 ;;
    *"Orin"*|*"orin"*|*"Thor"*|*"thor"*)
      IS_JETSON=1 ;;
  esac
fi

# Detect JetPack version from nv_tegra_release if available
if [ "$IS_JETSON" -eq 1 ] || [ "$FORCE_JETSON" -eq 1 ]; then
  if [ -r /etc/nv_tegra_release ]; then
    # Format: # R36 (release), REVISION: 2.2, GCID: 34907586, BOARD: ...
    # R36 = JetPack 6.x
    JETPACK_VERSION="$(head -1 /etc/nv_tegra_release | grep -oP 'REVISION: \K[0-9.]+' || true)"
  fi
  # Override with explicit arg
  if [ -n "$JETPACK_ARG" ]; then
    JETPACK_VERSION="$JETPACK_ARG"
  fi
fi

# -- Branch on hardware -------------------------------------------------------
if [ "$IS_OLD_NANO" -eq 1 ]; then
  echo
  fail "Detected: $JETSON_MODEL"
  echo
  warn "The original Jetson Nano (Maxwell, 2019) is too old for Reflex."
  note "  • NVIDIA EOL'd it at JetPack 4.6 (Ubuntu 18 / Python 3.6) in 2022."
  note "  • Maxwell GPU has no Tensor Cores; modern VLA models can't run usefully."
  note "  • 4 GB shared memory will OOM loading pi0 / SmolVLA in FP32."
  echo
  info "Recommended paths:"
  echo "  ${BOLD}1. Run on a Mac${RESET} (CPU path, works for chat + dev)"
  echo "       pip install 'reflex-vla[serve,onnx]'"
  echo "  ${BOLD}2. Run on Jetson Orin${RESET} (Orin Nano / NX / AGX) — full GPU support"
  echo "  ${BOLD}3. Run in the cloud${RESET} (Modal, Lambda, RunPod with NVIDIA T4+)"
  echo
  exit 1
fi

# -- Python version check -----------------------------------------------------
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  fail "Python $PY_VER is too old. Reflex requires Python 3.10+."
  echo
  info "Install a newer Python:"
  case "$OS" in
    Darwin)
      note "  brew install python@3.11"
      note "  # or: pyenv install 3.11.9 && pyenv global 3.11.9" ;;
    Linux)
      note "  pyenv install 3.11.9 && pyenv global 3.11.9"
      note "  # or via your distro: apt install python3.11   (Ubuntu 22+)" ;;
    *)
      note "  Use pyenv or your platform's package manager to install Python 3.10+." ;;
  esac
  echo
  exit 1
fi
ok "Python $PY_VER (>=3.10 required)"

# -- Jetson-specific Python version check ------------------------------------
# JetPack 6 ships Python 3.10. Jetson Zoo wheels are built for cp310.
# Running on Python 3.11+ means no pre-built GPU wheel and a source build.
if [ "$IS_JETSON" -eq 1 ] || [ "$FORCE_JETSON" -eq 1 ]; then
  if [ "$PY_MINOR" -ne 10 ]; then
    echo
    warn "Jetson detected but Python is $PY_VER."
    note "  Jetson Zoo provides GPU wheels for Python 3.10 only."
    note "  Python $PY_VER will work for CPU inference but GPU (CUDA) may fail."
    note "  Recommended: use Python 3.10, e.g. pyenv or a Docker image."
    echo
    if [ "$FORCE_JETSON" -eq 0 ]; then
      read -r -p "Continue anyway? [y/N] " response
      case "$response" in
        [yY][eE][sS]|[yY]) ;;  # continue
        *) exit 1 ;;
      esac
    fi
  fi
fi

# -- Pick install extras based on detected platform ---------------------------
if [ -z "$EXTRAS" ]; then
  if [ "$IS_JETSON" -eq 1 ] || [ "$FORCE_JETSON" -eq 1 ]; then
    # NEVER install [gpu] on Jetson — those are x86_64 wheels (nvidia-cu12,
    # tensorrt). They will either fail to install, segfault, or silently
    # fall back to CPU. Instead we install [serve,monolithic] and then
    # pull the Jetson Zoo onnxruntime-gpu wheel explicitly.
    EXTRAS="serve,monolithic"
    ok "Detected Jetson ($JETSON_MODEL) → installing with [serve,monolithic]"
    note "  Jetson-specific GPU runtime will be installed separately."
  elif [ "$OS" = "Darwin" ]; then
    EXTRAS="serve,onnx,monolithic"
    ok "Detected macOS → installing with [serve,onnx,monolithic] (CPU runtime)"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    EXTRAS="serve,gpu,monolithic"
    ok "Detected NVIDIA GPU → installing with [serve,gpu,monolithic]"
  else
    EXTRAS="serve,onnx,monolithic"
    ok "No GPU detected → installing with [serve,onnx,monolithic] (CPU runtime)"
  fi
  if [ "$EXTRAS" != "serve,monolithic" ] && [ "$IS_JETSON" -eq 0 ]; then
    note "  (monolithic adds the extras 'reflex go' needs to actually deploy a model — not just chat)"
  fi
fi
echo

# -- Ensure pip is available --------------------------------------------------
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  warn "pip not available for $PYTHON — trying to bootstrap via ensurepip"
  if "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
    ok "Installed pip via ensurepip"
  else
    fail "pip is missing and ensurepip can't bootstrap it on this system."
    echo
    info "Install pip for your distro, then re-run this installer:"
    note "  Arch:    sudo pacman -S python-pip"
    note "  Debian:  sudo apt install python3-pip"
    note "  Fedora:  sudo dnf install python3-pip"
    note "  Alpine:  sudo apk add py3-pip"
    note "  macOS:   pip ships with the Python.org installer / Homebrew Python"
    echo
    exit 1
  fi
fi

# -- Run pip install ----------------------------------------------------------
PIP_TARGET="reflex-vla[$EXTRAS]"
info "Installing: $PIP_TARGET"
echo
"$PYTHON" -m pip install --upgrade "$PIP_TARGET"

# -- Jetson: install Jetson Zoo onnxruntime-gpu -------------------------------
# This must happen AFTER reflex is installed so we override any CPU-only
# onnxruntime that may have been pulled transitively.
if [ "$IS_JETSON" -eq 1 ] || [ "$FORCE_JETSON" -eq 1 ]; then
  echo
  info "Installing Jetson-compatible onnxruntime-gpu..."

  # JetPack version → index URL mapping
  # Default to JP6.0/6.1 (cu126) since that's the current standard.
  # JP6.2+ uses cu129 and has cp310 + cp312 wheels.
  case "${JETPACK_VERSION:-}" in
    6.2*|6.3*|6.4*)
      JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu129"
      ok "Detected JetPack $JETPACK_VERSION → using cu129 index"
      ;;
    6.0*|6.1*|""|*)
      JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"
      if [ -n "${JETPACK_VERSION:-}" ]; then
        ok "Detected JetPack $JETPACK_VERSION → using cu126 index"
      else
        warn "Could not detect JetPack version. Assuming JetPack 6.0/6.1 (cu126)."
        note "  If you are on JetPack 6.2+, run with: --jetpack 6.2"
      fi
      ;;
  esac

  # Pin numpy<2 because Jetson Zoo wheels are compiled against numpy 1.x
  # and will segfault / throw ABI errors with numpy 2.x.
  # Reflex itself works fine with numpy 1.x.
  note "  Pinning numpy<2 for Jetson Zoo ABI compatibility..."
  "$PYTHON" -m pip install 'numpy<2' || warn "numpy pin failed — may cause runtime issues"

  # Install the GPU wheel from Jetson Zoo index
  if "$PYTHON" -m pip install --upgrade --index-url "$JETSON_INDEX" onnxruntime-gpu; then
    ok "Installed onnxruntime-gpu (Jetson Zoo, $JETSON_INDEX)"
  else
    echo
    fail "Jetson Zoo onnxruntime-gpu install failed."
    info "Manual install command:"
    note "  $PYTHON -m pip install numpy '<2'"
    note "  $PYTHON -m pip install --upgrade --index-url $JETSON_INDEX onnxruntime-gpu"
    echo
    warn "Reflex is installed but inference will fall back to CPU (slow)."
  fi
fi

echo
ok "Installed."
echo
info "Next:"
echo "  reflex doctor      # check your install (Jetson: should show CUDAExecutionProvider)"
echo "  reflex chat        # natural-language CLI"
echo "  reflex --help      # see all commands"
echo

if [ "$IS_JETSON" -eq 1 ] || [ "$FORCE_JETSON" -eq 1 ]; then
  info "Jetson deployment:"
  echo "  reflex go --preset franka    # one-command deploy (after ONNX export)"
  echo "  reflex serve /path/to/export  # serve a pre-exported model"
  echo
  note "For a fully-bundled JetPack Docker image:"
  note "  docker pull ghcr.io/fastcrest/reflex-vla:latest-jetpack"
fi

note "Source: https://github.com/FastCrest/reflex-vla"
