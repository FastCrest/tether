"""Tether accelerated engines (FP8): Triton engines for Jetson Thor, RTX 4090 and RTX 5090.

Each engine prepares a checkpoint's weights once (per-channel FP8), runs the
model with the kernels in :mod:`tether.accel.kernels`, and captures its whole
request path in CUDA graphs keyed only by buffer geometry (one graph per
prompt-length bucket). Activation scales are computed per token as data flows,
so there is no calibration pass on live observations.

Engines:

* :mod:`tether.accel.pi05` -- pi0.5 (PaliGemma prefix + Gemma action expert).
  :class:`~tether.accel.pi05.inference.AccelPi05Inference` implements the
  rollout/serving ``InferenceProtocol`` (``predict_action_chunk``).
* :mod:`tether.accel.groot_n17` -- GR00T N1.7 action head (VLSA + 4-step DiT),
  installed in place of a native head's ``get_action``.

Hardware: only the compute capabilities in :data:`SUPPORTED_DEVICES`. Jetson
Orin (sm_87) is deliberately absent: it has no FP8 tensor cores, and Orin
deployments use the TensorRT export path. Importing this package needs neither
Triton nor a GPU.

Provenance: ported from an Apache-2.0 upstream; see ``THIRD_PARTY_NOTICES`` in
this directory and ``docs/provenance/instinct-port.md``.
"""

from __future__ import annotations

#: CUDA compute capability -> the target the engines were qualified on upstream.
#: Other parts sharing a capability (L4/L40S at sm_89, RTX 5080 at sm_120) run
#: the same code path but are not qualified targets.
SUPPORTED_DEVICES: dict[tuple[int, int], str] = {
    (11, 0): "jetson_thor",
    (8, 9): "rtx4090",
    (12, 0): "rtx5090",
}


class UnsupportedAccelDevice(RuntimeError):
    """The CUDA device is not one the accel engines support. Never fall back silently."""


def require_supported_device(device=None) -> str:
    """Return the qualified target name for ``device`` or raise :class:`UnsupportedAccelDevice`."""
    import torch

    if not torch.cuda.is_available():
        raise UnsupportedAccelDevice(
            "tether.accel engines require a CUDA device; none is available"
        )
    cap = tuple(torch.cuda.get_device_capability(device))
    target = SUPPORTED_DEVICES.get(cap)
    if target is None:
        name = torch.cuda.get_device_name(device)
        hint = " Jetson Orin uses the TensorRT export path." if cap == (8, 7) else ""
        raise UnsupportedAccelDevice(
            f"{name} (sm_{cap[0]}{cap[1]}) is not supported by tether.accel; supported: "
            + ", ".join(f"sm_{a}{b} ({t})" for (a, b), t in SUPPORTED_DEVICES.items())
            + "."
            + hint
        )
    return target


__all__ = ["SUPPORTED_DEVICES", "UnsupportedAccelDevice", "require_supported_device"]
