"""Pi0.5 shape signature whitelist — guards the Triton fast-kernels path.

FluxVLA's vendored Triton kernels (in ``src/reflex/kernels/triton/``) hard-code
Pi0.5's PaliGemma SigLIP-base vision encoder shapes — see
``reference/FluxVLA/fluxvla/models/vlas/pi05_flowmatching_inference.py:23-30``:

    num_patches = 256          # (image_size / patch_size)^2 = (224/14)^2
    vit_hidden = 1152          # SigLIP-base hidden_size
    vit_intermediate = 4304    # SigLIP-base intermediate_size
    vit_num_heads = 16         # SigLIP-base num_attention_heads
    grid_size = 16             # image_size / patch_size = 224 / 14
    patch_size = 14            # SigLIP-base patch_size

A customer attempting ``reflex serve --fast-kernels`` on a non-PaliGemma SigLIP
model (DinoSigLIP, EVA-CLIP, SigLIP-large with different shapes) would trigger
**silently wrong outputs** because Triton block sizes baked into the kernels
won't match the new shapes — kernels would do partial writes and return
plausible-looking garbage.

Per FM-7 of ``features/01_serve/triton-fast-kernels_research.md``, the defense
is **refuse-to-fast-kernels**: validate the model's `reflex_config.json` against
this whitelist at runtime construction; if it doesn't match a known-good
signature, raise a clear error directing the user to fall back to ORT.

V1 scope is Pi0.5 with PaliGemma SigLIP-base only (T-2). GR00T + SmolVLA paths
defer to Phase 2.5+; their entries here are intentionally absent.
"""
from __future__ import annotations

from typing import Any


# Pi0.5 / PaliGemma SigLIP-base — the only shape signature V1 supports.
# Sourced from FluxVLA's pi05_flowmatching_inference.py:23-30 hard-codes.
PI05_SHAPE_SIGNATURES: dict[str, dict[str, int]] = {
    "paligemma_siglip_base": {
        "num_patches": 256,
        "vit_hidden": 1152,
        "vit_intermediate": 4304,
        "vit_num_heads": 16,
        "grid_size": 16,
        "patch_size": 14,
        "image_size": 224,
    },
}


# Required keys in `reflex_config.json` (or the kwargs passed to the runtime)
# that we read for shape validation. Missing keys → REJECT (don't guess).
REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "vit_hidden",
    "vit_intermediate",
    "vit_num_heads",
    "image_size",
    "patch_size",
)


def validate_shape_signature(
    reflex_config: dict[str, Any],
) -> tuple[bool, str]:
    """Validate a model's shape signature against the Pi0.5 V1 whitelist.

    Args:
        reflex_config: Parsed ``reflex_config.json`` (or any dict with the
            keys listed in ``REQUIRED_CONFIG_KEYS``).

    Returns:
        ``(True, "")`` if every key matches the ``paligemma_siglip_base``
        signature. ``(False, msg)`` otherwise — ``msg`` cites the offending
        key, its value, and the expected value. Caller is expected to surface
        ``msg`` in the refuse-to-load error.
    """
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in reflex_config]
    if missing:
        return (
            False,
            f"reflex_config missing required keys for fast-kernels: {sorted(missing)}. "
            f"--fast-kernels requires PaliGemma SigLIP-base shapes; fall back to ORT "
            f"or supply a richer reflex_config.json.",
        )

    sig = PI05_SHAPE_SIGNATURES["paligemma_siglip_base"]
    mismatches: list[str] = []
    for key in REQUIRED_CONFIG_KEYS:
        got = reflex_config[key]
        want = sig[key]
        if got != want:
            mismatches.append(f"{key}: got={got!r}, want={want!r}")

    # num_patches is derived (image_size / patch_size)^2 — verify the
    # derivation matches if both are present.
    if "num_patches" in reflex_config:
        derived = (reflex_config["image_size"] // reflex_config["patch_size"]) ** 2
        if reflex_config["num_patches"] != derived:
            mismatches.append(
                f"num_patches: got={reflex_config['num_patches']!r}, "
                f"want={derived!r} (=(image_size/patch_size)^2)"
            )

    if mismatches:
        return (
            False,
            "fast-kernels requires PaliGemma SigLIP-base shapes; got "
            + "; ".join(mismatches)
            + ". Fall back to ORT (drop --fast-kernels) for this model.",
        )

    return (True, "")


def supported_signatures() -> tuple[str, ...]:
    """List the model shape signatures supported by V1 fast-kernels.

    V1 = pi0.5 only per T-2. GR00T + SmolVLA defer to Phase 2.5+.
    """
    return tuple(PI05_SHAPE_SIGNATURES.keys())


__all__ = [
    "PI05_SHAPE_SIGNATURES",
    "REQUIRED_CONFIG_KEYS",
    "supported_signatures",
    "validate_shape_signature",
]
