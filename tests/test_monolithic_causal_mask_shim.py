"""The monolithic exporter's `create_causal_mask` stand-in.

Returning `None` unconditionally is what regressed pi0/pi0.5 `num_steps=10`
export parity from cos=1.000000 to cos=0.977: the block-causal mask the export
`denoise_step` patch assembles with `F.pad` is correct and concrete, and
discarding it silently removes prefix-pad masking on the PaliGemma + Gemma
path. `scripts/modal_pi0_monolithic_export.py` always dispatched on a prepared
4D mask; `src/tether/exporters/monolithic.py` lost that when it was extracted.
"""

from __future__ import annotations

import numpy as np

from tether.exporters.monolithic import _causal_mask_export_shim

PREFIX_LEN = 835
SUFFIX_LEN = 51
CHUNK = 50
BIG_NEG = -2.3819763e38  # lerobot OPENPI_ATTENTION_MASK_VALUE


def test_prepared_4d_mask_is_returned_unchanged() -> None:
    mask = np.zeros((1, 1, SUFFIX_LEN, PREFIX_LEN + SUFFIX_LEN), dtype=np.float32)
    assert _causal_mask_export_shim(attention_mask=mask) is mask
    # Positional call form: (config, inputs_embeds, attention_mask, ...).
    assert _causal_mask_export_shim(object(), object(), mask) is mask


def test_everything_not_already_4d_falls_back_to_none() -> None:
    # SmolVLA passes a 3D mask and applies it inside its own attention
    # implementation, so it must keep taking the None branch.
    for mask in (
        None,
        np.zeros((1, PREFIX_LEN), dtype=np.float32),
        np.zeros((1, SUFFIX_LEN, PREFIX_LEN), dtype=np.float32),
        np.zeros((1, 1, 1, SUFFIX_LEN, PREFIX_LEN), dtype=np.float32),
    ):
        assert _causal_mask_export_shim(attention_mask=mask) is None
    assert _causal_mask_export_shim() is None


def test_inputs_embeds_is_never_mistaken_for_a_mask() -> None:
    embeds = np.zeros((1, SUFFIX_LEN, 8), dtype=np.float32)
    assert _causal_mask_export_shim(inputs_embeds=embeds, attention_mask=None) is None


def _pi0_suffix_additive_mask() -> np.ndarray:
    """pi0's suffix block from `lerobot.policies.pi0.make_att_2d_masks`.

    `embed_suffix` emits `att_masks = [1] + [1] + [0] * (chunk - 1)`: the state
    token must not attend to the action tokens. The mask is therefore not
    all-allow, which is exactly why dropping it changes the result.
    """
    att_masks = np.array([1, 1] + [0] * (CHUNK - 1), dtype=np.int64)
    cumsum = np.cumsum(att_masks)
    allowed = cumsum[None, :] <= cumsum[:, None]
    assert not allowed.all(), "suffix mask must actually mask something"
    return np.where(allowed, 0.0, BIG_NEG).astype(np.float32)


def test_dropping_the_mask_materially_changes_attention() -> None:
    """Measure the cost of the None branch on pi0's real suffix mask."""
    rng = np.random.default_rng(0)
    dim = 64
    q = rng.standard_normal((SUFFIX_LEN, dim)).astype(np.float32)
    k = rng.standard_normal((SUFFIX_LEN, dim)).astype(np.float32)
    v = rng.standard_normal((SUFFIX_LEN, dim)).astype(np.float32)
    scores = (q @ k.T) / np.sqrt(dim, dtype=np.float32)

    def attend(logits: np.ndarray) -> np.ndarray:
        weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return (weights / weights.sum(axis=-1, keepdims=True)) @ v

    masked = attend(scores + _pi0_suffix_additive_mask()).ravel()
    unmasked = attend(scores).ravel()
    cosine = float(masked @ unmasked / (np.linalg.norm(masked) * np.linalg.norm(unmasked)))
    # The 0.999 floor is the pi0 export-parity threshold in
    # docs/pi0-export-parity.md. Dropping the mask does not clear it.
    assert cosine < 0.999, f"expected a real divergence, got cos={cosine}"
