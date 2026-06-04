"""Per-model seeded fixture generators for VLA validation.

Produces deterministic (image, prompt, state) tuples for SmolVLA, pi0, and GR00T.
pi0.5 and OpenVLA are explicitly unsupported in v1.
"""

from __future__ import annotations

import numpy as np
import torch

_PROMPTS: list[str] = [
    "pick up the red cup",
    "move the block to the left",
    "place the object on the shelf",
    "push the button",
    "grasp the handle",
]

# (image_hw, state_dim) per supported model
_MODEL_SHAPES: dict[str, tuple[int, int]] = {
    "smolvla": (512, 6),
    "pi0": (224, 14),
    "gr00t": (224, 64),
}

_UNSUPPORTED_MSG = (
    "tether validate v1 supports smolvla, pi0, gr00t. "
    "For pi0.5 / openvla see roadmap."
)


def load_fixtures(
    model_type: str,
    num: int,
    seed: int = 0,
) -> list[tuple[np.ndarray, str, np.ndarray]]:
    """Generate `num` deterministic (image, prompt, state) tuples.

    Args:
        model_type: One of "smolvla", "pi0", "gr00t".
        num: Number of fixtures to produce.
        seed: Seed for the CPU torch.Generator.

    Returns:
        List of (image, prompt, state) tuples where:
          - image is a float32 [H, W, 3] array in [0, 1]
          - prompt is a string drawn (with repetition) from the curated list
          - state is a float32 [action_dim] array

    Raises:
        ValueError: For pi0.5/openvla (unsupported in v1) or unknown model types.
    """
    if model_type in ("pi05", "openvla"):
        raise ValueError(_UNSUPPORTED_MSG)
    if model_type not in _MODEL_SHAPES:
        raise ValueError(f"unknown model_type: {model_type}")

    image_hw, state_dim = _MODEL_SHAPES[model_type]

    # Seeded CPU generator — torch.manual_seed not used to avoid global state mutation.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    fixtures: list[tuple[np.ndarray, str, np.ndarray]] = []
    for i in range(num):
        image = (
            torch.rand((image_hw, image_hw, 3), generator=g)
            .numpy()
            .astype(np.float32)
        )
        state = (
            torch.randn((state_dim,), generator=g).numpy().astype(np.float32)
        )
        prompt = _PROMPTS[i % len(_PROMPTS)]
        fixtures.append((image, prompt, state))

    return fixtures
