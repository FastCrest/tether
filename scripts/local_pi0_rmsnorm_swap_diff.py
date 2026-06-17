"""Pi0 native parity: PyTorch reference vs PyTorch with DecomposedRMSNorm swap.

Minimal isolation test: the claim of native-path-parity is that swapping
RMSNorm -> DecomposedRMSNorm is numerically equivalent. If cos=1.0000 holds
on pi0, the same SmolVLA pattern generalizes and multi-model-native-parity
is viable.

Critical: pi0 uses GemmaRMSNorm (PaliGemma backbone) with (1+weight)
parameterization. A naive Llama-style swap produces wrong output. This
script uses `swap_rmsnorm_variants` which dispatches correctly.

Fast local iteration. Does NOT need Modal. Requires `pip install 'fastcrest-tether[native]'`.
"""
import sys
import types

# Python 3.13 + lerobot 0.5.1 compat: upstream GR00TN15Config dataclass
# violates 3.13's strict non-default-after-default check. Stub before import.
for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
    _stub = types.ModuleType(_mod)
    _stub.GrootPolicy = None
    _stub.GR00TN15 = None
    sys.modules[_mod] = _stub

import copy
import numpy as np
import torch


def build_dummy_inputs(policy, device):
    """Construct minimal valid pi0 inputs for sample_actions.

    We bypass preprocessor to isolate the RMSNorm-swap question: if cos=1.0000
    here, the decomposition is numerically equivalent independent of preprocess.
    """
    cfg = policy.config
    bsize = 1

    # images: list of [B, 3, 224, 224] tensors (PaliGemma expects 3x224x224)
    # Count varies; use 1 camera for isolation
    image = torch.randn(bsize, 3, 224, 224, device=device, dtype=torch.float32)
    images = [image]
    img_masks = [torch.ones(bsize, dtype=torch.bool, device=device)]

    # lang_tokens + mask: short dummy prompt tokenized length
    seq_len = 16
    lang_tokens = torch.randint(0, 256000, (bsize, seq_len), device=device, dtype=torch.long)
    lang_masks = torch.ones(bsize, seq_len, dtype=torch.bool, device=device)

    # state: pi0 uses max_state_dim
    state_dim = getattr(cfg, "max_state_dim", 32)
    state = torch.randn(bsize, state_dim, device=device, dtype=torch.float32) * 0.1

    # noise: shape matches action sampling
    noise_shape = (bsize, cfg.chunk_size, cfg.max_action_dim)
    return images, img_masks, lang_tokens, lang_masks, state, noise_shape


def main():
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from tether.decompose import swap_rmsnorm_variants

    model_id = "lerobot/pi0_base"
    device = "cpu"  # CPU is fine for a numerical parity check; slow but deterministic

    print(f"Loading {model_id} ...")
    policy_ref = PI0Policy.from_pretrained(model_id)
    policy_ref.eval().to(dtype=torch.float32).to(device)

    images, img_masks, lang_tokens, lang_masks, state, noise_shape = build_dummy_inputs(
        policy_ref, device
    )

    # Fixed noise for deterministic comparison
    rng = np.random.RandomState(99)
    noise_np = rng.randn(*noise_shape).astype(np.float32)
    noise = torch.from_numpy(noise_np).to(device)

    print("Running reference (original GemmaRMSNorm) ...")
    with torch.no_grad():
        actions_ref = policy_ref.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
        )

    # Swap RMSNorm on a deep copy
    print("Copying policy + swapping RMSNorm ...")
    policy_swap = copy.deepcopy(policy_ref)
    n_swapped = swap_rmsnorm_variants(policy_swap)
    print(f"  swapped {n_swapped} RMSNorm layers")

    print("Running swapped (DecomposedGemmaRMSNorm) ...")
    with torch.no_grad():
        actions_swap = policy_swap.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
        )

    # Compare first action
    a_ref = actions_ref[0, 0].detach().cpu().numpy()
    a_swp = actions_swap[0, 0].detach().cpu().numpy()
    diff = a_ref - a_swp
    l2 = float(np.linalg.norm(diff))
    max_abs = float(np.max(np.abs(diff)))
    denom = np.linalg.norm(a_ref) * np.linalg.norm(a_swp) + 1e-8
    cos = float(np.dot(a_ref, a_swp) / denom)

    print()
    print(f"  ref  first action: {np.round(a_ref[:7], 4).tolist()} ...")
    print(f"  swap first action: {np.round(a_swp[:7], 4).tolist()} ...")
    print(f"  L2={l2:.6f}  max_abs={max_abs:.6e}  cos={cos:+.6f}")

    if cos > 0.9999 and max_abs < 1e-3:
        print("  PASS: pi0 RMSNorm swap is numerically equivalent (cos=1.0000)")
    else:
        print("  FAIL: swap introduced numerical drift — investigate")
        sys.exit(1)


if __name__ == "__main__":
    main()
