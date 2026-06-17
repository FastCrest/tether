"""Full-pipeline local diff: PyTorch policy action vs our ONNX.

Runs end-to-end on the SAME preprocessed batch, SAME noise, compares.
Fast local iteration (~1 min). Does NOT need Modal.
"""
import sys
import types

# Python 3.13 + lerobot 0.5.1 compat: lerobot's policies/__init__ eagerly
# imports GrootPolicy, whose GR00TN15Config dataclass violates Python 3.13's
# stricter non-default-after-default validation. We only need SmolVLAPolicy,
# so stub the broken groot modules before lerobot imports. Upstream bug.
for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
    _stub = types.ModuleType(_mod)
    _stub.GrootPolicy = None
    _stub.GR00TN15 = None
    sys.modules[_mod] = _stub

import numpy as np
import torch


def main():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    from huggingface_hub import snapshot_download

    repo = snapshot_download("lerobot/smolvla_libero")

    print("Loading policy ...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
    policy.eval().to(dtype=torch.float32).to("cpu")

    # Override device to cpu in processor config (default is cuda, blows up on mac)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    rng = np.random.RandomState(42)
    img = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    state8 = rng.randn(8).astype(np.float32) * 0.1

    batch_raw = {
        "observation.images.camera1": img_t.unsqueeze(0),
        "observation.images.camera2": img_t.unsqueeze(0),
        "observation.images.camera3": img_t.unsqueeze(0),
        "observation.state": torch.from_numpy(state8).unsqueeze(0),
        "task": ["put the red bowl on the plate"],
    }
    batch_pp = preprocessor(batch_raw)

    chunk = policy.config.chunk_size
    max_action = policy.config.max_action_dim
    noise_np = np.random.RandomState(99).randn(1, chunk, max_action).astype(np.float32)

    print("Running policy.predict_action_chunk ...")
    with torch.no_grad():
        torch_actions_norm = policy.predict_action_chunk(
            batch_pp, noise=torch.from_numpy(noise_np)
        ).cpu()
    # Apply postprocessor to get unnormalized actions (matches native server)
    torch_actions = postprocessor(torch_actions_norm)
    if hasattr(torch_actions, "detach"):
        torch_actions = torch_actions.detach().cpu().numpy()
    else:
        torch_actions = np.asarray(torch_actions)

    import os
    if os.environ.get("TETHER_NATIVE", "0") == "1":
        print("Running NATIVE path (lerobot SmolVLAPolicy + DecomposedRMSNorm) ...")
        from tether.runtime.smolvla_native import SmolVLANativeServer
        server = SmolVLANativeServer("/tmp/tether_libero_export3", device="cpu", strict_providers=False)
    else:
        print("Running decomposed ONNX pipeline ...")
        from tether.runtime.server import TetherServer
        server = TetherServer("/tmp/tether_libero_export3", device="cpu", strict_providers=False)
    server.load()

    # same image as raw batch
    result = server.predict(
        image=[img, img, img],
        instruction="put the red bowl on the plate",
        state=state8,
        noise=noise_np,
    )
    onnx_actions = np.asarray(result["actions"], dtype=np.float32)

    # Compare first action
    t_first = torch_actions[0, 0, :7]
    o_first = onnx_actions[0, :7]
    abs_diff = np.abs(t_first - o_first)
    l2 = float(np.linalg.norm(t_first - o_first))
    cos = float(np.dot(t_first, o_first) / (np.linalg.norm(t_first) * np.linalg.norm(o_first) + 1e-8))

    print(f"  torch first action: {np.round(t_first, 3).tolist()}")
    print(f"  onnx  first action: {np.round(o_first, 3).tolist()}")
    print(f"  L2={l2:.3f}  cos={cos:+.3f}")


if __name__ == "__main__":
    main()
