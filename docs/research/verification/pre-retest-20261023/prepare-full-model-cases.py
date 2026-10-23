import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from tether.exporters.monolithic import _force_eager_attn

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).parent
out = root / "full-model-cases"
out.mkdir(exist_ok=True)
thresholds = {
    "mixed_fp16": {"max_abs": 0.01, "cosine": 0.999},
    "fp32_vs_pytorch": {"max_abs": 0.0001, "cosine": 0.99999},
    "scope": "numerical engineering checks on eight generated raw-input cases; not task accuracy or robot safety",
}
(out / "acceptance.json").write_text(json.dumps(thresholds, indent=2) + "\n")
checkpoint = (
    root
    / "hf-cache/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
)
torch.set_num_threads(4)
policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).eval().to("cpu").to(torch.float32)
_force_eager_attn(policy.model)
policy.model.config.num_steps = 10
rows = []
for seed in range(8):
    r = np.random.default_rng(seed)
    x = dict(
        img_cam1=r.standard_normal((1, 3, 512, 512), dtype=np.float32),
        img_cam2=r.standard_normal((1, 3, 512, 512), dtype=np.float32),
        img_cam3=r.standard_normal((1, 3, 512, 512), dtype=np.float32),
        mask_cam1=np.ones(1, bool),
        mask_cam2=np.ones(1, bool),
        mask_cam3=np.array([seed % 2 == 0]),
        lang_tokens=r.integers(0, 49152, (1, 16), dtype=np.int64),
        lang_masks=np.ones((1, 16), bool),
        state=r.standard_normal((1, 32), dtype=np.float32),
        noise=r.standard_normal((1, 50, 32), dtype=np.float32),
    )
    np.savez(out / f"case-{seed}.npz", **x)
    tx = {k: torch.from_numpy(v) for k, v in x.items()}
    started = time.monotonic()
    with torch.inference_mode():
        ref = policy.model.sample_actions(
            [tx[k] for k in ["img_cam1", "img_cam2", "img_cam3"]],
            [tx[k] for k in ["mask_cam1", "mask_cam2", "mask_cam3"]],
            tx["lang_tokens"],
            tx["lang_masks"],
            tx["state"],
            noise=tx["noise"],
        ).numpy()
    assert np.isfinite(ref).all()
    np.save(out / f"pytorch-{seed}.npy", ref)
    row = {
        "seed": seed,
        "wall_s": time.monotonic() - started,
        "shape": list(ref.shape),
        "absmax": float(np.max(np.abs(ref))),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
(out / "pytorch-reference.json").write_text(json.dumps(rows, indent=2) + "\n")
