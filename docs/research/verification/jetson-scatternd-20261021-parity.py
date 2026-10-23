"""Diagnostic: strip ScatterND reduction='none' on the real SmolVLA export; parity vs original ORT and PyTorch."""

import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from tether.exporters.monolithic import _repair_exported_onnx  # noqa: E402
from tether.runtime.ort_providers import onnx_has_scatternd_reduction  # noqa: E402

d = Path(sys.argv[1])
seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
orig, fixed = d / "model.onnx", d / "model_stripped.onnx"
shutil.copyfile(orig, fixed)
t = time.time()
n = _repair_exported_onnx(fixed)
repair_s = time.time() - t
onnx.checker.check_model(str(fixed))  # path form handles >2GB external data
m = onnx.load(str(fixed), load_external_data=False)
left = sum(
    1
    for x in m.graph.node
    if x.op_type == "ScatterND" and any(a.name == "reduction" for a in x.attribute)
)
total = sum(1 for x in m.graph.node if x.op_type == "ScatterND")
mo = onnx.load(str(orig), load_external_data=False)
same_except_attr = len(mo.graph.node) == len(m.graph.node) and all(
    a.op_type == b.op_type and list(a.input) == list(b.input) and list(a.output) == list(b.output)
    for a, b in zip(mo.graph.node, m.graph.node)
)


def inputs(seed):
    r = np.random.default_rng(seed)
    return dict(
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


so = ort.SessionOptions()
so.log_severity_level = 3
s_orig = ort.InferenceSession(str(orig), so, providers=["CPUExecutionProvider"])
s_fix = ort.InferenceSession(str(fixed), so, providers=["CPUExecutionProvider"])

import torch  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval().to("cpu").to(torch.float32)
policy.model.config.num_steps = 10
for mod in policy.model.modules():
    if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
        mod.config._attn_implementation = "eager"


def stats(a):
    a = np.abs(a).ravel()
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "frac_gt_1e-6": float((a > 1e-6).mean()),
        "frac_gt_1e-4": float((a > 1e-4).mean()),
        "frac_gt_1e-3": float((a > 1e-3).mean()),
    }


d_strip, d_torch_fix, d_torch_orig, scale = [], [], [], []
for seed in range(seeds):
    x = inputs(seed)
    a = s_orig.run(None, x)[0]
    b = s_fix.run(None, x)[0]
    with torch.inference_mode():
        tx = {k: torch.from_numpy(v) for k, v in x.items()}
        ref = policy.model.sample_actions(
            [tx["img_cam1"], tx["img_cam2"], tx["img_cam3"]],
            [tx["mask_cam1"], tx["mask_cam2"], tx["mask_cam3"]],
            tx["lang_tokens"],
            tx["lang_masks"],
            tx["state"],
            noise=tx["noise"],
        ).numpy()
    d_strip.append(b - a)
    d_torch_fix.append(b - ref)
    d_torch_orig.append(a - ref)
    scale.append(np.abs(ref).max())
    print(
        f"seed {seed}: strip-vs-orig max {np.abs(b - a).max():.3e}, fixed-vs-torch max {np.abs(b - ref).max():.3e}",
        flush=True,
    )

print(
    json.dumps(
        {
            "repairs_applied": n,
            "repair_seconds": round(repair_s, 1),
            "scatternd_total": total,
            "scatternd_with_reduction_after": left,
            "graph_otherwise_identical": same_except_attr,
            "preflight_before": onnx_has_scatternd_reduction(orig),
            "preflight_after": onnx_has_scatternd_reduction(fixed),
            "checker": "ok",
            "seeds": seeds,
            "output_shape": list(a.shape),
            "stripped_vs_original_ort_cpu": stats(np.stack(d_strip)),
            "stripped_ort_cpu_vs_pytorch_cpu": stats(np.stack(d_torch_fix)),
            "original_ort_cpu_vs_pytorch_cpu": stats(np.stack(d_torch_orig)),
            "pytorch_output_absmax": float(max(scale)),
            "versions": {
                "onnxruntime": ort.__version__,
                "onnx": onnx.__version__,
                "torch": torch.__version__,
            },
        },
        indent=1,
    )
)
