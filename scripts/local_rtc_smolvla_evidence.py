"""Local RTC evidence: SmolVLA RTC-on vs RTC-off through the real /act path.

Runs the full-size SmolVLA architecture (450M parameters, LeRobot 0.6.1,
chunk 50, 10 denoise steps) with RANDOM weights: no checkpoint is downloaded,
so this measures the mechanism (seam continuity, boundary jerk, latency
overhead), not task success. The SmolVLM backbone config comes from the local
HF cache (HF_HUB_OFFLINE=1); only its two special-token ids are read from the
processor, which is patched.

Kinematic sim on a virtual clock: the robot tracks every commanded action
exactly (state = last executed action). It executes ``stride`` actions of the
current chunk, then requests the next one, reporting ``actions_executed`` +
``chunk_id``. While inference runs it keeps executing the old chunk for
``ceil(latency * hz)`` actions, then switches to the new chunk at that index
(naive async for RTC off; LeRobot guidance for RTC on). ``hz`` is chosen so
the median measured latency maps to the target delay.

Arms, on matched seeds per chunk index:
  off       plain chunking, async switch
  rtc       LeRobot 0.6.1 guidance (identity Jacobian) — what Tether serves
  rtc_vjp   paper/Kinetix guidance (full VJP through the denoiser), patched
            into LeRobot's RTCProcessor for this diagnostic only
  rtc_nograd  LeRobot's identity-Jacobian arithmetic without enable_grad
            (diagnostic only: isolates the autograd-graph cost)

    HF_HUB_OFFLINE=1 python scripts/local_rtc_smolvla_evidence.py \
        --device mps --chunks 100 --delays 3 6 10 --out evidence.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ACTION_DIM = 7
STATE_DIM = 8


def build_policy(device: str, seed: int):
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla import smolvlm_with_expert as swe
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.manual_seed(seed)
    cfg = SmolVLAConfig(
        device=device,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.image2": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            ),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )
    proc = SimpleNamespace(
        tokenizer=SimpleNamespace(fake_image_token_id=49189, global_image_token_id=49190)
    )
    with patch.object(swe.AutoProcessor, "from_pretrained", return_value=proc):
        policy = SmolVLAPolicy(cfg)
    return policy.to(device).eval(), cfg


def postprocessor(mean, std):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import PolicyProcessorPipeline, UnnormalizerProcessorStep
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )

    return PolicyProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(
                features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
                norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                stats={"action": {"mean": mean, "std": std}},
            )
        ],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )


def vjp_denoise_step(
    self,
    x_t,
    prev_chunk_left_over,
    inference_delay,
    time,
    original_denoise_step_partial,
    execution_horizon=None,
):
    """RTCProcessor.denoise_step with the paper's VJP (Kinetix model.py L237-251):
    v_t is computed AFTER x_t requires grad, so the correction is J^T W err."""
    import torch

    if prev_chunk_left_over is None:
        return original_denoise_step_partial(x_t)
    tau = 1 - time
    x_t = x_t.clone().detach()
    if prev_chunk_left_over.dim() < 3:
        prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)
    horizon = min(
        execution_horizon or self.rtc_config.execution_horizon, prev_chunk_left_over.shape[1]
    )
    b, t, a = x_t.shape
    y = torch.zeros(b, t, a, device=x_t.device)
    y[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
    w = self.get_prefix_weights(inference_delay, horizon, t).to(x_t.device)[None, :, None]
    with torch.enable_grad():
        x_t.requires_grad_(True)
        v_t = original_denoise_step_partial(x_t)
        x1 = x_t - time * v_t
        err = (y - x1) * w
        correction = torch.autograd.grad(x1, x_t, err.detach())[0]
    beta = torch.as_tensor(self.rtc_config.max_guidance_weight)
    tau_t = torch.as_tensor(tau)
    inv_r2 = ((1 - tau_t) ** 2 + tau_t**2) / ((1 - tau_t) ** 2)
    c = torch.nan_to_num((1 - tau_t) / tau_t, posinf=beta)
    g = torch.minimum(torch.nan_to_num(c * inv_r2, posinf=beta), beta)
    vjp_denoise_step.rel.append(float((correction - err).norm() / (err.norm() + 1e-12)))
    return v_t.detach() - g * correction


vjp_denoise_step.rel = []


def nograd_denoise_step(
    self,
    x_t,
    prev_chunk_left_over,
    inference_delay,
    time,
    original_denoise_step_partial,
    execution_horizon=None,
):
    """LeRobot 0.6.1's identity-Jacobian step without the autograd graph.

    Same arithmetic as RTCProcessor.denoise_step (correction == err), computed
    under no_grad. Diagnostic only: isolates what LeRobot's enable_grad costs."""
    import torch

    if prev_chunk_left_over is None:
        return original_denoise_step_partial(x_t)
    tau = 1 - time
    if prev_chunk_left_over.dim() < 3:
        prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)
    horizon = min(
        execution_horizon or self.rtc_config.execution_horizon, prev_chunk_left_over.shape[1]
    )
    b, t, a = x_t.shape
    y = torch.zeros(b, t, a, device=x_t.device)
    y[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
    w = self.get_prefix_weights(inference_delay, horizon, t).to(x_t.device)[None, :, None]
    with torch.no_grad():
        v_t = original_denoise_step_partial(x_t)
        err = (y - (x_t - time * v_t)) * w
    beta = torch.as_tensor(self.rtc_config.max_guidance_weight)
    tau_t = torch.as_tensor(tau)
    inv_r2 = ((1 - tau_t) ** 2 + tau_t**2) / ((1 - tau_t) ** 2)
    c = torch.nan_to_num((1 - tau_t) / tau_t, posinf=beta)
    g = torch.minimum(torch.nan_to_num(c * inv_r2, posinf=beta), beta)
    return v_t - g * err


def make_client(policy, device, rtc_config, export_dir):
    from fastapi.testclient import TestClient

    from tether.runtime.server import create_app

    os.environ["TETHER_NATIVE"] = "1"
    app = create_app(str(export_dir), device=device, rtc_config=rtc_config, prewarm=False)
    server = app.state.tether_server
    mean = np.linspace(-0.5, 0.5, ACTION_DIM).astype(np.float32)
    std = np.linspace(0.5, 2.0, ACTION_DIM).astype(np.float32)

    def load():
        import torch

        server.policy = policy
        if not os.environ.get("RTC_EVIDENCE_TRAINABLE_PARAMS"):
            server._freeze_policy()  # what SmolVLANativeServer.load does
        server.preprocessor = lambda batch: {
            **batch,
            "observation.language.tokens": torch.tensor([[1, 5, 9, 13]]),
            "observation.language.attention_mask": torch.ones((1, 4), dtype=torch.bool),
        }
        server.postprocessor = postprocessor(mean, std)
        server._device = device
        server.action_dim = ACTION_DIM
        server.chunk_size = int(policy.config.chunk_size)
        server.num_denoising_steps = int(policy.config.num_steps)
        if server._rtc_required:
            server._action_unnormalizer()
        server._ready = True

    server.load = load
    return TestClient(app)


def seed_all(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_arm(policy, device, arm, hz, chunks, stride, horizon, export_dir, seed0):
    from tether.runtime.rtc_adapter import RtcAdapterConfig

    rtc_cfg = None
    if arm != "off":
        rtc_cfg = RtcAdapterConfig(
            enabled=True,
            execute_hz=hz,
            rtc_execution_horizon=horizon,
            cold_start_discard=0,
            latency_percentile=95,
        )
    proc = policy.model.rtc_processor
    executed: list[np.ndarray] = []
    seams, latencies, delays, applied, prefix_err, seam_jump, starved = [], [], [], [], [], [], 0
    with make_client(policy, device, rtc_cfg, export_dir) as client:
        patcher = None
        if arm in ("rtc_vjp", "rtc_nograd"):
            from lerobot.policies.rtc.modeling_rtc import RTCProcessor

            step = vjp_denoise_step if arm == "rtc_vjp" else nograd_denoise_step
            patcher = patch.object(RTCProcessor, "denoise_step", step)
            patcher.start()
        try:
            state = np.zeros(STATE_DIM, np.float32)

            def act(report):
                body = {"instruction": "pick", "state": state.tolist(), "episode_id": "sim"}
                body.update(report)
                r = client.post("/act", json=body)
                r.raise_for_status()
                return r.json()

            seed_all(seed0)
            resp = act({})
            chunk = np.asarray(resp["actions"], np.float32)
            idx = 0
            for i in range(1, chunks + 1):
                for _ in range(stride):  # execute up to the replan point
                    executed.append(chunk[min(idx, len(chunk) - 1)])
                    idx += 1
                state[:ACTION_DIM] = executed[-1]
                report = (
                    {} if arm == "off" else {"actions_executed": idx, "chunk_id": resp["chunk_id"]}
                )
                seed_all(seed0 + i)
                new = act(report)
                lat = float(new["latency_ms"]) / 1000.0
                d = math.ceil(round(lat * hz, 6))
                # Inference runs: the old chunk keeps executing for d actions.
                for k in range(d):
                    j = idx + k
                    if j >= len(chunk):
                        starved += 1
                    executed.append(chunk[min(j, len(chunk) - 1)])
                new_chunk = np.asarray(new["actions"], np.float32)
                if d < len(new_chunk) and idx + d < len(chunk):
                    prefix_err.append(
                        float(np.abs(new_chunk[:d] - chunk[idx : idx + d]).max()) if d else 0.0
                    )
                    # Discontinuity at the switch: new[d] vs where the old plan went.
                    seam_jump.append(float(np.linalg.norm(new_chunk[d] - chunk[idx + d])))
                seams.append(len(executed))
                latencies.append(lat)
                delays.append(d)
                applied.append(bool(new.get("rtc", {}).get("applied", False)))
                chunk, resp, idx = new_chunk, new, d
        finally:
            if patcher is not None:
                patcher.stop()
                policy.model.rtc_processor = proc
    traj = np.stack(executed)
    jerk = traj[3:] - 3 * traj[2:-1] + 3 * traj[1:-2] - traj[:-3]  # per step^3
    jn = np.linalg.norm(jerk, axis=1)
    seam_set = set()
    for s in seams:
        seam_set.update(range(max(0, s - 3), s))  # windows straddling the switch
    at_seam = np.array([jn[k] for k in range(len(jn)) if k in seam_set])
    inside = np.array([jn[k] for k in range(len(jn)) if k not in seam_set])
    return {
        "arm": arm,
        "chunks": chunks,
        "rtc_applied_frac": float(np.mean(applied)) if applied else 0.0,
        "starved_actions": starved,
        "delay_actions": dist(delays),
        "latency_ms": dist([x * 1000 for x in latencies]),
        "_latencies_ms": [x * 1000 for x in latencies],
        "prefix_max_abs": dist(prefix_err),
        "seam_jump_l2": dist(seam_jump),
        "jerk_at_seam": dist(at_seam),
        "jerk_inside_chunk": dist(inside),
    }


def dist(values) -> dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def main() -> int:
    import lerobot
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--chunks", type=int, default=100)
    p.add_argument("--delays", type=int, nargs="+", default=[3, 6, 10])
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--arms", nargs="+", default=["off", "rtc", "rtc_vjp"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--interleave",
        type=int,
        default=0,
        help="repeat the arms ABAB... this many times per delay (one warm-up)",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    policy, cfg = build_policy(args.device, args.seed)
    export_dir = Path(tempfile.mkdtemp(prefix="rtc-evidence-"))
    (export_dir / "expert_stack.onnx").write_bytes(b"unused by the native path")
    (export_dir / "tether_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "random/smolvla",
                "model_type": "smolvla",
                "action_dim": ACTION_DIM,
                "num_denoising_steps": cfg.num_steps,
                "opset": 19,
                "export_kind": "decomposed_onnx",
                "artifacts": [{"path": "expert_stack.onnx", "role": "model"}],
                "io_contract": {
                    "inputs": [
                        {
                            "name": "input",
                            "dtype": "float32",
                            "shape": [1, cfg.chunk_size, ACTION_DIM],
                        }
                    ],
                    "outputs": [
                        {
                            "name": "actions",
                            "dtype": "float32",
                            "shape": [1, cfg.chunk_size, ACTION_DIM],
                        }
                    ],
                },
            }
        )
    )

    # Warm up and measure plain latency to map target delays onto hz.
    with make_client(policy, args.device, None, export_dir) as client:
        warm = []
        for i in range(8):
            seed_all(i)
            r = client.post("/act", json={"instruction": "pick", "state": [0.0] * STATE_DIM}).json()
            warm.append(float(r["latency_ms"]) / 1000.0)
    p50 = float(np.median(warm[3:]))

    results = []
    for d in args.delays:
        hz = d / p50
        if args.interleave:
            pooled: dict[str, list[float]] = {arm: [] for arm in args.arms}
            for rep in range(args.interleave):
                for arm in args.arms:
                    row = run_arm(
                        policy,
                        args.device,
                        arm,
                        hz,
                        args.chunks,
                        args.stride,
                        args.horizon,
                        export_dir,
                        args.seed * 100_000,
                    )
                    pooled[arm].extend(row.pop("_latencies_ms"))
                    row.update(target_delay=d, execute_hz=round(hz, 3), repeat=rep)
                    results.append(row)
                    print(
                        json.dumps({"rep": rep, "arm": arm, "mean_ms": row["latency_ms"]["mean"]}),
                        flush=True,
                    )
            for arm, lat in pooled.items():
                results.append({"arm": arm, "target_delay": d, "pooled_latency_ms": dist(lat)})
                print(json.dumps({"pooled": arm, **dist(lat)}), flush=True)
            continue
        for arm in args.arms:
            t0 = time.perf_counter()
            row = run_arm(
                policy,
                args.device,
                arm,
                hz,
                args.chunks,
                args.stride,
                args.horizon,
                export_dir,
                args.seed * 100_000,
            )
            row.pop("_latencies_ms")
            row.update(
                target_delay=d, execute_hz=round(hz, 3), wall_s=round(time.perf_counter() - t0, 1)
            )
            if arm == "rtc_vjp":
                row["vjp_vs_identity_rel_diff"] = dist(vjp_denoise_step.rel)
                vjp_denoise_step.rel.clear()
            results.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "target_delay",
                            "arm",
                            "prefix_max_abs",
                            "seam_jump_l2",
                            "jerk_at_seam",
                            "latency_ms",
                        )
                    }
                ),
                flush=True,
            )

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "lerobot": lerobot.__version__,
            "device": args.device,
        },
        "model": {
            "family": "smolvla",
            "weights": "random (seeded), no checkpoint",
            "params_m": round(sum(q.numel() for q in policy.parameters()) / 1e6, 1),
            "chunk_size": cfg.chunk_size,
            "num_steps": cfg.num_steps,
            "num_vlm_layers": cfg.num_vlm_layers,
        },
        "sim": {
            "stride": args.stride,
            "execution_horizon": args.horizon,
            "chunks_per_arm": args.chunks,
            "warm_p50_latency_ms": round(p50 * 1000, 1),
            "schedule": "LINEAR",
            "max_guidance_weight": 10.0,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
