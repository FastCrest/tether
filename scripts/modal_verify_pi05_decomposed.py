"""Modal: parity verification — monolithic pi0.5 ONNX vs decomposed chain
(``vlm_prefix.onnx`` → ``expert_denoise.onnx``). Feeds identical seeded
inputs through both paths and reports cos-sim + max_abs + mean_abs on the
action output.

Design doc: ``reflex_context/reflex_vla/01_architecture/prefix_kv_cache_reuse_design.md``

Closes the "decomposed = same work as monolithic" verification in the
design doc's ``Verification plan`` step 1. Expected cos=+1.0,
max_abs<1e-5 (no numerical loss from the split).

Usage:
    # SnapFlow student (num_steps=1):
    modal run scripts/modal_verify_pi05_decomposed.py \\
      --monolithic-onnx-dir /onnx_out/distill_v031_pi05_libero_r4/onnx_1nfe \\
      --decomposed-dir /onnx_out/distill_v031_pi05_libero_r4/decomposed_smoke \\
      --seed 7
"""
import os
import subprocess
import modal

app = modal.App("tether-verify-pi05-decomposed")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUST = _build_bust()

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
ONNX_OUT = "/onnx_out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "numpy",
    )
    .run_commands(f'echo "build_bust={_BUST}"')
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
    })
)


@app.function(
    image=image,
    cpu=8,
    memory=64 * 1024,
    timeout=3600,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def verify_modal(
    monolithic_onnx_dir: str,
    decomposed_dir: str,
    seed: int = 7,
):
    """Run monolithic ONNX and the decomposed chain on seeded inputs;
    compare action outputs."""
    import json
    import logging
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("verify")

    mono_path = Path(monolithic_onnx_dir) / "model.onnx"
    prefix_path = Path(decomposed_dir) / "vlm_prefix.onnx"
    expert_path = Path(decomposed_dir) / "expert_denoise.onnx"
    cfg_path = Path(decomposed_dir) / "tether_config.json"

    for p in (mono_path, prefix_path, expert_path, cfg_path):
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")

    cfg = json.loads(cfg_path.read_text())
    past_kv_names: list[str] = cfg["decomposed"]["past_kv_tensor_names"]
    n_layers: int = cfg["decomposed"]["paligemma_layers"]
    chunk: int = cfg["chunk_size"]
    action_dim: int = cfg["action_dim"]
    log.info(
        "decomposed config: layers=%d chunk=%d action_dim=%d num_steps=%d",
        n_layers, chunk, action_dim, cfg["num_denoising_steps"],
    )

    B = 1
    rng = np.random.default_rng(seed)
    inputs = {
        "img_base": rng.standard_normal((B, 3, 224, 224)).astype(np.float32),
        "img_wrist_l": rng.standard_normal((B, 3, 224, 224)).astype(np.float32),
        "img_wrist_r": rng.standard_normal((B, 3, 224, 224)).astype(np.float32),
        "mask_base": np.ones((B,), dtype=np.bool_),
        "mask_wrist_l": np.ones((B,), dtype=np.bool_),
        "mask_wrist_r": np.ones((B,), dtype=np.bool_),
        "lang_tokens": rng.integers(0, 257152, size=(B, 16)).astype(np.int64),
        "lang_masks": np.ones((B, 16), dtype=np.bool_),
        "noise": rng.standard_normal((B, chunk, action_dim)).astype(np.float32),
    }

    # ---- Monolithic ONNX ---------------------------------------------
    log.info("loading monolithic ONNX: %s", mono_path)
    sess_mono = ort.InferenceSession(str(mono_path), providers=["CPUExecutionProvider"])
    mono_input_names = {i.name for i in sess_mono.get_inputs()}
    log.info("monolithic inputs: %s", sorted(mono_input_names))
    mono_inputs = {k: v for k, v in inputs.items() if k in mono_input_names}
    missing_mono = mono_input_names - set(mono_inputs.keys())
    if missing_mono:
        raise RuntimeError(
            f"monolithic expects inputs we didn't prepare: {missing_mono}"
        )
    log.info("running monolithic forward ...")
    actions_mono = sess_mono.run(["actions"], mono_inputs)[0]
    log.info("monolithic output: %s %s", actions_mono.shape, actions_mono.dtype)
    del sess_mono  # free ~13GB before loading the next session

    # ---- Decomposed: vlm_prefix -------------------------------------
    log.info("loading vlm_prefix ONNX: %s", prefix_path)
    sess_prefix = ort.InferenceSession(str(prefix_path), providers=["CPUExecutionProvider"])
    prefix_input_names = [i.name for i in sess_prefix.get_inputs()]
    prefix_output_names = [o.name for o in sess_prefix.get_outputs()]
    log.info("prefix inputs:  %s", prefix_input_names)
    log.info("prefix outputs: %s", prefix_output_names[:4] + ["..."] + prefix_output_names[-1:])

    prefix_feed = {k: inputs[k] for k in prefix_input_names}
    log.info("running prefix forward ...")
    prefix_outputs = sess_prefix.run(prefix_output_names, prefix_feed)
    prefix_out_dict = dict(zip(prefix_output_names, prefix_outputs))
    # prefix_pad_masks is last output; past_k_i/past_v_i are the rest
    log.info(
        "prefix_pad_masks shape: %s; sample past_k_0 shape: %s",
        prefix_out_dict["prefix_pad_masks"].shape,
        prefix_out_dict["past_k_0"].shape,
    )
    del sess_prefix

    # ---- Decomposed: expert_denoise ---------------------------------
    log.info("loading expert_denoise ONNX: %s", expert_path)
    sess_expert = ort.InferenceSession(str(expert_path), providers=["CPUExecutionProvider"])
    expert_input_names = [i.name for i in sess_expert.get_inputs()]
    log.info("expert inputs count: %d (past_kv + pad + noise)", len(expert_input_names))

    # Build expert feed by mapping names → tensors. Most past_k_i / past_v_i
    # come straight from the prefix output; prefix_pad_masks pipes through;
    # noise comes from the seeded inputs.
    expert_feed: dict[str, "np.ndarray"] = {}
    for name in expert_input_names:
        if name in prefix_out_dict:
            expert_feed[name] = prefix_out_dict[name]
        elif name == "noise":
            expert_feed[name] = inputs["noise"]
        elif name == "prefix_pad_masks":
            expert_feed[name] = prefix_out_dict["prefix_pad_masks"]
        else:
            raise RuntimeError(f"expert input {name} not resolved")

    # ---- Shape-match past_kv to the expert ONNX's static shapes ------
    # vlm_prefix was exported shape-specialized to whatever seq_len the
    # tokenizer produced for its dummy inputs; expert_denoise was exported
    # at a fixed seq_len=1024 (see decomposed.py line 210). If the prefix
    # produced a different seq_len at verify time, pad or trim the past_kv
    # along the seq_len axis so the expert ONNX accepts them.
    for name in expert_input_names:
        if not (name.startswith("past_k_") or name.startswith("past_v_")):
            continue
        expected_shape = next(
            i.shape for i in sess_expert.get_inputs() if i.name == name
        )
        got = expert_feed[name]
        if got.shape[-2] == expected_shape[-2]:
            continue
        # expected_shape may have symbolic dims ('batch_size' etc); only
        # adjust when the seq_len dim is concrete.
        if isinstance(expected_shape[-2], int):
            want = expected_shape[-2]
            have = got.shape[-2]
            if have > want:
                expert_feed[name] = got[..., :want, :]
            else:
                pad_shape = list(got.shape)
                pad_shape[-2] = want - have
                expert_feed[name] = np.concatenate(
                    [got, np.zeros(pad_shape, dtype=got.dtype)],
                    axis=-2,
                )
    # prefix_pad_masks also needs to track seq_len.
    if "prefix_pad_masks" in expert_input_names:
        expected_pad_shape = next(
            i.shape for i in sess_expert.get_inputs() if i.name == "prefix_pad_masks"
        )
        got = expert_feed["prefix_pad_masks"]
        if isinstance(expected_pad_shape[-1], int) and got.shape[-1] != expected_pad_shape[-1]:
            want = expected_pad_shape[-1]
            have = got.shape[-1]
            if have > want:
                expert_feed["prefix_pad_masks"] = got[..., :want]
            else:
                pad_shape = list(got.shape)
                pad_shape[-1] = want - have
                expert_feed["prefix_pad_masks"] = np.concatenate(
                    [got, np.zeros(pad_shape, dtype=got.dtype)],
                    axis=-1,
                )

    log.info("running expert forward ...")
    actions_chain = sess_expert.run(["actions"], expert_feed)[0]
    log.info("chain output: %s %s", actions_chain.shape, actions_chain.dtype)

    # ---- Parity comparison -------------------------------------------
    assert actions_mono.shape == actions_chain.shape, (
        f"shape mismatch: mono {actions_mono.shape} vs chain {actions_chain.shape}"
    )
    diff = actions_mono.astype(np.float64) - actions_chain.astype(np.float64)
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())

    mo = actions_mono.reshape(-1).astype(np.float64)
    ch = actions_chain.reshape(-1).astype(np.float64)
    cos = float(np.dot(mo, ch) / (np.linalg.norm(mo) * np.linalg.norm(ch) + 1e-12))

    log.info("==== PARITY (monolithic vs decomposed-chain) ====")
    log.info("  shape:    %s", actions_mono.shape)
    log.info("  cos_sim:  %.10f", cos)
    log.info("  max_abs:  %.6e", max_abs)
    log.info("  mean_abs: %.6e", mean_abs)
    log.info("  mono  sample: %s", actions_mono.flatten()[:5])
    log.info("  chain sample: %s", actions_chain.flatten()[:5])

    return {
        "status": "ok",
        "shape": list(actions_mono.shape),
        "cos_sim": cos,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "monolithic_first_values": actions_mono.flatten()[:5].tolist(),
        "chain_first_values": actions_chain.flatten()[:5].tolist(),
    }


@app.local_entrypoint()
def main(
    monolithic_onnx_dir: str = "/onnx_out/distill_v031_pi05_libero_r4/onnx_1nfe",
    decomposed_dir: str = "/onnx_out/distill_v031_pi05_libero_r4/decomposed_smoke",
    seed: int = 7,
):
    """
    --monolithic-onnx-dir  Volume path containing model.onnx + model.onnx.data
                           from export_snapflow_student_monolithic.
    --decomposed-dir       Volume path containing vlm_prefix.onnx +
                           expert_denoise.onnx + tether_config.json from
                           export_pi05_decomposed.
    --seed                 Seed both forward paths consume identical inputs.
    """
    r = verify_modal.remote(
        monolithic_onnx_dir=monolithic_onnx_dir,
        decomposed_dir=decomposed_dir,
        seed=seed,
    )
    print("\n=== PARITY ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
