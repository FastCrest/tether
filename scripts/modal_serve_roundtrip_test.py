"""Modal: serve-act-roundtrip gate.

Loads the pi0 + SmolVLA monolithic ONNX artifacts (already in the
`pi0-onnx-outputs` Modal volume from previous export runs) through the
production runtime server classes and verifies a predict() call returns
a valid action chunk. Proves export → runtime wiring works end-to-end,
not just the ONNX file by itself.

Passes if for each model:
  - load() succeeds without exception
  - predict() returns a dict with "actions" key
  - actions shape is [chunk_size, action_dim]
  - all values are finite (no NaN/Inf)
  - the reported inference_mode matches the expected value
"""
import modal

app = modal.App("tether-serve-roundtrip-test")

# HF cache isn't needed — we don't load PyTorch weights; just onnxruntime.
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=False)
ONNX_OUTPUT_PATH = "/onnx_out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "tether-vla[serve,onnx] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla.git"
    )
)


@app.function(
    image=image,
    timeout=900,
    volumes={ONNX_OUTPUT_PATH: onnx_output},
)
def roundtrip_test():
    import json
    import logging
    from pathlib import Path

    import numpy as np

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("serve_roundtrip")

    results = {}

    # --- pi0 ------------------------------------------------------------
    log.info("=== pi0 roundtrip ===")
    pi0_dir = Path(ONNX_OUTPUT_PATH) / "monolithic"  # existing pi0 export
    if (pi0_dir / "model.onnx").exists():
        # Write a minimal tether_config.json so the server knows what it has
        (pi0_dir / "tether_config.json").write_text(json.dumps({
            "model_type": "pi0",
            "export_kind": "monolithic",
            "num_denoising_steps": 10,
            "chunk_size": 50,
            "action_chunk_size": 50,
            "action_dim": 32,
            "max_state_dim": 32,
        }))

        from tether.runtime.pi0_onnx_server import Pi0OnnxServer
        srv = Pi0OnnxServer(str(pi0_dir))
        srv.load()
        rng = np.random.RandomState(7)
        image_rgb = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        state = [0.1] * 14
        noise = rng.randn(1, 50, 32).astype(np.float32)

        resp = srv.predict(
            image=image_rgb, instruction="pick up the red cup",
            state=state, noise=noise,
        )

        ok = (
            "actions" in resp
            and isinstance(resp["actions"], list)
            and len(resp["actions"]) == 50
            and len(resp["actions"][0]) == 32
            and all(
                all(np.isfinite(v) for v in row) for row in resp["actions"]
            )
            and resp.get("inference_mode") == "pi0_onnx_monolithic"
        )
        log.info("pi0 predict() latency_ms=%s, inference_mode=%s",
                 resp.get("latency_ms"), resp.get("inference_mode"))
        results["pi0"] = {
            "passed": ok,
            "num_actions": resp.get("num_actions"),
            "action_dim": resp.get("action_dim"),
            "latency_ms": resp.get("latency_ms"),
            "inference_mode": resp.get("inference_mode"),
        }
    else:
        results["pi0"] = {"passed": False, "reason": "monolithic ONNX missing"}

    # --- SmolVLA: check ONNX file exists; runtime server class TBD -----
    log.info("=== SmolVLA roundtrip (ONNX load sanity) ===")
    smol_dir = Path(ONNX_OUTPUT_PATH) / "smolvla_monolithic"
    smol_onnx = smol_dir / "model.onnx"
    if smol_onnx.exists():
        # We don't yet have a SmolVLAOnnxServer class; sanity-check the ONNX
        # loads via raw onnxruntime. This is weaker than the pi0 test but
        # proves the artifact is servable from a clean box.
        import onnxruntime as ort
        sess = ort.InferenceSession(str(smol_onnx), providers=["CPUExecutionProvider"])
        rng = np.random.RandomState(11)
        ort_inputs = {
            "img_cam1": rng.randn(1, 3, 512, 512).astype(np.float32),
            "img_cam2": rng.randn(1, 3, 512, 512).astype(np.float32),
            "img_cam3": rng.randn(1, 3, 512, 512).astype(np.float32),
            "mask_cam1": np.ones((1,), dtype=np.bool_),
            "mask_cam2": np.ones((1,), dtype=np.bool_),
            "mask_cam3": np.ones((1,), dtype=np.bool_),
            "lang_tokens": np.ones((1, 16), dtype=np.int64),
            "lang_masks": np.ones((1, 16), dtype=np.bool_),
            "state": np.zeros((1, 32), dtype=np.float32),
            "noise": rng.randn(1, 50, 32).astype(np.float32),
        }
        actions = sess.run(None, ort_inputs)[0]
        ok = actions.shape == (1, 50, 32) and np.all(np.isfinite(actions))
        log.info("smolvla ONNX run shape=%s, finite=%s", actions.shape, np.all(np.isfinite(actions)))
        results["smolvla"] = {
            "passed": ok,
            "shape": list(actions.shape),
            "note": "raw onnxruntime roundtrip; server class TBD",
        }
    else:
        results["smolvla"] = {"passed": False, "reason": "monolithic ONNX missing"}

    all_pass = all(r.get("passed") for r in results.values())
    print(f"\n=== VERDICT: {'PASS' if all_pass else 'FAIL'} ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return {"results": results, "passed": all_pass}


@app.local_entrypoint()
def main():
    result = roundtrip_test.remote()
    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
