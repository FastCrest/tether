"""Phase I.2 verification: tether serve composes wedges via flags.

Runs `tether serve` with --safety-config, --adaptive-steps, and --deadline-ms
simultaneously, then POSTs /act to verify the response surfaces telemetry from
each wedge (safety_violations, adaptive_enabled, deadline_exceeded).

Usage:
    modal run scripts/modal_verify_wedge_compose.py
"""

import modal

app = modal.App("tether-wedge-compose-verify")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "safetensors", "huggingface_hub",
        "transformers>=4.40,<5.0",
        "onnx", "onnxscript",
        "onnxruntime-gpu==1.20.1",
        "numpy<2.0", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
        "fastapi", "uvicorn", "httpx",
        "yourdfpy", "trimesh",
    )
    .add_local_dir("src/tether", "/root/tether-vla/src/tether", copy=True)
    .add_local_file("pyproject.toml", "/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/tether-vla/README.md", copy=True)
    .run_commands("cd /root/tether-vla && pip install -e . --no-deps")
)


def _make_fake_export(export_dir: str, action_dim: int = 6):
    import json
    import os
    import onnx
    from onnx import helper, TensorProto

    os.makedirs(export_dir, exist_ok=True)
    input_tensor = helper.make_tensor_value_info(
        "noisy_actions", TensorProto.FLOAT, [1, 50, action_dim]
    )
    time_tensor = helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1])
    pos_tensor = helper.make_tensor_value_info("position_ids", TensorProto.INT64, [1, 50])
    output_tensor = helper.make_tensor_value_info(
        "velocity", TensorProto.FLOAT, [1, 50, action_dim]
    )
    # Identity — produces a velocity that matches noisy_actions. With random
    # input that has values around ~N(0,1), the integration loop produces
    # actions around ~N(0, 10), which puts them outside typical joint limits.
    # Good for testing the guard wedge actually fires.
    identity = helper.make_node(
        "Identity", inputs=["noisy_actions"], outputs=["velocity"], name="identity"
    )
    graph = helper.make_graph(
        [identity], "fake",
        [input_tensor, time_tensor, pos_tensor],
        [output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 10
    onnx.save(model, f"{export_dir}/expert_stack.onnx")
    with open(f"{export_dir}/tether_config.json", "w") as f:
        json.dump({
            "model_type": "smolvla",
            "action_chunk_size": 50,
            "action_dim": action_dim,
            "expert": {"action_dim": action_dim, "expert_hidden": 720},
        }, f)


def _make_safety_config(path: str, num_joints: int = 6):
    """Tight limits so guard definitely fires on random-ish actions."""
    from tether.safety import SafetyLimits
    limits = SafetyLimits.default(num_joints)
    # Crush the limits so any random action gets clamped
    limits.position_min = [-0.01] * num_joints
    limits.position_max = [0.01] * num_joints
    limits.save(path)


@app.function(image=image, cpu=2, timeout=600)
def test_compose():
    import subprocess
    import time
    import httpx

    results = []

    _make_fake_export("/tmp/fake_export", action_dim=6)
    _make_safety_config("/tmp/safety.json", num_joints=6)

    def _run_and_probe(cmd: list[str], port: int, label: str, timeout_health_s: int = 30):
        """Launch, wait for /health, POST /act once, return response dict."""
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        # Wait for /health
        ready = False
        t0 = time.time()
        while time.time() - t0 < timeout_health_s:
            if proc.poll() is not None:
                break
            try:
                resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                if resp.status_code == 200 and resp.json().get("model_loaded"):
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        act_resp = None
        error = None
        if ready:
            try:
                r = httpx.post(
                    f"http://127.0.0.1:{port}/act",
                    json={"instruction": "reach", "state": [0.1]*6},
                    timeout=10.0,
                )
                act_resp = r.json()
            except Exception as e:
                error = str(e)[:200]

        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate(timeout=5)

        return {
            "label": label,
            "ready": ready,
            "act_response": act_resp,
            "error": error,
            "exit_code": proc.returncode,
            "stdout_tail": out[-1000:] if out else "",
        }

    # Scenario 1: Plain serve (no wedges) — baseline
    print("[1/4] plain serve")
    r1 = _run_and_probe(
        ["tether", "serve", "/tmp/fake_export", "--port", "9100",
         "--host", "127.0.0.1", "--device", "cpu"],
        9100, "plain",
    )
    results.append(r1)

    # Scenario 2: --safety-config only
    print("[2/4] serve --safety-config")
    r2 = _run_and_probe(
        ["tether", "serve", "/tmp/fake_export", "--port", "9101",
         "--host", "127.0.0.1", "--device", "cpu",
         "--safety-config", "/tmp/safety.json"],
        9101, "safety",
    )
    results.append(r2)

    # Scenario 3: --adaptive-steps only
    print("[3/4] serve --adaptive-steps")
    r3 = _run_and_probe(
        ["tether", "serve", "/tmp/fake_export", "--port", "9102",
         "--host", "127.0.0.1", "--device", "cpu",
         "--adaptive-steps"],
        9102, "adaptive",
    )
    results.append(r3)

    # Scenario 4: ALL wedges composed
    print("[4/4] serve + all wedges (safety + adaptive + deadline + cloud-fallback stub)")
    r4 = _run_and_probe(
        ["tether", "serve", "/tmp/fake_export", "--port", "9103",
         "--host", "127.0.0.1", "--device", "cpu",
         "--safety-config", "/tmp/safety.json",
         "--adaptive-steps",
         "--deadline-ms", "1000000",  # absurd deadline so it never fires
         "--cloud-fallback", "http://not-actually-used:9999"],
        9103, "all-wedges",
    )
    results.append(r4)

    # Summary + assertions
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    passes = 0
    total = 4

    # 1. plain: ready, act returns, no wedge telemetry keys present
    ar = r1.get("act_response") or {}
    if r1["ready"] and ar.get("num_actions") == 50 and "safety_violations" not in ar:
        print(f"  1 PASS: plain serve works, no wedge keys leaking")
        passes += 1
    else:
        print(f"  1 FAIL: {r1}")

    # 2. safety: safety_violations surfaces
    ar = r2.get("act_response") or {}
    if r2["ready"] and "safety_violations" in ar and ar["safety_violations"] > 0:
        print(f"  2 PASS: guard wedge composed — {ar['safety_violations']} violations")
        passes += 1
    else:
        print(f"  2 FAIL: response={ar}, stdout_tail={r2.get('stdout_tail','')[-400:]}")

    # 3. adaptive: adaptive_enabled=True surfaces
    ar = r3.get("act_response") or {}
    if r3["ready"] and ar.get("adaptive_enabled") is True:
        print(f"  3 PASS: turbo adaptive wedge composed, steps_used={ar.get('denoising_steps')}")
        passes += 1
    else:
        print(f"  3 FAIL: response={ar}")

    # 4. all: all four keys surface; no crashes
    ar = r4.get("act_response") or {}
    needed = ["safety_violations", "adaptive_enabled", "deadline_exceeded", "split_enabled"]
    missing = [k for k in needed if k not in ar]
    if r4["ready"] and not missing:
        print(f"  4 PASS: all 4 wedges composed, response has {sorted(ar.keys())}")
        passes += 1
    else:
        print(f"  4 FAIL: ready={r4['ready']}, missing={missing}, response={ar}")

    print(f"\nOverall: {passes}/{total}")
    return {"passes": passes, "total": total, "results": results}


@app.local_entrypoint()
def main():
    print("Phase I.2 verification — tether serve wedge composition\n")
    result = test_compose.remote()
    print(f"\nFinal: {result['passes']}/{result['total']}")
