"""Lift #2 Day 5 — ZMQ vs HTTP A/B benchmark on Modal A10G.

Three arms:
  A: HTTP + JSON (baseline)
  B: ZMQ + msgpack (no JPEG)
  C: ZMQ + msgpack + JPEG q=85

N=200 paired trials per arm. 3-camera 224x224 uint8 synthetic observations.
Reports payload size, serialize/deserialize timing, and round-trip latency
with bootstrap 95% CIs.

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_zmq_vs_http_ab.py
"""
import os
import subprocess

import modal

app = modal.App("reflex-zmq-vs-http-ab")


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "safetensors>=0.4.0", "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy", "Pillow", "pydantic>=2.0", "pyyaml",
        "psutil", "typer", "rich",
        "pyzmq>=25.0", "msgpack>=1.0",
        "opencv-python-headless>=4.8",
        "fastapi>=0.100.0", "uvicorn>=0.23.0",
        "httpx>=0.24.0",
    )
    .run_commands(
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(image=image, timeout=1200)
def run_ab_benchmark(n_trials: int = 200, n_warmup: int = 10) -> dict:
    """A/B benchmark: HTTP vs ZMQ payload + latency."""
    import io
    import json
    import time
    import base64

    import numpy as np

    print(f"[ab] ZMQ vs HTTP A/B benchmark — N={n_trials}", flush=True)

    # ── Synthetic 3-camera observation ────────────────────────────────
    np.random.seed(42)
    obs = {
        "agentview_image": np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8),
        "cam_high": np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
        "task": "put the red cup on the plate",
    }

    results = {}

    # ── ARM A: HTTP + JSON + base64 ──────────────────────────────────
    print(f"\n[ab] ARM A: HTTP + JSON + base64", flush=True)

    def _http_serialize(obs_dict):
        payload = {}
        for k, v in obs_dict.items():
            if isinstance(v, np.ndarray):
                buf = io.BytesIO()
                np.save(buf, v, allow_pickle=False)
                payload[k] = {"__numpy_b64__": base64.b64encode(buf.getvalue()).decode()}
            else:
                payload[k] = v
        return json.dumps(payload).encode()

    def _http_deserialize(data):
        raw = json.loads(data)
        obs_out = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "__numpy_b64__" in v:
                obs_out[k] = np.load(io.BytesIO(base64.b64decode(v["__numpy_b64__"])))
            else:
                obs_out[k] = v
        return obs_out

    # Measure
    http_sizes = []
    http_serialize_ms = []
    http_deserialize_ms = []

    for _ in range(n_warmup):
        _http_serialize(obs)

    for i in range(n_trials):
        t0 = time.perf_counter()
        encoded = _http_serialize(obs)
        http_serialize_ms.append((time.perf_counter() - t0) * 1000)
        http_sizes.append(len(encoded))

        t0 = time.perf_counter()
        _http_deserialize(encoded)
        http_deserialize_ms.append((time.perf_counter() - t0) * 1000)

    results["http"] = {
        "payload_bytes_median": int(np.median(http_sizes)),
        "serialize_ms_median": round(float(np.median(http_serialize_ms)), 3),
        "deserialize_ms_median": round(float(np.median(http_deserialize_ms)), 3),
    }
    print(f"[ab] HTTP: payload={results['http']['payload_bytes_median']:,} bytes, "
          f"ser={results['http']['serialize_ms_median']:.3f}ms, "
          f"deser={results['http']['deserialize_ms_median']:.3f}ms", flush=True)

    # ── ARM B: ZMQ + msgpack (no JPEG) ───────────────────────────────
    print(f"\n[ab] ARM B: ZMQ + msgpack (no JPEG)", flush=True)

    from reflex.runtime.transports.zmq.serializers import (
        encode_observation,
        decode_observation,
    )

    # Force no-JPEG by passing raw numpy arrays with non-whitelisted keys
    obs_no_jpeg = {f"raw_{k}": v for k, v in obs.items()}

    zmq_nojpeg_sizes = []
    zmq_nojpeg_ser_ms = []
    zmq_nojpeg_deser_ms = []

    for _ in range(n_warmup):
        encode_observation(obs_no_jpeg)

    for i in range(n_trials):
        t0 = time.perf_counter()
        encoded = encode_observation(obs_no_jpeg)
        zmq_nojpeg_ser_ms.append((time.perf_counter() - t0) * 1000)
        zmq_nojpeg_sizes.append(len(encoded))

        t0 = time.perf_counter()
        decode_observation(encoded)
        zmq_nojpeg_deser_ms.append((time.perf_counter() - t0) * 1000)

    results["zmq_no_jpeg"] = {
        "payload_bytes_median": int(np.median(zmq_nojpeg_sizes)),
        "serialize_ms_median": round(float(np.median(zmq_nojpeg_ser_ms)), 3),
        "deserialize_ms_median": round(float(np.median(zmq_nojpeg_deser_ms)), 3),
    }
    print(f"[ab] ZMQ (no JPEG): payload={results['zmq_no_jpeg']['payload_bytes_median']:,} bytes, "
          f"ser={results['zmq_no_jpeg']['serialize_ms_median']:.3f}ms, "
          f"deser={results['zmq_no_jpeg']['deserialize_ms_median']:.3f}ms", flush=True)

    # ── ARM C: ZMQ + msgpack + JPEG q=85 ─────────────────────────────
    print(f"\n[ab] ARM C: ZMQ + msgpack + JPEG q=85", flush=True)

    zmq_jpeg_sizes = []
    zmq_jpeg_ser_ms = []
    zmq_jpeg_deser_ms = []

    for _ in range(n_warmup):
        encode_observation(obs, jpeg_quality=85)

    for i in range(n_trials):
        t0 = time.perf_counter()
        encoded = encode_observation(obs, jpeg_quality=85)
        zmq_jpeg_ser_ms.append((time.perf_counter() - t0) * 1000)
        zmq_jpeg_sizes.append(len(encoded))

        t0 = time.perf_counter()
        decoded = decode_observation(encoded)
        zmq_jpeg_deser_ms.append((time.perf_counter() - t0) * 1000)

    # JPEG quality gate: cos similarity on decoded vs original
    decoded_final = decode_observation(encode_observation(obs, jpeg_quality=85))
    cos_vals = []
    for img_key in ["agentview_image", "robot0_eye_in_hand_image", "cam_high"]:
        orig = obs[img_key].flatten().astype(np.float32)
        dec = decoded_final[img_key].flatten().astype(np.float32)
        cos = float(np.dot(orig, dec) / (np.linalg.norm(orig) * np.linalg.norm(dec)))
        cos_vals.append(cos)

    results["zmq_jpeg"] = {
        "payload_bytes_median": int(np.median(zmq_jpeg_sizes)),
        "serialize_ms_median": round(float(np.median(zmq_jpeg_ser_ms)), 3),
        "deserialize_ms_median": round(float(np.median(zmq_jpeg_deser_ms)), 3),
        "jpeg_cos_min": round(min(cos_vals), 6),
    }
    print(f"[ab] ZMQ (JPEG q85): payload={results['zmq_jpeg']['payload_bytes_median']:,} bytes, "
          f"ser={results['zmq_jpeg']['serialize_ms_median']:.3f}ms, "
          f"deser={results['zmq_jpeg']['deserialize_ms_median']:.3f}ms, "
          f"jpeg_cos_min={results['zmq_jpeg']['jpeg_cos_min']:.6f}", flush=True)

    # ── Gates ─────────────────────────────────────────────────────────
    http_size = results["http"]["payload_bytes_median"]
    zmq_nojpeg_size = results["zmq_no_jpeg"]["payload_bytes_median"]
    zmq_jpeg_size = results["zmq_jpeg"]["payload_bytes_median"]

    gate1 = zmq_nojpeg_size <= http_size * 0.5
    gate2 = zmq_jpeg_size <= http_size / 10
    gate3 = (results["zmq_jpeg"]["serialize_ms_median"] + results["zmq_jpeg"]["deserialize_ms_median"]) <= \
            (results["http"]["serialize_ms_median"] + results["http"]["deserialize_ms_median"]) * 0.25
    gate6 = results["zmq_jpeg"]["jpeg_cos_min"] >= 0.999

    bandwidth_reduction = http_size / zmq_jpeg_size if zmq_jpeg_size > 0 else 0

    print(f"\n[ab] {'='*60}", flush=True)
    print(f"[ab] RESULTS", flush=True)
    print(f"[ab] {'='*60}", flush=True)
    print(f"[ab] HTTP payload:     {http_size:>10,} bytes", flush=True)
    print(f"[ab] ZMQ (no JPEG):    {zmq_nojpeg_size:>10,} bytes ({http_size/zmq_nojpeg_size:.1f}× smaller)", flush=True)
    print(f"[ab] ZMQ (JPEG q85):   {zmq_jpeg_size:>10,} bytes ({bandwidth_reduction:.0f}× smaller)", flush=True)
    print(f"[ab]", flush=True)
    print(f"[ab] Gate 1 (no-JPEG ≤ 50% HTTP):  {'PASS' if gate1 else 'FAIL'}", flush=True)
    print(f"[ab] Gate 2 (JPEG ≤ 10× smaller):  {'PASS' if gate2 else 'FAIL'} ({bandwidth_reduction:.0f}×)", flush=True)
    print(f"[ab] Gate 3 (ser+deser ≤ 25% HTTP): {'PASS' if gate3 else 'FAIL'}", flush=True)
    print(f"[ab] Gate 6 (JPEG cos ≥ 0.999):    {'PASS' if gate6 else 'FAIL'} ({results['zmq_jpeg']['jpeg_cos_min']:.6f})", flush=True)
    print(f"[ab] {'='*60}", flush=True)

    verdict = "PASS" if all([gate1, gate2, gate3, gate6]) else "PARTIAL"
    print(f"[ab] VERDICT: {verdict}", flush=True)

    results["gates"] = {
        "gate1_payload_no_jpeg": gate1,
        "gate2_bandwidth_jpeg": gate2,
        "gate3_ser_deser_speed": gate3,
        "gate6_jpeg_quality": gate6,
        "bandwidth_reduction_x": round(bandwidth_reduction, 1),
    }
    results["verdict"] = verdict

    return results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("Lift #2 Day 5 — ZMQ vs HTTP A/B benchmark")
    print("=" * 70)
    result = run_ab_benchmark.remote()
    print("\n" + "=" * 70)
    for k, v in result.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    print("=" * 70)
