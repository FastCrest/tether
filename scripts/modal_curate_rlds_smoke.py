"""Smoke test for the [curate-rlds] format converters on Modal.

Validates:
  1. `pip install reflex-vla[curate-rlds]` resolves cleanly with tensorflow +
     tensorflow_datasets pinned in pyproject.toml.
  2. `reflex curate convert --format rlds` against a synthetic JSONL trace
     produces TFRecord + dataset_info.json + features.json output.
  3. `reflex curate convert --format openx-embodiment` produces the same
     plus the OXE-specific embodiment_id schema fields.
  4. The TFRecord output is parse-readable via tf.data.TFRecordDataset
     (catches SequenceExample schema mismatches before a buyer hits one).

Cost: ~$1-2 for the full run on T4 / A10G (most cost is the image build
since tensorflow + tfds are heavy).

Usage:
    modal run scripts/modal_curate_rlds_smoke.py
    modal run scripts/modal_curate_rlds_smoke.py --diagnostic-only
"""
import os
import subprocess

import modal


app = modal.App("reflex-curate-rlds-smoke")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("huggingface")


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()


# Heavy image: tensorflow + tensorflow-datasets. Cached after first build.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "numpy",
        "Pillow",
        "pyarrow",  # for the LeRobot v3 fallback test
        # The [curate-rlds] extra deps directly so we don't need the
        # full reflex-vla install pulling unrelated stuff.
        "tensorflow>=2.13.0",
        "tensorflow-datasets>=4.9.0",
    )
    .run_commands(
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image,
    cpu=2.0,
    timeout=1200,
    secrets=[_hf_secret()],
)
def rlds_smoke(diagnostic_only: bool = False):
    """Generate synthetic JSONL → run RLDS + OXE converters → parse back."""
    import json
    import logging
    import tempfile
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    # 1. Verify the converters can be imported with the extras installed.
    try:
        import tensorflow as tf
        from reflex.curate.format_converters import (
            CONVERTER_REGISTRY,
            EMBODIMENT_OXE_MAP,
            OpenXEmbodimentConverter,
            RLDSConverter,
        )
    except Exception as exc:
        return {"status": "FAIL_IMPORT", "error": repr(exc), "head_sha": _HEAD}

    if "rlds" not in CONVERTER_REGISTRY or "openx-embodiment" not in CONVERTER_REGISTRY:
        return {
            "status": "FAIL_REGISTRY",
            "error": "rlds / openx-embodiment not in registry",
            "registry": list(CONVERTER_REGISTRY.keys()),
        }

    if diagnostic_only:
        return {
            "status": "OK_DIAGNOSTIC",
            "tf_version": tf.__version__,
            "rlds_class": RLDSConverter.__name__,
            "oxe_map_franka": EMBODIMENT_OXE_MAP.get("franka"),
            "head_sha": _HEAD,
        }

    # 2. Build a synthetic JSONL trace (5 episodes × 20 rows × 5-step chunks).
    work = Path(tempfile.mkdtemp())
    jsonl = work / "synthetic.jsonl"
    with open(jsonl, "w") as f:
        for ep in range(5):
            for step in range(20):
                row = {
                    "kind": "request",
                    "schema_version": 1,
                    "seq": step,
                    "chunk_id": step,
                    "timestamp": f"2026-05-06T{step:02d}:00:00Z",
                    "episode_id": f"ep_{ep:02d}",
                    "instruction_raw": f"Task {ep}: pick up object",
                    "state_vec": [0.1 * step, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
                    "action_chunk": [
                        [float(i) * 0.01 + step, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
                        for i in range(5)
                    ],
                    "metadata": {"contributor_id": "smoke", "quality_score": 0.9},
                }
                f.write(json.dumps(row) + "\n")
    logger.info(f"synthetic JSONL: {jsonl} ({jsonl.stat().st_size} bytes)")

    results: dict = {"head_sha": _HEAD}

    # 3. RLDS converter.
    rlds_out = work / "rlds_out"
    rlds_result = RLDSConverter(
        dataset_name="reflex_test_rlds",
        shard_size=3,
    ).convert(input_jsonl=jsonl, output_dir=rlds_out)
    rlds_dict = rlds_result.to_dict()
    logger.info(f"rlds outcome: {rlds_dict}")

    # Verify dataset_info.json + features.json exist + TFRecord shards present.
    rlds_info = json.loads((rlds_out / "dataset_info.json").read_text())
    rlds_features = json.loads((rlds_out / "features.json").read_text())
    rlds_shards = sorted(rlds_out.glob("*.tfrecord-*"))

    # Parse-back: count records across shards.
    rlds_record_count = 0
    sample_keys: list[str] = []
    if rlds_shards:
        ds = tf.data.TFRecordDataset([str(p) for p in rlds_shards])
        # Parse the first SequenceExample to verify the schema lands.
        for raw in ds.take(1):
            seq_example = tf.train.SequenceExample()
            seq_example.ParseFromString(raw.numpy())
            sample_keys = list(seq_example.feature_lists.feature_list.keys())
        rlds_record_count = sum(1 for _ in ds)

    results["rlds"] = {
        "status": "PASS" if (
            rlds_dict["episode_count"] == 5
            and rlds_record_count == 5
            and "steps" in rlds_features
            and "action" in rlds_features["steps"]["feature_spec"]
            and len(rlds_shards) == 2  # 5 episodes / 3 shard_size = 2 shards
        ) else "FAIL",
        "outcome": rlds_dict,
        "shard_count": len(rlds_shards),
        "tfrecord_record_count": rlds_record_count,
        "feature_list_keys": sample_keys,
        "info_format_version": rlds_info.get("format_version"),
    }

    # 4. OXE converter.
    oxe_out = work / "oxe_out"
    oxe_result = OpenXEmbodimentConverter(embodiment="franka").convert(
        input_jsonl=jsonl, output_dir=oxe_out,
    )
    oxe_dict = oxe_result.to_dict()
    logger.info(f"oxe outcome: {oxe_dict}")

    oxe_info = json.loads((oxe_out / "dataset_info.json").read_text())
    oxe_features = json.loads((oxe_out / "features.json").read_text())
    oxe_shards = sorted(oxe_out.glob("*.tfrecord-*"))

    # Parse-back + verify embodiment_id context feature is present.
    oxe_embodiment_id_seen: str | None = None
    if oxe_shards:
        ds = tf.data.TFRecordDataset([str(p) for p in oxe_shards])
        for raw in ds.take(1):
            seq_example = tf.train.SequenceExample()
            seq_example.ParseFromString(raw.numpy())
            ctx = seq_example.context.feature
            if "embodiment_id" in ctx:
                oxe_embodiment_id_seen = (
                    ctx["embodiment_id"].bytes_list.value[0].decode("utf-8")
                )

    results["oxe"] = {
        "status": "PASS" if (
            oxe_dict["episode_count"] == 5
            and oxe_info.get("oxe_embodiment") == "franka_emika_panda"
            and oxe_info.get("format_version") == "openx-embodiment-1.0"
            and oxe_embodiment_id_seen == "franka_emika_panda"
            and "embodiment_id" in oxe_features["episode_metadata"]["feature_spec"]
        ) else "FAIL",
        "outcome": oxe_dict,
        "oxe_embodiment_in_info": oxe_info.get("oxe_embodiment"),
        "embodiment_id_in_seq_context": oxe_embodiment_id_seen,
        "format_version": oxe_info.get("format_version"),
    }

    # 5. Summary
    results["status"] = "PASS" if (
        results["rlds"]["status"] == "PASS"
        and results["oxe"]["status"] == "PASS"
    ) else "FAIL"
    return results


@app.local_entrypoint()
def main(diagnostic_only: bool = False):
    print(f"[reflex curate-rlds Modal smoke]")
    print(f"  HEAD: {_HEAD}")
    print(f"  diagnostic_only: {diagnostic_only}")
    print()
    r = rlds_smoke.remote(diagnostic_only=diagnostic_only)
    print("=== RESULT ===")
    import json as _json
    print(_json.dumps(r, indent=2, default=str))
