"""Modal: OpenVLA CPU export + numerics pre-check (Tether Studio M4 #54).

Cheap CPU job that de-risks the A100 parity run:

1. clones the exact Tether implementation commit,
2. snapshots `openvla/openvla-7b` at the exact pinned revision (full weights),
3. runs `tether.exporters.openvla.export_openvla_monolithic` in a subprocess
   to the shared output volume (`<PARITY_OUT_PATH>/openvla/export/`) and
   writes a provenance record beside it (`export.provenance.json`), which
   the A100 job requires to match before it will use the export,
4. prints an ONNX op census (Pad count must be 0; Concat inventory),
5. runs `scripts/local_openvla_monolithic_parity.py` with the CPU provider
   in a subprocess as a numerics pre-check (verdict signal only — external
   acceptance stays `not-run` off CUDA by design; gates unmodified).

No GPU is used. Wall time is bounded by per-phase subprocess timeouts, the
Modal function timeout and retries=0, plus a restart guard keyed by
--run-id (see EXPORT_TIMEOUT_SECONDS).

Usage:
    modal run scripts/modal_openvla_export_cpu_pinned.py \
        --tether-revision <40-hex> --run-id <id>
"""

import os

import modal


# `openvla/openvla-7b` HEAD (2026-02-17 README update; weights/code stable
# since 2024-09-16). Pinned exactly; floating HEAD is refused.
OPENVLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"

# Exact Tether implementation commit under test: the 1733d30
# re-qualification. Override with --tether-revision.
TETHER_REVISION = "1733d3045c98dfc1ed961763486acc5579a0b147"

TETHER_REPO = "https://github.com/FastCrest/tether.git"

app = modal.App("tether-openvla-export-cpu-pinned")
_hf_cache_volume = modal.Volume.from_name(
    "openvla-hf-cache", create_if_missing=True
)
_parity_out_volume = modal.Volume.from_name(
    "parity-gpu-receipts", create_if_missing=True
)

HF_CACHE_PATH = "/root/.cache/huggingface"
PARITY_OUT_PATH = "/parity_out/requal-1733d30-20261030"

# Libraries whose version is part of an export's identity. These are the
# packages the export path imports (torch.onnx.export on the transformers /
# timm reference); the A100 job compares the same set. `onnx` is not on the
# export path (tether/exporters/openvla.py does not import it) and the two
# images resolve it independently, so it is recorded but not compared.
PROVENANCE_PACKAGES = ("torch", "transformers", "tokenizers", "timm", "numpy")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


# modeling_prismatic.py (pinned snapshot) gates on exact contemporary deps:
#   timm in {0.9.10..0.9.16} (hard NotImplementedError otherwise),
#   transformers==4.40.1 / tokenizers==0.19.1 (warning otherwise).
# torch 2.4.x is the contemporary CUDA-capable build (export itself runs on
# CPU; the later parity job needs the same pinned source only).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.40.1",
        "tokenizers==0.19.1",
        "timm==0.9.16",
        "numpy==1.26.4",
        "Pillow",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.23",
        "accelerate",
        "scipy",
    )
    # numpy is re-pinned here: without it this layer upgraded numpy to 2.x, so the
    # export's provenance never matched the GPU parity image (numpy 1.26.4) and the
    # parity job refused it (2026-10-30, 1733d30 re-qualification).
    .pip_install(
        "onnx>=1.16",
        "onnxruntime>=1.20,<1.24",
        "numpy==1.26.4",
    )
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


# Phase budgets. NOT measured per phase: no doc records the OpenVLA CPU
# export or CPU pre-check duration on their own. The only bound on record
# is that the whole job (download + export + census + CPU pre-check) ran
# inside the old 5100 s in-container watchdog without tripping it
# (docs/parity/parity-round2-report.md:115, "never fired except as
# designed"). Stated, unmeasured bounds: export 3600 s, CPU pre-check
# 2400 s, and 1200 s for clone, the ~15 GB snapshot download and the
# census; 7200 s in all, about 1.4x that 5100 s whole-job ceiling. CPU only,
# so the cost of the ceiling is small. Each phase runs in a subprocess with
# its timeout, so an overrun ends the input as an ordinary failure; the old
# SIGKILL watchdog surfaced as a runner crash, which Modal's client retries
# as an internal failure regardless of the retry policy (modal 1.4.0
# `_functions.py`, MAX_INTERNAL_FAILURE_COUNT = 8).
EXPORT_TIMEOUT_SECONDS = 3600
PARITY_TIMEOUT_SECONDS = 2400
SETUP_MARGIN_SECONDS = 1200
MODAL_TIMEOUT_SECONDS = EXPORT_TIMEOUT_SECONDS + PARITY_TIMEOUT_SECONDS + SETUP_MARGIN_SECONDS


@app.function(
    image=image,
    timeout=MODAL_TIMEOUT_SECONDS,
    retries=0,
    volumes={
        HF_CACHE_PATH: _hf_cache_volume,
        "/parity_out": _parity_out_volume,
    },
    secrets=[_hf_secret()],
)
def run_export_cpu(
    model_revision: str = OPENVLA_REVISION,
    tether_revision: str = TETHER_REVISION,
    reuse_export: bool = False,
    run_harness: bool = True,
    export_timeout_seconds: float = EXPORT_TIMEOUT_SECONDS,
    parity_timeout_seconds: float = PARITY_TIMEOUT_SECONDS,
    run_id: str = "",
) -> dict:
    import json as _json
    import os as _os
    import shutil
    import subprocess
    import sys
    import time
    from importlib.metadata import version as _v
    from pathlib import Path

    _started_at = time.monotonic()

    def _progress(stage: str) -> None:
        print(f"[progress] {time.monotonic() - _started_at:7.1f}s  {stage}", flush=True)

    _progress("container up")
    summary: dict = {
        "model_source": "openvla/openvla-7b",
        "model_revision": model_revision,
        "tether_revision": tether_revision,
        "run_id": run_id,
        "vehicle": "openvla-cpu-export",
    }
    family_dir = Path(PARITY_OUT_PATH) / "openvla"

    # Restart guard: if Modal re-schedules this input after a container
    # crash, the second attempt finds its own marker and exits at once.
    if run_id:
        marker = family_dir / "attempts" / f"cpu-{run_id}"
        _parity_out_volume.reload()
        if marker.exists():
            _progress("restarted attempt detected; refusing to run again")
            summary["verdict"] = "aborted-restarted-attempt"
            return summary
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(time.time()))
        _parity_out_volume.commit()

    pin_dir = Path("/root/tether-pin")
    if pin_dir.exists():
        subprocess.run(["rm", "-rf", str(pin_dir)], check=True)
    subprocess.run(["git", "clone", "--quiet", TETHER_REPO, str(pin_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(pin_dir), "checkout", "--quiet", tether_revision],
        check=True,
    )
    got = subprocess.check_output(
        ["git", "-C", str(pin_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    assert got == tether_revision, f"tether pin mismatch: {got} != {tether_revision}"
    summary["tether_commit"] = got
    _progress(f"tether pinned at {got[:12]}")
    sys.path.insert(0, str(pin_dir / "src"))

    import torch as _torch

    _progress(f"torch {_torch.__version__} (cpu={not _torch.cuda.is_available()})")

    from huggingface_hub import snapshot_download

    _progress("snapshot_download openvla/openvla-7b@" + model_revision[:12])
    snap = snapshot_download("openvla/openvla-7b", revision=model_revision)
    summary["snapshot_dir"] = snap
    _progress("snapshot complete")

    export_dir = family_dir / "export"
    provenance_path = family_dir / "export.provenance.json"
    harness_env = dict(_os.environ)
    harness_env["PYTHONPATH"] = str(pin_dir / "src") + _os.pathsep + harness_env.get(
        "PYTHONPATH", ""
    )
    # Provenance sits beside the export, not inside it, so it never enters
    # the artifact identity the harness hashes over --onnx-dir.
    provenance = {
        "tether_revision": tether_revision,
        "model_revision": model_revision,
        **{p: _v(p) for p in PROVENANCE_PACKAGES},
    }
    summary["export_provenance"] = provenance
    summary["onnx_version_not_compared"] = _v("onnx")
    retained = (
        _json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
    )
    if reuse_export and retained == provenance:
        export_result = {"status": "reused", "onnx_path": str(export_dir / "model.onnx")}
        _progress("reusing retained export (provenance matches)")
    else:
        if reuse_export:
            _progress(f"not reusing: retained provenance {retained} != {provenance}")
        # Drop the old provenance first and clear the directory, so a
        # failed or partial export can never be mistaken for a matching one.
        provenance_path.unlink(missing_ok=True)
        shutil.rmtree(export_dir, ignore_errors=True)
        _parity_out_volume.commit()
        _progress("export_openvla_monolithic from pinned snapshot (CPU, subprocess)")
        code = (
            "import json, sys; from tether.exporters.openvla import export_openvla_monolithic; "
            "r = export_openvla_monolithic(sys.argv[1], sys.argv[2]); "
            "print('EXPORT_RESULT ' + json.dumps(r, default=str), flush=True)"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code, snap, str(export_dir)],
                cwd=str(pin_dir), env=harness_env, capture_output=True, text=True,
                timeout=export_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _parity_out_volume.commit()
            raise RuntimeError(
                f"export exceeded {export_timeout_seconds:.0f}s; stopped"
            ) from exc
        print(proc.stdout[-4000:], flush=True)
        print(proc.stderr[-4000:], flush=True)
        if proc.returncode != 0:
            raise RuntimeError(f"export failed with return code {proc.returncode}")
        line = next(x for x in proc.stdout.splitlines() if x.startswith("EXPORT_RESULT "))
        export_result = _json.loads(line.split(" ", 1)[1])
        provenance_path.write_text(_json.dumps(provenance, sort_keys=True) + "\n")
        _parity_out_volume.commit()
        _progress(f"export ok: {export_result.get('size_mb', 0):.1f}MB")
    summary["export"] = export_result

    # ONNX census: Pad must be 0; inventory every Concat by input dtype.
    import collections as _collections

    import onnx as _onnx
    from onnx import TensorProto as _TP

    _model = _onnx.load(str(export_dir / "model.onnx"), load_external_data=False)
    _info = {}
    for _vi in list(_model.graph.input) + list(_model.graph.output) + list(
        _model.graph.value_info
    ):
        _t = _vi.type.tensor_type
        _info[_vi.name] = (
            _TP.DataType.Name(_t.elem_type),
            [_d.dim_value for _d in _t.shape.dim],
        )
    for _init in _model.graph.initializer:
        _info[_init.name] = (_TP.DataType.Name(_init.data_type), list(_init.dims))
    _census = _collections.Counter(n.op_type for n in _model.graph.node)
    _pads = int(_census.get("Pad", 0))
    _concats = [
        (n.name, sorted({_info.get(i, ("?", []))[0] for i in n.input}))
        for n in _model.graph.node
        if n.op_type == "Concat"
    ]
    print(f"[census] nodes={len(_model.graph.node)} Pad={_pads}", flush=True)
    for _op, _count in _census.most_common(12):
        print(f"[census]   {_count:6d}  {_op}", flush=True)
    print(f"[census] inputs={[i.name for i in _model.graph.input]}", flush=True)
    print(f"[census] outputs={[(o.name, _info.get(o.name)) for o in _model.graph.output]}", flush=True)
    print(f"[census] concat_count={len(_concats)}", flush=True)
    for _name, _dtypes in _concats[:20]:
        print(f"[census]   {_name} dtypes={_dtypes}", flush=True)
    summary["census"] = {
        "nodes": len(_model.graph.node),
        "pad": _pads,
        "concat_count": len(_concats),
        "top_ops": dict(_census.most_common(12)),
    }

    if run_harness:
        receipt_path = family_dir / "openvla-cpu-precheck-receipt.json"
        cmd = [
            sys.executable,
            str(pin_dir / "scripts" / "local_openvla_monolithic_parity.py"),
            "--model-source", "openvla/openvla-7b",
            "--model-revision", model_revision,
            "--onnx-dir", str(export_dir),
            "--provider", "CPUExecutionProvider",
            "--action-dim", "7",
            "--trust-remote-code",
            "--receipt", str(receipt_path),
        ]
        _progress("running CPU numerics pre-check (acceptance stays not-run off CUDA)")
        receipt_path.unlink(missing_ok=True)
        try:
            proc = subprocess.run(
                cmd, cwd=str(pin_dir), capture_output=True, text=True, env=harness_env,
                timeout=parity_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _parity_out_volume.commit()
            raise RuntimeError(
                f"CPU pre-check exceeded {parity_timeout_seconds:.0f}s; stopped"
            ) from exc
        print(proc.stdout[-6000:], flush=True)
        print(proc.stderr[-3000:], flush=True)
        summary["harness_returncode"] = proc.returncode

        if receipt_path.is_file():
            receipt = _json.loads(receipt_path.read_text())
            summary["verdict"] = receipt.get("verdict")
            summary["action_token_agreement"] = (receipt.get("metrics") or {}).get(
                "action_token_agreement"
            )
            summary["full_cosine"] = (receipt.get("metrics") or {}).get("full_cosine")
            summary["full_max_abs"] = (receipt.get("metrics") or {}).get("full_max_abs")
            print("[receipt] " + _json.dumps(receipt.get("metrics", {}), sort_keys=True), flush=True)
        else:
            summary["verdict"] = "no-receipt"

    if run_id:
        run_record = family_dir / "runs" / f"cpu-{run_id}.json"
        run_record.parent.mkdir(parents=True, exist_ok=True)
        run_record.write_text(_json.dumps(summary, sort_keys=True, default=str) + "\n")
    _parity_out_volume.commit()
    _progress("done")
    return summary


@app.local_entrypoint()
def main(
    tether_revision: str = TETHER_REVISION,
    reuse_export: bool = False,
    run_harness: bool = True,
    export_timeout_seconds: float = EXPORT_TIMEOUT_SECONDS,
    parity_timeout_seconds: float = PARITY_TIMEOUT_SECONDS,
    run_id: str = "",
) -> None:
    import uuid

    result = run_export_cpu.remote(
        tether_revision=tether_revision,
        reuse_export=reuse_export,
        run_harness=run_harness,
        export_timeout_seconds=export_timeout_seconds,
        parity_timeout_seconds=parity_timeout_seconds,
        run_id=run_id or uuid.uuid4().hex,
    )
    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
