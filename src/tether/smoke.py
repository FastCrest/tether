"""End-to-end install smoke helpers for the ``tether smoke`` command."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tether import __version__


class SmokeError(RuntimeError):
    """Raised for smoke setup failures with a user-actionable message."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _default_smoke_export_dir() -> Path:
    home = Path(os.environ.get("TETHER_HOME", Path.home() / ".cache" / "tether")).expanduser()
    return home / "smoke" / "export"


def _require_smoke_export_deps() -> tuple[Any, Any, Any, Any, Any, Any]:
    missing: list[str] = []
    modules: dict[str, Any] = {}
    for import_name, package_name in (
        ("numpy", "numpy"),
        ("onnx", "onnx"),
        ("tokenizers", "tokenizers"),
        ("transformers", "transformers"),
    ):
        try:
            modules[import_name] = __import__(import_name)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{package_name} ({type(exc).__name__}: {exc})")
    if missing:
        raise SmokeError(
            "Cannot create the local smoke export because dependencies are missing: "
            + ", ".join(missing)
            + ". Install the serve extra with `pip install 'fastcrest-tether[serve]'`."
        )
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    return (
        modules["numpy"],
        modules["onnx"],
        modules["tokenizers"].Tokenizer,
        WordLevel,
        Whitespace,
        PreTrainedTokenizerFast,
    )


def _require_serve_runtime_deps() -> None:
    try:
        __import__("onnxruntime")
        __import__("fastapi")
        __import__("uvicorn")
    except Exception as exc:  # noqa: BLE001
        raise SmokeError(
            "Cannot start the smoke server because the serve runtime is missing: "
            f"{type(exc).__name__}: {exc}. Install with "
            "`pip install 'fastcrest-tether[serve]'`."
        ) from exc


def create_smoke_export(export_dir: str | Path) -> Path:
    """Create a tiny offline SmolVLA-compatible ONNX export for smoke tests."""

    (
        np,
        onnx,
        Tokenizer,
        WordLevel,
        Whitespace,
        PreTrainedTokenizerFast,
    ) = _require_smoke_export_deps()
    from onnx import TensorProto, helper

    export_path = Path(export_dir).expanduser().resolve()
    tokenizer_dir = export_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    inputs = [
        helper.make_tensor_value_info("img_cam1", TensorProto.FLOAT, [1, 3, 512, 512]),
        helper.make_tensor_value_info("img_cam2", TensorProto.FLOAT, [1, 3, 512, 512]),
        helper.make_tensor_value_info("img_cam3", TensorProto.FLOAT, [1, 3, 512, 512]),
        helper.make_tensor_value_info("mask_cam1", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info("mask_cam2", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info("mask_cam3", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info("lang_tokens", TensorProto.INT64, [1, 16]),
        helper.make_tensor_value_info("lang_masks", TensorProto.BOOL, [1, 16]),
        helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, 32]),
        helper.make_tensor_value_info("noise", TensorProto.FLOAT, [1, 50, 32]),
    ]
    output = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 50, 32])
    tensor = helper.make_tensor(
        "actions_value",
        TensorProto.FLOAT,
        [1, 50, 32],
        np.zeros((1, 50, 32), dtype=np.float32).reshape(-1).tolist(),
    )
    node = helper.make_node("Constant", inputs=[], outputs=["actions"], value=tensor)
    graph = helper.make_graph([node], "tether-smoke", inputs, [output])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, export_path / "model.onnx")

    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "reach": 2,
                "pick": 3,
                "place": 4,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(tokenizer_dir)

    config = {
        "model_type": "smolvla",
        "export_kind": "monolithic",
        "num_denoising_steps": 1,
        "chunk_size": 50,
        "action_chunk_size": 50,
        "action_dim": 32,
        "max_state_dim": 32,
        "tokenizer_ref": "HuggingFaceTB/SmolLM2-135M",
        "tokenizer_path": "tokenizer",
        "tokenizer_bundled": True,
        "smoke_export": True,
        "created_by": "tether smoke",
        "created_at": _now_iso(),
    }
    (export_path / "tether_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return export_path


def find_free_port() -> int:
    """Return an available localhost TCP port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    old: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _offline_env(enabled: bool) -> dict[str, str]:
    env = {
        "TETHER_SKIP_ONBOARDING": "1",
        "TETHER_NO_UPGRADE_CHECK": "1",
        "TETHER_NO_CONTRIB_NUDGE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if enabled:
        env.update(
            {
                "TETHER_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    return env


def _run_deploy_doctor(export_dir: Path, offline: bool) -> dict[str, Any]:
    from tether.diagnostics import exit_code, format_json, run_all_checks

    with _temporary_env(_offline_env(offline)):
        results = run_all_checks(str(export_dir), "custom")
    body = json.loads(format_json(results, model_path=str(export_dir), embodiment_name="custom"))
    body["exit_code"] = exit_code(results)
    return body


def _read_json_url(url: str, *, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _post_json_url(url: str, body: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _log_tail(path: Path | None, *, max_lines: int = 80) -> list[str]:
    if path is None or not path.exists():
        return []
    text = path.read_text(errors="replace")
    return text.splitlines()[-max_lines:]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 1)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return round(float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac), 1)


def summarize_latency_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize smoke /act samples without including raw action payloads."""

    def _values(key: str, source: list[dict[str, Any]]) -> list[float]:
        return [
            float(sample[key])
            for sample in source
            if isinstance(sample.get(key), (int, float))
        ]

    inference = _values("latency_ms", samples)
    roundtrip = _values("roundtrip_ms", samples)
    warm_samples = samples[1:]
    first = samples[0] if samples else {}

    def _summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        return {
            "p50_ms": _percentile(values, 50.0),
            "p95_ms": _percentile(values, 95.0),
            "max_ms": round(max(values), 1),
        }

    return {
        "samples": len(samples),
        "first_sample": {
            "inference_ms": (
                round(float(first["latency_ms"]), 1)
                if isinstance(first.get("latency_ms"), (int, float))
                else 0.0
            ),
            "roundtrip_ms": (
                round(float(first["roundtrip_ms"]), 1)
                if isinstance(first.get("roundtrip_ms"), (int, float))
                else 0.0
            ),
        },
        "inference_ms": _summary(inference),
        "roundtrip_ms": _summary(roundtrip),
        "warm_inference_ms": _summary(_values("latency_ms", warm_samples)),
        "warm_roundtrip_ms": _summary(_values("roundtrip_ms", warm_samples)),
    }


def _stop_process(process: subprocess.Popen[str] | None) -> int | None:
    if process is None:
        return None
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if process.returncode in {-signal.SIGINT, -signal.SIGTERM}:
        return 0
    return process.returncode


def _wait_for_health(
    process: subprocess.Popen[str],
    base_url: str,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SmokeError(
                "server exited before /health was ready "
                f"(exit_code={process.returncode})"
            )
        try:
            return _read_json_url(f"{base_url}/health", timeout_s=1.0)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.25)
    raise SmokeError(
        f"server did not become healthy within {timeout_s:.1f}s; "
        f"last_error={last_error}"
    )


def run_smoke(
    *,
    export_dir: str | Path | None = None,
    offline: bool = True,
    port: int = 0,
    timeout_s: float = 30.0,
    keep_export: bool = True,
    act_samples: int = 3,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Run export, doctor, serve, /health, and /act as a single smoke receipt."""

    if act_samples < 1:
        raise SmokeError(f"act_samples must be >= 1, got {act_samples}")

    started = time.monotonic()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    server_process: subprocess.Popen[str] | None = None
    log_path: Path | None = None
    if export_dir is None:
        if keep_export:
            export_path = _default_smoke_export_dir()
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="tether-smoke-")
            export_path = Path(temp_dir.name) / "export"
    else:
        export_path = Path(export_dir).expanduser()

    if port == 0:
        port = find_free_port()

    base_url = f"http://127.0.0.1:{port}"
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": _now_iso(),
        "passed": False,
        "tether_version": __version__,
        "python": sys.version.split()[0],
        "offline": bool(offline),
        "export_dir": str(export_path.resolve()),
        "server": {
            "url": base_url,
            "port": port,
            "started": False,
            "exit_code": None,
            "log_tail": [],
        },
        "doctor": None,
        "health": None,
        "act": None,
        "act_samples": [],
        "latency": None,
        "checks": [],
        "duration_ms": 0.0,
        "error": None,
    }

    try:
        export_path = create_smoke_export(export_path)
        receipt["export_dir"] = str(export_path)
        receipt["checks"].append({"name": "create_smoke_export", "status": "pass"})

        doctor = _run_deploy_doctor(export_path, offline)
        receipt["doctor"] = doctor
        if doctor["summary"]["fail"]:
            raise SmokeError("deploy doctor reported failing checks")
        receipt["checks"].append(
            {"name": "deploy_doctor", "status": "pass", "summary": doctor["summary"]}
        )

        _require_serve_runtime_deps()
        log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w",
            encoding="utf-8",
            prefix="tether-smoke-serve-",
            suffix=".log",
            delete=False,
        )
        log_path = Path(log_file.name)
        env = os.environ.copy()
        env.update(_offline_env(offline))
        cmd = [
            python_executable or sys.executable,
            "-m",
            "tether.cli",
            "serve",
            str(export_path),
            "--device",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-prewarm",
        ]
        server_process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        log_file.close()
        receipt["server"]["command"] = cmd
        receipt["server"]["started"] = True

        health = _wait_for_health(server_process, base_url, timeout_s=timeout_s)
        receipt["health"] = health
        receipt["checks"].append({"name": "server_health", "status": "pass"})

        for idx in range(act_samples):
            act_start = time.monotonic()
            act_body = _post_json_url(
                f"{base_url}/act",
                {"instruction": "reach", "state": [0, 0, 0, 0, 0, 0]},
                timeout_s=timeout_s,
            )
            roundtrip_ms = (time.monotonic() - act_start) * 1000.0
            act_sample = {
                "sample": idx + 1,
                "num_actions": act_body.get("num_actions"),
                "action_dim": act_body.get("action_dim"),
                "latency_ms": act_body.get("latency_ms"),
                "roundtrip_ms": round(roundtrip_ms, 1),
                "inference_mode": act_body.get("inference_mode"),
                "provider_mode": act_body.get("provider_mode"),
                "active_providers": act_body.get("active_providers", []),
                "denoising_steps": act_body.get("denoising_steps"),
                "error": act_body.get("error"),
            }
            receipt["act_samples"].append(act_sample)
            receipt["act"] = act_sample
            act_ok = (
                act_sample["error"] is None
                and act_sample["num_actions"] == 50
                and act_sample["action_dim"] == 32
                and bool(act_sample["active_providers"])
            )
            if not act_ok:
                raise SmokeError(f"/act response failed shape/provider checks: {act_sample}")
        receipt["latency"] = summarize_latency_samples(receipt["act_samples"])
        receipt["checks"].append({"name": "act_roundtrip", "status": "pass"})
        receipt["passed"] = True
    except Exception as exc:  # noqa: BLE001
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["checks"].append({"name": "smoke", "status": "fail", "error": receipt["error"]})
    finally:
        receipt["server"]["exit_code"] = _stop_process(server_process)
        receipt["server"]["log_tail"] = _log_tail(log_path)
        receipt["duration_ms"] = round((time.monotonic() - started) * 1000.0, 1)
        if temp_dir is not None:
            temp_dir.cleanup()
    return receipt


def format_smoke_human(receipt: dict[str, Any]) -> str:
    """Return a concise terminal receipt."""

    status = "PASS" if receipt.get("passed") else "FAIL"
    lines = [
        f"tether smoke - {status}",
        f"version: {receipt.get('tether_version')}",
        f"export:  {receipt.get('export_dir')}",
        f"server:  {receipt.get('server', {}).get('url')}",
        f"offline: {receipt.get('offline')}",
    ]
    doctor = receipt.get("doctor") or {}
    if doctor.get("summary"):
        summary = doctor["summary"]
        lines.append(
            "doctor:  "
            f"{summary.get('pass', 0)} pass, {summary.get('fail', 0)} fail, "
            f"{summary.get('warn', 0)} warn, {summary.get('skip', 0)} skip"
        )
    act = receipt.get("act") or {}
    if act:
        lines.append(
            "act:     "
            f"{act.get('num_actions')}x{act.get('action_dim')} "
            f"{act.get('provider_mode')} {act.get('roundtrip_ms')}ms"
        )
    latency = receipt.get("latency") or {}
    roundtrip = latency.get("roundtrip_ms") or {}
    inference = latency.get("inference_ms") or {}
    warm_roundtrip = latency.get("warm_roundtrip_ms") or {}
    first_sample = latency.get("first_sample") or {}
    if roundtrip or inference:
        lines.append(
            "latency: "
            f"n={latency.get('samples', 0)} "
            f"first={first_sample.get('roundtrip_ms', 0.0)}ms, "
            f"inference p50/p95={inference.get('p50_ms', 0.0)}/"
            f"{inference.get('p95_ms', 0.0)}ms, "
            f"roundtrip p50/p95={roundtrip.get('p50_ms', 0.0)}/"
            f"{roundtrip.get('p95_ms', 0.0)}ms, "
            f"warm roundtrip p50/p95={warm_roundtrip.get('p50_ms', 0.0)}/"
            f"{warm_roundtrip.get('p95_ms', 0.0)}ms"
        )
    if receipt.get("error"):
        lines.append(f"error:   {receipt['error']}")
    return "\n".join(lines)


def format_smoke_markdown(receipt: dict[str, Any]) -> str:
    """Return a markdown proof receipt suitable for attaching to PRs."""

    status = "PASS" if receipt.get("passed") else "FAIL"
    doctor = receipt.get("doctor") or {}
    summary = doctor.get("summary") or {}
    act = receipt.get("act") or {}
    latency = receipt.get("latency") or {}
    inference = latency.get("inference_ms") or {}
    roundtrip = latency.get("roundtrip_ms") or {}
    warm_inference = latency.get("warm_inference_ms") or {}
    warm_roundtrip = latency.get("warm_roundtrip_ms") or {}
    first_sample = latency.get("first_sample") or {}
    lines = [
        "# Tether Smoke Receipt",
        "",
        f"- Status: {status}",
        f"- Tether version: {receipt.get('tether_version')}",
        f"- Python: {receipt.get('python')}",
        f"- Offline mode: {receipt.get('offline')}",
        f"- Export dir: `{receipt.get('export_dir')}`",
        f"- Server URL: `{receipt.get('server', {}).get('url')}`",
        f"- Duration: {receipt.get('duration_ms')} ms",
        "",
        "## Doctor",
        "",
        f"- Pass: {summary.get('pass', 0)}",
        f"- Fail: {summary.get('fail', 0)}",
        f"- Warn: {summary.get('warn', 0)}",
        f"- Skip: {summary.get('skip', 0)}",
        "",
        "## /act",
        "",
        f"- Shape: {act.get('num_actions')} x {act.get('action_dim')}",
        f"- Provider mode: {act.get('provider_mode')}",
        f"- Active providers: `{act.get('active_providers', [])}`",
        f"- Runtime latency: {act.get('latency_ms')} ms",
        f"- Roundtrip latency: {act.get('roundtrip_ms')} ms",
        "",
        "## Latency Summary",
        "",
        f"- Samples: {latency.get('samples', 0)}",
        f"- First inference: {first_sample.get('inference_ms', 0.0)} ms",
        f"- First roundtrip: {first_sample.get('roundtrip_ms', 0.0)} ms",
        f"- Inference p50: {inference.get('p50_ms', 0.0)} ms",
        f"- Inference p95: {inference.get('p95_ms', 0.0)} ms",
        f"- Roundtrip p50: {roundtrip.get('p50_ms', 0.0)} ms",
        f"- Roundtrip p95: {roundtrip.get('p95_ms', 0.0)} ms",
        f"- Warm inference p50: {warm_inference.get('p50_ms', 0.0)} ms",
        f"- Warm inference p95: {warm_inference.get('p95_ms', 0.0)} ms",
        f"- Warm roundtrip p50: {warm_roundtrip.get('p50_ms', 0.0)} ms",
        f"- Warm roundtrip p95: {warm_roundtrip.get('p95_ms', 0.0)} ms",
    ]
    if receipt.get("error"):
        lines.extend(["", "## Error", "", f"`{receipt['error']}`"])
    return "\n".join(lines) + "\n"
