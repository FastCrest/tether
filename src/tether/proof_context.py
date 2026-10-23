"""Evidence scope and bounded export metadata for proof packets.

Returned plans and scheduler settings do not establish consumer execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SERVING_CONTRACT = {
    "scope": "serving_only",
    "input_source": "synthetic_zero_state",
    "request_sequence": "serial_identical_observations",
    "physical_execution_observed": False,
    "limitation": "HTTP responses measure serving; no robot consumer executes these chunks.",
}

MODEL_FIELDS = (
    "model_type",
    "model_id",
    "policy_id",
    "export_kind",
    "action_dim",
    "chunk_size",
    "action_horizon",
    "action_units",
    "precision",
    "denoising_steps",
    "num_denoising_steps",
)


def export_metadata(export_dir: Path) -> dict[str, Any]:
    path = export_dir / "tether_config.json"
    if not path.exists():
        return {"source": "tether_config.json", "available": False, "fields": {}}
    if path.stat().st_size > 1024 * 1024:
        return {
            "source": "tether_config.json",
            "available": False,
            "fields": {},
            "error": "metadata exceeds 1 MiB capture limit",
        }
    try:
        config = json.loads(path.read_text())
        if not isinstance(config, dict):
            raise ValueError("export config must be an object")
    except (OSError, ValueError) as exc:
        return {
            "source": "tether_config.json",
            "available": False,
            "fields": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "source": "tether_config.json",
        "available": True,
        "fields": {k: config[k] for k in MODEL_FIELDS if k in config},
    }


def certificate_context(receipt: dict[str, Any]) -> dict[str, Any]:
    """Copy recorded metadata; never infer model/precision from a directory name."""
    metadata = receipt.get("export_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    config = receipt.get("config") or {}
    config = config.get("body") if isinstance(config, dict) else {}
    config = config if isinstance(config, dict) else {}
    fields = metadata.get("fields")
    model = dict(fields) if isinstance(fields, dict) else {}
    model.update({k: config[k] for k in MODEL_FIELDS if k in config})
    samples = receipt.get("act_samples")
    samples = [s for s in samples if isinstance(s, dict)] if isinstance(samples, list) else []

    def modes(key):
        return sorted({str(s[key]) for s in samples if s.get(key) is not None})

    active = sorted(
        {
            str(p)
            for s in samples
            for p in (s["active_providers"] if isinstance(s.get("active_providers"), list) else [])
        }
    )
    contract = receipt.get("evidence_contract")
    server = receipt.get("server") or {}
    startup = server.get("startup")
    error = str(receipt.get("error") or "")
    if startup is None and "server exited before /health was ready" in error:
        startup = {
            "status": "failed",
            "diagnostic": "server_exit_before_ready",
            "source": "legacy_proof_error",
            "exit_code": server.get("exit_code"),
            "process_memory": {"source": "unavailable", "oom_confirmed": False},
        }
    return {
        "evidence_contract": dict(contract)
        if isinstance(contract, dict)
        else {
            "scope": "legacy_unclassified",
            "physical_execution_observed": False,
        },
        "model": model,
        "runtime": {
            "request": receipt.get("runtime_request"),
            "active_providers": active,
            "provider_modes": modes("provider_mode"),
            "inference_modes": modes("inference_mode"),
            "execution_providers": config.get("execution_providers"),
        },
        "provenance": {
            "model_metadata_source": "declared_export_and_server_config",
            "precision_verification": "not_measured",
            "tether_version": receipt.get("tether_version"),
            "python": receipt.get("python"),
            "platform": receipt.get("platform"),
            "export_manifest": receipt.get("export_manifest"),
            "startup": startup,
            "source_proof_error": receipt.get("error"),
        },
    }
