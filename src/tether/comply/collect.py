"""Collect Tether verification, audit, and ActionGuard evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from tether.comply.audit import summarize_audit_log
from tether.comply.schemas import ArtifactRef, EvidenceCollection
from tether.parity_cert import verify_parity_cert_signature
from tether.verification_report import _sha256


def _artifact_ref(path: Path, *, name: str | None = None, required: bool = False) -> ArtifactRef:
    return ArtifactRef(
        name=name or path.name,
        path=str(path),
        sha256=_sha256(path) if path.exists() and path.is_file() else "",
        size_bytes=path.stat().st_size if path.exists() and path.is_file() else 0,
        required=required,
        present=path.exists() and path.is_file(),
    )


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _summarize_actionguard(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"present": False, "status": "missing", "path": ""}
    p = Path(path)
    if not p.exists():
        return {"present": False, "status": "missing", "path": str(p)}
    try:
        cfg = _load_json(p)
    except Exception as exc:  # noqa: BLE001
        return {"present": False, "status": "unreadable", "path": str(p), "error": str(exc)}

    joint_names = cfg.get("joint_names") if isinstance(cfg.get("joint_names"), list) else []
    position_min = cfg.get("position_min") if isinstance(cfg.get("position_min"), list) else []
    position_max = cfg.get("position_max") if isinstance(cfg.get("position_max"), list) else []
    velocity_max = cfg.get("velocity_max") if isinstance(cfg.get("velocity_max"), list) else []
    effort_max = cfg.get("effort_max") if isinstance(cfg.get("effort_max"), list) else []
    workspace_min = cfg.get("workspace_min") if isinstance(cfg.get("workspace_min"), list) else []
    workspace_max = cfg.get("workspace_max") if isinstance(cfg.get("workspace_max"), list) else []
    return {
        "present": True,
        "status": "ok",
        "path": str(p),
        "sha256": _sha256(p),
        "joint_count": len(joint_names) or max(len(position_min), len(position_max), len(velocity_max)),
        "has_position_limits": bool(position_min and position_max),
        "has_velocity_limits": bool(velocity_max),
        "has_effort_limits": bool(effort_max),
        "has_workspace_limits": bool(workspace_min and workspace_max),
        "joint_names": joint_names,
        "config": cfg,
    }


def collect_evidence(
    *,
    verify_dir: str | Path,
    audit_log: str | Path | None = None,
    actionguard: str | Path | None = None,
) -> EvidenceCollection:
    verify = Path(verify_dir)
    if not verify.exists():
        raise FileNotFoundError(f"verify_dir does not exist: {verify}")

    parity_cert_path = verify / "parity.cert.json"
    parity_md_path = verify / "PARITY.md"
    parity_sig_path = verify / "parity.cert.sig"

    artifacts: list[ArtifactRef] = [
        _artifact_ref(parity_cert_path, required=True),
        _artifact_ref(parity_md_path, required=False),
    ]
    if parity_sig_path.exists():
        artifacts.append(_artifact_ref(parity_sig_path, required=False))

    parity_cert: dict[str, Any] | None = None
    cert_valid: bool | None = None
    cert_error = ""
    if parity_cert_path.exists():
        parity_cert = _load_json(parity_cert_path)
        if isinstance(parity_cert.get("signature"), dict):
            try:
                verify_parity_cert_signature(parity_cert)
                cert_valid = True
            except Exception as exc:  # noqa: BLE001
                cert_valid = False
                cert_error = str(exc)

    actionguard_summary = _summarize_actionguard(actionguard)
    if actionguard_summary.get("present"):
        artifacts.append(_artifact_ref(Path(str(actionguard)), name="actionguard_config.json", required=True))

    return EvidenceCollection(
        verify_dir=str(verify),
        parity_cert=parity_cert,
        parity_cert_signature_valid=cert_valid,
        parity_cert_signature_error=cert_error,
        parity_md_sha256=_sha256(parity_md_path) if parity_md_path.exists() else "",
        audit_summary=summarize_audit_log(audit_log),
        actionguard=actionguard_summary,
        source_artifacts=artifacts,
    )


def copy_source_artifacts(evidence: EvidenceCollection, artifacts_dir: str | Path) -> list[Path]:
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for artifact in evidence.source_artifacts:
        src = Path(artifact.path)
        if not src.exists() or not src.is_file():
            continue
        name = artifact.name
        if name == "actionguard_config.json":
            dest = out / "actionguard_config.json"
        else:
            dest = out / src.name
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


__all__ = ["collect_evidence", "copy_source_artifacts"]
