"""Shared constants and lightweight schema helpers for Tether Comply."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "tether.comply_bundle.v1"
SBOM_SCHEMA_VERSION = "CycloneDX-1.5"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class DeploymentMetadata:
    product_name: str = "Tether robot deployment"
    deployment_id: str = ""
    robot_id: str = ""
    manufacturer: str = ""
    operator: str = ""
    data_residency: str = "customer-controlled"
    retention_days: int = 30
    vulnerability_contact: str = "security@example.com"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: str
    sha256: str
    size_bytes: int
    required: bool = False
    present: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegulatoryControl:
    control_id: str
    regulation: str
    article: str
    requirement: str
    tether_evidence: list[str]
    customer_gap: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceCollection:
    verify_dir: str
    parity_cert: dict[str, Any] | None = None
    parity_cert_signature_valid: bool | None = None
    parity_cert_signature_error: str = ""
    parity_md_sha256: str = ""
    audit_summary: dict[str, Any] = field(default_factory=dict)
    actionguard: dict[str, Any] = field(default_factory=dict)
    source_artifacts: list[ArtifactRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verify_dir": self.verify_dir,
            "parity_cert": self.parity_cert,
            "parity_cert_signature_valid": self.parity_cert_signature_valid,
            "parity_cert_signature_error": self.parity_cert_signature_error,
            "parity_md_sha256": self.parity_md_sha256,
            "audit_summary": self.audit_summary,
            "actionguard": self.actionguard,
            "source_artifacts": [a.to_dict() for a in self.source_artifacts],
        }
