"""Machine-readable parity certificate for ``tether verify``.

``PARITY.md`` is the human receipt. ``parity.cert.json`` is the artifact Cloud
and Comply consume: stable schema, deterministic canonical bytes, optional
Ed25519 signature, and hashes for adjacent receipt files.
"""
from __future__ import annotations

import base64
import json
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from tether.verification_report import _sha256

if TYPE_CHECKING:
    from tether.verify import ParityVerdict

CERT_FILENAME = "parity.cert.json"
SIG_FILENAME = "parity.cert.sig"
SCHEMA_VERSION = "tether.parity_cert.v1"


class ParityCertError(Exception):
    """Raised when parity-cert signing or verification fails."""


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Canonical JSON used for signatures: sorted keys, no whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tether_version() -> str:
    try:
        from tether import __version__

        return __version__
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _read_signing_key(ref: str) -> bytes:
    if ref.startswith("env:"):
        import os

        name = ref.removeprefix("env:")
        value = os.environ.get(name)
        if not value:
            raise ParityCertError(f"Signing key environment variable is empty or missing: {name}")
        return value.replace("\\n", "\n").encode("utf-8")
    if ref.startswith("file:"):
        return Path(ref.removeprefix("file:")).expanduser().read_bytes()
    path = Path(ref).expanduser()
    if path.exists():
        return path.read_bytes()
    return ref.replace("\\n", "\n").encode("utf-8")


def load_ed25519_private_key(ref: str) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from env/file/raw text.

    Supported forms:
    - ``env:VAR`` where VAR contains PEM or base64 32-byte seed
    - ``file:path`` or an existing path containing PEM or base64 32-byte seed
    - raw PEM text
    - raw base64 32-byte seed
    """
    raw = _read_signing_key(ref).strip()
    try:
        key = serialization.load_pem_private_key(raw, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ParityCertError("Signing key is not an Ed25519 private key.")
        return key
    except ValueError:
        pass

    try:
        key = serialization.load_der_private_key(raw, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ParityCertError("Signing key is not an Ed25519 private key.")
        return key
    except ValueError:
        pass

    try:
        seed = base64.b64decode(raw, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ParityCertError(
            "Signing key must be PEM, DER, or base64-encoded 32-byte Ed25519 seed."
        ) from exc
    if len(seed) != 32:
        raise ParityCertError(
            f"Ed25519 seed must be 32 bytes after base64 decoding, got {len(seed)}."
        )
    return Ed25519PrivateKey.from_private_bytes(seed)


def build_parity_cert(verdict: "ParityVerdict", *, parity_md_path: str | Path | None = None) -> dict[str, Any]:
    """Build the unsigned parity certificate payload."""
    artifacts: dict[str, Any] = {}
    if parity_md_path is not None:
        p = Path(parity_md_path)
        if p.exists():
            artifacts["PARITY.md"] = {"sha256": _sha256(p)}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": verdict.generated_at,
        "tether_version": _tether_version(),
        "platform": platform.platform(),
        "optimized_ref": verdict.optimized_ref,
        "original_ref": verdict.original_ref,
        "suite": verdict.suite,
        "target": verdict.target,
        "verdict": "PASS" if verdict.passed else "FAIL",
        "passed": verdict.passed,
        "n_episodes": verdict.n_episodes,
        "success_rates": {
            "original": verdict.original_success_rate,
            "optimized": verdict.optimized_success_rate,
            "delta": verdict.success_rate_delta,
        },
        "first_failing_gate_id": verdict.first_failing_gate_id,
        "gates": [
            {
                "gate_id": g.gate_id,
                "gate_class": g.gate_class,
                "passed": g.passed,
                "measured": g.measured,
                "threshold": g.threshold,
                "message": g.message,
            }
            for g in verdict.eval_report.all_gates
        ],
        "eval_report": verdict.eval_report.to_dict(),
        "artifacts": artifacts,
    }


def sign_parity_cert(
    cert: dict[str, Any],
    *,
    signing_key: str,
    key_id: str = "",
) -> dict[str, Any]:
    """Return a copy of ``cert`` with an Ed25519 signature block."""
    payload = {k: v for k, v in cert.items() if k != "signature"}
    private_key = load_ed25519_private_key(signing_key)
    signature = private_key.sign(canonical_json_bytes(payload))
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signed = dict(payload)
    signed["signature"] = {
        "alg": "Ed25519",
        "key_id": key_id,
        "sig": base64.b64encode(signature).decode("ascii"),
        "public_key_b64": base64.b64encode(public_key).decode("ascii"),
    }
    return signed


def verify_parity_cert_signature(cert: dict[str, Any]) -> None:
    """Verify a cert's embedded signature. Raises ``ParityCertError`` on fail."""
    sig_block = cert.get("signature")
    if not isinstance(sig_block, dict):
        raise ParityCertError("Parity cert has no signature block.")
    if sig_block.get("alg") != "Ed25519":
        raise ParityCertError(f"Unsupported signature alg: {sig_block.get('alg')!r}")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(str(sig_block["public_key_b64"]), validate=True)
        )
        signature = base64.b64decode(str(sig_block["sig"]), validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ParityCertError(f"Malformed parity cert signature block: {exc}") from exc

    payload = {k: v for k, v in cert.items() if k != "signature"}
    try:
        public_key.verify(signature, canonical_json_bytes(payload))
    except InvalidSignature as exc:
        raise ParityCertError("Parity cert signature verification failed.") from exc


def write_parity_cert(
    output_dir: str | Path,
    verdict: "ParityVerdict",
    *,
    parity_md_path: str | Path | None = None,
    signing_key: str = "",
    key_id: str = "",
) -> tuple[Path, Path | None]:
    """Write ``parity.cert.json`` and optional detached signature.

    Returns ``(cert_path, sig_path_or_none)``.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cert = build_parity_cert(verdict, parity_md_path=parity_md_path)
    sig_path: Path | None = None
    if signing_key:
        cert = sign_parity_cert(cert, signing_key=signing_key, key_id=key_id)
        sig_path = output / SIG_FILENAME
        sig_path.write_text(cert["signature"]["sig"] + "\n")

    cert_path = output / CERT_FILENAME
    cert_path.write_text(json.dumps(cert, indent=2, sort_keys=True) + "\n")
    return cert_path, sig_path


__all__ = [
    "CERT_FILENAME",
    "SCHEMA_VERSION",
    "SIG_FILENAME",
    "ParityCertError",
    "build_parity_cert",
    "canonical_json_bytes",
    "load_ed25519_private_key",
    "sign_parity_cert",
    "verify_parity_cert_signature",
    "write_parity_cert",
]
