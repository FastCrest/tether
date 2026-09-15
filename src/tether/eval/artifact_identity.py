from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ARTIFACT_IDENTITY_SCHEMA = 1
MARKER = "TETHER_ARTIFACT_IDENTITY_JSON="


class ArtifactIdentityError(ValueError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifacts(root: Path) -> list[dict[str, Any]]:
    """Return the same sorted regular-file manifest Studio uses for imported checkpoints.

    Symbolic links are deliberately excluded instead of dereferenced. Studio's existing
    checkpoint import hash uses the same rule (`is_file() and not is_symlink()`), so a
    provider receipt can be compared byte-for-byte with the already registered candidate
    identity without inventing a second hashing convention.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactIdentityError(f"Artifact root is not a directory: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactIdentityError(f"Unsafe artifact path: {relative}")
        rows.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not rows:
        raise ArtifactIdentityError("Artifact root contains no regular non-symlink files.")
    return rows


def artifact_digest(artifacts: list[dict[str, Any]]) -> str:
    """Match Studio's existing training-import digest exactly."""
    return hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest()


def build_artifact_identity(
    root: Path,
    *,
    source: str,
    revision: str | None = None,
    kind: str = "checkpoint",
) -> dict[str, Any]:
    source = source.strip()
    if not source:
        raise ArtifactIdentityError("A logical artifact source is required.")
    artifacts = collect_artifacts(root)
    receipt = {
        "schema": ARTIFACT_IDENTITY_SCHEMA,
        "kind": kind,
        "source": source,
        "revision": revision or None,
        "artifact_sha256": artifact_digest(artifacts),
        "file_count": len(artifacts),
        "size_bytes": sum(int(item["size_bytes"]) for item in artifacts),
        "artifacts": artifacts,
        "hash_contract": "studio-training-artifacts-v1",
        "symlink_policy": "excluded",
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a deterministic byte identity receipt for an evaluation checkpoint."
    )
    parser.add_argument("path", help="Local materialized checkpoint directory")
    parser.add_argument("--source", required=True, help="Logical checkpoint path/HF repository")
    parser.add_argument("--revision", default="", help="Exact remote revision when applicable")
    parser.add_argument("--kind", default="checkpoint", help="Artifact role/kind")
    args = parser.parse_args(argv)
    receipt = build_artifact_identity(
        Path(args.path), source=args.source, revision=args.revision or None, kind=args.kind
    )
    print(MARKER + json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_IDENTITY_SCHEMA",
    "MARKER",
    "ArtifactIdentityError",
    "collect_artifacts",
    "artifact_digest",
    "build_artifact_identity",
]
