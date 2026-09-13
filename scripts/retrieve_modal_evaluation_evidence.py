"""Retrieve and validate a bounded evaluation-evidence directory from Modal."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import subprocess

from tether.eval.evidence_capture import validate_episode_evidence


VOLUME = "pi0-onnx-outputs"


def safe_remote_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "evaluation-evidence":
        raise ValueError("remote path must stay under evaluation-evidence/")
    return path.as_posix()


def retrieve(remote: str, destination: Path, *, runner=subprocess.run) -> list[Path]:
    remote = safe_remote_path(remote)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    runner(["modal", "volume", "get", VOLUME, remote, str(destination)], check=True)
    manifests = sorted(destination.rglob("episode-manifest.json"))
    if not manifests:
        raise ValueError("retrieved directory contains no episode evidence manifests")
    for manifest in manifests:
        validate_episode_evidence(manifest.parent)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("remote", help="Path below evaluation-evidence/ on the Modal output volume")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifests = retrieve(args.remote, args.destination)
    print(f"Validated {len(manifests)} episode evidence manifest(s) in {args.destination.resolve()}")


if __name__ == "__main__":
    main()
