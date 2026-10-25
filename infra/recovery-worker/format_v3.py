"""Bounded, checksum-pinned v3 manifest traversal shared by private local tools."""

import hashlib
import json
import pathlib

SHARD_BYTES = 4 * 1024 * 1024


def checked_json(path, descriptor):
    path = pathlib.Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > SHARD_BYTES:
        raise ValueError("Missing or oversized manifest shard")
    raw = path.read_bytes()
    if len(raw) != descriptor["size"] or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError("Manifest shard checksum differs")
    return json.loads(raw)


def entries(directory, manifest):
    directory = pathlib.Path(directory)
    for index, descriptor in enumerate(manifest["indexes"]):
        listing = checked_json(directory / "indexes" / f"{index:08d}.json", descriptor)
        for number, item in enumerate(listing["entries"]):
            if item["kind"] not in {"table", "object"}:
                raise ValueError("Unknown manifest entry")
            page = checked_json(directory / "shards" / f"{index:08d}-{number:08d}.json", item)
            yield index, number, item, page
