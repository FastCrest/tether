#!/usr/bin/env python3
"""Restore a checksum-pinned backup to NEW local files. Never connects to production."""

import base64
import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import sys

from format_v3 import entries


def fingerprint(rows):
    values = sorted(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def restore(snapshot_file, expected_sha256, target, media_directory=None):
    os.umask(0o077)
    snapshot_file = pathlib.Path(snapshot_file)
    if snapshot_file.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Snapshot exceeds reviewed size")
    raw = snapshot_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Snapshot checksum does not match publication receipt")
    snapshot = json.loads(raw)
    if snapshot.get("version") not in (1, 2, 3) or snapshot.get("complete") is not True:
        raise ValueError("Incomplete or unsupported snapshot")
    if snapshot["version"] == 3:
        return restore_v3(snapshot_file, snapshot, target)
    legacy_tables = {"contributors", "uploads", "daily_uploads", "revoke_requests"}
    auth_tables = legacy_tables | {"contributor_keys", "contributor_nonces"}
    expected_tables = set(snapshot["tables"])
    if expected_tables not in (legacy_tables, auth_tables):
        raise ValueError("Unexpected table set")
    target = pathlib.Path(target)
    target.mkdir(mode=0o700, exist_ok=False)

    def quote(value):
        return '"' + value.replace('"', '""') + '"'

    with sqlite3.connect(target / "metadata.sqlite") as db:
        db.execute("PRAGMA foreign_keys=OFF")
        for item in snapshot["schema"]:
            if item["type"] not in {"table", "index"}:
                raise ValueError("Unsupported schema object")
            if item["type"] == "table":
                if item["name"] not in expected_tables:
                    raise ValueError("Unexpected table")
                db.execute(item["sql"])
        for name, rows in snapshot["tables"].items():
            for row in rows:
                db.execute(
                    "INSERT INTO "
                    + quote(name)
                    + " ("
                    + ",".join(map(quote, row))
                    + ") VALUES ("
                    + ",".join("?" for _ in row)
                    + ")",
                    list(row.values()),
                )
        for item in snapshot["schema"]:
            if item["type"] == "index":
                db.execute(item["sql"])
        db.commit()
        if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("SQLite integrity check failed")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Foreign-key check failed")
        db.row_factory = sqlite3.Row
        for name, rows in snapshot["tables"].items():
            actual = [dict(row) for row in db.execute("SELECT * FROM " + quote(name))]
            if fingerprint(actual) != fingerprint(rows):
                raise ValueError("Restored row fingerprint differs")
    mapping = []
    media_directory = (
        pathlib.Path(media_directory) if media_directory else snapshot_file.parent / "objects"
    )
    seen = set()
    for index, item in enumerate(snapshot["objects"]):
        if item["key"] in seen:
            raise ValueError("Duplicate source object key")
        seen.add(item["key"])
        name = hashlib.sha256(item["key"].encode()).hexdigest()
        output = target / name
        if snapshot["version"] == 1:
            data = base64.b64decode(item["bodyBase64"], validate=True)
            output.write_bytes(data)
        else:
            # Numeric filenames are generated locally, never taken from an untrusted object key.
            source = media_directory / f"{index:06d}"
            if source.is_symlink() or not source.is_file():
                raise ValueError("Missing local backup object")
            with source.open("rb") as src, output.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        digest = hashlib.sha256()
        with output.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if output.stat().st_size != item["size"] or digest.hexdigest() != item["sha256"]:
            raise ValueError("Object checksum differs")
        mapping.append(
            {
                "key": item["key"],
                "file": name,
                "sha256": item["sha256"],
                "version": item.get("version"),
            }
        )
    (target / "object-map.json").write_text(json.dumps(mapping, indent=2))
    return {
        "verified": True,
        "tables": len(expected_tables),
        "rows": sum(len(rows) for rows in snapshot["tables"].values()),
        "objects": len(mapping),
        "sourceSnapshotAt": snapshot["capturedAt"],
        "productionModified": False,
        "revocationsExecuted": False,
    }


def restore_v3(snapshot_file, snapshot, target):
    expected_tables = {
        "contributors",
        "uploads",
        "daily_uploads",
        "revoke_requests",
        "contributor_keys",
        "contributor_nonces",
    }
    if set(snapshot["tables"]) != expected_tables:
        raise ValueError("Unexpected table set")
    target = pathlib.Path(target)
    target.mkdir(mode=0o700, exist_ok=False)
    directory = snapshot_file.parent

    def quote(value):
        return '"' + value.replace('"', '""') + '"'

    counts = dict.fromkeys(expected_tables, 0)
    with sqlite3.connect(target / "metadata.sqlite") as db:
        db.execute("PRAGMA foreign_keys=OFF")
        for item in snapshot["schema"]:
            if item["type"] not in {"table", "index"} or item["tbl_name"] not in expected_tables:
                raise ValueError("Unexpected schema object")
            if item["type"] == "table":
                db.execute(item["sql"])
        for _, _, descriptor, page in entries(directory, snapshot):
            if descriptor["kind"] != "table":
                continue
            table = page["table"]
            if table not in expected_tables or descriptor["table"] != table:
                raise ValueError("Unexpected table shard")
            for source_row in page["rows"]:
                row = {key: value for key, value in source_row.items() if key != "__backup_rowid"}
                db.execute(
                    "INSERT INTO "
                    + quote(table)
                    + " ("
                    + ",".join(map(quote, row))
                    + ") VALUES ("
                    + ",".join("?" for _ in row)
                    + ")",
                    list(row.values()),
                )
            counts[table] += len(page["rows"])
        if counts != snapshot["tables"]:
            raise ValueError("Restored table counts differ")
        for item in snapshot["schema"]:
            if item["type"] == "index":
                db.execute(item["sql"])
        db.commit()
        if (
            db.execute("PRAGMA integrity_check").fetchone() != ("ok",)
            or db.execute("PRAGMA foreign_key_check").fetchall()
        ):
            raise ValueError("Restored SQLite integrity or foreign keys differ")
        db.row_factory = sqlite3.Row
        offsets = dict.fromkeys(expected_tables, 0)
        for _, _, descriptor, page in entries(directory, snapshot):
            if descriptor["kind"] == "table":
                table = page["table"]
                expected = [
                    {key: value for key, value in row.items() if key != "__backup_rowid"}
                    for row in page["rows"]
                ]
                actual = [
                    dict(row)
                    for row in db.execute(
                        "SELECT * FROM " + quote(table) + " ORDER BY rowid LIMIT ? OFFSET ?",
                        (len(expected), offsets[table]),
                    )
                ]
                if fingerprint(actual) != fingerprint(expected):
                    raise ValueError("Restored row fingerprint differs")
                offsets[table] += len(expected)
    seen, mapping, object_bytes = set(), [], 0
    for index, number, descriptor, media in entries(directory, snapshot):
        if descriptor["kind"] != "object":
            continue
        if media["key"] in seen:
            raise ValueError("Duplicate source object key")
        seen.add(media["key"])
        name = hashlib.sha256(media["key"].encode()).hexdigest()
        full, offset = hashlib.sha256(), 0
        with (target / name).open("xb") as output:
            for chunk, piece in enumerate(media["chunks"]):
                source = directory / "media" / f"{index:08d}-{number:08d}-{chunk:08d}"
                if (
                    source.is_symlink()
                    or not source.is_file()
                    or piece["offset"] != offset
                    or source.stat().st_size != piece["size"]
                ):
                    raise ValueError("Missing or unordered media chunk")
                checksum = hashlib.sha256()
                with source.open("rb") as stream:
                    while data := stream.read(1024 * 1024):
                        checksum.update(data)
                        full.update(data)
                        output.write(data)
                        offset += len(data)
                if checksum.hexdigest() != piece["sha256"]:
                    raise ValueError("Media chunk checksum differs")
        if offset != media["size"]:
            raise ValueError("Restored media size differs")
        object_bytes += offset
        mapping.append(
            {
                "key": media["key"],
                "file": name,
                "sha256": full.hexdigest(),
                "version": media["version"],
            }
        )
    if len(mapping) != snapshot["objectCount"] or object_bytes != snapshot["objectBytes"]:
        raise ValueError("Restored media inventory differs")
    (target / "object-map.json").write_text(json.dumps(mapping, indent=2))
    return {
        "verified": True,
        "version": 3,
        "tables": len(counts),
        "rows": sum(counts.values()),
        "objects": len(mapping),
        "objectBytes": object_bytes,
        "sourceSnapshotAt": snapshot["capturedAt"],
        "consistency": snapshot["consistency"],
        "productionModified": False,
        "revocationsExecuted": False,
    }


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit(
            "Usage: restore.py SNAPSHOT_JSON PUBLICATION_SHA256 NEW_LOCAL_DIRECTORY [MEDIA_DIRECTORY]"
        )
    print(json.dumps(restore(*sys.argv[1:])))
