#!/usr/bin/env python3
"""Restore a checksum-pinned backup to NEW local files. Never connects to production."""

import base64
import hashlib
import json
import os
import pathlib
import sqlite3
import sys


def fingerprint(rows):
    values = sorted(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def restore(snapshot_file, expected_sha256, target):
    os.umask(0o077)
    snapshot_file = pathlib.Path(snapshot_file)
    if snapshot_file.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Snapshot exceeds reviewed size")
    raw = snapshot_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Snapshot checksum does not match publication receipt")
    snapshot = json.loads(raw)
    if snapshot.get("version") != 1 or snapshot.get("complete") is not True:
        raise ValueError("Incomplete or unsupported snapshot")
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
    for item in snapshot["objects"]:
        data = base64.b64decode(item["bodyBase64"], validate=True)
        if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Object checksum differs")
        name = hashlib.sha256(item["key"].encode()).hexdigest()
        (target / name).write_bytes(data)
        mapping.append({"key": item["key"], "file": name, "sha256": item["sha256"]})
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


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: restore.py SNAPSHOT_JSON PUBLICATION_SHA256 NEW_LOCAL_DIRECTORY")
    print(json.dumps(restore(*sys.argv[1:])))
