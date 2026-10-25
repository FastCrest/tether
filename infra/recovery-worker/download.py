#!/usr/bin/env python3
"""Download checksum-pinned private backups. RECOVERY_TOKEN stays in the environment."""

import hashlib
import json
import os
import pathlib
import re
import sys
import urllib.parse
import urllib.request

from format_v3 import SHARD_BYTES


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Backup endpoint redirects are not accepted")


def download(endpoint, key, expected_sha256, destination):
    os.umask(0o077)
    url = urllib.parse.urlsplit(endpoint)
    if (
        url.scheme != "https"
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ("", "/")
    ):
        raise ValueError("Expected an HTTPS Worker origin")
    if not re.fullmatch(r"snapshots/[A-Za-z0-9TZ-]+/snapshot\.json", key):
        raise ValueError("Invalid snapshot key")
    token = os.environ["RECOVERY_TOKEN"]
    destination = pathlib.Path(destination)
    destination.mkdir(mode=0o700, exist_ok=False)
    opener = urllib.request.build_opener(NoRedirect())

    def fetch(path, target, maximum):
        request = urllib.request.Request(
            endpoint.rstrip("/") + path,
            headers={"Authorization": "Bearer " + token, "User-Agent": "Tether recovery operator"},
        )
        digest, size = hashlib.sha256(), 0
        with opener.open(request, timeout=120) as response, target.open("xb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > maximum:
                    raise ValueError("Download exceeds manifest size")
                digest.update(chunk)
                output.write(chunk)
        return size, digest.hexdigest()

    query = urllib.parse.urlencode({"key": key})
    manifest_file = destination / "snapshot.json"
    _, actual = fetch("/snapshot?" + query, manifest_file, 8 * 1024 * 1024)
    if actual != expected_sha256:
        raise ValueError("Snapshot checksum does not match publication receipt")
    manifest = json.loads(manifest_file.read_bytes())
    if manifest.get("complete") is not True or manifest.get("version") not in (1, 2, 3):
        raise ValueError("Incomplete or unsupported snapshot")
    if manifest["version"] == 2:
        (destination / "objects").mkdir(mode=0o700)
        for index, item in enumerate(manifest["objects"]):
            size, checksum = fetch(
                "/object?" + query + "&index=" + str(index),
                destination / "objects" / f"{index:06d}",
                item["size"],
            )
            if size != item["size"] or checksum != item["sha256"]:
                raise ValueError("Downloaded object checksum differs")
    if manifest["version"] == 3:
        for name in ("indexes", "shards", "media"):
            (destination / name).mkdir(mode=0o700)
        for index, descriptor in enumerate(manifest["indexes"]):
            index_path = destination / "indexes" / f"{index:08d}.json"
            size, checksum = fetch(
                "/index?" + query + "&index=" + str(index), index_path, SHARD_BYTES
            )
            if size != descriptor["size"] or checksum != descriptor["sha256"]:
                raise ValueError("Index checksum differs")
            listing = json.loads(index_path.read_bytes())
            for number, item in enumerate(listing["entries"]):
                if item["kind"] not in {"table", "object"}:
                    raise ValueError("Unknown manifest entry")
                shard_path = destination / "shards" / f"{index:08d}-{number:08d}.json"
                suffix = query + "&index=" + str(index) + "&entry=" + str(number)
                size, checksum = fetch("/shard?" + suffix, shard_path, SHARD_BYTES)
                if size != item["size"] or checksum != item["sha256"]:
                    raise ValueError("Manifest shard checksum differs")
                if item["kind"] == "object":
                    media = json.loads(shard_path.read_bytes())
                    for chunk, piece in enumerate(media["chunks"]):
                        size, checksum = fetch(
                            "/chunk?" + suffix + "&chunk=" + str(chunk),
                            destination / "media" / f"{index:08d}-{number:08d}-{chunk:08d}",
                            piece["size"],
                        )
                        if size != piece["size"] or checksum != piece["sha256"]:
                            raise ValueError("Media chunk checksum differs")
    return {
        "verified": True,
        "objects": manifest["objectCount"]
        if manifest["version"] == 3
        else len(manifest["objects"]),
        "snapshot": str(manifest_file),
    }


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: download.py HTTPS_WORKER_ORIGIN SNAPSHOT_KEY PUBLICATION_SHA256 NEW_PRIVATE_DIRECTORY"
        )
    print(json.dumps(download(*sys.argv[1:])))
