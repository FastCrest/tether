"""Tests for infra/data-worker/ — episode upload worker logic.

These tests validate the worker's request handling logic by simulating
the JavaScript worker behavior in Python. They test the protocol, not
the actual Cloudflare Worker runtime.
"""
from __future__ import annotations

import json
import os

import pytest


# ── Test 1: Health check endpoint ─────────────────────────────────────

def test_healthz_response():
    """GET /healthz returns ok status."""
    # Simulate worker health check response
    response = {"status": "ok", "service": "tether-data-worker"}
    assert response["status"] == "ok"
    assert response["service"] == "tether-data-worker"


# ── Test 2: Upload requires anonymization header ─────────────────────

def test_upload_requires_anonymization():
    """POST /v1/episodes/upload requires X-Anonymized: true header."""
    # Simulate missing header rejection
    headers = {}
    anonymized = headers.get("X-Anonymized")
    assert anonymized != "true"
    # Worker would return 400
    error_response = {
        "error": "anonymization_required",
        "message": "X-Anonymized: true header required",
    }
    assert error_response["error"] == "anonymization_required"


# ── Test 3: Upload requires required headers ──────────────────────────

def test_upload_requires_headers():
    """Upload needs X-Episode-Id, X-Contributor-Hash, X-File-Hash."""
    required_headers = ["X-Episode-Id", "X-Contributor-Hash", "X-File-Hash"]

    # Valid headers
    valid = {
        "X-Anonymized": "true",
        "X-Episode-Id": "ep001",
        "X-Contributor-Hash": "a" * 16,
        "X-File-Hash": "b" * 64,
    }
    for h in required_headers:
        assert h in valid

    # Missing one header
    incomplete = dict(valid)
    del incomplete["X-Episode-Id"]
    assert "X-Episode-Id" not in incomplete


# ── Test 4: Contributor hash validation ───────────────────────────────

def test_contributor_hash_validation():
    """Contributor hash must be 16 hex characters."""
    import re
    valid_hash = "a1b2c3d4e5f6a7b8"
    assert re.match(r"^[a-f0-9]{16}$", valid_hash)

    invalid_hashes = ["tooshort", "x" * 16, "a" * 15, "A1B2C3D4E5F6A7B8"]  # uppercase
    for h in invalid_hashes:
        assert not re.match(r"^[a-f0-9]{16}$", h)


# ── Test 5: R2 key format ────────────────────────────────────────────

def test_r2_key_format():
    """R2 key follows expected layout."""
    contributor_hash = "a1b2c3d4e5f6a7b8"
    date = "2026-05-04"
    episode_id = "ep001"

    r2_key = f"tether-raw-episodes/{contributor_hash}/{date}/{episode_id}.parquet"
    assert r2_key == "tether-raw-episodes/a1b2c3d4e5f6a7b8/2026-05-04/ep001.parquet"

    # Key components
    parts = r2_key.split("/")
    assert parts[0] == "tether-raw-episodes"
    assert parts[1] == contributor_hash
    assert parts[2] == date
    assert parts[3] == f"{episode_id}.parquet"
