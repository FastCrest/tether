"""Tests for src/tether/pro/hf_hub.py — Phase 1 self-distilling-serve Day 8.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #6:
per-customer private repo, customer's own HF token, fail-loud on
HF-down (NEVER silent fallback).
"""
from __future__ import annotations

import pytest

from tether.pro.hf_hub import (
    DEFAULT_PRO_ORG,
    HfHubAuthFailure,
    HfHubClient,
    HfHubMissingToken,
    HfPullOutcome,
    HfPushOutcome,
    HfRepoSpec,
)


# ---------------------------------------------------------------------------
# HfRepoSpec
# ---------------------------------------------------------------------------


def test_repo_spec_full_name_format():
    repo = HfRepoSpec(org="tether-students", name="acme-prod-1")
    assert repo.full_name == "tether-students/acme-prod-1"


def test_repo_spec_rejects_empty_parts():
    with pytest.raises(ValueError):
        HfRepoSpec(org="", name="acme")
    with pytest.raises(ValueError):
        HfRepoSpec(org="org", name="")


def test_repo_spec_rejects_slash_in_parts():
    with pytest.raises(ValueError):
        HfRepoSpec(org="tether/students", name="acme")
    with pytest.raises(ValueError):
        HfRepoSpec(org="tether-students", name="acme/prod")


def test_repo_spec_rejects_space_in_parts():
    with pytest.raises(ValueError):
        HfRepoSpec(org="tether students", name="acme")


def test_repo_spec_for_customer_default_org():
    repo = HfRepoSpec.for_customer(
        customer_slug="acme", workspace_id="prod-1",
    )
    assert repo.org == DEFAULT_PRO_ORG
    assert repo.name == "acme-prod-1"


def test_repo_spec_for_customer_sanitizes_underscores():
    repo = HfRepoSpec.for_customer(
        customer_slug="acme_corp", workspace_id="ws_42",
    )
    # underscores → hyphens
    assert "_" not in repo.name
    assert repo.name == "acme-corp-ws-42"


def test_repo_spec_for_customer_rejects_empty_args():
    with pytest.raises(ValueError):
        HfRepoSpec.for_customer(customer_slug="", workspace_id="prod")
    with pytest.raises(ValueError):
        HfRepoSpec.for_customer(customer_slug="acme", workspace_id="")


# ---------------------------------------------------------------------------
# HfHubClient construction
# ---------------------------------------------------------------------------


def test_client_raises_missing_token_when_none(monkeypatch):
    """No HF_TOKEN env + no explicit token → HfHubMissingToken at __init__."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(HfHubMissingToken):
        HfHubClient(repo=HfRepoSpec(org="acme", name="repo"))


def test_client_accepts_explicit_token():
    client = HfHubClient(
        repo=HfRepoSpec(org="acme", name="repo"),
        token="hf_explicit_token",
    )
    assert client.repo.full_name == "acme/repo"


def test_client_uses_env_token_when_no_explicit(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")
    client = HfHubClient(repo=HfRepoSpec(org="acme", name="repo"))
    # Token resolved from env — no exception
    assert client.repo.full_name == "acme/repo"


def test_client_dry_run_skips_token_requirement(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # dry_run=True bypasses the missing-token error
    client = HfHubClient(
        repo=HfRepoSpec(org="acme", name="repo"), dry_run=True,
    )
    assert client.is_dry_run


def test_client_rejects_zero_retry_attempts():
    with pytest.raises(ValueError, match="retry_attempts"):
        HfHubClient(
            repo=HfRepoSpec(org="a", name="b"), token="t", retry_attempts=0,
        )


def test_client_rejects_negative_retry_backoff():
    with pytest.raises(ValueError, match="retry_backoff_s"):
        HfHubClient(
            repo=HfRepoSpec(org="a", name="b"), token="t",
            retry_backoff_s=-1.0,
        )


# ---------------------------------------------------------------------------
# push() — happy path + retry + fail
# ---------------------------------------------------------------------------


def _stub_upload_folder(commit_oid: str = "abc123def456"):
    """Returns a stub api_caller that succeeds + returns a CommitInfo-like."""
    class _CommitInfo:
        oid = commit_oid

    def _caller(**kwargs):
        return _CommitInfo()

    _caller.calls = []  # type: ignore[attr-defined]
    return _caller


def test_push_succeeds_on_clean_call(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"fake")
    client = HfHubClient(repo=HfRepoSpec(org="a", name="b"), token="t")
    api_call = _stub_upload_folder()
    outcome = client.push(
        local_dir=tmp_path, commit_message="initial", api_caller=api_call,
    )
    assert outcome.succeeded
    assert outcome.repo == "a/b"
    assert outcome.revision == "abc123def456"
    assert outcome.error is None


def test_push_returns_failure_on_missing_local_dir(tmp_path):
    client = HfHubClient(repo=HfRepoSpec(org="a", name="b"), token="t")
    outcome = client.push(
        local_dir=tmp_path / "nonexistent",
        commit_message="x",
        api_caller=lambda **k: None,
    )
    assert not outcome.succeeded
    assert "does not exist" in outcome.error


def test_push_dry_run_skips_actual_upload(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"fake")
    client = HfHubClient(
        repo=HfRepoSpec(org="a", name="b"), dry_run=True,
    )
    outcome = client.push(local_dir=tmp_path, commit_message="x")
    assert outcome.succeeded
    assert outcome.revision == "dry-run-revision"


def test_push_retries_on_transient_error(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"fake")
    client = HfHubClient(
        repo=HfRepoSpec(org="a", name="b"), token="t",
        retry_attempts=3, retry_backoff_s=0,  # no backoff for fast test
    )
    call_count = {"n": 0}

    def flaky_caller(**kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("transient network blip")
        class _CI:
            oid = "later_success"
        return _CI()

    outcome = client.push(
        local_dir=tmp_path, commit_message="x", api_caller=flaky_caller,
    )
    assert outcome.succeeded
    assert call_count["n"] == 3


def test_push_fails_after_exhausting_retries(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"fake")
    client = HfHubClient(
        repo=HfRepoSpec(org="a", name="b"), token="t",
        retry_attempts=2, retry_backoff_s=0,
    )

    def always_fails(**kwargs):
        raise ConnectionError("server offline")

    outcome = client.push(
        local_dir=tmp_path, commit_message="x", api_caller=always_fails,
    )
    assert not outcome.succeeded
    assert "server offline" in outcome.error


def test_push_raises_auth_failure_immediately_no_retry(tmp_path):
    """401/403 errors don't retry — token is wrong, not rate-limited."""
    (tmp_path / "model.onnx").write_bytes(b"fake")
    client = HfHubClient(
        repo=HfRepoSpec(org="a", name="b"), token="t",
        retry_attempts=5, retry_backoff_s=0,
    )
    call_count = {"n": 0}

    def auth_fail(**kwargs):
        call_count["n"] += 1
        raise PermissionError("403 Forbidden — invalid token")

    with pytest.raises(HfHubAuthFailure):
        client.push(
            local_dir=tmp_path, commit_message="x", api_caller=auth_fail,
        )
    assert call_count["n"] == 1  # no retries


# ---------------------------------------------------------------------------
# pull() — happy path + retry + fail
# ---------------------------------------------------------------------------


def test_pull_succeeds_on_clean_call(tmp_path):
    client = HfHubClient(repo=HfRepoSpec(org="a", name="b"), token="t")

    def stub_download(**kwargs):
        return str(tmp_path / "downloaded")

    outcome = client.pull(local_dir=tmp_path, api_caller=stub_download)
    assert outcome.succeeded
    assert outcome.local_path is not None


def test_pull_dry_run_skips_actual_download(tmp_path):
    client = HfHubClient(
        repo=HfRepoSpec(org="a", name="b"), dry_run=True,
    )
    outcome = client.pull(local_dir=tmp_path)
    assert outcome.succeeded


def test_pull_retries_on_transient_error(tmp_path):
    client = HfHubClient(
        repo=HfRepoSpec(org="a", name="b"), token="t",
        retry_attempts=3, retry_backoff_s=0,
    )
    call_count = {"n": 0}

    def flaky(**kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise OSError("network down")
        return str(tmp_path / "succeeded")

    outcome = client.pull(local_dir=tmp_path, api_caller=flaky)
    assert outcome.succeeded
    assert call_count["n"] == 3


def test_pull_fails_after_exhausting_retries(tmp_path):
    client = HfHubClient(
        repo=HfRepoSpec(org="a", name="b"), token="t",
        retry_attempts=2, retry_backoff_s=0,
    )

    def always_fails(**kwargs):
        raise OSError("offline")

    outcome = client.pull(
        local_dir=tmp_path, api_caller=always_fails,
    )
    assert not outcome.succeeded


def test_pull_raises_auth_failure_no_retry(tmp_path):
    client = HfHubClient(
        repo=HfRepoSpec(org="a", name="b"), token="t",
        retry_attempts=5, retry_backoff_s=0,
    )
    call_count = {"n": 0}

    def auth_fail(**kwargs):
        call_count["n"] += 1
        raise PermissionError("401 Unauthorized")

    with pytest.raises(HfHubAuthFailure):
        client.pull(local_dir=tmp_path, api_caller=auth_fail)
    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Outcome shapes
# ---------------------------------------------------------------------------


def test_push_outcome_is_frozen():
    o = HfPushOutcome(
        succeeded=True, repo="a/b", revision="abc",
        elapsed_s=0.1, error=None,
    )
    with pytest.raises(AttributeError):
        o.succeeded = False  # type: ignore[misc]


def test_pull_outcome_is_frozen():
    o = HfPullOutcome(
        succeeded=True, repo="a/b", local_path="/tmp/out",
        elapsed_s=0.1, error=None,
    )
    with pytest.raises(AttributeError):
        o.succeeded = False  # type: ignore[misc]
