"""Pro-tier HF Hub client — per-customer private repo for trained students.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #6:
- Per-customer private repo: `{org}/{customer_slug}-{workspace_id}`
- Customer's own HF token (NOT Tether's — regulated-industry compliance
  + no shared-credential liability)
- HF-down-mid-swap = FAIL-LOUD, abort swap, keep current model running.
  NEVER silent fallback.

Phase 1 ships the substrate: push (post-eval-pass) + pull (pre-swap) +
clear retry/fail semantics. Phase 1.5 wires this into the actual
distill_loop + rollback_handler in production.

The HF token is read from the `HF_TOKEN` env var per HF convention.
NEVER persisted to disk by Tether; never logged.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Default org for Pro customer student repos. Customers can override via
# `--hf-repo <org>/<repo-name>` to use their own org.
DEFAULT_PRO_ORG = "tether-students"

# Retry policy for transient HF errors (rate-limit, network blip).
# Hard-coded conservative values; not customer-tunable in Phase 1.
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_S = 2.0


class HfHubError(Exception):
    """Base for HF Hub failures. Caller maps to a fail-loud outcome —
    never silently swallows."""


class HfHubMissingToken(HfHubError):
    """HF_TOKEN env var unset OR explicit token=None at construction."""


class HfHubDown(HfHubError):
    """HF endpoint unreachable after retries. Caller (rollback_handler)
    aborts the swap; current model stays running."""


class HfHubAuthFailure(HfHubError):
    """401/403 — customer token invalid or doesn't have write access to
    the target repo. Operator must rotate the token."""


@dataclass(frozen=True)
class HfRepoSpec:
    """Frozen target-repo spec. Constructed at startup from customer's
    license + workspace_id + optional --hf-repo override."""

    org: str
    name: str

    def __post_init__(self) -> None:
        if not self.org or not self.name:
            raise ValueError(
                f"HfRepoSpec org and name must both be non-empty; got "
                f"org={self.org!r}, name={self.name!r}"
            )
        for ch in "/ ":
            if ch in self.org or ch in self.name:
                raise ValueError(
                    f"HfRepoSpec parts may not contain {ch!r}; got "
                    f"org={self.org!r}, name={self.name!r}"
                )

    @property
    def full_name(self) -> str:
        return f"{self.org}/{self.name}"

    @classmethod
    def for_customer(
        cls,
        *,
        customer_slug: str,
        workspace_id: str,
        org: str = DEFAULT_PRO_ORG,
    ) -> "HfRepoSpec":
        """Build from license + workspace_id per ADR convention."""
        if not customer_slug or not workspace_id:
            raise ValueError(
                "customer_slug and workspace_id must both be non-empty"
            )
        # Sanitize: replace any forbidden chars with hyphens
        safe = lambda s: s.replace("/", "-").replace(" ", "-").replace("_", "-")
        return cls(org=org, name=f"{safe(customer_slug)}-{safe(workspace_id)}")


@dataclass(frozen=True)
class HfPushOutcome:
    """Frozen output of HfHubClient.push()."""

    succeeded: bool
    repo: str
    revision: str | None  # commit hash on success
    elapsed_s: float
    error: str | None = None


@dataclass(frozen=True)
class HfPullOutcome:
    """Frozen output of HfHubClient.pull()."""

    succeeded: bool
    repo: str
    local_path: str | None  # path to downloaded snapshot on success
    elapsed_s: float
    error: str | None = None


class HfHubClient:
    """HF Hub client for the Pro-tier model storage.

    Lazy-imports `huggingface_hub` so tests + non-Pro deploys don't pay
    the import cost. Construction-time token validation: missing token =
    HfHubMissingToken at __init__, never silent.

    Lifecycle:
        client = HfHubClient(repo=HfRepoSpec.for_customer(...), token=...)
        outcome = client.push(local_dir="/tmp/student", commit_message="...")
        # Day 5+ wiring: gate the rollback_handler on outcome.succeeded
    """

    __slots__ = (
        "_repo", "_token", "_retry_attempts", "_retry_backoff_s",
        "_dry_run",
    )

    def __init__(
        self,
        *,
        repo: HfRepoSpec,
        token: str | None = None,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
        dry_run: bool = False,
    ):
        if retry_attempts < 1:
            raise ValueError(
                f"retry_attempts must be >= 1, got {retry_attempts}"
            )
        if retry_backoff_s < 0:
            raise ValueError(
                f"retry_backoff_s must be >= 0, got {retry_backoff_s}"
            )
        self._repo = repo
        # Token resolution: explicit > env. Empty string treated as missing.
        resolved_token = (token or os.environ.get("HF_TOKEN", "")).strip()
        if not resolved_token and not dry_run:
            raise HfHubMissingToken(
                "HF token unset. Set HF_TOKEN env var OR pass token= "
                "explicitly. Tether never falls back to a shared token."
            )
        self._token = resolved_token
        self._retry_attempts = int(retry_attempts)
        self._retry_backoff_s = float(retry_backoff_s)
        self._dry_run = bool(dry_run)

    @property
    def repo(self) -> HfRepoSpec:
        return self._repo

    @property
    def is_dry_run(self) -> bool:
        return self._dry_run

    def push(
        self,
        *,
        local_dir: str | Path,
        commit_message: str,
        api_caller: Callable[..., Any] | None = None,
    ) -> HfPushOutcome:
        """Push the local model dir to the customer's private repo.

        `api_caller` is a testable injection point — when provided, calls
        this instead of the real huggingface_hub.upload_folder. Production
        callers leave it None.

        Retries on transient errors (HfHubDown); 401/403 raises
        HfHubAuthFailure immediately (no retry — token is wrong, not
        rate-limited).
        """
        import time
        from pathlib import Path as _Path

        local_path = _Path(local_dir)
        if not local_path.exists():
            return HfPushOutcome(
                succeeded=False, repo=self._repo.full_name,
                revision=None, elapsed_s=0.0,
                error=f"local_dir does not exist: {local_path}",
            )

        if self._dry_run:
            return HfPushOutcome(
                succeeded=True, repo=self._repo.full_name,
                revision="dry-run-revision", elapsed_s=0.0, error=None,
            )

        t0 = time.perf_counter()
        last_error: str | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                if api_caller is None:
                    from huggingface_hub import upload_folder
                    api_caller = upload_folder
                commit_info = api_caller(
                    folder_path=str(local_path),
                    repo_id=self._repo.full_name,
                    repo_type="model",
                    token=self._token,
                    commit_message=commit_message,
                )
                # huggingface_hub returns a CommitInfo object with oid attribute
                revision = getattr(commit_info, "oid", None) or str(commit_info)
                elapsed = time.perf_counter() - t0
                logger.info(
                    "hf_hub.push succeeded repo=%s revision=%s elapsed_ms=%.1f",
                    self._repo.full_name, revision, elapsed * 1000,
                )
                return HfPushOutcome(
                    succeeded=True, repo=self._repo.full_name,
                    revision=str(revision), elapsed_s=elapsed,
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                last_error = error_msg
                # Auth errors: don't retry; token is wrong
                if _looks_like_auth_failure(exc):
                    elapsed = time.perf_counter() - t0
                    logger.error(
                        "hf_hub.push auth_failure repo=%s exc=%s",
                        self._repo.full_name, error_msg,
                    )
                    raise HfHubAuthFailure(
                        f"HF Hub auth failed for {self._repo.full_name}: "
                        f"{error_msg}. Rotate HF_TOKEN."
                    ) from exc
                if attempt < self._retry_attempts:
                    logger.warning(
                        "hf_hub.push attempt=%d/%d failed exc=%s — retrying in %.1fs",
                        attempt, self._retry_attempts, error_msg,
                        self._retry_backoff_s,
                    )
                    time.sleep(self._retry_backoff_s)
                else:
                    logger.error(
                        "hf_hub.push exhausted_retries repo=%s last_error=%s",
                        self._repo.full_name, error_msg,
                    )

        elapsed = time.perf_counter() - t0
        return HfPushOutcome(
            succeeded=False, repo=self._repo.full_name,
            revision=None, elapsed_s=elapsed, error=last_error,
        )

    def pull(
        self,
        *,
        local_dir: str | Path,
        revision: str | None = None,
        api_caller: Callable[..., Any] | None = None,
    ) -> HfPullOutcome:
        """Pull the latest (or pinned) model snapshot from the customer's
        repo to local_dir. Used by the rollback handler pre-swap.

        `revision` pins to a commit hash; None = latest.

        Same retry + auth semantics as push().
        """
        import time
        from pathlib import Path as _Path

        local_path = _Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        if self._dry_run:
            return HfPullOutcome(
                succeeded=True, repo=self._repo.full_name,
                local_path=str(local_path), elapsed_s=0.0, error=None,
            )

        t0 = time.perf_counter()
        last_error: str | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                if api_caller is None:
                    from huggingface_hub import snapshot_download
                    api_caller = snapshot_download
                downloaded_path = api_caller(
                    repo_id=self._repo.full_name,
                    repo_type="model",
                    token=self._token,
                    revision=revision,
                    local_dir=str(local_path),
                )
                elapsed = time.perf_counter() - t0
                logger.info(
                    "hf_hub.pull succeeded repo=%s revision=%s elapsed_ms=%.1f",
                    self._repo.full_name, revision or "latest", elapsed * 1000,
                )
                return HfPullOutcome(
                    succeeded=True, repo=self._repo.full_name,
                    local_path=str(downloaded_path), elapsed_s=elapsed,
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                last_error = error_msg
                if _looks_like_auth_failure(exc):
                    elapsed = time.perf_counter() - t0
                    raise HfHubAuthFailure(
                        f"HF Hub auth failed for {self._repo.full_name}: "
                        f"{error_msg}. Rotate HF_TOKEN."
                    ) from exc
                if attempt < self._retry_attempts:
                    logger.warning(
                        "hf_hub.pull attempt=%d/%d failed — retrying in %.1fs",
                        attempt, self._retry_attempts, self._retry_backoff_s,
                    )
                    time.sleep(self._retry_backoff_s)

        elapsed = time.perf_counter() - t0
        return HfPullOutcome(
            succeeded=False, repo=self._repo.full_name,
            local_path=None, elapsed_s=elapsed, error=last_error,
        )


def _looks_like_auth_failure(exc: Exception) -> bool:
    """Heuristic — `huggingface_hub` raises various exception types
    depending on version. We pattern-match on message content + exception
    name for the auth-failure cases (don't retry these)."""
    name = type(exc).__name__
    msg = str(exc).lower()
    return (
        "auth" in name.lower()
        or "401" in msg
        or "403" in msg
        or "unauthorized" in msg
        or "forbidden" in msg
        or "token" in msg
    )


__all__ = [
    "DEFAULT_PRO_ORG",
    "HfHubAuthFailure",
    "HfHubClient",
    "HfHubDown",
    "HfHubError",
    "HfHubMissingToken",
    "HfPullOutcome",
    "HfPushOutcome",
    "HfRepoSpec",
]
