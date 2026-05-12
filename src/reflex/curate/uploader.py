"""Background R2 uploader for Free-tier contributed episodes.

Per `data-collection-free-tier.md`:
- Cron-style: runs at server start, every 24h, and on graceful shutdown
- Reads `~/.reflex/contribute/queue/`, applies episode-level quality filter,
  uploads accepted JSONL → R2 via signed PUT URLs from the contribution worker
- Bandwidth-respecting: throttle to REFLEX_CONTRIB_MAX_MBPS (default 10 MB/s)
- Set REFLEX_NO_CONTRIB_UPLOAD=1 to keep collecting locally without uploading

Live transport: 3-step round-trip (sign → put → complete) against the
deployed contribution-worker at https://reflex-contributions.fastcrest
.workers.dev. Override the endpoint with REFLEX_CONTRIB_ENDPOINT.

Defaults to `live=True` from the server lifespan; set REFLEX_CURATE_DRY_RUN=1
to keep collecting locally without uploading. In dry-run mode the uploader
scans + filters + logs what *would* upload but does not mutate the queue.

Retry policy: transient transport errors + 5xx codes retry up to MAX_RETRIES
times with exponential backoff (2s, 4s). Terminal codes (400 / 403 / 410 /
412) skip retry. ContributorRevoked stops the pass.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_DIR = "~/.reflex/contribute/queue"
DEFAULT_UPLOADED_DIR = "~/.reflex/contribute/uploaded"
DEFAULT_REJECTED_DIR = "~/.reflex/contribute/rejected"

# Episode-level quality thresholds (per spec).
MIN_EPISODE_STEPS = 30
MAX_ZERO_ACTION_FRAC = 0.90
MAX_DUP_IMAGE_HASH_FRAC = 0.50
MAX_ACTION_Z_SCORE = 100.0  # any per-channel z-score exceeding this → reject

# Bandwidth + scheduling defaults.
DEFAULT_MAX_MBPS = 10.0
DEFAULT_DAILY_INTERVAL_S = 86_400.0  # 24h between scheduled passes
KILL_SWITCH_ENV = "REFLEX_NO_CONTRIB_UPLOAD"
THROTTLE_ENV = "REFLEX_CONTRIB_MAX_MBPS"

# Live contribution-worker endpoint. Override via REFLEX_CONTRIB_ENDPOINT for
# testing or self-hosting. Endpoints (per infra/contribution-worker/README.md):
#   POST /v1/uploads/sign         — issue signed PUT URL for an upload
#   PUT  /v1/uploads/put/<id>     — receive raw bytes (worker proxies to R2)
#   POST /v1/uploads/complete     — record success + update stats
#   POST /v1/revoke/cascade       — mark contributor for purge
#   GET  /v1/contributors/<id>/stats  — return contribution totals
DEFAULT_WORKER_URL = "https://reflex-contributions.fastcrest.workers.dev"
WORKER_URL_ENV = "REFLEX_CONTRIB_ENDPOINT"
HTTP_TIMEOUT_S = 30.0

# Retry policy for transient errors. Don't retry on 403 (revoke), 410 (expired),
# 412 (precondition failed), or 400 (bad request) — those are terminal. DO retry
# on httpx connect/timeout errors, 502/503/504, and 500s.
MAX_RETRIES = 2  # Initial attempt + 2 retries = 3 total tries
BACKOFF_BASE_S = 2.0  # First retry waits 2s, second waits 4s
RETRY_STATUS_CODES = frozenset({500, 502, 503, 504})
TERMINAL_STATUS_CODES = frozenset({400, 403, 410, 412})


def _worker_url() -> str:
    return os.environ.get(WORKER_URL_ENV, DEFAULT_WORKER_URL).rstrip("/")


def _kill_switch_active() -> bool:
    return os.environ.get(KILL_SWITCH_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _max_mbps() -> float:
    raw = os.environ.get(THROTTLE_ENV)
    if not raw:
        return DEFAULT_MAX_MBPS
    try:
        v = float(raw)
        if v < 0:
            return DEFAULT_MAX_MBPS
        return v
    except ValueError:
        logger.warning("invalid %s=%r, falling back to default %g", THROTTLE_ENV, raw, DEFAULT_MAX_MBPS)
        return DEFAULT_MAX_MBPS


@dataclass
class EpisodeStats:
    """Computed once per episode at filter time."""

    episode_id: str
    step_count: int
    has_nan_or_inf_action: bool
    zero_action_frac: float
    dup_image_hash_frac: float
    max_action_z_score: float

    def reject_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.step_count < MIN_EPISODE_STEPS:
            reasons.append(f"length<{MIN_EPISODE_STEPS}({self.step_count})")
        if self.has_nan_or_inf_action:
            reasons.append("nan_or_inf_in_actions")
        if self.zero_action_frac > MAX_ZERO_ACTION_FRAC:
            reasons.append(f"all_zero_actions({self.zero_action_frac:.2f})")
        if self.dup_image_hash_frac > MAX_DUP_IMAGE_HASH_FRAC:
            reasons.append(f"dup_images({self.dup_image_hash_frac:.2f})")
        if self.max_action_z_score > MAX_ACTION_Z_SCORE:
            reasons.append(f"action_z>{MAX_ACTION_Z_SCORE}({self.max_action_z_score:.1f})")
        return reasons

    @property
    def accepted(self) -> bool:
        return not self.reject_reasons()


@dataclass
class UploadOutcome:
    """One pass of the uploader. Aggregates across all queue files."""

    files_scanned: int = 0
    episodes_inspected: int = 0
    episodes_accepted: int = 0
    episodes_rejected: int = 0
    bytes_uploaded: int = 0
    files_uploaded: int = 0
    files_kept_in_queue: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "episodes_inspected": self.episodes_inspected,
            "episodes_accepted": self.episodes_accepted,
            "episodes_rejected": self.episodes_rejected,
            "bytes_uploaded": self.bytes_uploaded,
            "files_uploaded": self.files_uploaded,
            "files_kept_in_queue": self.files_kept_in_queue,
            "rejection_reasons": dict(self.rejection_reasons),
            "errors": list(self.errors),
        }


def _iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("uploader.skipping_bad_line file=%s err=%s", path, exc)


def _stats_for_episode(rows: list[dict[str, Any]]) -> EpisodeStats:
    """Compute per-episode quality stats from a list of /act event rows."""
    step_count = len(rows)
    has_nan_or_inf = False
    zero_action_count = 0
    z_max = 0.0
    image_hashes: list[str] = []

    # Pass 1: collect action vectors (flattened) for stats + check NaN/Inf.
    flat_actions: list[float] = []
    for row in rows:
        chunk = row.get("action_chunk") or []
        for action in chunk:
            if not isinstance(action, list):
                continue
            non_zero = False
            for v in action:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fv) or math.isinf(fv):
                    has_nan_or_inf = True
                    continue
                flat_actions.append(fv)
                if fv != 0.0:
                    non_zero = True
            if not non_zero:
                zero_action_count += 1
        img_hash = row.get("image_b64")
        if isinstance(img_hash, str):
            image_hashes.append(img_hash)

    total_action_count = sum(
        1 for r in rows for a in (r.get("action_chunk") or []) if isinstance(a, list)
    )
    zero_frac = (zero_action_count / total_action_count) if total_action_count else 0.0

    if flat_actions:
        n = len(flat_actions)
        mean = sum(flat_actions) / n
        var = sum((v - mean) ** 2 for v in flat_actions) / n
        std = math.sqrt(var) if var > 0 else 0.0
        if std > 0:
            z_max = max(abs(v - mean) / std for v in flat_actions)

    if image_hashes:
        unique = len(set(image_hashes))
        dup_frac = 1.0 - (unique / len(image_hashes))
    else:
        dup_frac = 0.0

    return EpisodeStats(
        episode_id=rows[0].get("episode_id", "") if rows else "",
        step_count=step_count,
        has_nan_or_inf_action=has_nan_or_inf,
        zero_action_frac=zero_frac,
        dup_image_hash_frac=dup_frac,
        max_action_z_score=z_max,
    )


def filter_episodes(
    rows_by_episode: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, EpisodeStats]]:
    """Split rows into accepted/rejected episodes. Returns (accepted_rows_by_episode, stats_by_episode)."""
    accepted: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, EpisodeStats] = {}
    for episode_id, rows in rows_by_episode.items():
        s = _stats_for_episode(rows)
        stats[episode_id] = s
        if s.accepted:
            accepted[episode_id] = rows
    return accepted, stats


# ── R2 transport via the contribution worker ─────────────────────────────────


class UploadStub(Exception):
    """Raised when the live transport can't be used (e.g. httpx missing,
    or kill switch active in a context where uploader was asked to run live)."""


class WorkerError(Exception):
    """Raised when the contribution worker returns a non-2xx response that
    isn't a recoverable rate limit. Caller logs + skips the file for this pass."""

    def __init__(self, status: int, body: dict[str, Any] | None = None):
        self.status = status
        self.body = body or {}
        super().__init__(f"worker_error status={status} body={self.body}")


class RateLimited(Exception):
    """Raised when the worker returns 429 — caller skips the file this pass
    and tries again on the next interval."""


class ContributorRevoked(Exception):
    """Raised when the worker returns 403 contributor_revoked — the local
    consent receipt should be cleared. Surface to operator via WARN log."""


def _request_signed_url(
    *,
    contributor_id: str,
    tier: str,
    opted_in_at: str,
    file_name: str,
    byte_size: int,
    episode_count: int,
    privacy_mode: str,
) -> dict[str, Any]:
    """POST /v1/uploads/sign. Returns the worker's response dict
    {upload_id, r2_key, put_url, expires_at}. Raises RateLimited on 429,
    ContributorRevoked on 403 contributor_revoked, WorkerError on other
    non-2xx after retry exhaustion."""
    try:
        import httpx
    except ImportError as exc:
        raise UploadStub(f"httpx not available — install reflex-vla[serve]: {exc}") from exc

    url = f"{_worker_url()}/v1/uploads/sign"
    payload = {
        "contributor_id": contributor_id,
        "tier": tier,
        "opted_in_at": opted_in_at,
        "file_name": file_name,
        "byte_size": int(byte_size),
        "episode_count": int(episode_count),
        "privacy_mode": privacy_mode,
    }
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = httpx.post(url, json=payload, timeout=HTTP_TIMEOUT_S)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_S * (2 ** attempt)
                logger.warning("uploader.sign_retrying attempt=%d wait=%.1fs err=%s", attempt + 1, wait, exc)
                time.sleep(wait)
                continue
            raise WorkerError(0, {"error": "transport", "message": str(exc)}) from exc

        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            raise RateLimited(r.text)
        if r.status_code == 403:
            body = _safe_json(r)
            if body.get("error") == "contributor_revoked":
                raise ContributorRevoked(str(body))
            raise WorkerError(r.status_code, body)
        if r.status_code in TERMINAL_STATUS_CODES:
            raise WorkerError(r.status_code, _safe_json(r))
        if r.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
            wait = BACKOFF_BASE_S * (2 ** attempt)
            logger.warning(
                "uploader.sign_retrying attempt=%d wait=%.1fs status=%d",
                attempt + 1, wait, r.status_code,
            )
            time.sleep(wait)
            continue
        raise WorkerError(r.status_code, _safe_json(r))
    if last_exc is not None:
        raise WorkerError(0, {"error": "transport", "message": str(last_exc)}) from last_exc
    raise WorkerError(0, {"error": "exhausted_without_response"})


def _put_bytes(*, put_url: str, file_bytes: bytes, max_mbps: float) -> int:
    """PUT the raw bytes to the worker's /v1/uploads/put/<id> endpoint.
    Throttles to `max_mbps` by chunked sends with sleep gating between chunks.
    Returns bytes uploaded on success.

    Retry policy: transient transport errors + 5xx retry up to MAX_RETRIES
    with exponential backoff. The R2 PUT is idempotent for the same upload_id
    so re-PUT on retry overwrites any partial bytes from the previous attempt.
    """
    try:
        import httpx
    except ImportError as exc:
        raise UploadStub(f"httpx not available: {exc}") from exc

    headers = {"Content-Type": "application/x-jsonlines"}
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            if max_mbps <= 0:
                r = httpx.put(put_url, content=file_bytes, headers=headers, timeout=HTTP_TIMEOUT_S * 4)
            else:
                chunk_bytes = max(64 * 1024, int(max_mbps * 1024 * 1024 * 0.25))

                def _chunked() -> Any:
                    for i in range(0, len(file_bytes), chunk_bytes):
                        yield file_bytes[i : i + chunk_bytes]
                        time.sleep(0.25)

                r = httpx.put(
                    put_url, content=_chunked(), headers=headers,
                    timeout=HTTP_TIMEOUT_S * 4,
                )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_S * (2 ** attempt)
                logger.warning("uploader.put_retrying attempt=%d wait=%.1fs err=%s", attempt + 1, wait, exc)
                time.sleep(wait)
                continue
            raise WorkerError(0, {"error": "transport", "message": str(exc)}) from exc

        if r.status_code == 200:
            body = r.json()
            return int(body.get("bytes_received", len(file_bytes)))
        if r.status_code == 410:
            raise WorkerError(r.status_code, _safe_json(r))  # URL expired — terminal
        if r.status_code == 403:
            body = _safe_json(r)
            if body.get("error") == "contributor_revoked_between_sign_and_put":
                raise ContributorRevoked(str(body))
            raise WorkerError(r.status_code, body)
        if r.status_code in TERMINAL_STATUS_CODES:
            raise WorkerError(r.status_code, _safe_json(r))
        if r.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
            wait = BACKOFF_BASE_S * (2 ** attempt)
            logger.warning(
                "uploader.put_retrying attempt=%d wait=%.1fs status=%d",
                attempt + 1, wait, r.status_code,
            )
            time.sleep(wait)
            continue
        raise WorkerError(r.status_code, _safe_json(r))
    if last_exc is not None:
        raise WorkerError(0, {"error": "transport", "message": str(last_exc)}) from last_exc
    raise WorkerError(0, {"error": "exhausted_without_response"})


def _complete_upload(*, upload_id: str, episode_count: int) -> dict[str, Any]:
    """POST /v1/uploads/complete. Retries on transient errors + 5xx."""
    try:
        import httpx
    except ImportError as exc:
        raise UploadStub(f"httpx not available: {exc}") from exc

    url = f"{_worker_url()}/v1/uploads/complete"
    payload = {"upload_id": upload_id, "episode_count": int(episode_count)}
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = httpx.post(url, json=payload, timeout=HTTP_TIMEOUT_S)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_S * (2 ** attempt)
                logger.warning("uploader.complete_retrying attempt=%d wait=%.1fs err=%s", attempt + 1, wait, exc)
                time.sleep(wait)
                continue
            raise WorkerError(0, {"error": "transport", "message": str(exc)}) from exc

        if r.status_code == 200:
            return r.json()
        if r.status_code in TERMINAL_STATUS_CODES:
            raise WorkerError(r.status_code, _safe_json(r))
        if r.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
            wait = BACKOFF_BASE_S * (2 ** attempt)
            logger.warning(
                "uploader.complete_retrying attempt=%d wait=%.1fs status=%d",
                attempt + 1, wait, r.status_code,
            )
            time.sleep(wait)
            continue
        raise WorkerError(r.status_code, _safe_json(r))
    if last_exc is not None:
        raise WorkerError(0, {"error": "transport", "message": str(last_exc)}) from last_exc
    raise WorkerError(0, {"error": "exhausted_without_response"})


def _safe_json(response: Any) -> dict[str, Any]:
    try:
        return response.json()
    except Exception:  # noqa: BLE001
        return {"raw": response.text[:500]}


# ── The uploader ──────────────────────────────────────────────────────────────


class Uploader:
    """Cron-driven background uploader for the Curate queue.

    Pass `live=True` (default in server lifespan) to perform the real
    sign + put + complete round-trip against the contribution worker. Pass
    `live=False` (or set REFLEX_CURATE_DRY_RUN=1) to scan + filter + log
    without mutating the queue — useful for inspection or offline replay.
    """

    __slots__ = (
        "_queue_dir", "_uploaded_dir", "_rejected_dir",
        "_contributor_id", "_tier", "_opted_in_at", "_privacy_mode",
        "_live", "_max_mbps",
        "_thread", "_stop_event", "_interval_s",
        "_last_outcome", "_lock",
    )

    def __init__(
        self,
        *,
        contributor_id: str,
        tier: str = "free",
        opted_in_at: str = "",
        privacy_mode: str = "hash_only",
        queue_dir: str | Path = DEFAULT_QUEUE_DIR,
        uploaded_dir: str | Path = DEFAULT_UPLOADED_DIR,
        rejected_dir: str | Path = DEFAULT_REJECTED_DIR,
        live: bool = False,
        max_mbps: float | None = None,
        interval_s: float = DEFAULT_DAILY_INTERVAL_S,
    ):
        self._queue_dir = Path(queue_dir).expanduser()
        self._uploaded_dir = Path(uploaded_dir).expanduser()
        self._rejected_dir = Path(rejected_dir).expanduser()
        self._contributor_id = contributor_id
        self._tier = tier
        self._opted_in_at = opted_in_at
        self._privacy_mode = privacy_mode
        self._live = bool(live)
        self._max_mbps = float(max_mbps) if max_mbps is not None else _max_mbps()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._interval_s = float(interval_s)
        self._last_outcome: UploadOutcome | None = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_outcome(self) -> UploadOutcome | None:
        return self._last_outcome

    def run_once(self) -> UploadOutcome:
        """Single pass: scan queue, filter episodes, upload accepted, archive rejected.
        Idempotent — safe to call multiple times."""
        outcome = UploadOutcome()
        if _kill_switch_active():
            logger.info("uploader.kill_switch_active queue_dir=%s", self._queue_dir)
            return outcome

        if not self._queue_dir.exists():
            return outcome

        for jsonl_path in sorted(self._queue_dir.glob("*.jsonl")):
            outcome.files_scanned += 1
            try:
                rows_by_episode = self._group_by_episode(jsonl_path)
            except Exception as exc:  # noqa: BLE001
                msg = f"failed to read {jsonl_path}: {exc}"
                logger.error("uploader.read_failed %s", msg)
                outcome.errors.append(msg)
                continue

            accepted, stats = filter_episodes(rows_by_episode)
            outcome.episodes_inspected += len(stats)
            for s in stats.values():
                if s.accepted:
                    outcome.episodes_accepted += 1
                else:
                    outcome.episodes_rejected += 1
                    for reason in s.reject_reasons():
                        outcome.rejection_reasons[reason] += 1

            if not accepted:
                # All episodes rejected — archive the file to rejected/ for audit.
                self._archive_rejected(jsonl_path)
                outcome.files_kept_in_queue += 0
                continue

            # Compute per-episode quality scores + stamp onto rows before upload.
            try:
                from reflex.curate.quality import quality_from_jsonl_rows
                for episode_id, episode_rows in accepted.items():
                    qresult = quality_from_jsonl_rows(episode_rows)
                    quality_payload = qresult.to_dict()
                    for r in episode_rows:
                        md = r.setdefault("metadata", {}) or {}
                        md["quality_score"] = quality_payload["quality_score"]
                        md["quality_components"] = quality_payload["quality_components"]
                        md["quality_version"] = quality_payload["quality_version"]
                    logger.info(
                        "uploader.quality file=%s episode=%s score=%.3f components=%s",
                        jsonl_path.name, episode_id, qresult.quality_score,
                        {k: round(v, 3) for k, v in qresult.components.items()},
                    )
            except Exception as exc:  # noqa: BLE001 — quality scoring never blocks upload
                logger.warning("uploader.quality_scoring_failed file=%s: %s", jsonl_path.name, exc)

            # Failure mode classification — flags is_failure + primary_failure_mode.
            # Per research sidecar open question 1: runs BEFORE dedup so
            # dedup canonical-selection can avoid picking a failed canonical.
            try:
                self._stamp_failure_modes(jsonl_path, accepted)
            except Exception as exc:  # noqa: BLE001 — classifier never blocks upload
                logger.warning("uploader.failure_classifier_failed file=%s: %s", jsonl_path.name, exc)

            # Episode-level dedup (within-file; cross-session dedup is Phase 1.5).
            # Stamps cluster_id + is_canonical on each row's metadata. NEVER
            # deletes data per the spec — flagging only.
            try:
                self._stamp_dedup(jsonl_path, accepted)
            except Exception as exc:  # noqa: BLE001 — dedup never blocks upload
                logger.warning("uploader.dedup_failed file=%s: %s", jsonl_path.name, exc)

            # Auto-tag metadata (task type / subtype / difficulty / language /
            # terminal gripper / action complexity). Stamps tags on each row's
            # metadata. Powers Tier-2 filterable dataset queries.
            try:
                self._stamp_metadata(jsonl_path, accepted)
            except Exception as exc:  # noqa: BLE001 — metadata enrichment never blocks upload
                logger.warning("uploader.metadata_failed file=%s: %s", jsonl_path.name, exc)

            # Build accepted-rows-only payload + try upload.
            accepted_bytes = self._build_payload(accepted)
            episode_count = len(accepted)
            try:
                if self._live:
                    bytes_up = self._upload_one(
                        file_name=jsonl_path.name,
                        file_bytes=accepted_bytes,
                        episode_count=episode_count,
                    )
                    outcome.bytes_uploaded += bytes_up
                    outcome.files_uploaded += 1
                    self._archive_uploaded(jsonl_path)
                else:
                    logger.info(
                        "uploader.dry_run file=%s episodes=%d bytes=%d (live=False)",
                        jsonl_path.name, episode_count, len(accepted_bytes),
                    )
                    outcome.files_kept_in_queue += 1
            except RateLimited as exc:
                logger.warning(
                    "uploader.rate_limited file=%s — retrying next pass: %s",
                    jsonl_path.name, exc,
                )
                outcome.files_kept_in_queue += 1
            except ContributorRevoked as exc:
                logger.error(
                    "uploader.contributor_revoked file=%s — stopping uploader: %s",
                    jsonl_path.name, exc,
                )
                outcome.errors.append(f"contributor_revoked: {exc}")
                outcome.files_kept_in_queue += 1
                # The worker says we're revoked; further attempts are pointless
                # this pass. Operator should `reflex contribute --opt-out` to
                # clear the local receipt.
                break
            except UploadStub as exc:
                logger.info("uploader.stub %s file=%s", exc, jsonl_path.name)
                outcome.files_kept_in_queue += 1
            except Exception as exc:  # noqa: BLE001
                msg = f"upload failed for {jsonl_path.name}: {exc}"
                logger.error("uploader.upload_failed %s", msg)
                outcome.errors.append(msg)
                outcome.files_kept_in_queue += 1

        with self._lock:
            self._last_outcome = outcome
        return outcome

    def start(self) -> None:
        """Spawn the background loop. Idempotent."""
        if self.is_running:
            return
        self._stop_event.clear()
        # Run an immediate pass in the loop, then wait `interval_s` between passes.
        self._thread = threading.Thread(
            target=self._loop, name="curate-uploader", daemon=True,
        )
        self._thread.start()

    def stop(self, *, drain: bool = True, timeout_s: float = 30.0) -> None:
        """Stop the background loop. If drain=True, runs one final pass before exit."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout_s)
        self._thread = None
        if drain:
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("uploader.drain_failed %s", exc)

    def _loop(self) -> None:
        # Always run an immediate pass so a fresh start picks up backlog.
        try:
            self.run_once()
        except Exception as exc:  # noqa: BLE001
            logger.error("uploader.run_failed %s", exc)
        # Then wait `interval_s` between passes; the stop_event provides a
        # responsive cancellation path.
        while not self._stop_event.wait(self._interval_s):
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.error("uploader.run_failed %s", exc)

    def _stamp_dedup(
        self,
        jsonl_path: Path,
        accepted: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Run the dedup pipeline on accepted episodes and stamp cluster_id +
        is_canonical onto each row's metadata. Logs cluster summary."""
        from reflex.curate.dedup import (
            compute_average_hash,
            dedup_episodes,
        )

        # Build the dedup input dict from accepted rows.
        episodes_input: dict[str, dict[str, Any]] = {}
        for episode_id, rows in accepted.items():
            # Flatten action chunks → (T, action_dim) for trajectory similarity.
            flat_actions: list[list[float]] = []
            for r in rows:
                chunk = r.get("action_chunk") or []
                for action in chunk:
                    if isinstance(action, list):
                        flat_actions.append(action)
            if not flat_actions:
                continue
            import numpy as np
            actions_arr = np.asarray(flat_actions, dtype=np.float32)

            # Try to compute phash from first row's image_b64. Falls back to
            # trajectory-only when image_b64 is None / hash_only / unloadable.
            phash: str | None = None
            first_image = rows[0].get("image_b64")
            if isinstance(first_image, str) and len(first_image) > 64:
                # Heuristic: full-mode base64 image data is much longer than
                # a 64-char SHA-256 hex (hash_only mode). Try to decode + hash.
                try:
                    import base64
                    img_bytes = base64.b64decode(first_image, validate=False)
                    phash = compute_average_hash(img_bytes)
                except Exception:  # noqa: BLE001
                    phash = None

            md0 = rows[0].get("metadata", {}) or {}
            quality_score_val = float(md0.get("quality_score") or 0.0)

            episodes_input[episode_id] = {
                "phash": phash,
                "actions": actions_arr,
                "quality_score": quality_score_val,
                "step_count": int(actions_arr.shape[0]),
                "first_seen_at": str(rows[0].get("timestamp") or ""),
            }

        if not episodes_input:
            return

        results = dedup_episodes(episodes_input)
        for episode_id, info in results.items():
            payload = info.to_dict()
            for r in accepted.get(episode_id, []):
                md = r.setdefault("metadata", {}) or {}
                md.update(payload)

        # Log cluster summary (only emit non-singleton clusters to keep logs tight).
        non_singletons = [
            (info.cluster_id, info.cluster_size)
            for info in results.values()
            if info.cluster_size > 1 and info.is_canonical
        ]
        if non_singletons:
            logger.info(
                "uploader.dedup file=%s clusters=%s",
                jsonl_path.name, non_singletons,
            )

    def _stamp_failure_modes(
        self,
        jsonl_path: Path,
        accepted: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Classify each accepted episode by failure mode + stamp result onto rows."""
        from reflex.curate.failure_classifier import classify_from_jsonl_rows

        for episode_id, rows in accepted.items():
            try:
                result = classify_from_jsonl_rows(rows)
            except Exception as exc:  # noqa: BLE001
                logger.debug("uploader.failure_classifier_episode_failed episode=%s: %s", episode_id, exc)
                continue
            payload = result.to_dict()
            for r in rows:
                md = r.setdefault("metadata", {}) or {}
                md["failure_modes"] = payload["failure_modes"]
                md["is_failure"] = payload["is_failure"]
                md["primary_failure_mode"] = payload["primary_failure_mode"]
                md["classifier_version"] = payload["classifier_version"]
            if result.is_failure:
                logger.info(
                    "uploader.failure_modes file=%s episode=%s primary=%s n_modes=%d",
                    jsonl_path.name, episode_id,
                    result.primary_failure_mode, len(result.failure_modes),
                )

    def _stamp_metadata(
        self,
        jsonl_path: Path,
        accepted: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Run auto-tagging on accepted episodes; stamp tag bundle onto each
        row's metadata for Tier-2 filterable dataset construction."""
        from reflex.curate.metadata import enrich_from_jsonl_rows

        for episode_id, rows in accepted.items():
            try:
                result = enrich_from_jsonl_rows(rows)
            except Exception as exc:  # noqa: BLE001
                logger.debug("uploader.metadata_episode_failed episode=%s: %s", episode_id, exc)
                continue
            payload = result.to_dict()
            for r in rows:
                md = r.setdefault("metadata", {}) or {}
                md["tags"] = payload["tags"]
                md["taxonomy_version"] = payload["taxonomy_version"]
            tags_summary = {
                k: v.get("value") for k, v in payload["tags"].items()
                if k in ("task_type", "task_subtype", "instruction_language", "difficulty")
            }
            logger.info(
                "uploader.metadata file=%s episode=%s tags=%s",
                jsonl_path.name, episode_id, tags_summary,
            )

    def _upload_one(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        episode_count: int,
    ) -> int:
        """3-step worker round-trip: sign → put → complete. Returns bytes uploaded."""
        sign_resp = _request_signed_url(
            contributor_id=self._contributor_id,
            tier=self._tier,
            opted_in_at=self._opted_in_at,
            file_name=file_name,
            byte_size=len(file_bytes),
            episode_count=episode_count,
            privacy_mode=self._privacy_mode,
        )
        put_url = sign_resp["put_url"]
        upload_id = sign_resp["upload_id"]
        bytes_up = _put_bytes(
            put_url=put_url, file_bytes=file_bytes, max_mbps=self._max_mbps,
        )
        _complete_upload(upload_id=upload_id, episode_count=episode_count)
        logger.info(
            "uploader.uploaded file=%s upload_id=%s bytes=%d",
            file_name, upload_id, bytes_up,
        )
        return bytes_up

    def _group_by_episode(self, path: Path) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _iter_jsonl_rows(path):
            episode_id = row.get("episode_id") or "anon"
            out[episode_id].append(row)
        return dict(out)

    def _build_payload(self, rows_by_episode: dict[str, list[dict[str, Any]]]) -> bytes:
        """Serialize accepted rows back to a JSONL byte payload for upload."""
        lines: list[str] = []
        for episode_id, rows in rows_by_episode.items():
            for row in rows:
                lines.append(json.dumps(row))
        return ("\n".join(lines) + "\n").encode("utf-8")

    def _archive_uploaded(self, src: Path) -> None:
        self._uploaded_dir.mkdir(parents=True, exist_ok=True)
        dest = self._uploaded_dir / src.name
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self._uploaded_dir / f"{stem}.{ts}{suffix}"
        shutil.move(str(src), str(dest))

    def _archive_rejected(self, src: Path) -> None:
        self._rejected_dir.mkdir(parents=True, exist_ok=True)
        dest = self._rejected_dir / src.name
        if dest.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self._rejected_dir / f"{src.stem}.{ts}{src.suffix}"
        shutil.move(str(src), str(dest))


__all__ = [
    "DEFAULT_QUEUE_DIR",
    "DEFAULT_UPLOADED_DIR",
    "DEFAULT_REJECTED_DIR",
    "DEFAULT_DAILY_INTERVAL_S",
    "DEFAULT_MAX_MBPS",
    "DEFAULT_WORKER_URL",
    "WORKER_URL_ENV",
    "MIN_EPISODE_STEPS",
    "MAX_ZERO_ACTION_FRAC",
    "MAX_DUP_IMAGE_HASH_FRAC",
    "MAX_ACTION_Z_SCORE",
    "ContributorRevoked",
    "EpisodeStats",
    "RateLimited",
    "UploadOutcome",
    "UploadStub",
    "Uploader",
    "WorkerError",
    "filter_episodes",
]
