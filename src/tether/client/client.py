"""TetherClient — sync + async client for `tether serve`.

Talks to the /act, /health, /config endpoints. Handles:
- Bearer auth via X-Tether-Key header
- Image encoding (numpy / PIL / file path / raw bytes / pre-encoded base64 str)
- Episode tracking (auto episode_id, context manager wraps a session)
- Retry with exponential backoff on 503 (respecting Retry-After header)
- Surfaces guard_violations + guard_clamped + injected_latency_ms response fields
- Distinguishes 401 / 422 / 503 (degraded vs warming) via typed exceptions

Stays a thin wrapper — no business logic on the client side. Customers can
extend by subclassing, but the default API matches the server's contract 1:1.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


# ---- Exceptions -----------------------------------------------------------

class TetherClientError(Exception):
    """Base for all SDK errors."""

    def __init__(self, message: str, *, status_code: int | None = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class TetherAuthError(TetherClientError):
    """401 — missing/wrong X-Tether-Key. Not retryable."""


class TetherValidationError(TetherClientError):
    """422 — bad request body. Not retryable."""


class TetherServerNotReadyError(TetherClientError):
    """503 with state in {initializing, loading, warming, warmup_failed} — retry-able."""

    def __init__(self, message: str, *, state: str = "", **kw):
        super().__init__(message, **kw)
        self.state = state


class TetherServerDegradedError(TetherClientError):
    """503 with state == degraded — circuit broken; restart needed before retries succeed."""

    def __init__(self, message: str, *, retry_after_s: float = 60.0, **kw):
        super().__init__(message, **kw)
        self.retry_after_s = retry_after_s


# ---- Image encoding -------------------------------------------------------

def encode_image(image: Any, jpeg_quality: int = 85) -> str:
    """Encode an image to base64-JPEG. Supports many input forms.

    Accepted shapes:
      - str              : if it parses as base64, returned as-is; else treated as file path
      - bytes            : if it looks like a JPEG/PNG file, b64-encoded; else passed through
      - pathlib.Path     : opened, encoded as JPEG
      - numpy.ndarray    : (H, W, 3) RGB uint8 → encoded as JPEG via PIL
      - PIL.Image.Image  : encoded as JPEG
      - None / ""        : returned as empty string (server treats as missing image)

    Raises TetherClientError on inputs that can't be encoded.
    """
    if image is None:
        return ""
    if isinstance(image, str) and image == "":
        return ""
    if isinstance(image, str):
        if image.startswith(("data:", "/9j/", "iVBOR")) or len(image) > 1024 and "/" not in image and "\\" not in image:
            return image  # already base64
        p = Path(image)
        if p.exists() and p.is_file():
            return base64.b64encode(p.read_bytes()).decode("ascii")
        return image  # caller knows best
    if isinstance(image, Path):
        if not image.exists():
            raise TetherClientError(f"image path does not exist: {image}")
        return base64.b64encode(image.read_bytes()).decode("ascii")
    if isinstance(image, bytes):
        if len(image) >= 4 and image[:3] == b"\xff\xd8\xff":
            return base64.b64encode(image).decode("ascii")
        if len(image) >= 8 and image[:8] == b"\x89PNG\r\n\x1a\n":
            return base64.b64encode(image).decode("ascii")
        return base64.b64encode(image).decode("ascii")
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise TetherClientError(
            "Pillow required to encode numpy arrays / PIL images; install tether[serve] or pip install Pillow"
        )
    if isinstance(image, PILImage.Image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    try:
        import numpy as np
    except ImportError:
        raise TetherClientError("numpy required for ndarray image encoding")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        buf = io.BytesIO()
        PILImage.fromarray(image).save(buf, format="JPEG", quality=jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    raise TetherClientError(f"unsupported image type: {type(image).__name__}")


# ---- Retry helpers --------------------------------------------------------

def _parse_retry_after(value: str | None) -> float | None:
    """Parse Retry-After header. Returns seconds or None."""
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify_response(resp: httpx.Response) -> TetherClientError | None:
    """Map HTTP status + body to a typed exception. Returns None for 2xx."""
    if 200 <= resp.status_code < 300:
        return None
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    msg = f"HTTP {resp.status_code}"
    if isinstance(body, dict):
        msg = f"{msg} — {body.get('error') or body.get('detail') or body}"
    if resp.status_code == 401:
        return TetherAuthError(msg, status_code=401, body=body)
    if resp.status_code == 422:
        return TetherValidationError(msg, status_code=422, body=body)
    if resp.status_code == 503:
        state = ""
        if isinstance(body, dict):
            state = body.get("state", "") or body.get("error", "")
        if state == "degraded" or (isinstance(body, dict) and body.get("error") == "server-degraded"):
            ra = _parse_retry_after(resp.headers.get("Retry-After")) or 60.0
            return TetherServerDegradedError(
                msg, status_code=503, body=body, retry_after_s=ra,
            )
        return TetherServerNotReadyError(msg, status_code=503, body=body, state=state)
    return TetherClientError(msg, status_code=resp.status_code, body=body)


# ---- Sync client ----------------------------------------------------------

class TetherClient:
    """Sync client for tether serve. Thread-safe via an internal httpx.Client lock."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        initial_backoff_s: float = 0.1,
        max_backoff_s: float = 1.6,
        retry_on_degraded: bool = False,
    ):
        """Construct a client.

        retry_on_degraded: when True, also retry the 503-degraded case using
        the server's Retry-After header. Default False because degraded is a
        circuit-breaker state — the server needs operator intervention. Set
        True only if you have an external orchestrator restarting servers.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s
        self.retry_on_degraded = retry_on_degraded
        headers: dict[str, str] = {}
        if api_key:
            headers["X-Tether-Key"] = api_key
        self._http = httpx.Client(headers=headers, timeout=timeout_s)
        self._closed = False

    def __enter__(self) -> TetherClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._http.close()
            self._closed = True

    # ---- Internals -------------------------------------------------------

    def _request_with_retry(self, method: str, path: str, **kw) -> httpx.Response:
        last_exc: TetherClientError | None = None
        attempt = 0
        backoff = self.initial_backoff_s
        while True:
            try:
                resp = self._http.request(method, f"{self.base_url}{path}", **kw)
            except (httpx.NetworkError, httpx.TimeoutException) as e:
                if attempt >= self.max_retries:
                    raise TetherClientError(f"network error after {attempt} retries: {e}") from e
                logger.warning("network error (attempt %d): %s; backing off %.2fs", attempt, e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
                attempt += 1
                continue
            err = _classify_response(resp)
            if err is None:
                return resp
            if isinstance(err, (TetherAuthError, TetherValidationError)):
                raise err
            if isinstance(err, TetherServerDegradedError) and not self.retry_on_degraded:
                raise err
            if attempt >= self.max_retries:
                raise err
            wait = _parse_retry_after(resp.headers.get("Retry-After"))
            if wait is None:
                wait = backoff
                backoff = min(backoff * 2, self.max_backoff_s)
            logger.warning(
                "%s %s returned %d (attempt %d); retrying after %.2fs",
                method, path, resp.status_code, attempt, wait,
            )
            time.sleep(wait)
            attempt += 1
            last_exc = err

    # ---- Public surface --------------------------------------------------

    def health(self) -> dict[str, Any]:
        """GET /health. Returns the JSON body. Does NOT raise on 503 — health
        is informational; the body's `state` field tells you what's up."""
        resp = self._http.get(f"{self.base_url}/health")
        try:
            return resp.json()
        except Exception:
            return {"status": "unknown", "state": "unknown", "raw": resp.text}

    def config(self) -> dict[str, Any]:
        """GET /config. Returns the server's config dict. Auth required if api_key set."""
        resp = self._request_with_retry("GET", "/config")
        return resp.json()

    def act(
        self,
        image: Any = None,
        instruction: str = "",
        state: list[float] | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """POST /act. Returns the parsed response dict.

        The result includes whatever the server returned plus normalizations:
        - `actions`: list of action chunks (always present on success)
        - `latency_ms`: real inference latency
        - `injected_latency_ms`: synthetic delay if `--inject-latency-ms` was set
        - `guard_violations`: list of strings if B.6 ActionGuard clamped anything
        - `guard_clamped`: bool flag mirroring guard_violations
        - `inference_mode`: "monolithic" / "decomposed" / etc.
        """
        body = {
            "image": encode_image(image),
            "instruction": instruction,
            "state": state,
            "episode_id": episode_id,
        }
        resp = self._request_with_retry("POST", "/act", json=body)
        return resp.json()

    @contextlib.contextmanager
    def episode(self, episode_id: str | None = None) -> Iterator[_EpisodeProxy]:
        """Context manager for a single episode. Auto-generates episode_id if
        not provided and propagates it on every .act() call inside the block.
        On exit, no /reset call is needed — the server's RTC adapter resets
        when episode_id changes on the next /act."""
        eid = episode_id or f"ep-{uuid.uuid4().hex[:12]}"
        proxy = _EpisodeProxy(self, eid)
        try:
            yield proxy
        finally:
            proxy._closed = True


class _EpisodeProxy:
    """Returned by TetherClient.episode() context manager."""

    def __init__(self, client: TetherClient, episode_id: str):
        self._client = client
        self._episode_id = episode_id
        self._closed = False
        self._call_count = 0

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def call_count(self) -> int:
        return self._call_count

    def act(
        self,
        image: Any = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise TetherClientError("episode context already closed")
        self._call_count += 1
        return self._client.act(
            image=image, instruction=instruction, state=state, episode_id=self._episode_id,
        )


# ---- Async client ---------------------------------------------------------

class TetherAsyncClient:
    """Async variant of TetherClient. Same API, awaits httpx.AsyncClient."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        initial_backoff_s: float = 0.1,
        max_backoff_s: float = 1.6,
        retry_on_degraded: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s
        self.retry_on_degraded = retry_on_degraded
        headers: dict[str, str] = {}
        if api_key:
            headers["X-Tether-Key"] = api_key
        self._http = httpx.AsyncClient(headers=headers, timeout=timeout_s)
        self._closed = False

    async def __aenter__(self) -> TetherAsyncClient:
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def close(self) -> None:
        if not self._closed:
            await self._http.aclose()
            self._closed = True

    async def _request_with_retry(self, method: str, path: str, **kw) -> httpx.Response:
        attempt = 0
        backoff = self.initial_backoff_s
        while True:
            try:
                resp = await self._http.request(method, f"{self.base_url}{path}", **kw)
            except (httpx.NetworkError, httpx.TimeoutException) as e:
                if attempt >= self.max_retries:
                    raise TetherClientError(f"network error after {attempt} retries: {e}") from e
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
                attempt += 1
                continue
            err = _classify_response(resp)
            if err is None:
                return resp
            if isinstance(err, (TetherAuthError, TetherValidationError)):
                raise err
            if isinstance(err, TetherServerDegradedError) and not self.retry_on_degraded:
                raise err
            if attempt >= self.max_retries:
                raise err
            wait = _parse_retry_after(resp.headers.get("Retry-After"))
            if wait is None:
                wait = backoff
                backoff = min(backoff * 2, self.max_backoff_s)
            await asyncio.sleep(wait)
            attempt += 1

    async def health(self) -> dict[str, Any]:
        resp = await self._http.get(f"{self.base_url}/health")
        try:
            return resp.json()
        except Exception:
            return {"status": "unknown", "state": "unknown", "raw": resp.text}

    async def config(self) -> dict[str, Any]:
        resp = await self._request_with_retry("GET", "/config")
        return resp.json()

    async def act(
        self,
        image: Any = None,
        instruction: str = "",
        state: list[float] | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "image": encode_image(image),
            "instruction": instruction,
            "state": state,
            "episode_id": episode_id,
        }
        resp = await self._request_with_retry("POST", "/act", json=body)
        return resp.json()

    @contextlib.asynccontextmanager
    async def episode(self, episode_id: str | None = None) -> AsyncIterator[_AsyncEpisodeProxy]:
        eid = episode_id or f"ep-{uuid.uuid4().hex[:12]}"
        proxy = _AsyncEpisodeProxy(self, eid)
        try:
            yield proxy
        finally:
            proxy._closed = True


class _AsyncEpisodeProxy:
    def __init__(self, client: TetherAsyncClient, episode_id: str):
        self._client = client
        self._episode_id = episode_id
        self._closed = False
        self._call_count = 0

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def call_count(self) -> int:
        return self._call_count

    async def act(
        self,
        image: Any = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise TetherClientError("episode context already closed")
        self._call_count += 1
        return await self._client.act(
            image=image, instruction=instruction, state=state, episode_id=self._episode_id,
        )
