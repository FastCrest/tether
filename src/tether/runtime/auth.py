"""Bearer auth + semaphore-based concurrency limiting for `tether serve`.

Upgrades the existing `X-Tether-Key` header to the standard
`Authorization: Bearer <token>` scheme (RFC 6750) while keeping the legacy
header working for back-compat. Adds a semaphore limiter so overloaded
servers return HTTP 429 + `Retry-After` instead of slowing down every
request (TGI's overload pattern).

Pattern source: TGI (HuggingFace text-generation-inference) streaming + auth
pattern. See tether-vla/reference/deep_dive_tgi_streaming_auth.md.

Feature spec: features/01_serve/subfeatures/_ecosystem/auth-bearer/
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """Return a short request id for error-response correlation.

    Format: 'req-{8-hex}'. Non-cryptographic — correlation only.
    """
    return f"req-{uuid.uuid4().hex[:8]}"


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Parse an `Authorization: Bearer <token>` header per RFC 6750.

    Returns the token string, or None if the header is missing or
    malformed. Case-insensitive on the 'Bearer' scheme.
    """
    if not authorization_header:
        return None
    parts = authorization_header.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token if token else None


def constant_time_token_match(provided: Optional[str], expected: Optional[str]) -> bool:
    """Compare tokens in constant time (resists timing attacks).

    Returns False if either is None or empty. Uses `secrets.compare_digest()`
    which is designed for this.
    """
    if not provided or not expected:
        return False
    # compare_digest requires equal-length inputs on some Python versions;
    # it doesn't leak length but we short-circuit on trivial mismatch.
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def resolve_request_token(
    authorization: Optional[str],
    x_tether_key: Optional[str],
) -> Optional[str]:
    """Return whichever auth token the caller provided, preferring the
    standard `Authorization: Bearer` header over the legacy `X-Tether-Key`.

    Back-compat semantics: a customer that migrated to Bearer can keep
    setting `X-Tether-Key` on clients we haven't rolled forward yet; both
    work.
    """
    bearer = extract_bearer_token(authorization)
    if bearer:
        return bearer
    if x_tether_key:
        return x_tether_key.strip() or None
    return None


@dataclass(frozen=True)
class AuthError:
    """Structured 401 response payload."""
    status: int
    error: str
    message: str
    request_id: str

    def to_dict(self) -> dict:
        return {
            "error": self.error,
            "message": self.message,
            "request_id": self.request_id,
        }


def make_401_payload(reason: str) -> AuthError:
    """Build a structured 401 payload with a fresh request_id."""
    return AuthError(
        status=401,
        error="unauthorized",
        message=reason,
        request_id=generate_request_id(),
    )


def make_429_payload(current: int, limit: int) -> dict:
    """Build a structured 429 payload."""
    return {
        "error": "too_many_requests",
        "message": (
            f"Server concurrency limit reached ({current}/{limit}); retry later"
        ),
        "request_id": generate_request_id(),
        "concurrent_requests": current,
        "max_concurrent": limit,
    }


class ConcurrencyLimiter:
    """Semaphore-based concurrency limiter.

    Usage:

        limiter = ConcurrencyLimiter(max_concurrent=32)
        # In middleware:
        async with limiter.try_acquire() as acquired:
            if not acquired:
                return JSONResponse(status_code=429, content=...)
            return await call_next(request)

    Unlike `asyncio.Semaphore`, `try_acquire()` is non-blocking — when the
    limit is saturated, the context returns immediately with `acquired=False`
    so the caller can return 429 instead of queueing the request. TGI's
    overload pattern: reject fast, let the client retry, rather than letting
    queue depth explode.
    """

    __slots__ = ("_max_concurrent", "_semaphore", "_in_flight")

    def __init__(self, max_concurrent: int):
        if max_concurrent <= 0:
            raise ValueError(f"max_concurrent must be > 0, got {max_concurrent}")
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._in_flight = 0  # observability counter

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def in_flight(self) -> int:
        """Current in-flight request count (approximate — not locked)."""
        return self._in_flight

    def try_acquire(self):
        """Async context manager. Non-blocking; yields an _AcquireResult
        with `.acquired` True or False.
        """
        return _LimiterContext(self)


class _LimiterContext:
    """Async CM for ConcurrencyLimiter.try_acquire()."""

    __slots__ = ("_limiter", "_acquired")

    def __init__(self, limiter: ConcurrencyLimiter):
        self._limiter = limiter
        self._acquired = False

    async def __aenter__(self) -> "_LimiterContext":
        # Non-blocking: locked() is True when count == 0 (exhausted)
        if self._limiter._semaphore.locked():
            self._acquired = False
        else:
            # Acquire without await in the exhausted case (doesn't block)
            await self._limiter._semaphore.acquire()
            self._acquired = True
            self._limiter._in_flight += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._acquired:
            self._limiter._in_flight -= 1
            self._limiter._semaphore.release()

    @property
    def acquired(self) -> bool:
        return self._acquired
