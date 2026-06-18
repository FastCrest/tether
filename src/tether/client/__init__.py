"""Tether Python SDK — sync + async clients for `tether serve`.

Designed against the production `/act`, `/health`, `/config` surface shipped
in src/tether/runtime/server.py. Consumes:

- `/health` 6-state machine (initializing/loading/warming/ready/warmup_failed/
  degraded) — `TetherClient.health()` returns the parsed state.
- `/act` 503 + Retry-After header on circuit-broken servers — retry loop
  respects the header.
- X-Tether-Key bearer auth — `api_key` constructor arg sets the header.
- `episode_id` field on /act request — `client.episode()` context manager
  auto-generates + propagates so RTC adapter resets correctly between episodes.
- `guard_violations` + `guard_clamped` response fields (B.6 ActionGuard) —
  surfaced on the result object as fields, not silently swallowed.

Usage:

    from tether.client import TetherClient

    client = TetherClient("http://localhost:8000", api_key="abc")
    result = client.act(image="frame.jpg", instruction="pick the cup", state=[...])
    print(result["actions"])

    # Episode tracking (auto episode_id, RTC reset):
    with client.episode() as ep:
        for frame in frames:
            ep.act(image=frame.image, state=frame.state)

    # Async:
    async with TetherAsyncClient("http://localhost:8000") as client:
        result = await client.act(...)
"""

from tether.client.client import (
    TetherClient,
    TetherAsyncClient,
    TetherClientError,
    TetherAuthError,
    TetherServerDegradedError,
    TetherServerNotReadyError,
    TetherValidationError,
    encode_image,
)

class ReflexClient(TetherClient):
    """Deprecated alias for :class:`TetherClient`. Removed in v0.14.0.

    Kept so pre-rename code (`from tether.client import ReflexClient`) keeps
    working through the v0.13.x compat window, matching the `reflex` import
    shim's removal schedule.
    """

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "ReflexClient is deprecated; use TetherClient. "
            "The alias is removed in tether v0.14.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class ReflexAsyncClient(TetherAsyncClient):
    """Deprecated alias for :class:`TetherAsyncClient`. Removed in v0.14.0."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "ReflexAsyncClient is deprecated; use TetherAsyncClient. "
            "The alias is removed in tether v0.14.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "TetherClient",
    "TetherAsyncClient",
    "TetherClientError",
    "TetherAuthError",
    "TetherServerDegradedError",
    "TetherServerNotReadyError",
    "TetherValidationError",
    "encode_image",
    # Deprecated rename-compat aliases (removed v0.14.0).
    "ReflexClient",
    "ReflexAsyncClient",
]
