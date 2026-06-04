"""Hook registry for `tether finetune` trainers.

Training backends fire named hooks at documented lifecycle points; the
HookRegistry routes each fire-event to zero-or-more handlers. Callers
register handlers by hook name.

## Lifecycle

| Hook | When | Payload |
|---|---|---|
| `on_start`       | Trainer begins | `config=FinetuneConfig` |
| `on_step`        | After each train step | `step: int, loss: float, lr: float` |
| `on_checkpoint`  | After a checkpoint is saved | `step: int, ckpt_path: Path` |
| `on_end`         | Trainer finishes (success OR failure) | `status: str, steps_completed: int` |
| `on_postprocess` | After auto-export + validate | `onnx_path: Path, parity_max_abs: float` |

`on_postprocess` is the hook slot where `libero_drop_gate`
(task-success kill-gate for distill) lives. See architecture doc
Section E.

## Handler contract

A handler is any callable `(ctx: TrainerContext, **payload) -> None`.
Handlers are called synchronously in registration order. If a handler
raises, the exception propagates — backends should wrap their hook
calls in try/except if they need resilience.

Handlers may MUTATE the registry (e.g., add a follow-up hook) but
must not remove hooks currently executing.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

# A handler is a callable with (ctx, **payload) signature.
Handler = Callable[..., None]

# Documented hook points. Handlers registered for any other name will
# still run (the registry doesn't whitelist), but they're off the
# contract and may silently stop firing in future versions.
LIFECYCLE_HOOKS: tuple[str, ...] = (
    "on_start",
    "on_step",
    "on_checkpoint",
    "on_end",
    "on_postprocess",
)


class HookRegistry:
    """Named multi-handler registry.

    Intentionally minimal — this is a shared-infra primitive. All the
    complexity (LIBERO gate, calibration drift, W&B logging) lives in
    the individual handlers, not the registry.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def register(self, name: str, handler: Handler) -> None:
        """Add a handler to a hook. Order-preserving."""
        self._handlers[name].append(handler)

    def clear(self, name: str | None = None) -> None:
        """Remove handlers. Pass a name to clear one hook, None for all.

        Useful in tests.
        """
        if name is None:
            self._handlers.clear()
        else:
            self._handlers.pop(name, None)

    def run(self, name: str, ctx: Any = None, **payload) -> None:
        """Fire a hook. All handlers run synchronously in registration
        order. Unknown hook names are logged at debug level but not
        rejected — the contract is soft to ease adding new hooks.
        """
        handlers = self._handlers.get(name, ())
        if not handlers:
            if name not in LIFECYCLE_HOOKS:
                logger.debug("[hooks] %r fired with no handlers", name)
            return
        for h in handlers:
            try:
                h(ctx, **payload)
            except Exception:
                # Don't swallow the exception — we want training to
                # surface it. Just attach a hint to the traceback.
                logger.exception(
                    "[hooks] handler %r raised on %r", h.__name__, name,
                )
                raise

    def handlers(self, name: str) -> list[Handler]:
        """Inspect registered handlers for a hook. Read-only view."""
        return list(self._handlers.get(name, ()))

    def __contains__(self, name: str) -> bool:
        return bool(self._handlers.get(name))


__all__ = ["HookRegistry", "Handler", "LIFECYCLE_HOOKS"]
