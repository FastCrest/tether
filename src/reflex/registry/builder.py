"""Minimal Registry + build_from_cfg.

Lifted PATTERN from FluxVLA's `engines/utils/{registry,builder}.py` (Apache-2.0;
upstream re-exports mmengine). Reimplemented here as a ~150-LOC zero-dep version
to avoid pulling in mmengine's ~50 transitive deps — see `01_decisions/
2026-05-19-fluxvla-lift-program.md` "Why mmengine is killed (K1)."

This is the foundation for the BaseVLA spine refactor (lift #1, see
`features/03_export/basevla-spine.md`). Components (vision backbones, LLM
backbones, action heads, etc.) register via `@VISION_BACKBONES.register`. VLA
specs are dict-of-dicts with a `type` field; `build_from_cfg()` recursively
resolves them through the right registry to instantiate the whole tree.

Hybrid registration per decision S-3:

- **Decorator (preferred):** `@VISION_BACKBONES.register` on the class declaration.
- **Explicit (testing + dynamic-import patterns):** `VISION_BACKBONES.register(cls)`
  is callable directly with the class as positional arg.

Both routes hit the same registry.

The minimal example:

    >>> from reflex.registry.builder import Registry, build_from_cfg
    >>> THINGS = Registry("things")
    >>> @THINGS.register
    ... class Foo:
    ...     def __init__(self, x: int):
    ...         self.x = x
    >>> foo = build_from_cfg({"type": "Foo", "x": 42}, THINGS)
    >>> foo.x
    42

Recursive resolution — a nested dict with `type` becomes a sub-call:

    >>> @THINGS.register
    ... class Bar:
    ...     def __init__(self, foo: Foo, y: int):
    ...         self.foo = foo
    ...         self.y = y
    >>> bar = build_from_cfg({"type": "Bar", "foo": {"type": "Foo", "x": 3}, "y": 7}, THINGS)
    >>> bar.foo.x, bar.y
    (3, 7)

What this does NOT do (deferred per decision S-5):

- JSON-Schema validation of spec dicts. V1 catches typos at instantiation time
  via the natural `TypeError` from missing keyword arguments. Schema validation
  ships in Phase 2 if contributor confusion becomes a measured pain point.

- mmengine's Hooks / Scope / Logger framework. We don't need them; they're the
  bulk of mmengine's transitive dep cost.

- Cross-registry resolution (e.g. type='Foo' in registry A but type happens to
  collide with a name in registry B). Each registry is independent; callers pass
  the registry explicitly.

Quality bar (per `features/03_export/basevla-spine_research.md` Lens 4):

- Foundation LOC budget ~150 LOC + tests ~200 LOC. This file is the budget.
- Zero new external dependencies (typing + collections only).
- No circular imports — this module imports nothing from `src/reflex/models/`.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class RegistryError(Exception):
    """Raised when a Registry operation fails.

    Examples:
        - Registering a class that's already in the registry
        - Looking up a name that's not registered
        - `build_from_cfg` encountering a dict without a `type` field where one
          is required
    """


class Registry:
    """Named registry of classes for build_from_cfg.

    Instances are typically module-level constants (e.g. `VISION_BACKBONES`)
    and components register themselves at import time via the `@register`
    decorator. Tests + dynamic-import patterns can also call `register()` as a
    plain method.

    A registry stores classes by their `__name__` attribute. Two classes with
    the same `__name__` cannot both register — that's an explicit error.

    Args:
        name: Human-readable name of the registry. Used in error messages.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._classes: dict[str, type] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, cls: type[T]) -> type[T]:
        """Register `cls` keyed by `cls.__name__`. Returns the class unchanged.

        Usable as a decorator OR as a plain method call:

            @REGISTRY.register
            class MyClass:
                ...

            # OR equivalent:
            REGISTRY.register(MyClass)

        Raises:
            RegistryError: if a class with the same `__name__` is already
                registered. Per decision S-3 we surface duplicate registration
                loud-and-early rather than silently overwriting.
        """
        if cls.__name__ in self._classes:
            existing = self._classes[cls.__name__]
            raise RegistryError(
                f"Cannot register {cls!r}: name {cls.__name__!r} already "
                f"registered in {self._name!r} as {existing!r}. "
                f"Either rename the new class or unregister the old one first."
            )
        self._classes[cls.__name__] = cls
        return cls

    def unregister(self, name: str) -> None:
        """Remove a registration by name. Used in tests.

        Raises:
            RegistryError: if `name` is not registered.
        """
        if name not in self._classes:
            raise RegistryError(
                f"Cannot unregister {name!r}: not in {self._name!r}. "
                f"Available: {sorted(self._classes)}"
            )
        del self._classes[name]

    def get(self, name: str) -> type:
        """Resolve a registered class by name.

        Raises:
            RegistryError: if `name` is not registered. The error message lists
                the available names — useful for typo debugging.
        """
        if name not in self._classes:
            raise RegistryError(
                f"{name!r} not in {self._name!r}. "
                f"Available: {sorted(self._classes)}"
            )
        return self._classes[name]

    def names(self) -> list[str]:
        """Sorted list of registered names. Useful for `reflex models list` UX."""
        return sorted(self._classes)

    def __contains__(self, name: str) -> bool:
        return name in self._classes

    def __len__(self) -> int:
        return len(self._classes)

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, {len(self._classes)} classes)"


def build_from_cfg(
    cfg: Any,
    registry: Registry,
    default_args: dict[str, Any] | None = None,
) -> Any:
    """Recursively build an object from a type-tagged dict.

    The contract:

    1. If `cfg` is not a dict, it's returned as-is. Lists are returned as-is;
       primitives are returned as-is. This lets users pass concrete values
       inline without `{type: ...}` wrapping.

    2. If `cfg` is a dict WITHOUT a `type` key, it's returned as-is. Callers
       pass plain config dicts (e.g. `{"learning_rate": 0.001}`) as a constructor
       arg. The Registry pattern is opt-in per key.

    3. If `cfg` is a dict WITH a `type` key, the resolver:
       a. Looks up `cfg["type"]` in the registry → class
       b. Recursively resolves each remaining key's value through this function
          (so nested `{type: ...}` dicts compose)
       c. Merges in `default_args` (the caller's defaults override CFG; see
          rationale below)
       d. Instantiates the class with the resolved kwargs

    Args:
        cfg: The config to build. Typically a dict-of-dicts with `type` tags.
        registry: The Registry to resolve `type` against.
        default_args: Optional defaults to merge under (i.e. cfg overrides
            defaults). Lifted from mmengine's `default_args` semantics so the
            pattern is familiar to anyone porting from there.

    Returns:
        An instance of the class identified by `cfg["type"]`, or `cfg` itself
        if no `type` key is present.

    Raises:
        RegistryError: if `cfg["type"]` is not registered.
        TypeError: if the class's `__init__` rejects the resolved kwargs. This
            is the typo-catching surface in V1; schema validation defers to
            Phase 2 (decision S-5).
    """
    # Bullets 1 + 2: only dict-with-type triggers resolution.
    if not isinstance(cfg, dict) or "type" not in cfg:
        return cfg

    type_name = cfg["type"]
    cls = registry.get(type_name)

    # Merge: caller's defaults first, then cfg keys override.
    merged: dict[str, Any] = dict(default_args or {})
    for key, value in cfg.items():
        if key == "type":
            continue
        merged[key] = value

    # Recursive resolution: each value that's a type-tagged dict gets built
    # through the same function. The registry passed down is the same one —
    # callers that need cross-registry resolution should compose explicitly.
    resolved = {
        key: build_from_cfg(value, registry) for key, value in merged.items()
    }

    try:
        return cls(**resolved)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate {type_name!r} from registry "
            f"{registry.name!r}: {exc}. "
            f"Resolved kwargs: {sorted(resolved)}"
        ) from exc


__all__ = ["Registry", "RegistryError", "build_from_cfg"]
