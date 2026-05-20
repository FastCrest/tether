"""Tests for src/reflex/registry/builder.py — Registry + build_from_cfg.

Foundation for the BaseVLA spine refactor (lift #1 per
`features/03_export/basevla-spine.md`). The Registry pattern is a 150-LOC
zero-dep clone of FluxVLA's `engines/utils/{registry,builder}.py` — we
must not import mmengine (decision K1 in the lift-program ADR).

Coverage:
- Registration (decorator + explicit), name collision, unregister
- Lookup (`get`, `__contains__`, `__len__`, `names`)
- `build_from_cfg` happy path, non-dict passthrough, nested resolution,
  default_args precedence
- Error surfaces (unknown type, TypeError pass-through with diagnostic)
- Independence: two registries don't collide on same class name

The contract tested here is what every downstream lift (#3, #5, #7)
will build on.
"""
from __future__ import annotations

import pytest

from reflex.registry.builder import (
    Registry,
    RegistryError,
    build_from_cfg,
)


# ---------------------------------------------------------------------------
# Test fixtures — minimal classes that exercise the Registry contract
# ---------------------------------------------------------------------------


class _Leaf:
    """Minimal class with two scalar args. Used in nested-build tests."""

    def __init__(self, x: int, y: str = "default"):
        self.x = x
        self.y = y


class _Branch:
    """Class that composes a _Leaf. Used to test recursive build_from_cfg."""

    def __init__(self, leaf: _Leaf, scale: float):
        self.leaf = leaf
        self.scale = scale


class _NoArgs:
    """Class with no constructor args. Used for empty-cfg test."""

    pass


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_register_as_decorator():
    """`@reg.register` on a class declaration adds it under the class's __name__."""
    reg = Registry("test_decorator")

    @reg.register
    class Widget:
        pass

    assert "Widget" in reg
    assert reg.get("Widget") is Widget


def test_register_as_method_call():
    """Explicit `reg.register(cls)` is equivalent to decorator usage."""
    reg = Registry("test_method")

    class Widget:
        pass

    reg.register(Widget)

    assert "Widget" in reg
    assert reg.get("Widget") is Widget


def test_register_returns_class_unchanged():
    """Decorator must return the class as-is so wrapped definitions still work."""
    reg = Registry("test_return")

    @reg.register
    class Widget:
        pass

    # The class object should be usable downstream (no proxy / metaclass shim)
    instance = Widget()
    assert isinstance(instance, Widget)


def test_duplicate_registration_raises():
    """Registering two classes with the same `__name__` is a loud error,
    not a silent overwrite. Per decision S-3 + the Registry's contract."""
    reg = Registry("test_dup")

    @reg.register
    class Widget:
        pass

    # Try to register a different class that happens to share the name
    class WidgetReplacement:
        pass

    WidgetReplacement.__name__ = "Widget"

    with pytest.raises(RegistryError, match="already registered"):
        reg.register(WidgetReplacement)


def test_unregister():
    """`unregister` removes a registration; tests use this for cleanup."""
    reg = Registry("test_unreg")

    @reg.register
    class Widget:
        pass

    assert "Widget" in reg
    reg.unregister("Widget")
    assert "Widget" not in reg


def test_unregister_unknown_name_raises():
    reg = Registry("test_unreg_err")
    with pytest.raises(RegistryError, match="not in"):
        reg.unregister("DoesNotExist")


def test_registries_are_independent():
    """Two registries can host classes with the same `__name__` without
    collision. Foundation for the component-level registry pattern."""
    reg_a = Registry("alpha")
    reg_b = Registry("beta")

    @reg_a.register
    class Widget:
        marker = "a"

    # Re-declare a different class with the same name in reg_b. In normal
    # code this would be a class defined in a different module; we simulate
    # it here.
    class WidgetB:
        marker = "b"

    WidgetB.__name__ = "Widget"
    reg_b.register(WidgetB)

    assert reg_a.get("Widget") is Widget
    assert reg_b.get("Widget") is WidgetB
    assert reg_a.get("Widget").marker == "a"
    assert reg_b.get("Widget").marker == "b"


# ---------------------------------------------------------------------------
# Lookup tests
# ---------------------------------------------------------------------------


def test_get_unknown_name_lists_available():
    """Error message on unknown name MUST list available names — this is the
    typo-debugging surface in V1 (per decision S-5, schema validation defers
    to Phase 2)."""
    reg = Registry("test_lookup")

    @reg.register
    class Foo:
        pass

    @reg.register
    class Bar:
        pass

    with pytest.raises(RegistryError, match=r"Available: \['Bar', 'Foo'\]"):
        reg.get("Bz")  # typo


def test_names_is_sorted():
    """`.names()` returns sorted list — used by `reflex models list` UX."""
    reg = Registry("test_names")

    @reg.register
    class Zebra:
        pass

    @reg.register
    class Alpha:
        pass

    @reg.register
    class Mango:
        pass

    assert reg.names() == ["Alpha", "Mango", "Zebra"]


def test_len_and_contains():
    reg = Registry("test_len")
    assert len(reg) == 0
    assert "Anything" not in reg

    @reg.register
    class Foo:
        pass

    assert len(reg) == 1
    assert "Foo" in reg
    assert "Bar" not in reg


def test_repr_shows_count():
    reg = Registry("vision_backbones")
    assert "vision_backbones" in repr(reg)
    assert "0 classes" in repr(reg)


# ---------------------------------------------------------------------------
# build_from_cfg tests
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_registry():
    """A registry with _Leaf, _Branch, _NoArgs registered."""
    reg = Registry("populated")
    reg.register(_Leaf)
    reg.register(_Branch)
    reg.register(_NoArgs)
    return reg


def test_build_simple_flat_dict(populated_registry):
    """Type-tagged dict with scalar args → instance."""
    obj = build_from_cfg({"type": "_Leaf", "x": 42, "y": "test"}, populated_registry)
    assert isinstance(obj, _Leaf)
    assert obj.x == 42
    assert obj.y == "test"


def test_build_uses_class_defaults(populated_registry):
    """Omitted kwargs fall through to the class's defaults."""
    obj = build_from_cfg({"type": "_Leaf", "x": 1}, populated_registry)
    assert obj.x == 1
    assert obj.y == "default"


def test_build_recursive(populated_registry):
    """Nested type-tagged dict resolves recursively."""
    cfg = {
        "type": "_Branch",
        "leaf": {"type": "_Leaf", "x": 5},
        "scale": 1.5,
    }
    obj = build_from_cfg(cfg, populated_registry)
    assert isinstance(obj, _Branch)
    assert isinstance(obj.leaf, _Leaf)
    assert obj.leaf.x == 5
    assert obj.leaf.y == "default"
    assert obj.scale == 1.5


def test_build_non_dict_passthrough(populated_registry):
    """Non-dict values pass through unchanged. Lets callers pass scalars,
    lists, and non-type dicts inline without `{type: ...}` wrapping."""
    assert build_from_cfg(42, populated_registry) == 42
    assert build_from_cfg("hello", populated_registry) == "hello"
    assert build_from_cfg([1, 2, 3], populated_registry) == [1, 2, 3]


def test_build_dict_without_type_key_passthrough(populated_registry):
    """A dict WITHOUT `type` key is returned as-is. Lets callers pass plain
    config dicts (e.g. {"lr": 0.001}) as constructor args without
    accidentally triggering registry resolution."""
    raw = {"lr": 0.001, "momentum": 0.9}
    assert build_from_cfg(raw, populated_registry) == raw


def test_build_default_args(populated_registry):
    """`default_args` provides fallback values; cfg keys override defaults."""
    obj = build_from_cfg(
        {"type": "_Leaf", "x": 99},
        populated_registry,
        default_args={"y": "from-defaults"},
    )
    assert obj.x == 99
    assert obj.y == "from-defaults"


def test_build_cfg_overrides_default_args(populated_registry):
    """When both default_args and cfg provide the same key, cfg wins."""
    obj = build_from_cfg(
        {"type": "_Leaf", "x": 1, "y": "from-cfg"},
        populated_registry,
        default_args={"y": "from-defaults"},
    )
    assert obj.y == "from-cfg"


def test_build_no_args_class(populated_registry):
    """A class with no constructor args builds from a bare type-tag dict."""
    obj = build_from_cfg({"type": "_NoArgs"}, populated_registry)
    assert isinstance(obj, _NoArgs)


def test_build_unknown_type_raises(populated_registry):
    """Typo in `type` field surfaces a clear error with available names."""
    with pytest.raises(RegistryError, match="Available"):
        build_from_cfg({"type": "DoesNotExist", "x": 1}, populated_registry)


def test_build_unexpected_kwarg_raises_with_diagnostic(populated_registry):
    """Wrong kwarg name in cfg → TypeError with diagnostic listing resolved
    kwargs. This is the V1 typo-catching surface (decision S-5)."""
    with pytest.raises(TypeError, match="Failed to instantiate"):
        build_from_cfg(
            {"type": "_Leaf", "x": 1, "wrong_arg": "oops"},
            populated_registry,
        )


def test_build_nested_with_default_args_does_not_leak_to_inner(populated_registry):
    """default_args applies to the OUTER call only — not to recursively-resolved
    inner dicts. Otherwise a default named `x` would mask the inner spec."""
    cfg = {
        "type": "_Branch",
        "leaf": {"type": "_Leaf", "x": 1},
        # scale provided via default_args
    }
    obj = build_from_cfg(
        cfg,
        populated_registry,
        default_args={"scale": 0.5, "x": 999},  # x=999 must NOT leak into _Leaf
    )
    assert obj.scale == 0.5
    assert obj.leaf.x == 1  # inner _Leaf saw x=1 from cfg, not 999 from defaults


# ---------------------------------------------------------------------------
# Component-registry sanity (proves the typed registries in components.py
# are independent + the global is what __init__ re-exports)
# ---------------------------------------------------------------------------


def test_component_registries_are_distinct_objects():
    """Each module-level registry is a separate Registry instance."""
    from reflex.registry import (
        VISION_BACKBONES,
        LLM_BACKBONES,
        VLM_BACKBONES,
        PROJECTORS,
        VLA_HEADS,
        TEXT_ENCODERS,
        VLAS,
    )

    all_regs = [
        VISION_BACKBONES, LLM_BACKBONES, VLM_BACKBONES,
        PROJECTORS, VLA_HEADS, TEXT_ENCODERS, VLAS,
    ]
    # All distinct objects
    assert len({id(r) for r in all_regs}) == len(all_regs)
    # All Registry instances
    assert all(isinstance(r, Registry) for r in all_regs)


def test_component_registries_named_correctly():
    """Each registry's name matches its module-level identifier (lowercased)."""
    from reflex.registry import (
        VISION_BACKBONES, LLM_BACKBONES, VLM_BACKBONES,
        PROJECTORS, VLA_HEADS, TEXT_ENCODERS, VLAS,
    )
    assert VISION_BACKBONES.name == "vision_backbones"
    assert LLM_BACKBONES.name == "llm_backbones"
    assert VLM_BACKBONES.name == "vlm_backbones"
    assert PROJECTORS.name == "projectors"
    assert VLA_HEADS.name == "vla_heads"
    assert TEXT_ENCODERS.name == "text_encoders"
    assert VLAS.name == "vlas"
