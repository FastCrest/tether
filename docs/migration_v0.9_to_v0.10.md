# Migration guide: v0.9 → v0.10

> If you only use `tether` as a CLI, you don't need to change anything — `tether export`, `tether models *`, `tether serve`, `tether chat` all keep working. This guide is for **library users** who import from `tether.exporters.*`.

## TL;DR

v0.10.0 lands the **BaseVLA spine refactor** (lift #1). Several exporter modules were renamed or deleted as part of the cleanup. Update your imports as below.

## The 5 import renames

| v0.9 import path | v0.10 path | Day |
|---|---|---|
| `tether.exporters.pi0_exporter` | `tether.exporters.pi0` | 11 |
| `tether.exporters.smolvla_exporter` | `tether.exporters.smolvla` | 11 |
| `tether.exporters.gr00t_exporter` | `tether.exporters.gr00t` | 11 |
| `tether.exporters.openvla_exporter` | `tether.exporters.openvla` | 8 |
| `tether.exporters.pi0_prefix_exporter` | `tether.exporters.pi0_prefix` | 9 |

The legacy modules are **gone** — they raise `ModuleNotFoundError`. No compat aliases. The test `tests/test_day10_cli_vla_type.py::test_legacy_exporter_modules_deleted` pins this so future regressions are caught.

## What stayed the same

- `build_pi0_expert_stack` / `build_pi05_expert_stack` / `build_expert_stack` (SmolVLA) / `build_gr00t_expert_stack` / `build_gr00t_full_stack` — same builders, just in the renamed modules
- `export_pi0` / `export_pi05` / `export_smolvla` / `export_gr00t` / `export_gr00t_full` — same export functions, same I/O shapes, same ONNX bytes byte-for-byte
- `Pi0ExpertStackWithPrefix` / `Pi05ExpertStackWithPrefix` / `ExpertStack` classes — same classes, exposed from their new home modules

The numerics didn't change. The PRs that landed the rename also passed bit-identical parity vs the reference checkpoints — pi0 max 1.13e-6, pi0.5 max 2.74e-6, SmolVLA max 0.0, GR00T N1.6 max 0.0.

## Two examples

**Before (v0.9):**

```python
from tether.exporters.smolvla_exporter import build_expert_stack
from tether.exporters.pi0_exporter import build_pi0_expert_stack, PI0_ACTION_KEYS
from tether.exporters.gr00t_exporter import build_gr00t_full_stack

# ...your code unchanged
```

**After (v0.10):**

```python
from tether.exporters.smolvla import build_expert_stack
from tether.exporters.pi0 import build_pi0_expert_stack, PI0_ACTION_KEYS
from tether.exporters.gr00t import build_gr00t_full_stack

# ...your code unchanged
```

A bulk `sed` covers it:

```bash
find . -name '*.py' -exec sed -i \
    -e 's|tether\.exporters\.pi0_exporter|tether.exporters.pi0|g' \
    -e 's|tether\.exporters\.smolvla_exporter|tether.exporters.smolvla|g' \
    -e 's|tether\.exporters\.gr00t_exporter|tether.exporters.gr00t|g' \
    -e 's|tether\.exporters\.openvla_exporter|tether.exporters.openvla|g' \
    -e 's|tether\.exporters\.pi0_prefix_exporter|tether.exporters.pi0_prefix|g' \
    {} +
```

## What's new

If you want to build VLAs through the new spine composition (rather than the legacy direct-build path), see `src/tether/models/vlas/` for the worked examples and [`docs/adding_a_vla.md`](./adding_a_vla.md) for the cookbook. The TL;DR is:

```python
from tether.models.vlas.pi05 import Pi05VLA

vla = Pi05VLA.from_pretrained("lerobot/pi05_libero_finetuned_v044")
actions = vla.predict_action(
    images=[...], lang_tokens=..., lang_masks=...,
)
```

`predict_action` is bit-identical to lerobot's `PI05Policy.predict_action` on the same checkpoint + inputs (max diff 2.74e-6 measured on Modal A10G).

## Why the rename

The legacy `*_exporter.py` modules grew to host both **builders** (load checkpoint → reconstruct the action expert as a PyTorch module) and **exporters** (build expert → write ONNX). The spine refactor split these concerns:

- **Builders** stay in `tether.exporters.<family>` (renamed from `<family>_exporter` for brevity).
- **Composition classes** live in `tether.models.vlas.<family>`. They wrap the builders behind a uniform `BaseVLA` 6-slot interface.

The drop of the `_exporter` suffix reflects this: the module is no longer JUST an exporter. It's the family's source of truth for builders + classes + constants.

## Help

If something else broke that isn't on this list, the rename is mechanical — the change is in the import path, not the API. Open an issue with the failing import + we'll add it here.

## See also

- [`CHANGELOG.md`](../CHANGELOG.md#v0100--2026-05-22) — full v0.10.0 release notes
- [`docs/adding_a_vla.md`](./adding_a_vla.md) — cookbook for adding a new VLA to the spine
- `reflex_context/features/03_export/basevla-spine_plan.md` — the lift #1 12-day plan that landed this refactor (in the design vault)
