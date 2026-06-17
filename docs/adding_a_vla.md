# Adding a new VLA model

This cookbook walks through the steps to add a new vision-language-action (VLA) model to Tether. Sibling to [`adding_a_robot.md`](./adding_a_robot.md) (embodiments). As of v0.10.0, adding a new VLA backbone is a ~100 LOC composition class on the **BaseVLA spine**, not a duplicated 600-1000 LOC exporter pipeline.

## The 6-slot taxonomy

Every VLA on the spine declares which of 6 component slots it uses:

| Slot | Used by |
|---|---|
| `vision_backbone` | pi0, pi0.5, smolvla (SigLIP / SmolVLM2 vision tower) |
| `llm_backbone` | pi0, pi0.5, smolvla (PaliGemma / SmolLM2 text decoder) |
| `vlm_backbone` | GR00T (Eagle = SigLIP + Qwen2 + mlp1, **fused**) |
| `projector` | most VLAs (action/state projection) |
| `vla_head` | every VLA (flow-matching / DiT / argmax) |
| `text_encoder` | DreamZero / future T5-only VLAs |

`REQUIRED_SLOTS` + `OPTIONAL_SLOTS` declare which slots your new VLA needs. The spine refuses construction if a required slot is missing OR if an undeclared slot is provided.

## Step 1 — Pick your slot pattern

Look at the existing VLA classes in `src/tether/models/vlas/`:

| Family | Pattern | File |
|---|---|---|
| Pi0VLA | 2-tower (vision + llm + head) | `src/tether/models/vlas/pi0.py` |
| Pi05VLA | 2-tower, state-in-language | `src/tether/models/vlas/pi05.py` |
| SmolVLA | 2-tower + state projector | `src/tether/models/vlas/smolvla.py` |
| GR00TVLA | fused VLM + DiT head | `src/tether/models/vlas/gr00t.py` |

Pick the closest match. Mirror its `REQUIRED_SLOTS` declaration.

## Step 2 — Build the components

If your VLA needs a slot type that doesn't already exist as a concrete class, build a thin wrapper:

- **Vision backbones** in `src/tether/models/vision/` (e.g., `SigLIPBackbone`, `SmolVLAVisionBackbone`)
- **LLM backbones** in `src/tether/models/llm/` (e.g., `PaliGemmaBackbone`, `SmolVLALLMBackbone`)
- **Fused VLMs** in `src/tether/models/vlm/` (e.g., `EagleBackbone`)
- **Projectors** in `src/tether/models/projectors/` (e.g., `LinearProjector`)
- **Heads** in `src/tether/models/heads/` (e.g., `FlowMatchingHead`, `DITHead`)

Each component inherits from its slot's ABC (`VisionBackbone`, `LLMBackbone`, etc) and registers via the matching component Registry decorator (`@VISION_BACKBONES.register`, etc).

## Step 3 — Write your composition class

Template (~100 LOC):

```python
# src/tether/models/vlas/myvla.py
from typing import Any, ClassVar
import torch
from tether.models.base_vla import BaseVLA
from tether.registry.components import VLAS


@VLAS.register
class MyVLA(BaseVLA):
    """One-line summary of this VLA's architecture."""

    REQUIRED_SLOTS: ClassVar[tuple[str, ...]] = (
        "vision_backbone", "llm_backbone", "vla_head",
    )
    OPTIONAL_SLOTS: ClassVar[tuple[str, ...]] = ()
    NAME_MAPPING: ClassVar[dict[str, str]] = {}

    @classmethod
    def from_pretrained(cls, hf_id: str, *, state_dict=None) -> "MyVLA":
        # 1. Load checkpoint
        # 2. Build each slot's component from the state_dict
        # 3. Return cls(vision_backbone=..., llm_backbone=..., vla_head=...)
        ...

    def forward(self, batch): ...
    def predict_action(self, ...): ...
```

## Step 4 — Register in `data.py`

Add a `ModelEntry` to `src/tether/registry/data.py`:

```python
ModelEntry(
    model_id="myvla-base",
    hf_repo="my-org/myvla-base",
    family="myvla",  # MUST be in models.py family check list — see Step 5
    action_dim=32,
    size_mb=3000,
    supported_embodiments=("franka",),
    supported_devices=("a10g", "a100"),
    description="...",
    license="apache-2.0",
    # vla_type=None → resolves to spine class name from family
    # vla_type="_myvla_shim" → marks non-spine shim like OpenVLA
),
```

## Step 5 — Update validators

Add the new family to the allowed list in `src/tether/registry/models.py::ModelEntry.__post_init__`:

```python
if self.family not in ("pi0", "pi05", "smolvla", "openvla", "groot", "myvla"):
    raise ValueError(...)
```

And to `ModelEntry.resolved_vla_type`:

```python
return {
    "pi0": "Pi0VLA",
    "pi05": "Pi05VLA",
    "smolvla": "SmolVLA",
    "groot": "GR00TVLA",
    "myvla": "MyVLA",  # new
}.get(self.family, self.family)
```

## Step 6 — Add a parity test

Mirror `tests/test_gr00t_spine.py` — registration + slot declarations + construction + `predict_action` smoke. For bit-identical parity vs a reference checkpoint, add a Modal script in `scripts/modal_myvla_parity.py` modeled on `scripts/modal_gr00t_spine_parity.py`.

## Step 7 — Validate

Local:

```bash
PYTHONPATH=src python3 -m pytest tests/test_myvla_spine.py -x
```

Modal (real-checkpoint parity, optional):

```bash
modal profile activate novarepmarketing  # or your profile
modal run scripts/modal_myvla_parity.py
```

Write the result to `reflex_context/03_experiments/YYYY-MM-DD-myvla-spine-parity.md` per the experiment-note convention. Bit-identical (`max_diff=0.0`, `cos=1.000000`) is the strongest signal that your composition class correctly wraps the underlying weights.

## Step 8 — Add to CLI integration

`tether models list` + `tether models info` already pick up new entries automatically via `ModelEntry.resolved_vla_type`. `tether export <model_id>` dispatch happens by `model_type` (== family) in `src/tether/cli.py::export()` — add a branch there if your family needs custom orchestration; otherwise the default fall-through works.

## See also

- [`adding_a_robot.md`](./adding_a_robot.md) — sibling cookbook for embodiments
- [`architecture_wedges.md`](./architecture_wedges.md) — server-level composition pattern
- `src/tether/models/base_vla.py` — the spine ABC (REQUIRED_SLOTS / OPTIONAL_SLOTS / from_config)
- `reflex_context/01_decisions/2026-05-19-fluxvla-lift-program-decisions.md` — the 19 architectural decisions that shape the spine
- `reflex_context/features/03_export/basevla-spine_plan.md` — the lift #1 12-day plan that landed this refactor
