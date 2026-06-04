"""LeRobot v3.0 dataset validation checks.

Format reference (LeRobot 0.5.1 / v3.0 codebase):
    dataset_root/
        meta/
            info.json                                 # required — schema declaration
            stats.json                                # optional — feature stats
            tasks.parquet                             # optional v3.0 — task lookup
            episodes/chunk-NNN/file-NNN.parquet       # episode metadata (length, etc.)
        data/chunk-NNN/file-NNN.parquet               # frames (action, observation.*, etc.)
        videos/                                       # optional, image features

info.json keys we depend on: codebase_version, fps, total_episodes, total_frames,
features (dict of feature_name -> {dtype, shape, names?}). Standard feature names:
    action                 — vector, shape (action_dim,)
    observation.state      — vector, shape (state_dim,)
    observation.images.*   — image, shape (H, W, 3)
    timestamp              — float
    frame_index            — int
    episode_index          — int
    task_index             — int

Each check returns a CheckResult with a Decision in {ok, warn, blocker}.
Overall: any blocker -> blocker; any warn -> warn; else ok. Exit codes from
the CLI map: ok->0, warn->1, blocker->2.

Pyarrow is OPTIONAL. When pyarrow is missing, parquet-reading checks degrade
to "skipped" (not failures); structural checks (info.json present, schema
declared) still run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except ImportError:
    pq = None  # type: ignore[assignment]
    _HAS_PYARROW = False


class Decision(str, Enum):
    OK = "ok"
    WARN = "warn"
    BLOCKER = "blocker"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Outcome of a single check. Falsifiable: each check_id maps to a known
    failure mode the customer would otherwise hit at training/serve time."""

    check_id: str
    decision: Decision
    summary: str
    details: list[str] = field(default_factory=list)
    fix_hint: str = ""

    def __post_init__(self):
        # Falsifiability gate: every result must declare its check_id + decision
        if not self.check_id:
            raise ValueError("CheckResult requires non-empty check_id")
        if not isinstance(self.decision, Decision):
            self.decision = Decision(self.decision)


@dataclass
class DatasetContext:
    """Everything checks may need about a dataset under examination."""

    root: Path
    info: dict[str, Any] | None = None
    info_load_error: str = ""
    embodiment_config: Any = None  # Optional EmbodimentConfig
    strict: bool = False
    data_files: list[Path] = field(default_factory=list)
    episode_meta_files: list[Path] = field(default_factory=list)


CheckFn = Callable[[DatasetContext], CheckResult]
REGISTERED_CHECKS: list[CheckFn] = []


def register_dataset_check(fn: CheckFn) -> CheckFn:
    """Decorator: register a dataset check. Order = registration order."""
    REGISTERED_CHECKS.append(fn)
    return fn


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _gather_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Find data + episode-metadata parquet files."""
    data = sorted((root / "data").glob("chunk-*/file-*.parquet")) if (root / "data").exists() else []
    episodes = (
        sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if (root / "meta" / "episodes").exists() else []
    )
    return data, episodes


def _load_parquet_table(path: Path):
    """Returns (table, error_str). Caller checks error_str."""
    if not _HAS_PYARROW:
        return None, "pyarrow not installed"
    try:
        return pq.read_table(str(path)), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

@register_dataset_check
def check_info_json(ctx: DatasetContext) -> CheckResult:
    """info.json must exist, parse, and declare codebase_version + fps + features."""
    if ctx.info_load_error:
        return CheckResult(
            check_id="dataset.info-json-parseable",
            decision=Decision.BLOCKER,
            summary="meta/info.json not loadable",
            details=[ctx.info_load_error],
            fix_hint=(
                "Confirm the path is a LeRobot v3.0 dataset root. "
                "Expected layout: meta/info.json, meta/episodes/chunk-NNN/, "
                "data/chunk-NNN/. See lerobot.datasets.dataset_metadata."
            ),
        )
    info = ctx.info or {}
    missing = [k for k in ("codebase_version", "fps", "features") if k not in info]
    if missing:
        return CheckResult(
            check_id="dataset.info-json-parseable",
            decision=Decision.BLOCKER,
            summary=f"meta/info.json missing required keys: {missing}",
            details=[f"present keys: {sorted(info.keys())}"],
            fix_hint="Re-export the dataset with `lerobot dataset export` or fix info.json by hand.",
        )
    return CheckResult(
        check_id="dataset.info-json-parseable",
        decision=Decision.OK,
        summary=f"info.json OK (codebase {info['codebase_version']}, fps {info['fps']})",
    )


@register_dataset_check
def check_schema_completeness(ctx: DatasetContext) -> CheckResult:
    """info.json features must declare 'action' and 'observation.state' at minimum."""
    if not ctx.info:
        return CheckResult(
            check_id="dataset.schema-completeness",
            decision=Decision.SKIPPED,
            summary="info.json not loaded; cannot inspect schema",
        )
    features = ctx.info.get("features", {})
    if not isinstance(features, dict):
        return CheckResult(
            check_id="dataset.schema-completeness",
            decision=Decision.BLOCKER,
            summary="info.json 'features' is not a dict",
            details=[f"got type: {type(features).__name__}"],
        )
    missing = [k for k in ("action", "observation.state") if k not in features]
    image_keys = sorted(k for k in features if k.startswith("observation.images."))
    if missing:
        return CheckResult(
            check_id="dataset.schema-completeness",
            decision=Decision.BLOCKER,
            summary=f"required features missing: {missing}",
            details=[f"declared features: {sorted(features.keys())}"],
            fix_hint="A LeRobot dataset must declare 'action' and 'observation.state'. "
                     "See lerobot.datasets.feature_utils for the canonical schema.",
        )
    details = [f"action shape: {features['action'].get('shape', '?')}"]
    details.append(f"observation.state shape: {features['observation.state'].get('shape', '?')}")
    if image_keys:
        details.append(f"image features: {image_keys}")
    else:
        details.append("no observation.images.* features (state-only dataset)")
    return CheckResult(
        check_id="dataset.schema-completeness",
        decision=Decision.OK,
        summary=f"schema declares action + observation.state ({len(image_keys)} image features)",
        details=details,
    )


@register_dataset_check
def check_data_files_present(ctx: DatasetContext) -> CheckResult:
    """data/chunk-NNN/file-NNN.parquet must exist."""
    if not ctx.data_files:
        return CheckResult(
            check_id="dataset.data-files-present",
            decision=Decision.BLOCKER,
            summary="no data/chunk-*/file-*.parquet files found",
            details=[f"checked: {ctx.root / 'data'}"],
            fix_hint="Confirm dataset layout — data/ must contain chunked parquet files.",
        )
    return CheckResult(
        check_id="dataset.data-files-present",
        decision=Decision.OK,
        summary=f"{len(ctx.data_files)} data parquet file(s) present",
    )


@register_dataset_check
def check_shape_consistency(ctx: DatasetContext) -> CheckResult:
    """Per-row action shape in the first data parquet must match info.json action shape."""
    if not ctx.info or not ctx.data_files:
        return CheckResult(
            check_id="dataset.shape-consistency",
            decision=Decision.SKIPPED,
            summary="info.json or data files missing; cannot cross-check shapes",
        )
    if not _HAS_PYARROW:
        return CheckResult(
            check_id="dataset.shape-consistency",
            decision=Decision.SKIPPED,
            summary="pyarrow not installed; install tether-vla[validation] to enable",
        )
    declared_shape = ctx.info["features"]["action"].get("shape", [])
    declared_dim = declared_shape[0] if declared_shape else None
    table, err = _load_parquet_table(ctx.data_files[0])
    if err or table is None:
        return CheckResult(
            check_id="dataset.shape-consistency",
            decision=Decision.BLOCKER,
            summary=f"cannot read first data parquet: {err}",
            details=[str(ctx.data_files[0])],
        )
    if "action" not in table.column_names:
        return CheckResult(
            check_id="dataset.shape-consistency",
            decision=Decision.BLOCKER,
            summary="data parquet missing 'action' column",
            details=[f"columns: {table.column_names}"],
        )
    sample = table["action"][0].as_py()
    sample_dim = len(sample) if isinstance(sample, list) else None
    if sample_dim is None:
        return CheckResult(
            check_id="dataset.shape-consistency",
            decision=Decision.WARN,
            summary="action column is not a list/array per row",
            details=[f"first value type: {type(sample).__name__}"],
        )
    if declared_dim is not None and declared_dim != sample_dim:
        return CheckResult(
            check_id="dataset.shape-consistency",
            decision=Decision.BLOCKER,
            summary=f"action dim mismatch: info.json declares {declared_dim}, parquet row 0 has {sample_dim}",
            details=[
                f"file: {ctx.data_files[0]}",
                "Re-export the dataset; do not finetune until this is fixed.",
            ],
            fix_hint="Action-dim mismatch is the #1 cause of mid-training crashes "
                     "(LeRobot Issue #2531). Fix at the data layer, not the model layer.",
        )
    return CheckResult(
        check_id="dataset.shape-consistency",
        decision=Decision.OK,
        summary=f"action dim consistent: declared={declared_dim}, parquet row 0={sample_dim}",
    )


@register_dataset_check
def check_action_finite(ctx: DatasetContext) -> CheckResult:
    """No NaN/Inf in actions across the first data parquet (sampled fast)."""
    if not ctx.data_files:
        return CheckResult(
            check_id="dataset.action-finite",
            decision=Decision.SKIPPED,
            summary="no data files; skipped",
        )
    if not _HAS_PYARROW:
        return CheckResult(
            check_id="dataset.action-finite",
            decision=Decision.SKIPPED,
            summary="pyarrow not installed; install tether-vla[validation] to enable",
        )
    table, err = _load_parquet_table(ctx.data_files[0])
    if err or table is None:
        return CheckResult(
            check_id="dataset.action-finite",
            decision=Decision.SKIPPED,
            summary=f"cannot read first data parquet: {err}",
        )
    if "action" not in table.column_names:
        return CheckResult(
            check_id="dataset.action-finite",
            decision=Decision.SKIPPED,
            summary="no 'action' column; skipped",
        )
    import math

    actions = table["action"].to_pylist()
    n_nan = sum(1 for row in actions if any(_v is None or (isinstance(_v, float) and not math.isfinite(_v)) for _v in (row or [])))
    if n_nan > 0:
        return CheckResult(
            check_id="dataset.action-finite",
            decision=Decision.BLOCKER,
            summary=f"{n_nan}/{len(actions)} action rows in first parquet contain NaN/Inf/None",
            details=[f"file: {ctx.data_files[0]}"],
            fix_hint="NaN actions cause silent training divergence. Drop or repair affected episodes.",
        )
    return CheckResult(
        check_id="dataset.action-finite",
        decision=Decision.OK,
        summary=f"{len(actions)} action rows in first parquet are finite",
    )


@register_dataset_check
def check_embodiment_action_dim_match(ctx: DatasetContext) -> CheckResult:
    """If --embodiment passed, dataset action_dim must match embodiment config."""
    if ctx.embodiment_config is None:
        return CheckResult(
            check_id="dataset.embodiment-action-dim-match",
            decision=Decision.SKIPPED,
            summary="--embodiment not passed; skipped",
        )
    if not ctx.info:
        return CheckResult(
            check_id="dataset.embodiment-action-dim-match",
            decision=Decision.SKIPPED,
            summary="info.json not loaded; cannot compare",
        )
    declared_shape = ctx.info["features"]["action"].get("shape", [])
    declared_dim = declared_shape[0] if declared_shape else None
    emb_dim = getattr(ctx.embodiment_config, "action_dim", None)
    if declared_dim is None or emb_dim is None:
        return CheckResult(
            check_id="dataset.embodiment-action-dim-match",
            decision=Decision.WARN,
            summary=f"could not extract dims (declared={declared_dim}, embodiment={emb_dim})",
        )
    if declared_dim != emb_dim:
        emb_name = getattr(ctx.embodiment_config, "embodiment", "<unknown>")
        return CheckResult(
            check_id="dataset.embodiment-action-dim-match",
            decision=Decision.BLOCKER,
            summary=f"embodiment '{emb_name}' has action_dim={emb_dim}; dataset has action_dim={declared_dim}",
            fix_hint=(
                "Either retrain a student sized for this embodiment, or re-export the "
                "dataset with the matching action shape, or pass a different --embodiment."
            ),
        )
    return CheckResult(
        check_id="dataset.embodiment-action-dim-match",
        decision=Decision.OK,
        summary=f"embodiment action_dim={emb_dim} matches dataset",
    )


@register_dataset_check
def check_episode_count_matches(ctx: DatasetContext) -> CheckResult:
    """info.json total_episodes vs episodes-meta parquet row count."""
    if not ctx.info:
        return CheckResult(
            check_id="dataset.episode-count-matches",
            decision=Decision.SKIPPED,
            summary="info.json not loaded; cannot cross-check episode count",
        )
    declared = ctx.info.get("total_episodes")
    if declared is None:
        return CheckResult(
            check_id="dataset.episode-count-matches",
            decision=Decision.WARN,
            summary="info.json missing total_episodes",
        )
    if not _HAS_PYARROW:
        return CheckResult(
            check_id="dataset.episode-count-matches",
            decision=Decision.SKIPPED,
            summary="pyarrow not installed; cannot count episode rows",
        )
    if not ctx.episode_meta_files:
        return CheckResult(
            check_id="dataset.episode-count-matches",
            decision=Decision.WARN,
            summary="no meta/episodes/chunk-*/file-*.parquet files found",
            details=[f"info.json declares {declared} episode(s)"],
        )
    actual = 0
    for p in ctx.episode_meta_files:
        table, err = _load_parquet_table(p)
        if err or table is None:
            return CheckResult(
                check_id="dataset.episode-count-matches",
                decision=Decision.WARN,
                summary=f"cannot read episode metadata parquet: {err}",
                details=[str(p)],
            )
        actual += table.num_rows
    if actual != declared:
        return CheckResult(
            check_id="dataset.episode-count-matches",
            decision=Decision.WARN,
            summary=f"episode count mismatch: info.json={declared}, episodes parquet={actual}",
            fix_hint="Re-run lerobot dataset stats to refresh info.json totals, OR check if some episode files are missing.",
        )
    return CheckResult(
        check_id="dataset.episode-count-matches",
        decision=Decision.OK,
        summary=f"episode count consistent: {actual}",
    )


@register_dataset_check
def check_timing_monotonic(ctx: DatasetContext) -> CheckResult:
    """Within the first data parquet, timestamp is strictly non-decreasing."""
    if not ctx.data_files:
        return CheckResult(
            check_id="dataset.timing-monotonic",
            decision=Decision.SKIPPED,
            summary="no data files; skipped",
        )
    if not _HAS_PYARROW:
        return CheckResult(
            check_id="dataset.timing-monotonic",
            decision=Decision.SKIPPED,
            summary="pyarrow not installed; cannot inspect timestamps",
        )
    table, err = _load_parquet_table(ctx.data_files[0])
    if err or table is None:
        return CheckResult(
            check_id="dataset.timing-monotonic",
            decision=Decision.SKIPPED,
            summary=f"cannot read first data parquet: {err}",
        )
    if "timestamp" not in table.column_names:
        return CheckResult(
            check_id="dataset.timing-monotonic",
            decision=Decision.SKIPPED,
            summary="no 'timestamp' column; skipped",
        )
    ts = table["timestamp"].to_pylist()
    if "episode_index" in table.column_names:
        eps = table["episode_index"].to_pylist()
    else:
        eps = [0] * len(ts)
    n_violations = 0
    last_t: float | None = None
    last_ep: int | None = None
    for t, ep in zip(ts, eps):
        if last_t is not None and ep == last_ep and t < last_t:
            n_violations += 1
        last_t, last_ep = t, ep
    if n_violations > 0:
        return CheckResult(
            check_id="dataset.timing-monotonic",
            decision=Decision.WARN,
            summary=f"{n_violations} non-monotonic timestamp transition(s) within episodes",
            fix_hint="Non-monotonic timestamps can cause RTC chunk-boundary jumps. "
                     "Re-export with sorted timestamps.",
        )
    return CheckResult(
        check_id="dataset.timing-monotonic",
        decision=Decision.OK,
        summary=f"timestamps monotonic across {len(ts)} frame(s)",
    )


# --------------------------------------------------------------------------
# Runner + reporters
# --------------------------------------------------------------------------

def _load_info(root: Path) -> tuple[dict[str, Any] | None, str]:
    """Load meta/info.json. Returns (parsed_dict, error_str)."""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None, f"missing file: {info_path}"
    try:
        return json.loads(info_path.read_text()), ""
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def run_all_checks(
    dataset_root: str | Path,
    embodiment_config: Any = None,
    strict: bool = False,
) -> list[CheckResult]:
    """Run every registered check against the dataset. Returns ordered results.

    Pass embodiment_config (an EmbodimentConfig from src/tether/embodiments/) to
    enable cross-checks between dataset action_dim and embodiment action_dim.
    """
    root = Path(dataset_root)
    info, err = _load_info(root)
    data_files, episode_files = _gather_files(root)
    ctx = DatasetContext(
        root=root,
        info=info,
        info_load_error=err,
        embodiment_config=embodiment_config,
        strict=strict,
        data_files=data_files,
        episode_meta_files=episode_files,
    )
    results: list[CheckResult] = []
    for fn in REGISTERED_CHECKS:
        try:
            results.append(fn(ctx))
        except Exception as e:  # noqa: BLE001
            logger.exception("check %s raised", fn.__name__)
            results.append(
                CheckResult(
                    check_id=f"dataset.{fn.__name__}",
                    decision=Decision.BLOCKER,
                    summary=f"check raised exception: {type(e).__name__}: {e}",
                )
            )
    return results


def overall_decision(results: list[CheckResult], strict: bool = False) -> Decision:
    """Reduce list of results to one decision.

    blocker if any blocker; warn if any warn; ok otherwise. In strict mode,
    warns escalate to blockers.
    """
    has_blocker = any(r.decision == Decision.BLOCKER for r in results)
    has_warn = any(r.decision == Decision.WARN for r in results)
    if has_blocker:
        return Decision.BLOCKER
    if has_warn:
        return Decision.BLOCKER if strict else Decision.WARN
    return Decision.OK


def format_human(results: list[CheckResult], dataset_root: str | Path = "") -> str:
    """Plain-text report with color-free status markers (CI-friendly)."""
    lines = [f"Dataset validation report: {dataset_root}"]
    counts = {d: 0 for d in Decision}
    for r in results:
        counts[r.decision] = counts.get(r.decision, 0) + 1
    summary = ", ".join(f"{counts[d]} {d.value}" for d in Decision if counts[d] > 0)
    lines.append(f"  {len(results)} checks: {summary}")
    lines.append("")
    for r in results:
        marker = {
            Decision.OK: "[OK]",
            Decision.WARN: "[WARN]",
            Decision.BLOCKER: "[BLOCKER]",
            Decision.SKIPPED: "[SKIPPED]",
        }[r.decision]
        lines.append(f"  {marker} {r.check_id}")
        lines.append(f"      {r.summary}")
        for d in r.details:
            lines.append(f"        - {d}")
        if r.fix_hint:
            lines.append(f"      fix: {r.fix_hint}")
    return "\n".join(lines)


def format_json(
    results: list[CheckResult],
    dataset_root: str | Path = "",
    decision: Decision | None = None,
) -> str:
    """Machine-readable JSON. Stable schema for CI integrations."""
    if decision is None:
        decision = overall_decision(results)
    return json.dumps(
        {
            "dataset_root": str(dataset_root),
            "decision": decision.value,
            "n_checks": len(results),
            "results": [
                {
                    "check_id": r.check_id,
                    "decision": r.decision.value,
                    "summary": r.summary,
                    "details": r.details,
                    "fix_hint": r.fix_hint,
                }
                for r in results
            ],
        },
        indent=2,
    )
