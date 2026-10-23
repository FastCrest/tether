"""Opt-in mixed FP16 ONNX conversion with explicit numerical validation.

Conversion reduces floating weight storage; it does not establish TensorRT build
memory, target latency, physical safety, or the accuracy of the FP32 reference.
The supported export-directory path preserves FP32 I/O and publishes only after
all supplied validation cases meet the caller's explicit tolerances.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Legacy research-only dtype-repair helper. The supported conversion pipeline
# never uses these guesses; graph/checker/session errors stop publication.
MIXED_DTYPE_OPS = frozenset({
    "Mul", "MatMul", "Add", "Sub", "Div", "Pow", "Gemm",
    "Concat", "Where",
})

# ONNX TensorProto dtype IDs (avoiding the import for lightweight usage).
_DT_FLOAT = 1       # TensorProto.FLOAT
_DT_FLOAT16 = 10    # TensorProto.FLOAT16
_DT_BFLOAT16 = 16   # TensorProto.BFLOAT16
_DT_INT64 = 7
_FLOATING_DTYPES = frozenset({_DT_FLOAT, _DT_FLOAT16, _DT_BFLOAT16})


# Keep numerically sensitive reductions/nonlinear operations at FP32 boundaries.
# The converter's own unsupported-op list is added at conversion time. Invalid
# boundaries are errors: the supported path never guesses operand dtypes.
FP16_OP_BLOCKLIST: tuple[str, ...] = (
    "LayerNormalization", "Softmax", "Pow", "ReduceMean", "Sqrt", "Reciprocal",
    "Add", "Div", "Sin", "Cos", "Unsqueeze", "Cast",
)


def estimate_fp16_size_bytes(fp32_total_bytes: int) -> int:
    """Rough estimate of post-conversion size.

    This legacy heuristic assumes weights dominate storage and adds 5%
    overhead. Models with retained FP32 weights or substantial graph metadata
    can differ. It is not a RAM estimate or evidence of target memory fit.
    The supported conversion reports measured file sizes instead.
    """
    halved = fp32_total_bytes // 2
    overhead = halved // 20  # 5% slop for metadata + blocklist residuals
    return halved + overhead


def parity_gate(
    max_abs_diff: float,
    cos_sim: float,
    *,
    max_abs_threshold: float = 1e-2,
    cos_threshold: float = 0.999,
) -> dict[str, Any]:
    """Apply the FP16-vs-FP32 parity gate.

    Returns a verdict dict compatible with VERIFICATION.md seeding.
    PASS requires cos >= cos_threshold AND max_abs <= max_abs_threshold.
    Either failure flips to FAIL with a human-readable reason.

    These are numerical tolerances, not a robot safety guarantee. The supported
    conversion API requires the caller to choose thresholds explicitly.
    """
    _validate_thresholds(max_abs_threshold, cos_threshold)
    cos_ok = math.isfinite(cos_sim) and -1 <= cos_sim <= 1 and cos_sim >= cos_threshold
    maxabs_ok = math.isfinite(max_abs_diff) and 0 <= max_abs_diff <= max_abs_threshold
    passed = cos_ok and maxabs_ok

    reasons: list[str] = []
    if not cos_ok:
        reasons.append(
            f"cos_sim {cos_sim:.6f} below threshold {cos_threshold}"
        )
    if not maxabs_ok:
        reasons.append(
            f"max_abs_diff {max_abs_diff:.2e} above threshold "
            f"{max_abs_threshold:.0e}"
        )

    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "cos_sim": cos_sim,
        "max_abs_diff": max_abs_diff,
        "cos_threshold": cos_threshold,
        "max_abs_threshold": max_abs_threshold,
        "reasons": reasons,
    }


def _remove_stale_external_data(dst: Path) -> list[Path]:
    """Delete external-data leftovers from a PRIOR conversion of *this* model.

    ``onnx.save`` writes the FP16 weights to ``{dst.stem}.bin`` next to ``dst``.
    A leftover ``.bin``/``.data`` from an earlier run at the same path would sit
    alongside the new one and inflate our size accounting, so we clear it first.

    Scope is an exact allowlist of names written by current and legacy saves:
    ``{dst.stem}.bin``, ``{dst.stem}.data``, and ``{dst.name}.data``. The
    previous implementation globbed external-data files and could destroy
    another model's weights in a shared export directory. ``dst`` itself (the
    ``.onnx``) is never touched.

    Returns the list of paths actually removed (for logging / tests).
    """
    removed: list[Path] = []
    candidates = (
        dst.with_suffix(".bin"),
        dst.with_suffix(".data"),
        Path(f"{dst}.data"),
    )
    for old in candidates:
        try:
            old.unlink()
            removed.append(old)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return removed


def _validate_thresholds(max_abs_threshold: float, cos_threshold: float) -> None:
    for value, name in ((max_abs_threshold, "max_abs_threshold"), (cos_threshold, "cos_threshold")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if max_abs_threshold < 0 or not -1 <= cos_threshold <= 1:
        raise ValueError("max_abs_threshold must be nonnegative; cos_threshold must be in [-1, 1]")


def _external_paths(onnx_path: Path) -> list[Path]:
    """Read exact referenced weight locations, including nested graph tensors."""
    import onnx

    root = onnx_path.parent.resolve()
    model = onnx.load(str(onnx_path), load_external_data=False)
    paths: set[Path] = set()

    def tensor(t: Any) -> None:
        if t.data_location != onnx.TensorProto.EXTERNAL:
            return
        location = next((v.value for v in t.external_data if v.key == "location"), None)
        if not location:
            raise ValueError(f"External tensor {t.name!r} has no location")
        candidate = (root / location).resolve()
        if Path(location).is_absolute() or not candidate.is_relative_to(root):
            raise ValueError(f"External tensor {t.name!r} resolves outside the model directory")
        if not candidate.is_file():
            raise ValueError(f"External tensor data is missing: {location}")
        paths.add(candidate)

    def graph(g: Any) -> None:
        for t in g.initializer:
            tensor(t)
        for t in g.sparse_initializer:
            tensor(t.values)
            tensor(t.indices)
        for node in g.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    tensor(attr.t)
                elif attr.type == onnx.AttributeProto.TENSORS:
                    for t in attr.tensors:
                        tensor(t)
                elif attr.type == onnx.AttributeProto.GRAPH:
                    graph(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        graph(subgraph)

    graph(model.graph)
    return sorted(paths)


def _io_contract(model: Any) -> tuple[bytes, bytes]:
    return (
        b"".join(value.SerializeToString() for value in model.graph.input),
        b"".join(value.SerializeToString() for value in model.graph.output),
    )


def _cpu_session(path: Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _deduplicate_conversion_casts(model: Any) -> int:
    """Reuse only identical Cast definitions emitted for a shared blocked input.

    ORT's converter emits the same named Cast once per blocked consumer. Keeping
    one producer is an SSA repair, not dtype inference: conflicting definitions
    and duplicate non-Cast producers are errors, never guessed replacements.
    """
    import onnx

    removed = 0

    def graph(g: Any) -> None:
        nonlocal removed
        producers: dict[str, Any] = {}
        kept = []
        reserved = {value.name for value in g.input} | {value.name for value in g.initializer}
        for node in g.node:
            outputs = [name for name in node.output if name]
            duplicates = [name for name in outputs if name in producers or name in reserved]
            if duplicates:
                prior = producers.get(outputs[0]) if len(outputs) == 1 else None
                def signature(value: Any) -> bytes | None:
                    if (value is None or value.op_type != "Cast" or value.domain not in ("", "ai.onnx")
                            or len(value.input) != 1 or len(value.output) != 1
                            or len(value.attribute) != 1 or value.attribute[0].name != "to"
                            or value.attribute[0].type != onnx.AttributeProto.INT):
                        return None
                    copied = onnx.NodeProto()
                    copied.CopyFrom(value)
                    copied.ClearField("name")
                    return copied.SerializeToString()
                prior_signature = signature(prior)
                if prior_signature is None or prior_signature != signature(node):
                    raise ValueError(f"Conflicting ONNX output definition after FP16 conversion: {duplicates}")
                removed += 1
                continue
            for name in outputs:
                producers[name] = node
            kept.append(node)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    graph(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for nested in attr.graphs:
                        graph(nested)
        del g.node[:]
        g.node.extend(kept)
        declarations: dict[str, Any] = {}
        kept_values = []
        for value in g.value_info:
            prior = declarations.get(value.name)
            if prior is not None:
                if prior.SerializeToString() != value.SerializeToString():
                    raise ValueError(f"Conflicting ONNX value_info after FP16 conversion: {value.name}")
                continue
            declarations[value.name] = value
            kept_values.append(value)
        del g.value_info[:]
        g.value_info.extend(kept_values)

    graph(model.graph)
    return removed


def _preserve_fp32_island_edges(
    model: Any, original_names: set[str], blocked_ops: set[str],
    preserved_edges: list[dict[str, str]] | None = None,
) -> int:
    """Keep explicitly FP32 operations connected without half-rounding edges.

    ORT inserts a half-output Cast followed by a full-input Cast even between
    blocked operations. Bypass that generated round trip only when its complete
    type declarations establish FP32 -> FP16 -> FP32 from a blocked producer.
    Parallel FP16 consumers retain their original half Cast.
    """
    import onnx

    graph = model.graph
    types = {v.name: v.type.tensor_type.elem_type for v in [*graph.value_info, *graph.input, *graph.output]}
    types.update({v.name: v.data_type for v in graph.initializer})
    producers = {name: node for node in graph.node for name in node.output if name}

    def generated_cast(node: Any, dtype: int) -> bool:
        return (node is not None and node.name not in original_names and node.op_type == "Cast"
                and node.domain in ("", "ai.onnx") and len(node.input) == len(node.output) == 1
                and len(node.attribute) == 1 and node.attribute[0].name == "to"
                and node.attribute[0].type == onnx.AttributeProto.INT and node.attribute[0].i == dtype)

    preserved = 0
    for cast32 in graph.node:
        if not generated_cast(cast32, onnx.TensorProto.FLOAT):
            continue
        cast16 = producers.get(cast32.input[0])
        if cast16 is None or not generated_cast(cast16, onnx.TensorProto.FLOAT16):
            continue
        producer = producers.get(cast16.input[0])
        if producer is None or producer.op_type not in blocked_ops:
            continue
        if (types.get(cast16.input[0]), types.get(cast16.output[0]), types.get(cast32.output[0])) != (1, 10, 1):
            continue
        if preserved_edges is not None:
            preserved_edges.append({
                "cast_node": cast32.name, "fp32_producer": producer.name,
                "producer_op": producer.op_type, "original_fp16_edge": cast32.input[0],
                "preserved_fp32_edge": cast16.input[0],
            })
        cast32.input[0] = cast16.input[0]
        preserved += 1
    return preserved


def convert_fp32_to_fp16(
    fp32_onnx_path: str | Path,
    fp16_onnx_path: str | Path,
    *,
    keep_io_types: bool = True,
    op_block_list: tuple[str, ...] | None = None,
    min_positive_val: float = 5.960464477539063e-8,
    max_finite_val: float = 65504.0,
) -> dict[str, Any]:
    """Convert into new files and validate the graph/session before publishing.

    Existing files are never replaced. The original and all its external weights
    remain untouched. This low-level helper establishes loadability only; use
    ``convert_monolithic_export_fp16`` for a validated serving export and receipt.
    """
    try:
        import onnx
        from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16
        from onnxruntime.transformers.onnx_model import OnnxModel
    except ImportError as exc:
        raise ImportError("FP16 conversion requires onnx and onnxruntime with its transformers conversion tools") from exc
    if not keep_io_types:
        raise ValueError("Supported conversion requires keep_io_types=True; the serving contract must not change")
    src, dst = Path(fp32_onnx_path).resolve(), Path(fp16_onnx_path).absolute()
    if src == dst.resolve():
        raise ValueError("FP16 output must be separate from the FP32 source")
    if dst.exists() or dst.is_symlink() or dst.with_suffix(".bin").exists():
        raise FileExistsError(f"Conversion destination already exists: {dst}")
    source_paths = [src, *_external_paths(src)]
    if dst.with_suffix(".bin").resolve() in source_paths:
        raise ValueError("FP16 output would replace source external data")
    dst.parent.mkdir(parents=True, exist_ok=True)
    blocklist = sorted(set(DEFAULT_OP_BLOCK_LIST).union(
        op_block_list if op_block_list is not None else FP16_OP_BLOCKLIST,
    ))
    model = onnx.load(str(src), load_external_data=True)
    contract = _io_contract(model)
    original_node_names = {node.name for node in model.graph.node}
    converted = convert_float_to_float16(
        model, keep_io_types=True, op_block_list=blocklist,
        min_positive_val=min_positive_val, max_finite_val=max_finite_val,
        # Large in-memory protos exceed shape inference's protobuf size limit.
        # Never compensate by changing I/O or guessing casts: validation refuses
        # any graph the upstream converter cannot represent correctly.
        disable_shape_infer=sum(p.stat().st_size for p in source_paths) > 1_800_000_000,
    )
    duplicate_casts_removed = _deduplicate_conversion_casts(converted)
    fp32_edges: list[dict[str, str]] = []
    fp32_edges_preserved = _preserve_fp32_island_edges(converted, original_node_names, set(blocklist), fp32_edges)
    OnnxModel(converted).topological_sort()
    if _io_contract(converted) != contract:
        raise ValueError("FP16 converter changed the source graph I/O contract")
    del model
    with tempfile.TemporaryDirectory(prefix=".tether-fp16-", dir=dst.parent) as temporary:
        staged = Path(temporary) / dst.name
        onnx.save_model(
            converted, str(staged), save_as_external_data=True,
            all_tensors_to_one_file=True, location=f"{dst.stem}.bin", size_threshold=1024,
        )
        del converted
        onnx.checker.check_model(str(staged))
        session = _cpu_session(staged)
        del session
        staged_weights = _external_paths(staged)
        published: list[Path] = []
        try:
            # Link without replacement; graph appears only after its weights.
            for path in [*staged_weights, staged]:
                target = dst.parent / path.name
                os.link(path, target)
                published.append(target)
        except BaseException:
            for path in reversed(published):
                path.unlink()
            raise
    src_bytes = sum(p.stat().st_size for p in source_paths)
    dst_bytes = _size_with_external(dst)
    return {
        "src_path": str(src), "dst_path": str(dst),
        "src_bytes": src_bytes, "dst_bytes": dst_bytes,
        "reduction_ratio": 1 - dst_bytes / src_bytes,
        "op_block_list": blocklist, "cast_nodes_inserted": 0,
        "keep_io_types": True, "numerical_parity": "not_measured",
        "duplicate_casts_removed": duplicate_casts_removed,
        "fp32_island_edges_preserved": fp32_edges_preserved,
        "fp32_island_edges": fp32_edges,
    }


class FP16ParityError(ValueError):
    """Validation failed; the destination was not published."""

    def __init__(self, receipt: dict[str, Any]):
        self.receipt = receipt
        super().__init__("FP16 conversion failed numerical parity; destination was not published")


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _conversion_versions() -> dict[str, str]:
    versions = {name: version(name) for name in ("onnx", "numpy")}
    for name in ("onnxruntime", "onnxruntime-gpu"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def _artifact_identities(root: Path, paths: Sequence[str]) -> dict[str, Any]:
    return {path: _file_identity(root / path) for path in sorted(set(paths))}


def _case_identity(case: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    result = {}
    for name in sorted(case):
        if not isinstance(case[name], np.ndarray):
            raise TypeError(f"Validation input {name!r} must be a NumPy array; implicit casts are not supported")
        array = case[name]
        if array.dtype.hasobject or array.dtype.kind not in "biuf":
            raise ValueError(f"Validation input {name!r} must be a numeric or bool array")
        if not np.isfinite(array).all():
            raise ValueError(f"Validation input {name!r} contains non-finite values")
        result[name] = {
            "dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        }
    return result


def _output_parity(reference: Any, candidate: Any, *, max_abs: float, cosine: float) -> dict[str, Any]:
    import numpy as np

    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("Conversion changed output shape or dtype")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {"passed": False, "reason": "non-finite output"}
    if reference.dtype.kind not in "f":
        return {"passed": bool(np.array_equal(reference, candidate)), "comparison": "exact"}
    ref, cand = reference.astype(np.float64).ravel(), candidate.astype(np.float64).ravel()
    if not ref.size:
        raise ValueError("Cannot establish numerical parity on an empty output")
    with np.errstate(over="ignore", invalid="ignore"):
        max_diff = float(np.max(np.abs(ref - cand)))
    if not math.isfinite(max_diff):
        return {"passed": False, "reason": "non-finite numerical difference"}
    # Normalize before dot product to avoid overflow on finite large outputs.
    ref_scale, cand_scale = float(np.max(np.abs(ref))), float(np.max(np.abs(cand)))
    if ref_scale == 0 or cand_scale == 0:
        cos = 1.0 if ref_scale == cand_scale else 0.0
    else:
        a, b = ref / ref_scale, cand / cand_scale
        cos = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    return parity_gate(max_diff, cos, max_abs_threshold=max_abs, cos_threshold=cosine)


def convert_monolithic_export_fp16(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_cases: Sequence[Mapping[str, Any]],
    max_abs_threshold: float,
    cos_threshold: float,
) -> dict[str, Any]:
    """Create a separate serving export only after measured FP32/FP16 parity.

    Each case maps the exact ONNX input names to correctly typed arrays. No
    input casts, random input generation or tolerance guesses are performed.
    A passing receipt applies only to these cases on CPU, not unseen inputs,
    TensorRT, target memory fit, or physical execution.
    """
    from tether.export_config import build_producer_config, load_tether_config, write_tether_config

    _validate_thresholds(max_abs_threshold, cos_threshold)
    if not validation_cases:
        raise ValueError("At least one explicit validation case is required")
    src, dst = Path(source_dir).resolve(), Path(output_dir).absolute()
    if src == dst.resolve() or dst.resolve().is_relative_to(src):
        raise ValueError("Output must be a separate directory outside the FP32 export")
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Output already exists: {dst}")
    config = load_tether_config(src)
    models = [a["path"] for a in config["artifacts"] if a["role"] == "model"]
    if config["export_kind"] != "monolithic_onnx" or models != ["model.onnx"]:
        raise ValueError("FP16 conversion supports a monolithic export containing only model.onnx")
    declared = {a["path"] for a in config["artifacts"]}
    external = _external_paths(src / "model.onnx")
    if any(p.relative_to(src).as_posix() not in declared for p in external):
        raise ValueError("Source manifest omits ONNX external weights")
    tokenizer_path = config.get("tokenizer_path")
    if tokenizer_path:
        token_dir = (src / tokenizer_path).resolve()
        if Path(tokenizer_path).is_absolute() or not token_dir.is_relative_to(src):
            raise ValueError("Tokenizer path must stay inside the source export")
        token_artifacts = [a["path"] for a in config["artifacts"] if a["role"] == "tokenizer"]
        if config.get("tokenizer_bundled") and not any((src / name).resolve().is_relative_to(token_dir) for name in token_artifacts):
            raise ValueError("Bundled tokenizer has no declared tokenizer artifacts")
    source_identity = _artifact_identities(src, ["tether_config.json", *declared])
    # Snapshot before hashing: both sessions and the receipt must use precisely
    # these arrays even if a caller changes its buffers during conversion.
    import numpy as np

    cases: list[dict[str, Any]] = []
    for case in validation_cases:
        snapshot = {}
        for name, value in case.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Validation input {name!r} must be a NumPy array; implicit casts are not supported")
            snapshot[name] = value.copy()
        cases.append(snapshot)
    case_ids = [_case_identity(case) for case in cases]
    reference_session = _cpu_session(src / "model.onnx")
    input_names = {v.name for v in reference_session.get_inputs()}
    for case in cases:
        if set(case) != input_names:
            raise ValueError(f"Validation inputs must match ONNX input names exactly: {sorted(input_names)}")
    output_names = [v.name for v in reference_session.get_outputs()]
    reference_outputs = [reference_session.run(output_names, case) for case in cases]
    del reference_session
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tether-export-fp16-", dir=dst.parent) as temporary:
        staged = Path(temporary) / "export"
        staged.mkdir()
        conversion = convert_fp32_to_fp16(src / "model.onnx", staged / "model.onnx")
        if conversion["dst_bytes"] >= conversion["src_bytes"]:
            raise ValueError("FP16 conversion did not reduce model storage; destination was not published")
        optional: list[tuple[str, str]] = []
        for artifact in config["artifacts"]:
            if artifact["role"] != "tokenizer":
                continue
            relative = artifact["path"]
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / relative, target)
            optional.append((relative, "tokenizer"))
        # Preserve serving semantics, not old parity/certificate/provider claims.
        fields = (
            "target", "chunk_size", "action_chunk_size", "max_state_dim", "action_units",
            "tokenizer_ref", "tokenizer_path", "tokenizer_bundled", "tokenizer_error",
        )
        metadata = {key: config[key] for key in fields if key in config}
        metadata["precision"] = "mixed_float16"
        metadata["precision_conversion"] = {"receipt": "precision-conversion.json", "validation_cases": len(cases)}
        candidate_config = build_producer_config(
            staged, producer="monolithic", model_id=config["model_id"], model_type=config["model_type"],
            action_dim=config["action_dim"], num_denoising_steps=config["num_denoising_steps"],
            opset=config["opset"], optional_artifacts=optional, metadata=metadata,
        )
        if candidate_config["io_contract"] != config["io_contract"]:
            raise ValueError("Source config does not match the preserved ONNX I/O contract")
        for artifact in candidate_config["artifacts"]:
            artifact["sha256"] = _file_identity(staged / artifact["path"])["sha256"]
        write_tether_config(staged, candidate_config)
        candidate_session = _cpu_session(staged / "model.onnx")
        measurements: list[dict[str, Any]] = []
        for index, case in enumerate(cases):
            actual = candidate_session.run(output_names, case)
            outputs = {
                name: _output_parity(ref, got, max_abs=max_abs_threshold, cosine=cos_threshold)
                for name, ref, got in zip(output_names, reference_outputs[index], actual, strict=True)
            }
            measurements.append({"case": index, "inputs": case_ids[index], "outputs": outputs})
        del candidate_session
        passed = all(output["passed"] for case in measurements for output in case["outputs"].values())
        receipt = {
            "schema_version": 1, "operation": "monolithic_mixed_fp16_conversion",
            "parity_verdict": "PASS" if passed else "FAIL", "provider": "CPUExecutionProvider",
            "source_artifacts": source_identity,
            "candidate_artifacts": _artifact_identities(staged, ["tether_config.json", *[a["path"] for a in candidate_config["artifacts"]]]),
            "max_abs_threshold": max_abs_threshold, "cos_threshold": cos_threshold,
            "cases": measurements, "keep_io_types": True, "op_block_list": conversion["op_block_list"],
            "model_bytes_before": conversion["src_bytes"], "model_bytes_after": conversion["dst_bytes"],
            "float16_min_positive_val": 5.960464477539063e-8, "float16_max_finite_val": 65504.0,
            "converter": "onnxruntime.transformers.float16",
            "duplicate_casts_removed": conversion["duplicate_casts_removed"],
            "fp32_island_edges_preserved": conversion["fp32_island_edges_preserved"],
            "fp32_island_edges": conversion["fp32_island_edges"],
            "versions": _conversion_versions(),
            "limits": ["supplied inputs only", "FP32 reference accuracy not established", "target memory fit not measured", "TensorRT not qualified", "physical execution not qualified"],
        }
        if not passed:
            raise FP16ParityError(receipt)
        if _artifact_identities(src, list(source_identity)) != source_identity:
            raise ValueError("FP32 source changed during conversion; destination was not published")
        (staged / "precision-conversion.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Reserve the destination without replacing an existing directory. The
        # staged tree is complete; rename cannot expose half-written weights.
        dst.mkdir()
        try:
            os.replace(staged, dst)
        except BaseException:
            dst.rmdir()
            raise
    return receipt


def _build_dtype_map(model: Any) -> dict[str, int]:
    """Forward-propagate tensor dtypes through the graph.

    Returns a mapping from tensor name → ONNX TensorProto dtype int.
    Seeds from graph inputs + initializers, then propagates through
    nodes using simple per-op rules (output dtype = input dtype for
    most ops; Cast reads its `to` attribute; Shape/Size always emit
    int64; ConstantOfShape gets its dtype from the `value` tensor).
    """
    dtype_map: dict[str, int] = {}

    # Seed from graph inputs
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        if tt.elem_type:
            dtype_map[inp.name] = tt.elem_type

    # Seed from initializers
    for init in model.graph.initializer:
        dtype_map[init.name] = init.data_type

    # Sparse initializers (rare but legal)
    for sinit in getattr(model.graph, "sparse_initializer", []):
        dtype_map[sinit.values.name] = sinit.values.data_type

    # Propagate through nodes in order (graph is topologically sorted).
    for node in model.graph.node:
        out_dtype = _infer_node_output_dtype(node, dtype_map)
        if out_dtype is None:
            continue
        for out_name in node.output:
            if out_name:
                dtype_map[out_name] = out_dtype

    return dtype_map


def _infer_node_output_dtype(node: Any, dtype_map: dict[str, int]) -> int | None:
    """Return the output dtype of a node, or None if we can't determine."""
    op = node.op_type
    if op == "Cast":
        for attr in node.attribute:
            if attr.name == "to":
                return int(attr.i)
        return None
    if op == "Shape" or op == "Size":
        return _DT_INT64
    if op == "ConstantOfShape":
        for attr in node.attribute:
            if attr.name == "value" and attr.t.data_type:
                return int(attr.t.data_type)
        return _DT_FLOAT  # onnx default
    if op == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                return int(attr.t.data_type)
            if attr.name in ("value_float", "value_floats"):
                return _DT_FLOAT
            if attr.name in ("value_int", "value_ints"):
                return _DT_INT64
        return None
    # Default: output dtype = first input with a known dtype.
    for name in node.input:
        if name and name in dtype_map:
            return dtype_map[name]
    return None


def fix_fp16_dtype_mismatches(
    onnx_path: str | Path,
    target_dtype: int = _DT_FLOAT16,
) -> int:
    """Legacy research-only heuristic, unused by supported FP16 conversion.

    Guessed propagation can be wrong for arbitrary operators. This helper is
    retained for existing experimental scripts; it does not validate parity or
    make an artifact eligible for serving. It inserts Cast nodes at guessed
    FP32/FP16 operand mismatches.

    `convert_float_to_float16` doesn't always rewire Cast nodes at all
    boundaries — common failure modes on large graphs:
      * A Constant node emits FP32 but feeds a Mul whose other input got
        converted to FP16.
      * A graph input stays FP32 (when keep_io_types=True) but feeds
        downstream FP16 ops without a Cast.
      * An op in the user's blocklist stays FP32 but its output feeds
        a converted MatMul/Mul.

    We walk the graph, detect mismatches on MIXED_DTYPE_OPS, and insert
    Cast(to=target_dtype) on the offending FP32 operands.

    Modifies the model in place on disk. Returns the count of Cast nodes
    inserted. If 0, the model was already consistent and no write
    occurred.
    """
    try:
        import onnx
        from onnx import helper
    except ImportError as e:
        raise ImportError("requires `onnx`: pip install onnx") from e

    onnx_path = Path(onnx_path)
    # Load WITHOUT rehydrating external weight data. Graph surgery only
    # touches nodes + value_info; initializers stay as external-data
    # placeholders. This keeps the subsequent save tiny (just the proto,
    # not the 6-12 GB .bin file) — and avoids onnx.save's float_data
    # re-serialization path that silently writes FP16 weights as FP32.
    model = onnx.load(str(onnx_path), load_external_data=False)

    dtype_map = _build_dtype_map(model)
    new_nodes: list[Any] = []
    cast_counter = 0

    for node in model.graph.node:
        if node.op_type not in MIXED_DTYPE_OPS:
            new_nodes.append(node)
            continue

        # For Where, only inputs[1] and inputs[2] need consistent dtype
        # (inputs[0] is a bool mask). All other ops: check every input.
        if node.op_type == "Where":
            check_indices = [i for i in (1, 2) if i < len(node.input)]
        else:
            check_indices = list(range(len(node.input)))

        # Collect floating dtypes among checked inputs.
        float_dtypes: set[int] = set()
        for idx in check_indices:
            name = node.input[idx]
            if not name:
                continue
            dt = dtype_map.get(name)
            if dt in _FLOATING_DTYPES:
                float_dtypes.add(dt)

        if len(float_dtypes) <= 1:
            # Already consistent (or non-floating).
            new_nodes.append(node)
            continue

        # Mismatch — cast non-target floating inputs to target_dtype.
        new_input_names = list(node.input)
        for idx in check_indices:
            name = node.input[idx]
            if not name:
                continue
            dt = dtype_map.get(name)
            if dt not in _FLOATING_DTYPES or dt == target_dtype:
                continue
            cast_out = f"{name}_fix_fp16_{cast_counter}"
            cast_node = helper.make_node(
                "Cast",
                inputs=[name],
                outputs=[cast_out],
                name=f"cast_fix_fp16_{cast_counter}",
                to=target_dtype,
            )
            new_nodes.append(cast_node)
            new_input_names[idx] = cast_out
            dtype_map[cast_out] = target_dtype
            cast_counter += 1

        # Mutate the node's input list in place.
        del node.input[:]
        node.input.extend(new_input_names)
        new_nodes.append(node)

    if cast_counter == 0:
        logger.info("[fp16-fix] graph already dtype-consistent — no changes.")
        return 0

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    logger.info(
        "[fp16-fix] inserted %d Cast node(s); re-saving %s...",
        cast_counter, onnx_path,
    )
    # Save WITHOUT toggling external data — initializers were loaded as
    # external-data placeholders and stay that way. This writes only the
    # graph proto (tiny), leaving the existing .bin file on disk untouched.
    onnx.save(model, str(onnx_path))
    return cast_counter


def _size_with_external(onnx_path: Path) -> int:
    """Count the graph and each actually referenced external file once."""
    return sum(p.stat().st_size for p in [onnx_path, *_external_paths(onnx_path)])


__all__ = [
    "FP16_OP_BLOCKLIST",
    "FP16ParityError",
    "convert_fp32_to_fp16",
    "convert_monolithic_export_fp16",
    "estimate_fp16_size_bytes",
    "fix_fp16_dtype_mismatches",
    "parity_gate",
]
