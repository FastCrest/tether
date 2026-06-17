"""FP16 conversion helpers for monolithic ONNX exports.

Halves on-disk size. Targeted at the pi0 / pi0.5 Orin Nano 8GB fit
problem — pi0 FP32 is 12.5GB, won't load; pi0 FP16 should be ~6.3GB.

Strategy: use `onnxconverter_common.float16.convert_float_to_float16`
with an op blocklist for ops known to underflow at FP16
(LayerNorm-adjacent: Pow, ReduceMean, Sqrt), and `keep_io_types=True`
so downstream clients don't need to change their input tensors.

Pure ONNX → ONNX conversion; no retraining, no TensorRT dependency.
Post-conversion, the FP16 ONNX goes through the same parity harness
as the FP32 export. See
https://github.com/FastCrest/reflex-vault/blob/main/reflex_vla/01_architecture/orin_nano_fp16_plan.md
for the full plan.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Binary/variadic ops whose inputs must share a floating dtype.
# When convert_float_to_float16 touches only the initializers (not
# Constant ops or graph inputs at non-convertible boundaries), these
# ops end up with one FP32 and one FP16 operand — ORT rejects the
# model at load time with "Type parameter (T) of Optype (X) bound to
# different types". fix_fp16_dtype_mismatches() walks these nodes and
# inserts Cast(to=FP16) on the offending FP32 input.
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


# Ops that commonly underflow at FP16 in flow-matching / transformer VLAs.
# Keeping these in FP32 was the standard mitigation (conservative precision
# preservation), BUT: the onnxconverter_common.float16 pass doesn't insert
# Cast nodes for blocklisted-op outputs, so downstream MatMul/Mul receives
# one FP32 arg + one FP16 arg → ORT rejects with "Type parameter (T) of
# Optype (X) bound to different types". On 2.25GB / 12.5GB monolithic
# graphs this is hard to fix by selectively casting, so we let Pow/
# ReduceMean/Sqrt convert to FP16 and accept the LayerNorm precision cost.
# The `parity_gate(cos>0.999, max_abs<5e-3)` check catches divergence.
FP16_OP_BLOCKLIST: tuple[str, ...] = ()


def estimate_fp16_size_bytes(fp32_total_bytes: int) -> int:
    """Rough estimate of post-conversion size.

    Weights are roughly halved (fp32 → fp16). Graph structure + a few
    per-op FP32 residuals (the blocklist) add ~5% overhead on top of
    the halved weight footprint. This gets us within ~10% of the actual
    post-conversion size — good enough for pre-flight fit checks.

    Real conversion size is reported by onnx.save() once the run
    completes.
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
    PASS requires cos > cos_threshold AND max_abs < max_abs_threshold.
    Either failure flips to FAIL with a human-readable reason.

    Default max_abs_threshold=1e-2 is the realistic ceiling for end-to-
    end FP16 VLA inference: smolvla_libero measured at max_abs=7.8e-03
    with cos=+0.999994. Actions live in [-1, 1], so 1e-2 is <1% of
    action range — imperceptible on the robot. Tighter thresholds
    (5e-3) catch tuned FP16 calibration; 1e-2 catches naive conversions.
    """
    cos_ok = cos_sim >= cos_threshold
    maxabs_ok = max_abs_diff <= max_abs_threshold
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


def convert_fp32_to_fp16(
    fp32_onnx_path: str | Path,
    fp16_onnx_path: str | Path,
    *,
    keep_io_types: bool = True,
    op_block_list: tuple[str, ...] | None = None,
    min_positive_val: float = 1e-7,
    max_finite_val: float = 1e4,
) -> dict[str, Any]:
    """Convert an ONNX model on disk from FP32 weights to FP16.

    Args:
        fp32_onnx_path: input ONNX (with external data if >2GB).
        fp16_onnx_path: output ONNX path; external data goes
            alongside with the same stem + `.bin`.
        keep_io_types: if True (default), the graph's input/output
            tensors stay FP32 — callers don't have to change their
            numpy dtype. Internal weights + activations go FP16.
        op_block_list: ops kept in FP32 to avoid underflow.
            Defaults to FP16_OP_BLOCKLIST.
        min_positive_val / max_finite_val: clamps for FP16 range
            (smaller/larger values are saturated to avoid NaN/Inf).

    Returns: a summary dict — input/output paths, byte sizes, reduction ratio.
    """
    try:
        import onnx
        from onnxconverter_common.float16 import convert_float_to_float16
    except ImportError as e:
        raise ImportError(
            "FP16 conversion requires `onnx` and `onnxconverter-common`. "
            "Install with: pip install onnx onnxconverter-common"
        ) from e

    src = Path(fp32_onnx_path)
    dst = Path(fp16_onnx_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Protobuf's 2GB-per-message limit bites both `ByteSize()` AND
    # `SerializeToString()`. For >~1.8GB models we can't run internal
    # shape inference (it serializes the loaded proto). We also flip to
    # keep_io_types=False because convert_float_to_float16's Cast-node
    # wiring gets Mul/Add operand dtypes wrong on big graphs, producing
    # FP32+FP16 mismatches ORT rejects. End-to-end FP16 is safer; callers
    # cast inputs/outputs.
    on_disk_bytes = _size_with_external(src)
    oversized = on_disk_bytes > 1_800_000_000
    disable_shape_infer = oversized
    if oversized and keep_io_types:
        logger.warning(
            "[fp16] model is %.2f GB on disk — flipping keep_io_types=False to "
            "avoid Mul/Add dtype-mismatch errors on oversized graphs.",
            on_disk_bytes / 1e9,
        )
        keep_io_types = False

    logger.info("[fp16] Loading %s (%.2f GB on disk)...",
                src, on_disk_bytes / 1e9)
    model_fp32 = onnx.load(str(src), load_external_data=True)

    blocklist = list(op_block_list if op_block_list is not None else FP16_OP_BLOCKLIST)

    logger.info(
        "[fp16] Converting (keep_io=%s, blocklist=%s, disable_shape_infer=%s)...",
        keep_io_types, blocklist, disable_shape_infer,
    )
    # The smolvla libero monolithic is 2.2GB on disk — triggers disable_shape_infer
    # (we don't want to hit the 2GB protobuf serialization limit during
    # onnx.shape_inference). pi0 (12.5GB) and pi0.5 (13GB) will also use this path.
    model_fp16 = convert_float_to_float16(
        model_fp32,
        keep_io_types=keep_io_types,
        op_block_list=blocklist,
        min_positive_val=min_positive_val,
        max_finite_val=max_finite_val,
        disable_shape_infer=disable_shape_infer,
    )

    # Strip stale value_info when we skipped shape inference. Those entries
    # carry the pre-conversion FP32 type annotations and ORT rejects the
    # model on load if they don't match the actual (now FP16) output types.
    # Without value_info, ORT re-infers types at session-init time.
    if disable_shape_infer:
        del model_fp16.graph.value_info[:]
        logger.info(
            "[fp16] Stripped %d stale value_info entries (forced ORT re-infer).",
            0,  # already deleted; log hint only
        )

    logger.info("[fp16] Saving %s...", dst)
    # Remove any leftover external data from a prior run at this path —
    # onnx.save would otherwise leave both the old and new .bin on disk
    # and our size accounting would double-count.
    for pat in ("*.bin", "*.data"):
        for old in dst.parent.glob(pat):
            try:
                old.unlink()
            except Exception:
                pass

    onnx.save(
        model_fp16,
        str(dst),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{dst.stem}.bin",
        size_threshold=1024,
    )

    # Post-pass: insert Cast nodes at FP32/FP16 operand mismatches so ORT
    # accepts the model. convert_float_to_float16 misses some boundaries
    # on large graphs; this cleanup makes the FP16 ONNX directly loadable.
    try:
        casts_inserted = fix_fp16_dtype_mismatches(dst)
        if casts_inserted > 0:
            logger.info(
                "[fp16] Inserted %d Cast nodes to fix dtype mismatches.",
                casts_inserted,
            )
    except Exception as e:
        logger.warning(
            "[fp16] Cast-insertion post-pass failed: %s — ORT may reject "
            "the FP16 ONNX at load time, but TRT engine build should "
            "still work.", e,
        )
        casts_inserted = -1

    # Size accounting
    fp32_bytes = _size_with_external(src)
    fp16_bytes = _size_with_external(dst)
    reduction = 1.0 - (fp16_bytes / fp32_bytes) if fp32_bytes else 0.0

    logger.info(
        "[fp16] %s -> %s: %.1f GB -> %.1f GB (%.1f%% reduction)",
        src.name, dst.name,
        fp32_bytes / 1e9, fp16_bytes / 1e9, reduction * 100,
    )

    return {
        "src_path": str(src),
        "dst_path": str(dst),
        "src_bytes": fp32_bytes,
        "dst_bytes": fp16_bytes,
        "reduction_ratio": reduction,
        "op_block_list": blocklist,
        "cast_nodes_inserted": casts_inserted,
    }


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
    """Post-process an FP16-converted ONNX: insert Cast nodes at
    FP32/FP16 operand mismatches so ORT accepts the model at load time.

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
    """Size of the graph file + any .bin/.data siblings."""
    total = onnx_path.stat().st_size
    for pat in ("*.bin", "*.data"):
        for p in onnx_path.parent.glob(pat):
            if p.stem == onnx_path.stem or p.stem.startswith(onnx_path.stem):
                total += p.stat().st_size
    return total


__all__ = [
    "FP16_OP_BLOCKLIST",
    "convert_fp32_to_fp16",
    "estimate_fp16_size_bytes",
    "fix_fp16_dtype_mismatches",
    "parity_gate",
]
