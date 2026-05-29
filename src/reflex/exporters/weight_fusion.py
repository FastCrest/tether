"""Post-export ONNX weight fusion pass.

Lifted from dexmal/realtime-vla (MIT) pattern analysis. Their key
insight: fuse RMSNorm scales into adjacent MatMul weights at export
time, pre-compute time embeddings for fixed step counts, and fold
the Euler step dt scalar into the output projection.

These are ONNX graph rewrites that reduce operator count and eliminate
runtime normalization multiplies at zero accuracy cost (algebraically
equivalent transformations).

Usage:
    from reflex.exporters.weight_fusion import fuse_weights
    fused_path = fuse_weights(onnx_path)
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _load_onnx(path: Path):
    import onnx
    return onnx.load(str(path))


def _iter_graph_tensors(graph):
    """Yield every TensorProto reachable from a graph, including attributes."""
    for tensor in graph.initializer:
        yield tensor
    for tensor in graph.sparse_initializer:
        yield tensor.values
        yield tensor.indices
    for node in graph.node:
        for attr in node.attribute:
            yield from _iter_attribute_tensors(attr)


def _iter_attribute_tensors(attr):
    if attr.HasField("t"):
        yield attr.t
    yield from attr.tensors
    if attr.HasField("g"):
        yield from _iter_graph_tensors(attr.g)
    for graph in attr.graphs:
        yield from _iter_graph_tensors(graph)


def _inline_tensor_bytes(tensor) -> int:
    """Approximate protobuf payload bytes for TensorProto inline data fields."""
    total = len(tensor.raw_data) if tensor.HasField("raw_data") else 0
    total += len(tensor.float_data) * 4
    total += len(tensor.int32_data) * 4
    total += len(tensor.int64_data) * 8
    total += len(tensor.double_data) * 8
    total += len(tensor.uint64_data) * 8
    total += sum(len(item) for item in tensor.string_data)
    return total


def _coerce_inline_tensor_data_to_raw(model, *, min_inline_bytes: int = 1024) -> tuple[int, int]:
    """Convert repeated numeric TensorProto fields to raw_data before save.

    ONNX's external-data conversion only externalizes tensors that use the
    ``raw_data`` field. Some exporter paths leave large Constant attribute
    tensors in repeated fields such as ``float_data`` or ``int64_data``; those
    remain embedded in ``model.onnx`` even when ``convert_attribute=True`` is
    passed to ``onnx.save``. Coercing them to raw_data lets the standard ONNX
    writer move them into ``model.onnx.data``.
    """
    from onnx import numpy_helper

    converted = 0
    converted_bytes = 0
    for tensor in _iter_graph_tensors(model.graph):
        inline_bytes = _inline_tensor_bytes(tensor)
        if inline_bytes < min_inline_bytes:
            continue
        if tensor.HasField("raw_data"):
            continue
        if len(tensor.string_data) > 0:
            continue
        try:
            array = numpy_helper.to_array(tensor)
            replacement = numpy_helper.from_array(array, name=tensor.name)
            tensor.CopyFrom(replacement)
        except Exception as exc:
            logger.debug("Could not normalize inline ONNX tensor %s: %s", tensor.name, exc)
            continue
        converted += 1
        converted_bytes += inline_bytes

    if converted:
        logger.info(
            "Normalized %d inline ONNX tensor(s) (%.1f MB) to raw_data for external-data save",
            converted,
            converted_bytes / 1e6,
        )
    return converted, converted_bytes


def _save_onnx(model, path: Path):
    import onnx
    external_data = path.with_suffix(path.suffix + ".data")
    _coerce_inline_tensor_data_to_raw(model)

    with tempfile.TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as tmp_dir:
        tmp_model = Path(tmp_dir) / path.name
        onnx.save(
            model,
            str(tmp_model),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data.name,
            size_threshold=1024,
            convert_attribute=True,
        )
        # onnx.save writes external data relative to the temporary model path,
        # but preserves the relative location string that will also be correct
        # once the model file is moved back beside the final external file.
        tmp_external = Path(tmp_dir) / external_data.name
        proto_size = tmp_model.stat().st_size
        max_proto_size = 2 * 1024 * 1024 * 1024
        if proto_size >= max_proto_size:
            raise RuntimeError(
                f"Saved ONNX protobuf is still {proto_size / 1e9:.2f}GB after "
                "external-data conversion. ONNX Runtime cannot load protobufs "
                ">=2GB; inspect embedded Constant tensors before benchmarking."
            )

        os.replace(tmp_model, path)
        if tmp_external.exists():
            os.replace(tmp_external, external_data)

    proto_size = path.stat().st_size
    max_proto_size = 2 * 1024 * 1024 * 1024
    if proto_size >= max_proto_size:
        raise RuntimeError(
            f"Saved ONNX protobuf is still {proto_size / 1e9:.2f}GB after "
            "external-data conversion. ONNX Runtime cannot load protobufs "
            ">=2GB; inspect embedded Constant tensors before benchmarking."
        )
    total_size = proto_size
    external_size = 0
    if external_data.exists():
        external_size = external_data.stat().st_size
        total_size += external_size
    logger.info(
        "Saved fused ONNX: %s (proto=%.1f MB external=%.1f MB total=%.1f MB)",
        path,
        proto_size / 1e6,
        external_size / 1e6,
        total_size / 1e6,
    )


def _get_initializer(model, name: str) -> np.ndarray | None:
    for init in model.graph.initializer:
        if init.name == name:
            import onnx
            return np.frombuffer(init.raw_data, dtype=_onnx_dtype_to_numpy(init.data_type)).reshape(init.dims)
    return None


def _set_initializer(model, name: str, data: np.ndarray):
    import onnx
    from onnx import numpy_helper
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            model.graph.initializer[i].CopyFrom(
                numpy_helper.from_array(data, name=name)
            )
            return True
    return False


def _onnx_dtype_to_numpy(dt: int):
    _MAP = {
        1: np.float32,
        10: np.float16,
        16: np.float32,  # bf16 stored as fp32 in ONNX initializers
        7: np.int64,
        6: np.int32,
        9: bool,
        2: np.uint8,
    }
    return _MAP.get(dt, np.float32)


def fuse_rmsnorm_into_matmul(model, dry_run: bool = False) -> int:
    """Fuse RMSNorm scale weights into adjacent MatMul/Gemm weights.

    Pattern: RMSNorm(x) * scale -> MatMul(x_normed, W)
    Becomes: MatMul(x_normed, W * scale) -- eliminates the Mul op.

    This is the core optimization from dexmal/realtime-vla:
      w_q *= (1 + w_scale[:, :, None])  (convert_from_jax.py:46-54)

    In ONNX graph form, we look for:
      Mul(rmsnorm_output, scale_initializer) -> MatMul(_, weight_initializer)
    and fold the scale into the weight.

    Returns the number of fusions applied.
    """
    import onnx
    fused = 0
    nodes_to_remove = []

    node_output_map = {}
    for node in model.graph.node:
        for output in node.output:
            node_output_map[output] = node

    node_input_consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp not in node_input_consumers:
                node_input_consumers[inp] = []
            node_input_consumers[inp].append(node)

    for node in model.graph.node:
        if node.op_type != "Mul":
            continue

        mul_output = node.output[0]
        consumers = node_input_consumers.get(mul_output, [])
        if len(consumers) != 1:
            continue
        consumer = consumers[0]
        if consumer.op_type not in ("MatMul", "Gemm"):
            continue

        scale_data = None
        other_input = None
        for inp in node.input:
            data = _get_initializer(model, inp)
            if data is not None and data.ndim <= 1:
                scale_data = data
            else:
                other_input = inp

        if scale_data is None or other_input is None:
            continue

        weight_name = None
        for inp in consumer.input:
            w = _get_initializer(model, inp)
            if w is not None and w.ndim == 2:
                weight_name = inp
                break

        if weight_name is None:
            continue

        if dry_run:
            logger.info("Would fuse: Mul(%s) -> %s(%s)", node.name, consumer.op_type, consumer.name)
            fused += 1
            continue

        weight_data = _get_initializer(model, weight_name)
        if weight_data is None:
            continue

        try:
            if scale_data.shape[0] == weight_data.shape[0]:
                fused_weight = weight_data * scale_data[:, np.newaxis]
            elif scale_data.shape[0] == weight_data.shape[1]:
                fused_weight = weight_data * scale_data[np.newaxis, :]
            else:
                logger.debug("Shape mismatch: scale %s vs weight %s, skipping", scale_data.shape, weight_data.shape)
                continue

            _set_initializer(model, weight_name, fused_weight.astype(weight_data.dtype))

            consumer_input_idx = list(consumer.input).index(mul_output)
            consumer.input[consumer_input_idx] = other_input
            nodes_to_remove.append(node)
            fused += 1
            logger.debug("Fused Mul(%s) scale into %s weight %s", node.name, consumer.op_type, weight_name)
        except Exception as e:
            logger.debug("Fusion failed for %s: %s", node.name, e)

    for node in nodes_to_remove:
        model.graph.node.remove(node)

    if fused:
        logger.info("Fused %d RMSNorm scale(s) into MatMul weights", fused)
    return fused


def fold_euler_dt_into_output_proj(
    model,
    num_steps: int = 10,
    dt_override: float | None = None,
) -> int:
    """Fold the Euler step dt = -1/num_steps into output projection weights.

    Pattern from dexmal:
      decoder_action_fused_out_proj_w *= -0.1  (for num_steps=10)

    In ONNX: finds Mul(output_proj, dt_constant) and absorbs the scalar
    into the weight initializer.

    Returns the number of fusions applied.
    """
    dt = dt_override or (-1.0 / num_steps)
    fused = 0

    for node in model.graph.node:
        if node.op_type != "Mul":
            continue

        scalar_val = None
        weight_name = None
        for inp in node.input:
            data = _get_initializer(model, inp)
            if data is not None:
                if data.size == 1:
                    scalar_val = float(data.flat[0])
                elif data.ndim == 2:
                    weight_name = inp

        if scalar_val is None or weight_name is None:
            continue

        if not (abs(scalar_val - dt) < 1e-6 or abs(scalar_val - (-dt)) < 1e-6):
            continue

        weight_data = _get_initializer(model, weight_name)
        if weight_data is None:
            continue

        _set_initializer(model, weight_name, (weight_data * scalar_val).astype(weight_data.dtype))
        node.op_type = "Identity"
        del node.input[:]
        node.input.append(weight_name)
        fused += 1
        logger.info("Folded dt=%.4f into weight %s", scalar_val, weight_name)

    return fused


def precompute_time_embeddings(
    model,
    num_steps: int = 10,
) -> int:
    """Pre-compute time embeddings for fixed-step flow matching.

    For flow matching with a fixed step schedule (linspace(1, 0, num_steps+1)),
    the time MLP produces the same embeddings every inference call. If we can
    identify the time MLP subgraph in ONNX, we can constant-fold it.

    This is a lighter version of dexmal's approach — they do it at weight
    conversion time (convert_from_jax.py:176-196). We do it as an ONNX
    graph pass which is model-agnostic.

    Returns the number of time embeddings pre-computed.
    """
    # This is a graph search problem that depends heavily on the specific
    # ONNX export structure. For now, rely on onnxsim's constant folding
    # which achieves the same result when time values are graph constants.
    # The explicit pre-computation (like dexmal's fused_time_biases) would
    # require model-specific knowledge of which nodes form the time MLP.
    logger.info("Time embedding pre-computation deferred to onnxsim constant folding")
    return 0


def eliminate_identity_nodes(model) -> int:
    """Remove Identity nodes left behind by other fusion passes."""
    removed = 0
    nodes_to_remove = []

    output_to_input = {}
    for node in model.graph.node:
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
            output_to_input[node.output[0]] = node.input[0]
            nodes_to_remove.append(node)

    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in output_to_input:
                node.input[i] = output_to_input[inp]

    for node in nodes_to_remove:
        model.graph.node.remove(node)
        removed += 1

    if removed:
        logger.info("Eliminated %d Identity nodes", removed)
    return removed


def fuse_weights(
    onnx_path: Path | str,
    output_path: Path | str | None = None,
    num_steps: int = 10,
    dry_run: bool = False,
) -> Path:
    """Run all weight fusion passes on an ONNX model.

    Args:
        onnx_path: Input ONNX model path.
        output_path: Where to save the fused model. Defaults to overwriting input.
        num_steps: Number of flow matching steps (for dt folding).
        dry_run: If True, report what would be fused without modifying.

    Returns:
        Path to the fused ONNX model.
    """
    onnx_path = Path(onnx_path)
    output_path = Path(output_path) if output_path else onnx_path

    logger.info("Running weight fusion on %s", onnx_path)
    model = _load_onnx(onnx_path)

    total_fused = 0
    total_fused += fuse_rmsnorm_into_matmul(model, dry_run=dry_run)
    total_fused += fold_euler_dt_into_output_proj(model, num_steps=num_steps)
    total_fused += precompute_time_embeddings(model, num_steps=num_steps)

    if not dry_run:
        total_fused += eliminate_identity_nodes(model)
        _save_onnx(model, output_path)

    logger.info("Weight fusion complete: %d transformations applied", total_fused)
    return output_path
