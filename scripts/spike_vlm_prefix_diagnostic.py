#!/usr/bin/env python3
"""Diagnostic spike: validate the Where+Cast hypothesis for vlm_prefix capture divergence.

Per vlm_prefix capture divergence investigation (2026-04-24), the suspected
root cause is the post-export Where+Cast workaround in
`_repair_exported_onnx()`. CUDA graph capture bakes dtype
assumptions at capture time; replay reads with wrong precision.

This script:
1. Counts Where + Cast ops in vlm_prefix.onnx.
2. Identifies Where→Cast chains (post-export fix signature).
3. Reports input dtypes on each Where op (mixed-dtype = suspect).
4. Runs eager, capture, replay through ORT + enable_cuda_graph=1.
5. Computes cosine similarity per output (vs bit-identical atol check).

Output validates or falsifies the hypothesis cheaply before committing to
M1 (re-export skip-list) or other mitigation.

Usage:
    python scripts/spike_vlm_prefix_diagnostic.py --export-dir <path>
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def count_ops(model: onnx.ModelProto) -> dict:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def find_where_cast_chains(model: onnx.ModelProto) -> list[tuple]:
    """Return list of (where_node_name, downstream_cast_node_name) tuples.

    A Where → Cast chain is the signature of the post-export workaround:
    the Cast was inserted AFTER a Where to coerce its output dtype.
    """
    # Map output tensor name → node that produces it
    producer: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node

    chains = []
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        if not node.input:
            continue
        parent = producer.get(node.input[0])
        if parent is not None and parent.op_type == "Where":
            chains.append((parent.name or "<unnamed>", node.name or "<unnamed>"))
    return chains


def where_input_dtypes(model: onnx.ModelProto) -> list[tuple]:
    """Return list of (where_name, [input dtypes as strings]) for each Where op."""
    # Build initializer dtype map
    init_dtype: dict[str, int] = {}
    for init in model.graph.initializer:
        init_dtype[init.name] = init.data_type

    # Build value_info dtype map (intermediate tensor types)
    vi_dtype: dict[str, int] = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.type.tensor_type.elem_type:
            vi_dtype[vi.name] = vi.type.tensor_type.elem_type

    def dtype_str(elem_type: int) -> str:
        # Per ONNX TensorProto.DataType
        mapping = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64",
                   9: "bool", 10: "float16", 11: "float64"}
        return mapping.get(elem_type, f"type_{elem_type}")

    rows = []
    for node in model.graph.node:
        if node.op_type != "Where":
            continue
        dts = []
        for in_name in node.input:
            if in_name in init_dtype:
                dts.append(dtype_str(init_dtype[in_name]))
            elif in_name in vi_dtype:
                dts.append(dtype_str(vi_dtype[in_name]))
            else:
                dts.append("unknown")
        rows.append((node.name or "<unnamed>", dts))
    return rows


def make_random_feed(session: ort.InferenceSession, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    feed: dict = {}
    for inp in session.get_inputs():
        shape = [1 if (isinstance(d, str) or d is None) else int(d) for d in inp.shape]
        dtype = _ORT_TO_NP.get(inp.type, np.float32)
        if np.issubdtype(dtype, np.floating):
            feed[inp.name] = rng.standard_normal(shape).astype(dtype)
        elif dtype == np.bool_:
            feed[inp.name] = (rng.integers(0, 2, size=shape) > 0)
        else:
            feed[inp.name] = rng.integers(0, 100, size=shape, dtype=dtype)
    return feed


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64).flatten()
    b_f = b.astype(np.float64).flatten()
    denom = (np.linalg.norm(a_f) * np.linalg.norm(b_f))
    if denom == 0:
        return float("nan")
    return float(np.dot(a_f, b_f) / denom)


def error_distribution(eager: np.ndarray, other: np.ndarray) -> dict:
    """Per-output absolute-error stats: mean, p50, p95, p99, max, fraction > 1e-6."""
    diff = np.abs(eager.astype(np.float64) - other.astype(np.float64)).flatten()
    denom = np.maximum(np.abs(eager.astype(np.float64)).flatten(), 1e-12)
    rel = diff / denom
    return {
        "abs_mean": float(diff.mean()),
        "abs_p50": float(np.percentile(diff, 50)),
        "abs_p95": float(np.percentile(diff, 95)),
        "abs_p99": float(np.percentile(diff, 99)),
        "abs_max": float(diff.max()),
        "rel_p50": float(np.percentile(rel, 50)),
        "rel_p95": float(np.percentile(rel, 95)),
        "rel_p99": float(np.percentile(rel, 99)),
        "rel_max": float(rel.max()),
        "frac_above_1em6": float((diff > 1e-6).mean()),
        "frac_above_1em4": float((diff > 1e-4).mean()),
        "frac_above_1em3": float((diff > 1e-3).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-dir", type=Path, required=True,
                    help="Dir containing vlm_prefix.onnx")
    ap.add_argument("--file", default="vlm_prefix.onnx",
                    help="ONNX file under export-dir to diagnose (default vlm_prefix.onnx)")
    args = ap.parse_args()

    model_path = args.export_dir / args.file
    if not model_path.exists():
        print(f"FAIL: {model_path} does not exist")
        return 1

    print("=" * 60)
    print(f"Diagnostic spike: {model_path}")
    print("=" * 60)

    # Part 1: ONNX IR analysis — count Where + Cast, find Where→Cast chains
    print("\n[1/3] Analyzing ONNX IR...")
    model = onnx.load(str(model_path))
    op_counts = count_ops(model)
    where_count = op_counts.get("Where", 0)
    cast_count = op_counts.get("Cast", 0)
    print(f"  Total nodes: {sum(op_counts.values())}")
    print(f"  Where ops:   {where_count}")
    print(f"  Cast ops:    {cast_count}")

    chains = find_where_cast_chains(model)
    print(f"  Where → Cast chains (post-export fix signature): {len(chains)}")
    if chains:
        print("    (first 5):")
        for (w, c) in chains[:5]:
            print(f"      Where '{w}' -> Cast '{c}'")

    where_dtypes = where_input_dtypes(model)
    mixed = [(name, dts) for (name, dts) in where_dtypes
             if len(set(d for d in dts[1:] if d != "unknown")) > 1]
    print(f"  Where ops with mixed-dtype branches (suspect): {len(mixed)}")
    if mixed:
        print("    (first 3):")
        for (w, dts) in mixed[:3]:
            print(f"      Where '{w}' inputs: {dts}")

    # Part 2: Eager reference run
    print("\n[2/3] Running eager (no cuda graph)...")
    eager_providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    eager_sess = ort.InferenceSession(str(model_path), providers=eager_providers)
    if "CUDAExecutionProvider" not in eager_sess.get_providers():
        print(f"  FAIL: CUDA EP not active. Got {eager_sess.get_providers()}")
        return 1
    feed = make_random_feed(eager_sess, seed=0)
    t0 = time.perf_counter()
    eager_out = eager_sess.run(None, feed)
    eager_ms = (time.perf_counter() - t0) * 1000
    print(f"  Eager: {eager_ms:.1f} ms, {len(eager_out)} outputs")

    # Part 3: CUDA graph capture + replay
    # Test with MINIMAL provider options (just enable_cuda_graph) to isolate
    # whether the M2 options (cudnn_conv_algo_search=HEURISTIC, arena tuning)
    # are the cause of divergence vs cuda_graph itself.
    print("\n[3/3] Running with enable_cuda_graph=1 (MINIMAL options, no M2 tuning)...")
    cg_providers = [
        ("CUDAExecutionProvider", {"enable_cuda_graph": "1"}),
        "CPUExecutionProvider",
    ]
    try:
        cg_sess = ort.InferenceSession(str(model_path), providers=cg_providers)
    except Exception as e:
        print(f"  FAIL: session creation raised {type(e).__name__}: {e}")
        return 1
    try:
        t0 = time.perf_counter()
        cg_capture = cg_sess.run(None, feed)
        capture_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        cg_replay = cg_sess.run(None, feed)
        replay_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"  FAIL: capture/replay raised {type(e).__name__}: {e}")
        return 1
    print(f"  Capture: {capture_ms:.1f} ms, Replay: {replay_ms:.1f} ms")

    # Part 4: Per-output error distribution (eager vs captured)
    print("\n" + "=" * 60)
    print("ERROR DISTRIBUTION (eager vs captured) per output")
    print("=" * 60)
    print(f"  {'out':>4}  {'cos':>8}  {'abs_p50':>10}  {'abs_p99':>10}  {'abs_max':>10}  {'frac>1e-6':>10}  {'frac>1e-3':>10}")
    all_close = True
    min_cos = 1.0
    worst_abs_max = 0.0
    for i, (eager, captured) in enumerate(zip(eager_out, cg_capture)):
        cos_ec = cosine_similarity(eager, captured)
        dist = error_distribution(eager, captured)
        min_cos = min(min_cos, cos_ec)
        worst_abs_max = max(worst_abs_max, dist["abs_max"])
        flag = "" if dist["abs_max"] <= 1e-6 else "  DIV"
        print(f"  {i:4d}  {cos_ec:8.6f}  {dist['abs_p50']:10.2e}  {dist['abs_p99']:10.2e}  "
              f"{dist['abs_max']:10.2e}  {dist['frac_above_1em6']:10.2%}  {dist['frac_above_1em3']:10.2%}{flag}")
        if dist["abs_max"] > 1e-6:
            all_close = False

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Where ops total: {where_count}")
    print(f"  Where -> Cast chains (post-export fix): {len(chains)}")
    print(f"  Where ops with mixed-dtype inputs: {len(mixed)}")
    print(f"  Min cosine (eager vs captured): {min_cos:.6f}")
    print(f"  Max abs error across all outputs: {worst_abs_max:.2e}")
    print(f"  Parity status at atol=1e-6: {'PASS' if all_close else 'DIVERGES'}")
    print()
    if len(chains) > 0 and not all_close:
        print("  HYPOTHESIS CONFIRMED: Where+Cast chains present AND capture diverges.")
        return 0
    if len(chains) == 0 and not all_close:
        print("  HYPOTHESIS REJECTED (Where+Cast): 0 Where ops but capture still diverges.")
        print(f"  Max abs err = {worst_abs_max:.2e} — assess practical impact:")
        print(f"    - If abs_max < 1e-3 and cos >= 0.9999, likely non-deterministic reduction")
        print(f"      (cudnn conv algo selection or matmul algo); SHIP may be acceptable if")
        print(f"      action-level downstream impact is small (validate via LIBERO eval).")
        print(f"    - If abs_max > 1e-2 or cos < 0.999, likely a structural op mismatch —")
        print(f"      investigate stream sync, cudnn_conv_algo_search, or specific ops.")
        return 2
    if all_close:
        print("  SURPRISE: this run shows NO divergence at atol=1e-6.")
        print("  -> The M2 provider options (cudnn_conv_algo_search=HEURISTIC or")
        print("     arena_extend_strategy) were likely the cause of prior divergence.")
        print("     Next step: drop M2 options in production, capture with minimal flags.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
