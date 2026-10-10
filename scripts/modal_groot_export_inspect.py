"""Modal inspect (CPU-only): op census + per-node input dtypes of the retained
GR00T monolithic export, to root-cause its no-CPU-fallback gate failure.

Usage:
    modal run scripts/modal_groot_export_inspect.py
"""

import modal

app = modal.App("tether-groot-export-inspect")
_parity_out_volume = modal.Volume.from_name(
    "parity-gpu-receipts", create_if_missing=True
)

PARITY_OUT_PATH = "/parity_out"

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "onnx>=1.16", "numpy"
)


@app.function(
    image=image,
    timeout=600,
    volumes={PARITY_OUT_PATH: _parity_out_volume},
)
def run_inspect() -> dict:
    import collections
    from pathlib import Path

    import onnx
    from onnx import TensorProto

    onnx_path = Path(PARITY_OUT_PATH) / "groot" / "export" / "model.onnx"
    model = onnx.load(str(onnx_path), load_external_data=False)

    def dtype_name(num):
        try:
            return TensorProto.DataType.Name(num)
        except Exception:
            return "?"

    info = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(
        model.graph.value_info
    ):
        t = vi.type.tensor_type
        info[vi.name] = (
            dtype_name(t.elem_type),
            [d.dim_value for d in t.shape.dim],
        )
    for init in model.graph.initializer:
        info[init.name] = (dtype_name(init.data_type), list(init.dims))

    census: dict = collections.Counter(n.op_type for n in model.graph.node)
    print(f"[groot] nodes={len(model.graph.node)}", flush=True)
    for op, count in census.most_common():
        print(f"[groot]   {count:6d}  {op}", flush=True)
    print(f"[groot] inputs={[i.name for i in model.graph.input]}", flush=True)
    print(f"[groot] outputs={[o.name for o in model.graph.output]}", flush=True)

    # Non-float nodes are the prime CPU-fallback suspects (cf. pi0.5 bool Pad).
    print("[groot] non-float-tensor nodes:", flush=True)
    odd = []
    for idx, node in enumerate(model.graph.node):
        dtypes = {info.get(n, ("?", []))[0] for n in list(node.input) + list(node.output)}
        if dtypes - {"FLOAT", "?" }:
            odd.append((idx, node.op_type, node.name, sorted(dtypes)))
    for idx, op, name, dtypes in odd[:60]:
        print(f"[groot]   idx={idx} {op} {name} dtypes={dtypes}", flush=True)
    print(f"[groot] non-float node count: {len(odd)}", flush=True)
    return {
        "node_count": len(model.graph.node),
        "op_census": dict(census.most_common()),
        "non_float_nodes": odd[:60],
        "non_float_count": len(odd),
    }


@app.local_entrypoint()
def main() -> None:
    result = run_inspect.remote()
    print("\n=== RESULT ===")
    print(f"  node_count: {result['node_count']}")
    print(f"  non_float_count: {result['non_float_count']}")
