"""Modal inspect (CPU-only): dump the Pad node(s) of the retained pi0.5
monolithic export -- inputs, dtypes, shapes, attrs, neighbours.

Usage:
    modal run scripts/modal_pi05_pad_inspect.py
"""

import modal

app = modal.App("tether-pi05-pad-inspect")
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
    from pathlib import Path

    import onnx
    from onnx import TensorProto

    onnx_path = Path(PARITY_OUT_PATH) / "pi05" / "export" / "model.onnx"
    model = onnx.load(str(onnx_path), load_external_data=False)

    def dtype_name(num):
        return TensorProto.DataType.Name(num) if num else "?"

    # Index value-info + initializers for dtype/shape lookup.
    info = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(
        model.graph.value_info
    ):
        t = vi.type.tensor_type
        info[vi.name] = (dtype_name(t.elem_type), [d.dim_value for d in t.shape.dim])
    for init in model.graph.initializer:
        info[init.name] = (dtype_name(init.data_type), list(init.dims))

    out: dict = {"pad_nodes": []}
    nodes = model.graph.node
    for idx, node in enumerate(nodes):
        if node.op_type != "Pad":
            continue
        entry: dict = {
            "index": idx,
            "name": node.name,
            "attrs": {a.name: list(a.ints) if a.ints else a.s for a in node.attribute},
            "inputs": [
                {"name": n, "info": info.get(n, ("unknown", []))} for n in node.input
            ],
            "outputs": [
                {"name": n, "info": info.get(n, ("unknown", []))} for n in node.output
            ],
        }
        # Neighbour producers/consumers.
        prev = str(nodes[idx - 1].op_type) if idx > 0 else None
        nxt = str(nodes[idx + 1].op_type) if idx + 1 < len(nodes) else None
        entry["prev_op"] = prev
        entry["next_op"] = nxt
        out["pad_nodes"].append(entry)
        print(f"[pad] idx={idx} name={node.name}", flush=True)
        print(f"[pad]   attrs={entry['attrs']}", flush=True)
        for i in entry["inputs"]:
            print(f"[pad]   in  {i['name']} {i['info']}", flush=True)
        for o in entry["outputs"]:
            print(f"[pad]   out {o['name']} {o['info']}", flush=True)
        print(f"[pad]   prev={prev} next={nxt}", flush=True)
    out["pad_count"] = len(out["pad_nodes"])
    print(f"[pad] total Pad nodes: {out['pad_count']}", flush=True)
    return out


@app.local_entrypoint()
def main() -> None:
    result = run_inspect.remote()
    print("\n=== RESULT ===")
    print(f"  pad_count: {result['pad_count']}")
