"""Modal inspect (CPU-only): neighborhood of the forced-CPU node_Concat_43 in
the retained GR00T monolithic export, to guide the exporter-side fix (M4 #53).

ORT's verbose EP log (round 1) names exactly one node:
  Force fallback to CPU execution for node: node_Concat_43 because the CPU
  execution path is deemed faster than overhead involved with execution on
  other EPs
This job dumps that node's inputs (producers, dtypes, shapes, constant
values), its consumers, and every other Concat in the graph. No GPU, no
tether checkout, no gate touched.

Usage:
    modal run scripts/modal_groot_concat43_inspect.py
"""

import modal

app = modal.App("tether-groot-concat43-inspect")
_parity_out_volume = modal.Volume.from_name(
    "parity-gpu-receipts", create_if_missing=True
)

PARITY_OUT_PATH = "/parity_out"

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "onnx>=1.16", "numpy"
)


@app.function(
    image=image,
    timeout=900,
    volumes={PARITY_OUT_PATH: _parity_out_volume},
)
def run_inspect(target: str = "node_Concat_43") -> dict:
    from pathlib import Path

    import onnx
    from onnx import TensorProto, numpy_helper

    onnx_path = Path(PARITY_OUT_PATH) / "groot" / "export" / "model.onnx"
    model = onnx.load(str(onnx_path), load_external_data=False)
    nodes = list(model.graph.node)
    print(f"[groot] nodes={len(nodes)} target={target}", flush=True)

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
            [d.dim_value if d.dim_value else d.dim_param for d in t.shape.dim],
        )
    inits = {}
    for init in model.graph.initializer:
        info[init.name] = (dtype_name(init.data_type), list(init.dims))
        try:
            inits[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass

    producers = {}
    for idx, n in enumerate(nodes):
        for o in n.output:
            producers[o] = idx
    consumers: dict[str, list[int]] = {}
    for idx, n in enumerate(nodes):
        for i in n.input:
            consumers.setdefault(i, []).append(idx)

    hits = [i for i, n in enumerate(nodes) if n.name == target]
    print(f"[groot] target indices={hits}", flush=True)
    summary: dict = {"target_indices": hits, "targets": []}
    for idx in hits:
        n = nodes[idx]
        print(f"[groot] idx={idx} op={n.op_type} name={n.name}", flush=True)
        print(f"[groot]   inputs={list(n.input)} outputs={list(n.output)}", flush=True)
        entry: dict = {
            "idx": idx,
            "op": n.op_type,
            "inputs": list(n.input),
            "outputs": list(n.output),
            "input_detail": [],
        }
        for i in n.input:
            det: dict = {"name": i, "dtype_shape": info.get(i, ("?", []))}
            p = producers.get(i)
            det["producer"] = None if p is None else (p, nodes[p].op_type, nodes[p].name)
            if i in inits:
                arr = inits[i]
                det["const"] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "values": arr.reshape(-1)[:16].tolist(),
                }
            entry["input_detail"].append(det)
            print(f"[groot]   in {i}: {det}", flush=True)
        # transitive consumers, 2 levels
        for lvl, frontier in enumerate([[idx], []]):
            pass
        lvl1 = sorted({c for o in n.output for c in consumers.get(o, [])})
        print(f"[groot]   consumers L1={[(c, nodes[c].op_type, nodes[c].name) for c in lvl1]}", flush=True)
        lvl2 = sorted({c2 for c in lvl1 for o in nodes[c].output for c2 in consumers.get(o, [])} - set(lvl1) - {idx})
        print(f"[groot]   consumers L2={[(c, nodes[c].op_type, nodes[c].name) for c in lvl2][:12]}", flush=True)
        entry["consumers_L1"] = [(c, nodes[c].op_type, nodes[c].name) for c in lvl1]
        entry["consumers_L2"] = [(c, nodes[c].op_type, nodes[c].name) for c in lvl2][:12]
        summary["targets"].append(entry)

    # Every Concat in the graph with input dtypes: is 43 the only INT64 one?
    concats = []
    for idx, n in enumerate(nodes):
        if n.op_type == "Concat":
            dtypes = sorted({info.get(i, ("?", []))[0] for i in n.input})
            concats.append((idx, n.name, dtypes, list(n.input), list(n.output)))
    print(f"[groot] concat_count={len(concats)}", flush=True)
    for idx, name, dtypes, ins, outs in concats:
        print(f"[groot]   idx={idx} {name} dtypes={dtypes} ins={ins} outs={outs}", flush=True)
    summary["concat_count"] = len(concats)
    summary["concats"] = [
        {"idx": i, "name": nm, "dtypes": dt, "inputs": ins, "outputs": outs}
        for i, nm, dt, ins, outs in concats
    ]
    return summary


@app.local_entrypoint()
def main() -> None:
    result = run_inspect.remote()
    print("\n=== RESULT ===")
    print(f"  target_indices: {result['target_indices']}")
    print(f"  concat_count: {result['concat_count']}")
