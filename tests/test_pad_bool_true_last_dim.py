"""The Pad-free bool mask helper must equal F.pad(value=True) and emit no ONNX Pad."""
import io

import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
import torch.nn.functional as F

from tether.exporters.monolithic import _pad_bool_true_last_dim


@pytest.mark.parametrize("left,right", [(0, 5), (7, 0), (3, 4), (0, 0)])
@pytest.mark.parametrize("shape", [(1, 6, 9), (2, 3, 4)])
def test_matches_f_pad_true(shape, left, right):
    generator = torch.Generator().manual_seed(sum(shape) + left * 31 + right)
    masks = torch.rand(shape, generator=generator) > 0.5
    expected = F.pad(masks, (left, right), value=True)
    actual = _pad_bool_true_last_dim(masks, left, right)
    assert actual.dtype == torch.bool
    assert torch.equal(actual, expected)


def test_exports_without_a_pad_node():
    class Masks(torch.nn.Module):
        def forward(self, masks):
            return _pad_bool_true_last_dim(masks, 2, 3) & _pad_bool_true_last_dim(masks, 0, 5)

    buffer = io.BytesIO()
    torch.onnx.export(Masks(), (torch.ones(1, 4, 6, dtype=torch.bool),), buffer,
                      opset_version=17, dynamo=False)
    ops = {node.op_type for node in onnx.load_from_string(buffer.getvalue()).graph.node}
    assert "Pad" not in ops, ops
