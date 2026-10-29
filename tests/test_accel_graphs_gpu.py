"""GPU: the accel CUDA-graph cache captures once per key and replays with fresh inputs."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tether.accel._common import Arena, GraphCache  # noqa: E402


def test_graph_cache_captures_once_per_bucket_and_tracks_inputs():
    cache = GraphCache("cuda")
    x = torch.zeros(64, 32, device="cuda")
    y = torch.zeros(64, 32, device="cuda")
    calls = []

    def fn(rows):
        calls.append(rows)
        y[:rows].copy_(x[:rows] * 2 + 1)

    for value in (1.0, 3.0):
        x.fill_(value)
        cache.run(16, fn)
        torch.cuda.synchronize()
        assert torch.all(y[:16] == value * 2 + 1)
    assert len(cache.graphs) == 1 and calls == [16, 16, 16]  # 2 warmups + 1 capture, then replay
    cache.run(32, fn)
    assert sorted(cache.graphs) == [16, 32]


def test_arena_views_share_storage():
    a = Arena(1024, torch.float32, "cuda")
    a(4, 8).fill_(2.0)
    assert a(2, 16).sum().item() == 64.0
