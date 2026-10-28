# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Pieces shared by the accel engines: static buffer arenas and a CUDA-graph cache."""

from __future__ import annotations

import torch


class Arena:
    """One flat static buffer viewed as contiguous ``[rows, cols]`` matrices of any width."""

    def __init__(self, elems, dtype, device):
        self.buf = torch.zeros(int(elems), dtype=dtype, device=device)

    def __call__(self, rows, cols):
        return self.buf[: rows * cols].view(rows, cols)


class GraphCache:
    """Captures ``fn(key)`` once per key (after two warmup runs on a side stream) and replays it.

    All graphs share one memory pool: an engine never replays two of its graphs concurrently.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.graphs = {}
        self.pool = None

    def run(self, key, fn):
        graph = self.graphs.get(key)
        if graph is None:
            graph = self._capture(key, fn)
        graph.replay()

    def _capture(self, key, fn):
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                fn(key)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            fn(key)
        self.graphs[key] = graph
        return graph

    def clear(self):
        self.graphs.clear()


def bucket(n, size):
    return -(-int(n) // int(size)) * int(size)
