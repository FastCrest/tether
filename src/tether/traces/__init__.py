"""Customer trace archive (Phase 1.5 v1).

Searchable + summarizable view over JSONL traces written by
`tether serve --record <dir>`. Built directly on existing JSONL
storage (parquet+DuckDB migration deferred to v2 per spec
features/01_serve/subfeatures/_ecosystem/customer-trace-archive/).

Public surface:
    from tether.traces.archive import (
        TraceFilter, TraceReader, query_traces, summarize_traces,
    )

Used by CLI commands `tether traces query` and `tether traces summary`.
"""
from tether.traces.archive import (
    TraceFilter,
    TraceReader,
    TraceRecord,
    TraceSummary,
    query_traces,
    summarize_traces,
)

__all__ = [
    "TraceFilter",
    "TraceReader",
    "TraceRecord",
    "TraceSummary",
    "query_traces",
    "summarize_traces",
]
