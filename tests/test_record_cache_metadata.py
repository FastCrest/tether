"""`/act` must report the cache decision it actually made.

Before this, `write_request(...)` at server.py never passed `cache=`, so
`evidence.cache.status` read `"n/a"` on every record ever written, the
"Cache hit rate (5m)" Grafana panel rendered empty, and `tether replay
--diff cache` compared a recording against itself (docs/record_replay.md:125).

These tests fail if the per-request decision stops being recorded, stops
distinguishing hit from miss, stops naming *which* key component invalidated
the entry, or stops reaching the JSONL record.
"""

from __future__ import annotations

import concurrent.futures

import numpy as np
import pytest

from tether.runtime.pi05_decomposed_server import Pi05DecomposedInference


class _Stats:
    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions_ttl = 0
        self.evictions_lang = 0
        self.evictions_phash = 0


class _PrefixSession:
    """Stands in for vlm_prefix.onnx: one past_kv tensor + prefix_pad_masks."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, output_names, feed):
        self.calls += 1
        return [np.zeros((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.bool_)]


def _inference(*, enable_cache=True, max_age_steps=1000, hamming=0):
    inst = Pi05DecomposedInference.__new__(Pi05DecomposedInference)
    inst.cache_level = "prefix"
    inst.enable_cache = enable_cache
    inst._cache = None
    inst._stats = _Stats()
    inst._call_index = 0
    inst.cache_max_age_steps = max_age_steps
    inst.cache_ttl_sec = 60.0
    inst.phash_hamming_threshold = hamming
    inst._sess_prefix = _PrefixSession()
    inst._prefix_output_names = ["past_kv_0", "prefix_pad_masks"]
    inst._past_kv_names = ["past_kv_0"]
    return inst


_IMG = np.zeros((1, 3, 4, 4), dtype=np.float32)
_MASK = np.ones((1,), dtype=np.bool_)
_TOK = np.zeros((1, 4), dtype=np.int64)


def _prefix(inst, *, phash=b"\x00" * 8, lang=b"\x01" * 16):
    return inst._get_or_run_prefix(
        img_base=_IMG,
        img_wrist_l=_IMG,
        img_wrist_r=_IMG,
        mask_base=_MASK,
        mask_wrist_l=_MASK,
        mask_wrist_r=_MASK,
        lang_tokens=_TOK,
        lang_masks=_MASK,
        image_phashes=(phash,),
        lang_hash=lang,
    )


def test_cold_then_hit():
    inst = _inference()
    _prefix(inst)
    cold = inst.last_cache_decision()
    assert cold["status"] == "miss"
    assert cold["reason"] == "cold"
    assert cold["level"] == "prefix"

    _prefix(inst)
    hit = inst.last_cache_decision()
    assert hit["status"] == "hit"
    assert hit["reason"] == "key_match"
    # Same inputs must produce the same joinable key.
    assert hit["key"] == cold["key"]
    assert inst._sess_prefix.calls == 1, "hit must skip the VLM forward"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"lang": b"\x02" * 16}, "language_changed"),
        ({"phash": b"\xff" * 8}, "image_changed"),
    ],
)
def test_miss_names_the_invalidating_key_component(kwargs, reason):
    inst = _inference()
    _prefix(inst)
    _prefix(inst, **kwargs)
    decision = inst.last_cache_decision()
    assert decision["status"] == "miss"
    assert decision["reason"] == reason
    assert inst._sess_prefix.calls == 2


def test_stale_entry_is_a_miss_not_a_hit():
    inst = _inference(max_age_steps=1)
    _prefix(inst)
    inst._call_index = 99
    _prefix(inst)
    assert inst.last_cache_decision()["reason"] == "stale"


def test_cache_disabled_reports_off_not_miss():
    """`off` and `miss` are different facts. A run with caching disabled must
    not read as a run whose cache never hit."""
    inst = _inference(enable_cache=False)
    _prefix(inst)
    decision = inst.last_cache_decision()
    assert decision["status"] == "off"
    assert decision["level"] == "none"
    assert decision["reason"] == "disabled"


def test_decision_is_per_thread():
    """predict_from_base64_async offloads to a thread pool, so one request's
    hit must never be attributed to a concurrent request's record."""
    inst = _inference()
    _prefix(inst)  # cold
    _prefix(inst)  # hit — this thread's decision from here on

    def miss_on_worker():
        _prefix(inst, lang=b"\x09" * 16)
        return inst.last_cache_decision()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        worker_decision = pool.submit(miss_on_worker).result()

    assert worker_decision["status"] == "miss"
    # The main thread's own last decision is untouched by the worker's.
    assert inst.last_cache_decision()["status"] == "hit"


def test_reset_cache_clears_the_decision():
    inst = _inference()
    inst._action_cache = None
    inst._last_temporal_action_decision = None
    inst._episode_cache = None
    from tether.runtime.action_fast_path import ActionFastPath

    inst._fast_path = ActionFastPath(threshold=0.0, max_skips=0, enabled=False)
    _prefix(inst)
    assert inst.last_cache_decision() is not None
    inst.reset_cache()
    assert inst.last_cache_decision() is None


def test_evidence_cache_status_is_not_permanently_na(tmp_path):
    """The end of the chain: a decision reaching write_request(cache=...) must
    land in evidence.cache, which was hardcoded to "n/a" for every record."""
    from tether.runtime.record import RecordWriter

    inst = _inference()
    _prefix(inst)
    _prefix(inst)
    decision = inst.last_cache_decision()

    rec = RecordWriter(
        tmp_path,
        model_hash="abc123def4567890",
        config_hash="0123456789abcdef",
        export_dir=str(tmp_path / "export"),
        model_type="pi0.5",
        export_kind="decomposed",
        providers=["CPUExecutionProvider"],
        gzip_output=False,
    )
    rec.write_request(
        chunk_id=0,
        image_b64=None,
        instruction="pick",
        state=None,
        actions=[[0.0]],
        action_dim=1,
        latency_total_ms=1.0,
        cache=decision,
    )
    rec.close()

    import json

    records = [json.loads(line) for line in rec.filepath.read_text().splitlines() if line.strip()]
    request = next(r for r in records if r["kind"] == "request")
    assert request["cache"]["status"] == "hit"
    assert request["cache"]["reason"] == "key_match"
    assert request["evidence"]["cache"]["status"] == "hit", (
        "evidence.cache.status regressed back to a constant"
    )
