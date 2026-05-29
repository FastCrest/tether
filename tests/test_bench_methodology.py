"""Tests for the bench-revamp methodology + report layer (Phase 1).

ISB-1 methodology lifted from EasyInference (sibling project) with VLA-aware
semantics. Verifies:
  - LatencyStats math: percentiles, mean, std, jitter, 95% CI
  - Warmup discard: head samples actually dropped; raises if warmup >= n
  - Report serialization: Markdown + JSON, both consume the same dataclass
  - Environment capture: doesn't blow up on missing nvidia-smi / non-git dir
  - Reproducibility envelope: required fields present in JSON output
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from reflex.bench import (
    BenchEnvironment,
    BenchReport,
    LatencyStats,
    capture_environment,
    compute_stats,
    confidence_interval_95,
)


# ---------- methodology ----------

class TestComputeStats:
    def test_basic_stats(self):
        # Constant latencies → p50/p95/p99 all equal, std=0, jitter=0
        s = compute_stats([10.0] * 100, warmup_n=10)
        assert s.n == 90
        assert s.warmup_discarded == 10
        assert s.mean_ms == 10.0
        assert s.p50_ms == 10.0
        assert s.p99_ms == 10.0
        assert s.std_ms == 0.0
        assert s.jitter == 0.0
        assert s.hz_mean == 100.0  # 1000ms / 10ms

    def test_percentile_ordering(self):
        # 1..100 → p50≈50.5, p99≈99.01, max=100
        s = compute_stats(list(range(1, 101)))
        assert 50 <= s.p50_ms <= 51
        assert 98 <= s.p99_ms <= 100
        assert s.max_ms == 100.0
        assert s.min_ms == 1.0

    def test_warmup_n_discards_head_samples(self):
        # First 10 latencies are 1000ms outliers; should be discarded
        latencies = [1000.0] * 10 + [10.0] * 90
        s = compute_stats(latencies, warmup_n=10)
        assert s.n == 90
        assert s.mean_ms == 10.0  # outliers gone
        assert s.max_ms == 10.0

    def test_warmup_zero_keeps_everything(self):
        s = compute_stats([5.0, 10.0, 15.0])
        assert s.n == 3
        assert s.warmup_discarded == 0

    def test_warmup_too_large_raises(self):
        with pytest.raises(ValueError, match="nothing left to measure"):
            compute_stats([1.0, 2.0, 3.0], warmup_n=3)

    def test_warmup_negative_raises(self):
        with pytest.raises(ValueError, match=">=.*0"):
            compute_stats([1.0, 2.0, 3.0], warmup_n=-1)

    def test_jitter_is_coefficient_of_variation(self):
        # Two-value alternation: mean = 15, std ≈ 5, jitter ≈ 0.333
        s = compute_stats([10.0, 20.0] * 50)
        assert abs(s.mean_ms - 15.0) < 1e-9
        assert abs(s.jitter - (s.std_ms / s.mean_ms)) < 1e-9
        assert 0.3 < s.jitter < 0.4

    def test_ci95_bounds_mean(self):
        s = compute_stats([10.0, 11.0, 12.0, 13.0, 14.0] * 30)
        assert s.ci95_low_ms <= s.mean_ms <= s.ci95_high_ms

    def test_p99_9_present_and_above_p99(self):
        s = compute_stats(list(range(1, 1001)))
        assert s.p99_9_ms >= s.p99_ms

    def test_hz_mean_inversion(self):
        s = compute_stats([20.0] * 50)  # 20ms → 50Hz
        assert abs(s.hz_mean - 50.0) < 1e-9

    def test_to_dict_roundtrip(self):
        s = compute_stats([10.0, 11.0, 12.0])
        d = s.to_dict()
        assert d["n"] == 3
        assert "p50_ms" in d
        # JSON-serializable
        json.dumps(d)


class TestConfidenceInterval:
    def test_returns_tuple(self):
        lo, hi = confidence_interval_95([10.0] * 30)
        assert lo == 10.0 and hi == 10.0  # zero spread

    def test_n_le_1_returns_mean_mean(self):
        lo, hi = confidence_interval_95([42.0])
        assert lo == 42.0 == hi
        # Empty case
        lo, hi = confidence_interval_95([])
        assert math.isnan(lo) and math.isnan(hi)

    def test_normal_approx_widens_with_spread(self):
        narrow = confidence_interval_95([10.0, 10.5, 9.5] * 20)
        wide = confidence_interval_95([5.0, 15.0] * 30)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# ---------- environment capture ----------

class TestCaptureEnvironment:
    def test_capture_does_not_raise_on_nonexistent_export(self, tmp_path):
        # Even if export dir is empty, capture should work (envelope just has
        # empty onnx_files list); raising would defeat the bench-on-broken-
        # state debugging value.
        env = capture_environment(export_dir=tmp_path, device="cpu", seed=42)
        assert env.seed == 42
        assert env.device == "cpu"
        assert env.export_dir == str(tmp_path)
        assert env.timestamp_utc.endswith("Z")
        assert env.onnx_files == []

    def test_capture_records_seed_and_device(self, tmp_path):
        env = capture_environment(export_dir=tmp_path, device="cuda", seed=7, inference_mode="onnx_trt_fp16")
        assert env.seed == 7
        assert env.device == "cuda"
        assert env.inference_mode == "onnx_trt_fp16"

    def test_capture_includes_python_and_platform(self, tmp_path):
        env = capture_environment(export_dir=tmp_path)
        assert env.python_version  # non-empty
        assert env.platform         # non-empty

    def test_capture_handles_onnx_files(self, tmp_path):
        # Drop a fake ONNX
        (tmp_path / "vlm_prefix.onnx").write_bytes(b"fake-onnx-bytes-1234")
        (tmp_path / "expert_denoise.onnx").write_bytes(b"more-fake-bytes")
        env = capture_environment(export_dir=tmp_path)
        assert len(env.onnx_files) == 2
        # Sorted by name
        names = [f["name"] for f in env.onnx_files]
        assert names == sorted(names)
        for f in env.onnx_files:
            assert f["bytes"] > 0
            assert len(f["sha256_prefix"]) == 16

    def test_capture_includes_external_data_files(self, tmp_path):
        # Large monolithic exports save weights as model.onnx.data; the bench
        # receipt must hash that file too, not only the lightweight protobuf.
        (tmp_path / "model.onnx").write_bytes(b"protobuf")
        (tmp_path / "model.onnx.data").write_bytes(b"external-weights")
        env = capture_environment(export_dir=tmp_path)
        names = {f["name"] for f in env.onnx_files}
        assert names == {"model.onnx", "model.onnx.data"}


# ---------- report rendering ----------

class TestBenchReport:
    def _stats(self):
        return compute_stats([10.0, 11.0, 12.0, 13.0, 14.0] * 20)

    def _env(self, tmp_path):
        return capture_environment(export_dir=tmp_path, device="cuda", seed=42)

    def test_to_json_is_valid_json(self, tmp_path):
        r = BenchReport(stats=self._stats(), environment=self._env(tmp_path))
        parsed = json.loads(r.to_json())
        assert parsed["schema_version"] == 1
        assert "stats" in parsed
        assert "environment" in parsed
        assert parsed["parity"] is None

    def test_json_has_canonical_keys(self, tmp_path):
        r = BenchReport(stats=self._stats(), environment=self._env(tmp_path))
        parsed = json.loads(r.to_json())
        for key in ("n", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "p99_9_ms",
                    "max_ms", "std_ms", "jitter", "ci95_low_ms", "ci95_high_ms"):
            assert key in parsed["stats"]
        for key in ("timestamp_utc", "git_sha", "python_version", "platform",
                    "device", "seed", "onnx_files"):
            assert key in parsed["environment"]

    def test_to_markdown_contains_methodology_section(self, tmp_path):
        r = BenchReport(stats=self._stats(), environment=self._env(tmp_path))
        md = r.to_markdown()
        assert "# Reflex Bench Report" in md
        assert "Per-chunk latency" in md
        assert "Reproducibility envelope" in md
        assert "What this measures" in md
        assert "ISB-1" in md  # methodology source citation

    def test_markdown_includes_stats_values(self, tmp_path):
        r = BenchReport(stats=self._stats(), environment=self._env(tmp_path))
        md = r.to_markdown()
        # mean = 12.0 ± something; should appear somewhere
        assert "12.00" in md

    def test_parity_renders_when_set(self, tmp_path):
        r = BenchReport(
            stats=self._stats(),
            environment=self._env(tmp_path),
            parity={"cos": 0.999999, "passed": True, "threshold": 0.9999},
        )
        md = r.to_markdown()
        assert "Parity check" in md
        assert "PASS" in md
        assert "0.999999" in md

    def test_parity_fail_renders_fail(self, tmp_path):
        r = BenchReport(
            stats=self._stats(),
            environment=self._env(tmp_path),
            parity={"cos": 0.5, "passed": False, "threshold": 0.9999},
        )
        md = r.to_markdown()
        assert "FAIL" in md

    def test_notes_propagate_to_markdown(self, tmp_path):
        r = BenchReport(
            stats=self._stats(),
            environment=self._env(tmp_path),
            notes=["warmup=20 discarded", "TRT engine cached"],
        )
        md = r.to_markdown()
        assert "warmup=20 discarded" in md
        assert "TRT engine cached" in md

    def test_write_markdown_creates_parent_dir(self, tmp_path):
        r = BenchReport(stats=self._stats(), environment=self._env(tmp_path))
        out = tmp_path / "deep" / "nested" / "bench.md"
        r.write_markdown(out)
        assert out.exists()
        assert "Reflex Bench Report" in out.read_text()

    def test_write_json_roundtrips(self, tmp_path):
        r = BenchReport(stats=self._stats(), environment=self._env(tmp_path))
        out = tmp_path / "bench.json"
        r.write_json(out)
        parsed = json.loads(out.read_text())
        assert parsed["schema_version"] == 1
        assert parsed["stats"]["n"] == 100  # 5 values × 20
