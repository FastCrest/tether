# Record & replay

Every `/act` request + response from `tether serve` can be captured to a JSONL trace file so physical-robot bugs become reproducible on a dev laptop. Traces can be:

- **Replayed** against the same model to verify determinism (`cos ≈ 1.0`)
- **Diffed** against a different model to spot regression (e.g. did v0.5 regress vs v0.4?)
- **Compared** with `tether policy diff` before promotion, including shadow-policy action deltas, latency regressions, and guard regressions
- **Fed to A2C2** as a training corpus for the asynchronous action correction head
- **Handed to Tether support** as a reproduction artifact for a bug

Two layers, two jobs — don't confuse them:

| Layer | Purpose | Format | Sink | Use case |
|---|---|---|---|---|
| **Record/replay (this doc)** | Bit-exact replay + cosine diff | Custom JSONL schema v1 | Local `.jsonl.gz` file | "Did model B reproduce model A's actions on this trace?" |
| **OTel tracing** (see `src/tether/runtime/tracing.py`) | Live observability + debug UI | OTel spans (`gen_ai.*` + `tether.*`) | Phoenix / any OTLP backend | "Why did /act take 800ms at 14:32 yesterday?" |

Both can run simultaneously. The `/act` hook emits `tether.record.seq` on the OTel span so a single record can be grepped in either ledger by seq.

## Record

```bash
tether serve ./export/pi05 --record /var/log/tether/traces --embodiment franka
```

One file per server session, named `<YYYYMMDD>-<HHMMSS>-<model_hash>-<session_id>.jsonl.gz` (UTC). Default: gzipped + image hashes only.

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--record <dir>` | disabled | Path to the directory that will hold the trace file. Directory is auto-created. |
| `--record-images <mode>` | `hash_only` | `full` (base64 JPEG kept; ~40 MB/1k calls gzipped), `hash_only` (image_sha256 only; ~0.9 MB/1k calls gzipped), `none` (image field dropped entirely). |
| `--record-no-gzip` | off | Write plain `.jsonl` instead of `.jsonl.gz`. Useful for `grep` during dev. Production should keep gzip. |

### Size guidance

Per 1000 `/act` calls, pi0.5 decomposed, 50-action chunks, 7-dim action, 224×224 RGB:

| Mode | Uncompressed | Gzipped |
|---|---|---|
| `full` | ~45 MB | ~40 MB |
| `hash_only` (default) | ~2.8 MB | ~0.9 MB |
| `none` | ~2.1 MB | ~0.7 MB |

Default of `hash_only` is sufficient for replay against a fixed image corpus and small enough for indefinite retention. Full images are only needed when you plan to replay without the original image source.

### Privacy & compliance

Image frames can capture people or proprietary workcells. Default `hash_only` is the safe ship — image content never leaves the robot. If an enterprise customer requires a documented data-retention policy:

- Traces are filesystem-permissioned; wrap the recording dir in your own LUKS / ecryptfs if at-rest encryption is required.
- `--record-images none` drops all image data (even the hash).
- Instruction text is kept as-is. To redact, pre-process instructions before hitting `/act`.

### Degraded mode

If the disk fills mid-session, the recorder catches `OSError`, logs `slug=record-disk-full`, and stops writing. The server continues serving `/act` — recording is never load-bearing for inference. Inspect via `getattr(server, '_recorder', None).degraded` if your orchestrator wants to react.

## Replay

```bash
tether replay ./traces/20260424-171305-7a8b3c1d-<session>.jsonl.gz \
    --model ./export/pi05 \
    --diff all
```

Loads the trace, re-invokes `predict_from_base64` on the target model for each request, compares against the recorded response, prints a per-request line + summary footer.

### Flags

| Flag | Default | Notes |
|---|---|---|
| `trace_file` | *(required)* | Path to a `.jsonl` or `.jsonl.gz` trace. |
| `--model <dir>` | *(required unless --no-replay)* | Target export directory. Usually the same model that was recording (round-trip), but can be different (regression diff). |
| `--diff <mode>` | `actions` | `actions` (cosine + max_abs on action chunks), `latency` (total_ms within ±5%), `cache` (status match), or `all`. |
| `--n <int>` | 0 (all) | Replay first N records only. |
| `--output <path>` | unset | Write machine-readable JSON report — `{summary, header, per_request_diffs}`. CI-friendly. |
| `--fail-on <mode>` | unset | Exit 3 if any diff of that mode fails. Use one of `actions`, `latency`, `cache`. |
| `--no-replay` | off | Parse the trace, print header + record count, skip model load. Useful for inspecting traces and validating their schema. |

### Example output

```
Replay: traces/20260424-171305-7a8b3c1d-c9d4f0ea.jsonl.gz
  reflex_version: 0.3.1
  model_hash:     7a8b3c1d9f2e4a55
  config_hash:    e12f44c7b1a93802
  model_type:     pi0.5
  export_kind:    decomposed
  embodiment:     franka
  redaction:      {'image': 'hash_only', 'instruction': 'full'}
  started_at:     2026-04-24T17:13:05.241Z

Loading target model: ./export/pi05
WARN: trace was recorded with image redaction='hash_only';
      actions diff needs full images. Pass --no-replay to inspect
      the trace, or re-record with --record-images full.

Replaying requests (--n=all, --diff=all):
  seq=   0   actions: cos=1.000000 max_abs=2.09e-07 [PASS]   latency: 98→101ms (+3.1%) [PASS]   cache: hit→hit [PASS]
  seq=   1   actions: cos=1.000000 max_abs=2.11e-07 [PASS]   latency: 103→104ms (+1.0%) [PASS]   cache: miss→miss [PASS]
  ...

Summary:
  replayed: 1843
  diffed:   1843
  actions:  1843/1843 pass (cos≥0.999, max_abs<1e-3)
  latency:  1821/1843 pass (within ±5% of recorded total_ms)
  cache:    1843/1843 pass (status match)
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | All diffs passed (or `--no-replay` completed). |
| 1 | Trace file error (missing, malformed, unknown schema version, bad `--diff` value). |
| 2 | Target model load failed. |
| 3 | `--fail-on <mode>` triggered — at least one request failed that mode's threshold. |

### Known limitations (as of 2026-04-24)

- **Images**: replay requires `--record-images full` traces. `hash_only` traces can be inspected with `--no-replay` but can't be re-invoked through the model.
- **Cache diff**: Day-3 stub; currently compares recorded-vs-recorded (always passes). Real comparison lands when serve surfaces its own cache state on each `/act`.
- **No `--image-dir` hash-keyed lookup** for replaying `hash_only` traces against an external image corpus — Day-5+ feature.
- **No `--seed` override** for diffable determinism — replay runs with whatever RNG state the target happens to produce. Deterministic models (`onnx_gpu` with fixed inputs) don't need this; flow-matching with fresh noise per call does.

## JSONL format (schema v1)

Authoritative spec: `reflex_context/features/01_serve/TECHNICAL_PLAN.md` §D.1 (D.1.3 header, D.1.4 request, D.1.5 latency object, D.1.6 footer, D.1.11 failure modes, D.1.12 non-goals).

Short version:

```jsonl
{"kind":"header","schema_version":1,"reflex_version":"0.3.1","model_hash":"...","config_hash":"...","session_id":"...","started_at":"...", ...}
{"kind":"request","schema_version":1,"seq":0,"chunk_id":0,"timestamp":"...","request":{...},"response":{...},"latency":{...},"denoise":{...},"mode":"onnx_gpu"}
{"kind":"request","schema_version":1,"seq":1, ...}
...
{"kind":"footer","schema_version":1,"ended_at":"...","total_requests":1843, ...}
```

All records carry `schema_version` so readers can dispatch across versions. Additive fields don't bump the version; readers must ignore unknown fields.

### Additive rollout evidence

New request records may include an `evidence` block with stable, compact fields
for deployment proof and rollout gates:

```json
{
  "kind": "tether.rollout_evidence",
  "schema_version": 1,
  "policy": {"model_hash": "...", "config_hash": "...", "routing_slot": "prod"},
  "request": {"episode_id": "ep-1", "request_id": "req-1", "image_sha256": "..."},
  "action": {
    "num_actions": 50,
    "action_dim": 7,
    "raw_present": true,
    "raw_sha256": "...",
    "guarded_sha256": "...",
    "modified_by_guard": true
  },
  "safety": {"guard_present": true, "clamped": true, "clamp_count": 1, "violation_count": 1},
  "latency": {"total_ms": 31.2, "rolling_p95_ms": 38.4, "rolling_p99_ms": 42.0},
  "cache": {"status": "hit"},
  "outcome": {"status": "success", "error_slug": null}
}
```

When a runtime guard modifies the action chunk, records may also include
`action_trace` with `raw_actions`, `guarded_actions`, and their hashes. This is
what lets `tether policy diff` and deployment-proof packets distinguish "policy
changed" from "safety layer clamped it."

## Adding a new schema version

When a breaking change lands (field rename, semantic change):

1. Bump `SCHEMA_VERSION` in `src/tether/runtime/record.py` to the new int.
2. Add a new reader at `src/tether/replay/readers/v<N>.py` following the v1 pattern.
3. Register it in `src/tether/replay/readers/__init__.py` `_READERS` dict.
4. Old traces still work — `load_reader()` dispatches on the file's header.
5. Ship a migration script `scripts/migrate_trace_v<M>_to_v<N>.py` if users have old traces worth upgrading.
6. Document the delta in this file's "Adding a new schema version" section.

## Related

- `src/tether/runtime/record.py` — writer
- `src/tether/replay/readers/v1.py` — reader
- `src/tether/replay/cli.py` — replay CLI + diff functions
- `tests/test_record.py` / `tests/test_replay_reader.py` / `tests/test_replay_diffs.py` — coverage
- `scripts/local_record_smoke.py` — standalone writer smoke test
- `reflex_context/features/01_serve/subfeatures/_rtc_a2c2/record-replay.md` — canonical feature page
- `reflex_context/features/01_serve/TECHNICAL_PLAN.md` §D.1 — wire format spec (authoritative)
- `reflex_context/03_experiments/2026-04-23-phoenix-record-replay-smoke.md` — OTel/Phoenix decision + coexistence rationale
