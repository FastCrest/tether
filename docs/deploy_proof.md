# Deployment proof

`tether prove` collects evidence for one question: can this policy move forward?
It proves a specific export can run as a deployment, then writes a packet that
another engineer or CI job can audit. `tether promote` consumes that packet and
returns `PROMOTE`, `BLOCK`, or `ROLLBACK`. `tether deploy-proof` is the explicit
backend command and remains supported for scripts.

```bash
tether prove ./export \
  --embodiment franka \
  --api-key "$TETHER_API_KEY" \
  --record-dir /tmp/tether-proof-traces \
  --policy-diff-baseline ./traces/current.jsonl.gz \
  --policy-diff-candidate ./traces/candidate.jsonl.gz \
  --profile production.yml \
  --samples 100 \
  --output-dir /tmp/tether-deploy-proof
```

The packet contains:

- `deployment-proof.json` - machine-readable receipt.
- `deployment-proof.md` - human-readable summary for PRs or customer handoff.
- `export-manifest.json` - SHA-256 and size for every export file.
- `policy-diff.json` - optional candidate/shadow policy rollout report.
- `profile.json` - effective profile after defaults are merged.
- `server.log` - server log tail from the proof run.
- `MANIFEST.json` - SHA-256 and size for every packet artifact.

## Promotion decision

Use `tether promote` after the packet is written:

```bash
tether promote /tmp/tether-deploy-proof --profile warehouse-safe.yml
```

The command verifies `MANIFEST.json`, reads `deployment-proof.json` and optional
`policy-diff.json`, applies the promotion profile, and returns one decision:

- `PROMOTE` - proof and policy gates passed.
- `BLOCK` - do not promote the candidate.
- `ROLLBACK` - gates failed while the candidate is already active
  (`--candidate-active`).

## What it checks

- Deploy diagnostics via the same checks as `tether doctor --json --model`.
- Server readiness via `/health`.
- `/act` roundtrip samples with TTFA, p50/p95/p99, warm p95, jitter, deadline
  misses, and optional control-Hz budget misses.
- API-key boundary when `--api-key` is supplied: `/act`, `/config`,
  `/guard/status`, and `/guard/reset` must reject unauthenticated calls.
- Prometheus readiness by scraping `/metrics` and requiring Tether metric
  families when the profile requires metrics.
- Optional trace recording when `--record-dir` is supplied or the profile
  requires a record trace.
- Optional policy-diff gate when `--policy-diff-baseline` is supplied. Use
  `--policy-diff-candidate` for baseline-vs-candidate traces, or
  `--policy-diff-shadow` for shadow actions embedded in one trace.
- ActionGuard stress when `--safety-config`, `--embodiment`, or
  `--custom-embodiment-config` is supplied: out-of-range clamp, non-finite
  rejection, and repeated-clamp trip.

## Profile

Profiles are JSON or YAML. Values below override the default profile.

```yaml
name: production
thresholds:
  max_doctor_failures: 0
  max_act_errors: 0
  require_auth: true
  require_metrics: true
  require_record_trace: true
  require_policy_diff: true
  policy_diff_fail_on: any
  require_guard: true
  control_hz: 20
  max_first_roundtrip_ms: 1000
  max_roundtrip_p95_ms: 80
  max_warm_roundtrip_p95_ms: 40
  max_jitter_p95_minus_p50_ms: 10
  max_deadline_misses: 0
  max_missed_control_budget: 0
```

The default profile is intentionally permissive on latency because hardware and
model family vary widely. Production profiles should set concrete p95 and
control-rate thresholds for the robot cell being deployed.
