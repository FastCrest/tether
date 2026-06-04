# Self-distilling serve (Pro tier)

`tether serve --pro --collect-data` turns Tether into a continuous-learning loop. Customer points at production traffic; every N hours Tether distills a customer-specific 1-NFE student from the customer's own (state, action, reward_proxy) tuples; gates via a 9-gate methodology; hot-reloads the new student. Weekly customer report.

Per ADR `2026-04-25-self-distilling-serve-architecture`. Pro tier ($99/mo). No VLA peer ships this — strategic moat that only Tether can build because we own both the serve runtime and the distill pipeline.

## Quick start

```bash
# 1. Issue a Pro license bound to this hardware (Phase 1: dev license; Phase 1.5
#    moves to a real license server)
python -c "from tether.pro import issue_dev_license, HardwareFingerprintLite; \
    issue_dev_license(customer_id='acme', \
        hardware=HardwareFingerprintLite(gpu_uuid='...', gpu_name='A10G', cpu_count=8))"

# 2. Set your HF token (we use YOUR token, never Tether's)
export HF_TOKEN=hf_your_personal_token

# 3. Serve with Pro + data collection
tether serve ./my-export/ --embodiment franka \
    --pro --collect-data \
    --distill-schedule nightly \
    --hf-repo tether-students/acme-prod

# 4. Inspect last week's auto-distill activity
tether report last-week
```

## What gets shipped automatically

The 4-stage loop runs continuously while the server is up:

| Stage | Trigger | What happens | Where data lives |
|---|---|---|---|
| **Collect** | Every `/act` | One row appended to `~/.tether/pro-data/YYYY-MM-DD.jsonl` (state, action_chunk, reward_proxy, optional image, optional instruction). PII handled per `--pro-collect-faces` / `--pro-collect-instructions`. | Customer disk only |
| **Distill** | Per `--distill-schedule` (nightly / cron / N samples / quality-drop) | SnapFlow + teacher-supervised dual-loss against base + customer mix (default 50/50). Runs on `--distill-runtime modal` (default A100) or local. | Customer's HF repo |
| **Eval** | Post-distill | 9-gate methodology against LIBERO + customer's last 100 episodes. SAFETY gates non-overridable; PERFORMANCE gates `--pro-force` overridable with audit. | Customer's HF repo |
| **Swap** | When eval gate passes | Atomic warm-swap to the new student via the policy-versioning secondary slot. ≤60s SLA. | Live serve |

24h post-swap monitoring window watches for regressions; auto-rollback fires on 2-in-a-row trip signals (configurable via `--rollback-sensitivity`).

## The 9-gate eval methodology

The load-bearing safety primitive. **A bad gate that passes a regressing model = silent customer-model degradation = trust destroyed.** Two-class design with first-failing-gate precedence.

### 3 SAFETY gates (non-overridable)

| Gate | What it checks | Threshold |
|---|---|---|
| **S1** safety-clamp rate | Candidate's action_guard trip rate | ≤ 1.1× baseline AND ≤ 2 / 100 episodes absolute |
| **S2** velocity Wasserstein | Per-joint velocity distribution shift | Wasserstein-1 ≤ 0.15 |
| **S3** per-task no-cliff | Worst single-task regression | ≤ 5pp drop on ANY task |

`--pro-force` cannot bypass these. Even with explicit operator override, a SAFETY failure rejects the swap. (Wasserstein over KL because Wasserstein is bounded on disjoint support — KL fails when the two distributions barely overlap, which is exactly when we most need a real number.)

### 6 PERFORMANCE gates (overridable with `--pro-force` + audit)

| Gate | What it checks | Threshold |
|---|---|---|
| **P1** aggregate success | Wilson 95% lower bound on overall success rate | ≥ baseline lower bound |
| **P2** latency p99 | Inference latency tail | ≤ 1.10× baseline |
| **P3** memory | Process RSS at steady state | ≤ baseline (default; tunable) |
| **P4** action cos | Action-trajectory cos vs teacher on held-out | ≥ 0.85 |
| **P5** per-task Wilson | Per-task lower bound | ≥ baseline − 3pp every task |
| **P6** safety-guard reset rate | Episodes that BOTH failed AND clamped | ≤ baseline |

`--pro-force` requires `--pro-force-audit "<operator-id> + <reason>"` — bypass without audit raises an error. Bypasses log at WARN with full context.

### Statistical knobs

- **Wilson score interval** for proportions (better than normal-approx at small n / extreme p; handles n=0 + p=0/1 edges)
- **Bootstrap 10k resamples** for distributions (configurable via `confidence_level` + `bootstrap_n_resamples`)
- **Refuse swap at < 30 customer episodes** — insufficient statistical power to detect a 3pp regression at 95% confidence per Lens 5

### LIBERO veto

When evaluating against LIBERO, **any safety-gate failure rejects the swap regardless of customer-suite performance**. Prevents customer-distribution overfitting from destroying generalization on the canonical benchmark.

## Post-swap monitoring + auto-rollback

After a successful swap, a 24h / 500-episode rolling window watches three trip signals:

| Signal | Description | Default threshold |
|---|---|---|
| **T1** safety-clamp p95 | Rolling p95 of safety-clamp count | > 2× pre-swap baseline |
| **T2** action cos to previous | Avg cos similarity to the previous-deployed model | < 0.85 |
| **T3** webhook violations | safety_violation count in any 5-min window | > 5 |

`--rollback-sensitivity` controls how many consecutive trips fire the rollback:
- `aggressive` (1): single-strike — fastest reaction, noisiest
- `normal` (2, default): 2-in-a-row — protects against single-blip noise
- `tolerant` (3): 3-in-a-row — most conservative

Rollback target is the secondary slot in the policy-versioning router (the previous-deployed model, kept warm). ≤60s SLA.

## Pro license

```bash
# Phase 1: dev license bound to current hardware
python -m tether.pro.license issue_dev --customer-id acme --tier pro --days 30

# Phase 1.5: real license from the dashboard
# (URL TBD — see Romir's open-items list in the ADR)
```

License lives at `~/.tether/pro.license`. Bound to GPU UUID + GPU name + CPU count. Driver / kernel patches don't invalidate the license; major-version GPU swap or new host does.

24h heartbeat: every successful validation refreshes the local timestamp. After 24h without restart, the license is considered stale and refuses to load. (Phase 1.5 wires a remote heartbeat endpoint for tamper detection; Phase 1 is purely local.)

License absence at startup = exit 1. **Never silent fallback to non-Pro mode.**

## Data residency & privacy

- All data stays on customer disk by default (`~/.tether/pro-data/`)
- Tether never ingests parquet; the distill pipeline reads it locally OR uploads to YOUR HF Hub repo (customer's token, not Tether's)
- 90-day rolling retention (configurable)
- PII handling defaults:
  - **face_blur_mode = blur** (MediaPipe; Phase 1.5 wiring) — opt-in `raw` requires explicit consent re-prompt
  - **instruction_mode = hashed** (SHA-256) — opt-in `raw` requires re-prompt
  - **state_mode = raw** — required for distribution-shift detection; opt-in `hashed` emits a fail-loud warning that drift detection is disabled

GDPR/CCPA: `tether pro consent --revoke` wipes the receipt + the data directory. Idempotent.

## CLI reference

```text
tether serve --pro                                # gate Pro features on a valid license
            --collect-data                        # explicit opt-in (NEVER default-on)
            --pro-license <path>                  # default ~/.tether/pro.license
            --data-dir <path>                     # default ~/.tether/pro-data/
            --distill-schedule <spec>             # nightly | cron:M H * * * | samples:N | quality-drop | manual
            --distill-runtime <local|modal|hf>    # default modal (~$8-12/wk/customer)
            --eval-suite <libero|customer|both>   # default both
            --rollback-sensitivity <agg|normal|tolerant>  # default normal
            --rollback-sla <seconds>              # max wall-clock; default 120
            --hf-repo <org>/<name>                # default tether-students/{customer}-{workspace}
            --pro-force                           # bypass PERFORMANCE gates (NOT safety)
            --pro-force-audit "<operator-id> + <reason>"   # required when --pro-force is set

tether rollback --to <slot>                       # manual rollback CLI
tether report last-week                           # CLI weekly report (default channel)
                --format text|json                # human-readable or scriptable
                --report-channel cli|email:to|slack:webhook   # email + slack are Phase 1.5 stubs

tether pro consent --revoke                       # GDPR/CCPA wipe
```

## Pricing

$99/mo includes:
- 4 distill runs per month (Modal A100, ~$8-12 each)
- 8 eval runs per month (Modal A10G, ~$1-4 each)
- $60 Modal spend cap (overage: distill runs queued + warned; eval runs hard-capped)

Net gross margin: **37-57%** at typical $10-15/wk/customer Modal burn.

The `tether report last-week` CLI shows a budget bar for distill runs / eval runs / Modal $ — operators see when they're approaching the cap.

## What's NOT in Phase 1

- **Email + Slack channel adapters** — Phase 1.5; CLI default works for everyone today
- **Federated distill** across customers — Phase 3 (cross-customer aggregation with privacy posture TBD)
- **On-prem-only mode** (no HF Hub; MinIO/S3-compatible artifact store) — Phase 2; today's HF-private-repo path is the regulated-industry workaround
- **Customer-data fine-tuning composition** with auto-calibration — Phase 1.5
- **Real-customer Modal validation** — gated on Romir-authorized Modal A100 burn ($10-30 estimated)

## Architectural commitments

Per `01_decisions/2026-04-25-self-distilling-serve-architecture.md`:

- Schema v1 parquet + 9-gate methodology + Pro license format are **LOCKED** for Phase 1
- Phase 2 evolution is **additive-only** on the parquet schema (`extra_metadata` field reserved)
- HF Hub default; on-prem documented but not auto-configured Phase 1
- CLI-only weekly report default; email/Slack opt-in
- SAFETY gates are NEVER overridable, even with `--pro-force`
- Customer's HF token, never Tether's (regulated-industry compliance + no shared-credential liability)
- HF-down-mid-swap = FAIL-LOUD abort, NEVER silent fallback

Test surface: 1571/1571 passing across substrate (parquet + consent + license + scheduler + dual-loss + dataloader mix + 9-gate eval + post-swap monitor + rollback + HF Hub + weekly report + drift detection). Modal cross-validation gated on user authorization.
