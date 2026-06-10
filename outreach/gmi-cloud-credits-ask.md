# Tether × GMI Cloud — compute credits ask

**To:** GMI Cloud (sales@gmicloud.ai + LinkedIn route to head of developer relations / partnerships)
**From:** Romir Jain — solo founder, Tether — suranjana.jain@gmail.com
**Date:** 2026-04-24
**Ask:** $100K H100/H200 credits over 12 months (≈50,000 H100-hours at $2/hr), tiered milestones below + a separate hardware ask: brokered access to NVIDIA Jetson dev kits (Orin Nano $249, AGX Orin ~$2K, Thor ~$3.5K) if GMI has an Inception / NVIDIA Embedded relationship

---

## TL;DR

Tether is an open-source CLI that takes any Vision-Language-Action robot model (pi0, pi0.5, SmolVLA, GR00T) and makes it deployable on cheap edge hardware (Jetson Orin Nano, $249). Customer-facing inference happens at the edge; **the heavy lifting — distillation, benchmarking, cross-architecture training, dataset ingestion, parity validation — happens on H100/H200 clouds**, which is exactly GMI's category.

We've shipped 8+ features in the last 14 days, including the first public reproduction of SnapFlow distillation. We need cloud GPU compute to finish Phase 1 (months 0-6) and Phase 2 (hardware bundles, months 6-12). In return: arxiv preprints + workshop papers (CoRL, RSS, NeurIPS efficient-inference) with "compute provided by GMI Cloud" footer; HuggingFace model cards crediting GMI; permanent compute-partner section on docs.tether.dev; GMI logo on `reflex --version` and the `tether serve` startup banner.

---

## Why GMI specifically (and where in the stack you fit)

Tether's whole reason for existing is that VLA inference is hard, expensive, and customer-hostile today. Customers train a 7B-parameter pi0.5 model in PyTorch, then can't deploy it because it takes 30 seconds per action chunk on the Jetson they bought. We turn that into 0.5 seconds via decomposed ONNX export + 1-NFE distilled student + RTC chunking.

The deployment target for Tether's *customers* is cheap edge hardware — Jetson Orin Nano ($249), AGX Orin (~$2K), eventually Thor + custom silicon (Phase 4). **The cloud H100/H200 layer is where the producer-side heavy lifting happens**: SnapFlow distillation runs (multi-day, multi-GPU), cross-architecture training, LIBERO benchmark matrices, parity validation across 4 VLAs × 2 precisions, and customer-trace fine-tuning. That's the workload we'd run on GMI.

GMI's positioning ("inference-first by design," 3.7× throughput claim, scaling-to-zero) is also the right cloud profile for Tether's *direct cloud customers* — robotics startups who don't yet have edge hardware and want to serve their VLA from a hosted endpoint while they prototype. Every Tether customer that ships a hosted endpoint is a candidate to run on GMI directly. Giving us credits gets us into that flywheel — which means GMI logos on the artifacts robotics customers reach for.

There are other GPU clouds we could ask. We're asking GMI because we want to be associated with the inference-first one.

**Separate hardware ask (independent of cloud credits):** the Phase 1 roadmap has Jetson benchmarks blocked on access to physical hardware (Orin Nano $249 today, AGX Orin ~$2K, Thor ~$3.5K Phase 2 prep). We can't run our own E.2/E.3 milestones (published Jetson latency table, real-hardware benchmark CI) without it. **If GMI has an NVIDIA Inception / NVIDIA Embedded partner relationship that could broker a hardware loan or discounted-purchase pipeline, that would be enormously valuable.** Different ask, different SKU, but the same partnership story — Tether's edge deployment numbers are the customer-facing artifact, and GMI logos sit alongside NVIDIA logos on every published latency chart. The credits ask stands regardless of whether the hardware path works out.

## What we've actually shipped (proof of seriousness, not slideware)

Everything below is on GitHub today at `github.com/FastCrest/tether` (MIT license, 0 contributors besides me, ~12K LoC):

| What | Measured outcome | Where |
|---|---|---|
| Decomposed pi0.5 ONNX (split VLM prefix from expert denoise) | **9.79× per-call speedup** on Modal A10G vs monolithic baseline | `src/reflex/exporters/`, ADR `2026-04-21-decomposed-pi05-static-shape-ship` |
| **First public SnapFlow reproduction** (1-NFE student, arxiv 2604.05656) | 96.7% LIBERO @ 1-step inference, vs 93.3% pi0.5 teacher @ 10 steps | `src/reflex/distill/snapflow*.py` + `scripts/modal_export_snapflow_student.py` |
| Per-embodiment configs + JSON schema (Franka / SO-100 / UR5) | 40 tests passing | `src/reflex/embodiments/` |
| Record/replay (JSONL `--record`, `reflex replay`) | 80 tests passing | `src/reflex/runtime/record.py` + `src/reflex/replay/` |
| `tether doctor` — 10 falsifiable diagnostic checks | Maps to 10 specific LeRobot GitHub issues | `src/reflex/diagnostics/` + `docs/doctor_check_list.md` |
| Prometheus `/metrics` + Grafana template | 12 metrics, 90-series cardinality budget | `src/reflex/observability/` + `dashboards/grafana_template.json` |
| Per-axis ActionGuard (NaN/Inf zero-out + clamp + EU AI Act audit) | 16 tests | `src/reflex/safety/guard.py` |
| Prewarm + 6-state health machine + circuit breaker on `/act` | 22 tests; fixes a real "load-balancer-thinks-server-ready-during-30s-warmup" bug | `src/reflex/runtime/server.py` |
| A2C2 transfer-validation gate harness (arxiv 2509.23224) | 42 tests + Modal first-fire findings | `src/reflex/correction/` + `scripts/modal_b4_gate_fire.py` |

Test sweep across the above: **267/267 passing in 2.95s, 0 flakes**.

We also publish honest negative results. Two examples in the public vault:
- `2026-04-22-prefix-cache-moat-honest-finding.md` — falsified our own 5× cross-timestep cache claim after LIBERO showed 0/5 task success
- `2026-04-24-b4-gate-fire-attempt-and-findings.md` — first Modal fire of the A2C2 gate hit 2 unexpected gaps; documented + closed the gate as soft PROCEED with 3 design constraints

That second one is from this week. Discipline shows.

## What we need cloud compute for (concrete, costed at GMI's listed pricing)

Every line below is a *training / distillation / benchmarking* workload that runs on GMI cloud H100/H200. None of these are customer-facing inference (that runs on edge Jetson). GMI's role here is the producer substrate: we run the heavy training and validation; customers deploy the resulting artifacts.

Phase 1 remaining work (months 1.5 → 6):

| Experiment | Workload type | Estimated H100-hours | Cost @ $2/hr |
|---|---|---:|---:|
| B.5 — A2C2 head training + LIBERO eval pass with on/off | training + bench | 100 | $200 |
| E.1 — Cloud latency matrix (A100 + H100 + H200) across 4 VLAs × 2 precisions | benchmark | 50 | $100 |
| Multi-task SnapFlow distillation runs (pi0, SmolVLA, GR00T variants) | training (multi-day) | 2,000 | $4,000 |
| Customer-trace fine-tuning experiments for A2C2 transfer | training | 500 | $1,000 |
| C-series perf compound wins (CUDA graphs, FA3, compile cache) bench matrix | benchmark | 500 | $1,000 |
| Auto-calibration from 10 episodes (D.2) | training | 200 | $400 |
| Self-distilling serve MVP (D.3) | training | 1,000 | $2,000 |
| **Phase 1 subtotal** | | **4,350** | **$8,700** |

Phase 2 (hardware bundles, months 6-12) — partner SKUs are edge devices (Seeed reComputer, SO-ARM, Trossen, ADLINK); the H100 work below is the *training* needed to validate Tether on each bundle's edge target:

| Experiment | Workload type | H100-hours | Cost |
|---|---|---:|---:|
| Per-bundle SKU validation (train students sized for each edge target) | training + bench | 5,000 | $10,000 |
| Long-tail customer dogfood + benchmark publishing | bench + training | 5,000 | $10,000 |
| Continued SnapFlow + A2C2 expansion to GR00T N1.7 + new VLAs | training | 10,000 | $20,000 |
| **Phase 2 subtotal** | | **20,000** | **$40,000** |

Phase 3 prep (Tether Compute Pack appliance, months 12-18):

| Experiment | H100-hours | Cost |
|---|---:|---:|
| Multi-VLA distillation lab (cross-architecture student training) | 15,000 | $30,000 |
| Customer pilot programs with first 10 paid Pro subscribers | 7,500 | $15,000 |
| **Phase 3 prep subtotal** | **22,500** | **$45,000** |

**Total runway ask: ~47,000 H100-hours = ~$94K at $2/hr = $100K credits with margin.**

If $100K is too much, here's the milestone-gated ladder:

- **$10K starter** — closes Phase 1 software (B.5, E.1, perf compound). Gives GMI the case study from a complete v0.4 sprint within 60 days.
- **$25K** — adds SnapFlow expansion to all 4 VLAs + customer-trace fine-tuning. Two arxiv preprints in Q3.
- **$50K** — adds Phase 2 hardware-bundle work. GMI logo on first commercial Tether deployments.
- **$100K** — full ladder above. Phase 1 → 3 prep coverage. Multiple workshop papers.

We'd want H200 access for the larger distillation runs (memory-bound) and H100 for everything else. If GMI has **dedicated cluster availability for short bursts** (e.g., 8x H100 for 24 hours at a stretch), that's load-bearing for the LIBERO matrix work — happy to discuss.

## What GMI gets in return

Concrete and load-bearing — every artifact the robotics community sees from us this year:

1. **arxiv preprint footer:** "Compute provided by GMI Cloud" on every preprint we publish. Phase 1 will produce at least 2: (a) SnapFlow first-public-reproduction methodology + measurements, (b) decomposed VLA serving + RTC + A2C2 stack as a coherent runtime story.
2. **Workshop submissions in ack section:** target venues are CoRL workshops, RSS efficient-robotics workshops, NeurIPS efficient-inference / robot-learning workshops. ~3-month cycle, ~3-5 submissions across Phase 1 + 2.
3. **HuggingFace model cards:** every distilled student we upload to HuggingFace gets a "Compute partner: GMI Cloud" badge in the model card. We currently have weights uploaded; this would add to all existing + new ones.
4. **`reflex --version` + `tether serve` startup banner:** GMI logo / "compute partner: GMI Cloud" on the CLI itself. Every customer who runs Tether sees it. This is permanent, not a campaign.
5. **docs.tether.dev compute-partner section:** dedicated page, top-level nav.
6. **Co-published blog posts** on GMI's site for landmark ships (Phase 1 close, Phase 2 first bundle, etc.). We write the technical content; GMI gets the SEO + customer pipeline.
7. **Customer referral pipeline:** Tether's deployment story is split — academic robotics labs run on edge (Jetson, no cloud), but a growing tier of robotics startups want a hosted inference endpoint while they prototype. For that second tier the current honest answer is "Modal or Lambda or your own H100." With this partnership it becomes "GMI Cloud." That's net-new inference spend GMI gets that wouldn't otherwise come through.
8. **Cloud-vs-edge benchmark content:** every Phase 1 + Phase 2 benchmark publishes side-by-side numbers — H100/H200 (your hardware) vs Orin Nano vs AGX Orin. That's content GMI can use in marketing: "here's exactly where cloud H100 wins vs where edge wins, on the workload of the year (robot foundation models)." We benefit from the data either way; GMI gets first-party benchmark content.

We will not sign exclusivity. We will sign mutual case study + co-marketing rights.

## What we'd like next

A 30-min call to go through the experiment-by-experiment plan and confirm GMI infrastructure access (H100 vs H200, dedicated cluster availability, region for low-latency to a Hugging Face Hub mirror).

Email: suranjana.jain@gmail.com
GitHub: github.com/FastCrest/tether
Hugging Face: (add user's HF org link before sending)

Happy to send a 1-pager version, share the public technical-plan vault, or jump on a call this week.

---

*Footer for the email send: "Tether is MIT-licensed and 100% open source. We are not raising venture funding at this time; this is a compute-partnership ask, not a financing event."*
