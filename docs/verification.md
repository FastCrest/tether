# How to Read Your VERIFICATION.md

The `VERIFICATION.md` file is your **trust receipt** — it proves that the exported ONNX model produces the same outputs as the original PyTorch checkpoint. This guide explains every section in plain English.

The exact rendering code lives in [`src/tether/verification_report.py`](../src/tether/verification_report.py). If you ever see a mismatch between this doc and an actual report, the source file wins.

---

## When is it created?

`VERIFICATION.md` is auto-generated at two points:

1. **`tether export`** — creates a skeleton with file hashes but no parity numbers yet.
2. **`tether validate`** — fills in the numerical parity results.

Until you run `tether validate`, the parity section will say _"Not yet verified. Run `tether validate <export_dir>` to populate."_ The Files + Export metadata sections are populated at export time regardless.

---

## Section-by-section breakdown

### Export metadata

```markdown
- **Model:** `lerobot/smolvla-base`
- **Model type:** smolvla
- **Target:** orin-nano
- **ONNX opset:** 19
- **Denoising steps (baked in):** 10
- **Action chunk size:** 50
- **Tether version:** 0.9.6
- **Platform:** Linux-5.15.0-aarch64
```

| Field | What it means |
|---|---|
| **Model** | The HuggingFace model ID or local path that was exported |
| **Model type** | Architecture family: `smolvla`, `pi0`, `pi05`, `groot` |
| **Target** | Hardware the export was optimized for: `orin-nano`, `desktop`, `thor`, etc. |
| **ONNX opset** | ONNX operator set version. Higher = more ops available. Standard: 19 |
| **Denoising steps** | Number of flow-matching denoise iterations baked into the ONNX graph. More steps = higher quality but slower inference |
| **Action chunk size** | How many future actions the model predicts per inference call |
| **Tether version** | The `tether-vla` package version used for export (auto-filled from `tether.__version__`) |
| **Platform** | `platform.platform()` output where the export was run |

> **For drones:** The action chunk size is typically smaller (20 vs 50) because flight dynamics require faster replanning. The denoising steps may also be lower for latency-sensitive aerial deployments.

---

### Files table

```markdown
## Files

Total: **3 files, 247.5MB**

| File | Size | SHA256 |
|---|---|---|
| `model.onnx` | 245.3MB | `a1b2c3d4...` |
| `reflex_config.json` | 1.2KB | `e5f6a7b8...` |
| `tokenizer.json` | 957KB  | `f9a0b1c2...` |
```

| Column | What it means |
|---|---|
| **File** | Every file in your export directory (excluding `VERIFICATION.md` itself, which is regenerated each time) |
| **Size** | Human-readable file size |
| **SHA256** | Cryptographic hash — if even one byte changes, this hash changes completely |

**Why SHA256 matters:**

- **Integrity:** If you download an export from a teammate or CI, compare the SHA256 to confirm nothing was corrupted in transit or tampered with at rest.
- **Reproducibility:** Two exports from the same model + same settings should produce identical hashes (given identical opset + precision + chunk_size).
- **Audit trail:** For regulated verticals (warehouse safety, traffic management, defense), the SHA256 chain provides verifiable custody from model authorship → export run → fleet deployment.

---

### Parity section

This is the most important part — it appears after running `tether validate`.

```markdown
## Parity

**Verdict:** PASS
**Threshold:** 1e-04
**Fixtures:** 5
**Seed:** 42
**max_abs_diff across all fixtures:** 2.384e-07

| Fixture | max_abs_diff | mean_abs_diff | Passed |
|---|---|---|---|
| 0 | 1.192e-07 | 3.576e-08 | PASS |
| 1 | 2.384e-07 | 4.768e-08 | PASS |
| 2 | 1.192e-07 | 2.980e-08 | PASS |
| 3 | 1.788e-07 | 4.172e-08 | PASS |
| 4 | 1.192e-07 | 3.278e-08 | PASS |
```

The values in this example are float32 ULP multiples (1.192e-07 is the float32 machine epsilon at this magnitude) — characteristic of a correctly-exported model where the only disagreement between PyTorch and ONNX is at the floating-point precision floor. Tether's parity ledger reports first-action `max_abs` of **5.96e-07** for SmolVLA, **2.09e-07** for pi0, **2.38e-07** for pi0.5, and **8.34e-07** for GR00T — all at machine precision, all reproducible with the same seed.

#### Key metrics explained

**`max_abs_diff` (Maximum Absolute Difference)**

The largest difference between any single output value from PyTorch vs ONNX, across all action dimensions.

- `2.384e-07` means the biggest disagreement was 0.000000238 — practically zero.
- **Good values:** `< 1e-04` (the default threshold)
- **Concerning values:** `> 1e-03` — the ONNX model may behave differently in deployment
- **Failing values:** `> 1e-02` — the export is unreliable; do not deploy

> Mental model: "In the worst case, across all test inputs, how far off was any single predicted joint angle (or thrust value for drones)?"

**`mean_abs_diff` (Mean Absolute Difference)**

The average difference across all output values. Always smaller than or equal to `max_abs_diff`.

- Useful for seeing if the error is concentrated in one spot or spread evenly.
- If `mean_abs_diff` ≈ `max_abs_diff`, the error is spread evenly (usually fine).
- If `mean_abs_diff` << `max_abs_diff`, one outlier dimension is noisy (investigate before deploying).

**`Threshold`**

The configurable pass/fail cutoff. Default: `1e-04` (0.0001).

- If `max_abs_diff` < threshold → **PASS**
- If `max_abs_diff` ≥ threshold → **FAIL**

```bash
# Override the threshold
tether validate ./tether_export/ --threshold 1e-3  # more lenient
tether validate ./tether_export/ --threshold 1e-5  # stricter
```

**`Fixtures`**

The number of random test inputs used. Each fixture is a synthetic (image, instruction, state) tuple. More fixtures = higher confidence the parity holds across input distribution.

**`Seed`**

The random seed used to generate fixtures. Same seed + same model + same export settings = identical results. This is what makes the verification **reproducible** by anyone with the same `tether-vla` version.

---

### Reproducer

```markdown
## Reproducer

```bash
tether export lerobot/smolvla-base --target orin-nano --output <dir>
tether validate <dir>
```
```

Anyone with the model ID + target + opset listed in the metadata block can reproduce the entire export + validation pipeline from scratch and verify the SHA256s + parity numbers match. If they don't match, either (a) the model on HF was updated, or (b) the runner is on a different Tether version — the version field at the top tells them which.

---

## Interpreting results by vertical

Different deployments tolerate different levels of `max_abs_diff`. The shipped 1e-04 default is comfortable for most edge-VLA use cases; tightening it for high-precision deployments or loosening it for perception-only models is a deliberate trade-off.

| Vertical | Acceptable `max_abs_diff` | Notes |
|---|---|---|
| **Warehouse arms** (Franka / UR5 class) | `< 1e-04` | Tight tolerance for precise pick-and-place at sub-millimeter scale |
| **Farm robotics / SO-100 class** | `< 1e-03` | More tolerant for coarse outdoor manipulation |
| **Aerial drones** | `< 1e-04` | Flight control requires high fidelity — small per-step diffs compound at 50 Hz |
| **Smart-camera deployments** (retail, traffic management) | `< 1e-03` | Classification-oriented — tolerant of small numerical drift in attention layers |
| **Mining / heavy industrial** | `< 1e-04` | Safety-critical — minimize any deployment-time surprises |

---

## What to do if validation fails

1. **Re-export with default settings:**

   ```bash
   tether export <model> --target desktop --precision fp16
   tether validate ./tether_export/
   ```

   Some custom targets or precisions (`fp8`, `int8`) widen the tolerance gap. Start with `fp16` on the `desktop` target to isolate model issues from quantization issues.

2. **Try a lower opset:**

   ```bash
   tether export <model> --opset 17
   ```

   Some attention-heavy models hit precision-sensitive ops in opset 19; opset 17 falls back to slower but more numerically stable kernels.

3. **Run `tether doctor`:**

   ```bash
   tether doctor --export-dir ./tether_export/
   ```

   Catches the common silent-failure modes (cuDNN version skew, TRT EP loadchain breakage, JetPack mismatches) that manifest as parity failures.

4. **File an issue:** If a shipped model in the registry consistently fails validation, open a GitHub issue with the full `VERIFICATION.md` attached + your `tether doctor` output. The Tether maintainers can reproduce against the registry's expected hash.

---

## See also

- [`docs/cli_reference.md`](./cli_reference.md) — full flag list for `tether export` and `tether validate`
- [`docs/eval.md`](./eval.md) — task-success eval (parity is a necessary but not sufficient condition; pass eval also)
- [`docs/doctor_check_list.md`](./doctor_check_list.md) — what `tether doctor` checks
- [`docs/adding_a_robot.md`](./adding_a_robot.md) — embodiment cookbook (the embodiment config affects state shape passed to the policy)
- [`src/tether/verification_report.py`](../src/tether/verification_report.py) — the renderer (authoritative)
