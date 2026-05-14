# How to Read Your VERIFICATION.md

> The `VERIFICATION.md` file is your **trust receipt** — it proves that the exported ONNX model produces the same outputs as the original PyTorch checkpoint. This guide explains every section in plain English.

---

## When Is It Created?

`VERIFICATION.md` is auto-generated at two points:

1. **`reflex export`** — creates a skeleton with file hashes but no parity numbers yet
2. **`reflex validate`** — fills in the numerical parity results

Until you run `reflex validate`, the parity section will say _"Not yet verified."_

---

## Section-by-Section Breakdown

### Export Metadata

```markdown
- **Model:** `lerobot/smolvla-base`
- **Model type:** smolvla
- **Target:** orin-nano
- **ONNX opset:** 19
- **Denoising steps (baked in):** 10
- **Action chunk size:** 50
- **Reflex version:** 0.2.1
- **Platform:** Linux-5.15.0-aarch64
```

| Field | What It Means |
|---|---|
| **Model** | The HuggingFace model ID or local path that was exported |
| **Model type** | Architecture family: `smolvla`, `pi0`, `pi05`, `groot` |
| **Target** | Hardware the export was optimized for: `orin-nano`, `desktop`, etc. |
| **ONNX opset** | ONNX operator set version. Higher = more ops available. Standard: 19 |
| **Denoising steps** | Number of flow-matching denoise iterations baked into the ONNX graph. More steps = higher quality but slower inference |
| **Action chunk size** | How many future actions the model predicts per inference call |
| **Reflex version** | The `reflex-vla` package version used for export |
| **Platform** | OS and architecture where the export was run |

> **For drones:** The action chunk size is typically smaller (20 vs 50) because flight dynamics require faster replanning. The denoising steps may also be lower for latency-sensitive aerial deployments.

---

### Files Table

```markdown
| File | Size | SHA256 |
|---|---|---|
| `model.onnx` | 245.3MB | `a1b2c3d4...` |
| `reflex_config.json` | 1.2KB | `e5f6a7b8...` |
```

| Column | What It Means |
|---|---|
| **File** | Every file in your export directory (excluding VERIFICATION.md itself) |
| **Size** | Human-readable file size |
| **SHA256** | Cryptographic hash — if even one byte changes, this hash changes completely |

**Why SHA256 matters:**
- **Integrity:** If you download an export from a teammate or CI, compare the SHA256 to confirm nothing was corrupted or tampered with
- **Reproducibility:** Two exports from the same model + settings should produce identical hashes
- **Audit trail:** For regulated verticals (warehouse safety, traffic management), SHA256 provides a verifiable chain of custody

---

### Parity Section

This is the most important part — it appears after running `reflex validate`.

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

#### Key Metrics Explained

**`max_abs_diff` (Maximum Absolute Difference)**

The largest difference between any single output value from PyTorch vs ONNX, across all action dimensions.

- `2.384e-07` means the biggest disagreement was 0.000000238 — practically zero
- **Good values:** `< 1e-04` (the default threshold)
- **Concerning values:** `> 1e-03` — the ONNX model may behave differently
- **Failing values:** `> 1e-02` — the export is unreliable; do not deploy

> Think of it as: "In the worst case, across all test inputs, how far off was any single predicted joint angle (or thrust value for drones)?"

**`mean_abs_diff` (Mean Absolute Difference)**

The average difference across all output values. Always smaller than `max_abs_diff`.

- Useful for seeing if the error is concentrated in one spot or spread evenly
- If `mean_abs_diff` ≈ `max_abs_diff`, the error is spread evenly (usually fine)
- If `mean_abs_diff` << `max_abs_diff`, one outlier dimension is noisy (investigate)

**`Threshold`**

The configurable pass/fail cutoff. Default: `1e-04` (0.0001).

- If `max_abs_diff` < threshold → **PASS**
- If `max_abs_diff` ≥ threshold → **FAIL**

```bash
# Override the threshold
reflex validate ./reflex_export/ --threshold 1e-3  # more lenient
reflex validate ./reflex_export/ --threshold 1e-5  # stricter
```

**`Fixtures`**

The number of random test inputs used. Each fixture is a synthetic (image, instruction, state) tuple. More fixtures = higher confidence.

**`Seed`**

The random seed used to generate fixtures. Same seed + same model = identical results. This is what makes the verification **reproducible**.

---

### Reproducer

```markdown
## Reproducer

```bash
reflex export lerobot/smolvla-base --target orin-nano --output <dir>
reflex validate <dir>
```
```

This section gives anyone the exact commands to reproduce the entire export + validation pipeline from scratch.

---

## Interpreting Results by Vertical

| Vertical | Acceptable `max_abs_diff` | Notes |
|---|---|---|
| **Warehouse arms** | `< 1e-04` | Tight tolerance for precise pick-and-place |
| **Farm robotics** | `< 1e-03` | Slightly more tolerant for coarse outdoor manipulation |
| **Aerial drones** | `< 1e-04` | Flight control requires high fidelity — small diffs compound at 50 Hz |
| **Retail cameras** | `< 1e-03` | Perception tasks are more tolerant of small numerical diffs |
| **Traffic AI** | `< 1e-03` | Classification-oriented — tolerant of action-space drift |

---

## What To Do If Validation Fails

1. **Re-export with default settings:**
   ```bash
   reflex export <model> --target desktop --precision fp16
   reflex validate ./reflex_export/
   ```

2. **Try a lower opset:**
   ```bash
   reflex export <model> --opset 17
   ```

3. **Check for known issues:** Some models have precision-sensitive attention layers. See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

4. **File a bug:** If a shipped model consistently fails validation, open an issue with the full VERIFICATION.md attached.

---

## Further Reading

- [CLI Command Reference](./cli_reference.md) — `reflex export` and `reflex validate` flags
- [Troubleshooting](./TROUBLESHOOTING.md) — CUDA and export errors
- [Adding a Robot](./adding_a_robot.md) — embodiment cookbook
