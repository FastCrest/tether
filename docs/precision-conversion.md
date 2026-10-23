# Validate a smaller monolithic export

`tether export-precision` creates a separate mixed-FP16 serving export from an
existing FP32 monolithic ONNX directory. Input/output names, shapes and dtypes remain unchanged, including integer token
IDs and Boolean masks. Floating public I/O and sensitive normalization/rotary
phase calculations remain FP32. The original directory is preserved.
`export --precision fp16` remains unsupported for monolithic export: changing a
label is not a precision conversion.

Use an export environment with ONNX and ONNX Runtime (`[monolithic,onnx]` when also
exporting a policy). Prepare `.npz` files containing the exact graph input names,
shapes and NumPy dtypes. Supply representative inputs appropriate to your policy,
including its camera/token/mask/noise contracts. Object/pickle arrays are refused.
The command does not generate inputs or choose accuracy tolerances for you.

```bash
tether export-precision ./smolvla-fp32 --output ./smolvla-mixed-fp16 \
  --validation-inputs ./case-1.npz --validation-inputs ./case-2.npz \
  --max-abs-threshold 0.01 --cos-threshold 0.999
```

The thresholds above are an example, not a task or physical safety standard.
Choose and record tolerances before collecting outcomes. Every output of every
supplied case must be finite and meet both explicit tolerances against the FP32
CPU reference. The candidate must preserve the exact I/O contract, pass ONNX and
ORT session validation, and contain fewer measured model bytes. The source must
remain unchanged throughout the operation. An occupied destination is refused.

A passing directory contains fresh `tether_config.json`, graph/external weights,
bundled tokenizer assets and `precision-conversion.json`. The receipt records
source/candidate hashes, input hashes/shapes/dtypes, output comparisons, tolerances,
converter/runtime versions and blocklist. Old provider, parity and certificate
claims are not copied. Failed measured parity prints the failed receipt and leaves
no serving directory. Converter/load errors also refuse publication.

Inspect the receipt before serving the new directory through Studio's **Existing
local export directory** flow. Keep both exports so a stopped server can be
reviewed and restarted against the original. This is an artifact comparison, not
an automatic update or rollback qualification.

Smaller weight storage does not establish TensorRT builder peak memory, Orin fit,
GPU/provider parity, latency, policy accuracy or physical safety. A successful CPU
comparison applies only to its supplied inputs. GPU and target qualification need
separate retained evidence; existing FAIL certificates are not revised by this
conversion.
