# Windows console and CLI distribution follow-up

Observed 2026-10-23 on `fix/whatsapp-device-followup-20261023`, based on public
Tether `14c3a917e4118f1a0a1d47e4d3a6cde4a3fc53c5` (#363). The customer needs
the corrected Studio runtime and a usable desktop test, rather than another
sequence against an old public wheel. This receipt covers source diagnostics
and distribution identity, not a new PyPI release or native Windows/GPU proof.

## Windows Unicode crash

The older tester's report remains reproducible in current source: rendering
`✓ export complete → model.onnx` with Rich into a strict cp1252 TextIOWrapper
raises `UnicodeEncodeError` at the first checkmark. Rich already chooses ASCII
table borders for narrow encodings, but does not escape arbitrary status glyphs
or customer model paths. The CLI's default Console construction dates to
`e8fe39f14893135340c4c6ab1cccd8ea9dfe6f01` (2026-04-13); the stderr console
was added in `39ed011ad` (2026-05-22). Neither configured safe human output.

`tether.console.Console` now adapts only Rich's destination: characters that
cannot be encoded are represented by backslash escapes. Encodable characters
and normal UTF-8 output remain intact. Main CLI status, stderr and logging,
plus chat, evaluation, fine-tuning, distillation and contribution consoles use
this boundary. Chat's direct human token stream uses the same writer while
preserving exact whitespace, leading/trailing newlines and flush calls.

The process stdout/stderr objects and their error policies are unchanged.
JSON and binary output do not pass through the human writer. Before the fix,
the initial five focused cases produced 3 failures and 2 passes; after the fix,
all ten final encoding cases pass. They exercise real Rich output against
strict cp1252/ASCII streams, Unicode paths/status, stderr logging beside JSON
stdout, a real CLI subprocess, chat tool output and token streaming, and a
real evaluation error path. Native Windows console behavior is not measured
by these Mac-hosted TextIOWrapper tests.

## Doctor's five ModelProto failures

This issue is already fixed in #363. The TRT probe used the local name `model`
for its generated ONNX ModelProto, overwriting the optional customer model
path before deployment diagnostics ran. It is now named `probe_model`.
`tests/test_doctor_model_path_followup.py` covers preservation of both an empty
and supplied path, in human and JSON formats, while forcing the TRT probe to
execute. That existing unchanged coverage is reused; no duplicate fix was made.

## Missing `prove` and `bench realtime`

The public distribution gap is real. Reading the live PyPI JSON and inspecting
its wheel without installing it on 2026-10-23 found:

- Latest version: `fastcrest-tether 0.12.0`.
- Wheel: `fastcrest_tether-0.12.0-py3-none-any.whl`.
- Upload time: `2026-06-10T07:47:38.494425Z`.
- Wheel SHA-256: `c9e29b0389a3bfea365ff20a6ef73f74431d9956b0fd641779122bfad79c5261`.
- Its `tether/cli.py` lacks `deploy-proof`, the `prove` registration, and
  `_bench_realtime_cmd`.

Deployment proof was added in `3f47fbac2c0e422b5ed1c7b777f8a39e09adcdae`
(2026-06-12), after that wheel. The realtime command was added in
`87ba890614ffc0daee75e49a82afff093ab304ff` (2026-06-13). Current source still
reports `0.12.0`, so version text or `pip install --upgrade` cannot establish
that a user has the corrected commands. The Studio candidate must bind the
new public source revision and expose its producer identity. No PyPI upload,
package version bump, or package installation was performed here.

## Verification

Interpreter: `/Users/romirjain/dev/studio-wt-lerobot06/.sol-scratch/lerobot06/venv-061/bin/python`
(Python 3.12), with `PYTHONPATH=src`.

```
python -m pytest tests/test_console_encoding.py tests/test_cli.py \
  tests/test_cli_verb_noun.py tests/test_eval_cli.py \
  tests/test_curate_opt_in_cli.py tests/test_distill_cli_integration.py \
  tests/test_chat_regression.py tests/test_finetune.py -q
185 passed, 4 warnings in 5.88s
```

The four warnings are existing `datetime.utcnow()` deprecations in export-cache
metadata. Ruff check and format pass for the new helper and encoding tests.
Existing touched CLI modules retain 38 baseline Ruff findings; comparison to
HEAD found no added diagnostic codes/messages. `git diff --check` passes.
No training, model downloads, paid service, robot or GitHub Actions run occurred.
