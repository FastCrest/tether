# Dependency security and compatibility

This version requires Python 3.11 or newer. The patched filelock 4.2.1 and
Pillow 13.0.0 releases require that floor; base cache locks and all declared
image-serving/export extras now require those patched versions. Existing
Python 3.10 environments remain older environments, not patched installations.

## JetPack 6 boundary

JetPack 6 bundles Python 3.10. The observed CUDA 12.6 vendor ONNX Runtime
1.24.0 wheel is cp310-only. The installer checks for a binary matching the
selected Python before installing numpy, torch or Tether. A missing matching
wheel stops the installation; it never recommends a Python downgrade, changes
CUDA, or silently substitutes CPU inference. A matching downloadable wheel is
availability evidence, not native execution qualification.

The thin ARM64 Dockerfile now selects Python 3.11 and checks that vendor wheel
before package installation. Its CUDA execution remains unqualified on that
new interpreter. The historical L4T default in `Dockerfile.jetpack` bundles
Python 3.10 and now fails with an explicit version message; a replacement
Python/GPU-wheel base requires separate qualification. No container was
published by this change.

For an explicit CPU path, use a separate Python 3.11+ environment and install
`fastcrest-tether[serve,onnx]`. Do not replace the JetPack system interpreter.
LeRobot model export still requires a Python 3.12+ export host.

## Remaining dependency work

Transformers 5.5.4 remains the qualified LeRobot 0.6.1 export cohort. It has
[chat-template file-write advisory GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w)
and [custom-generation cache-write advisory GHSA-x9r9-c232-4q39](https://github.com/advisories/GHSA-x9r9-c232-4q39).
The first is patched in 5.10.0; stable 5.10.4 is outside both published affected
ranges. A direct candidate upgrade failed eight existing native/golden tests
because LeRobot still passes `cache_position` to the changed causal-mask API.
An independently tested compatibility change is required before publishing a
new model-export dependency cohort. No advisory is suppressed in CI.

The checked-in `uv.lock` predates the package rename and names `reflex-vla`.
A complete refresh remains blocked by an existing manifest conflict:

```sh
uv lock --upgrade --dry-run --no-progress
```

For Linux aarch64 with Python >=3.12, the base NumPy <2 JetPack ABI constraint
conflicts with LeRobot 0.6.1 requiring NumPy >=2 through `eval-local`, native and
monolithic extras. This change does not override NumPy ABI constraints, omit a
platform, remove an extra, or relabel that stale lock as current. A separate
cohort decision is needed before a truthful universal lock can be generated.
Use the exact installed-runtime audit artifact to assess the tested CPU serving
environment; it does not prove all optional GPU/export cohorts are patched.
