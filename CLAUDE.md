# CLAUDE.md

Notes for Claude Code, and for anyone new, working in this repository.

## What this is

`fastcrest-tether` is a CLI and library that exports VLA (vision-language-action)
robot policies to ONNX/TensorRT, serves them on edge hardware (Jetson, RTX,
Apple Silicon, AMD), and produces numerical-parity evidence against the
reference checkpoint. Roughly 116k lines of Python under `src/tether/`.

The thing to understand up front is that the product is the evidence, not just
the deployment. Most of the surface area (`prove`, `verify`, `release assure`,
`promote`, `rollout gate`, and the parity / latency / realtime certs) exists to
produce signed, provenance-bound receipts that a change is safe to ship. So
treat anything named `*_cert`, `*_receipt`, `*_proof`, `verification_*` or
`parity_*` as load-bearing. It isn't scaffolding.

## Commands

```bash
uv pip install -e ".[dev,onnx,serve]"   # what CI installs, use this locally too
pytest tests/                            # full suite
pytest tests/test_foo.py -v              # single file
ruff check . && ruff format --check .    # lint (CI pins ruff==0.15.10)
mypy --config-file=pyproject.toml src/   # types (CI pins mypy==1.20.1)
pre-commit run --all-files               # ruff + ruff-format + mypy + hygiene
tether doctor                            # probe the local box for install and GPU evidence
```

Python 3.10+, line length 100. CI runs 3.10 and 3.12.

## Layout

* `src/tether/` is the package. Biggest subsystems by size: `runtime/` (21k
  lines, serving plus ZMQ and transports), `models/` (15k, the VLA family
  implementations), `exporters/` (10k, ONNX/TRT export including the monolithic
  path), then `curate/`, `pro/` (licensing, Ed25519), `finetune/`, `eval/`,
  `embodiments/`, and `kernels/` (JIT CUDA).
* `src/reflex/` is a backwards-compat shim and nothing else. The package was
  renamed from `reflex` to `tether` in v0.12.0, and the shim just re-exports and
  emits a `DeprecationWarning`. It goes away in v0.14.0. Never add logic there.
  Change `tether` and let the shim re-export.
* The top-level `src/tether/*.py` files are the evidence and verification layer:
  `parity_cert.py`, `realtime_cert.py`, `deploy_proof.py`, `promote.py`,
  `release_assurance.py`, `receipt_provenance.py`, `verify*.py` and friends.
* `tests/` is 239 mostly flat files with a nearly empty `conftest.py`. All it
  does is pin `COLUMNS` and `TERM` so Rich renders plain text (see #299).
* `scripts/` is 143 files, a lot of it one-off local investigation named
  `local_*.py` or `spike_*.py`. It isn't a public surface, so don't assume any
  of it still runs.
* `.github/workflows/` has `pytest`, `quality-ratchet`, `quality-baseline-update`
  (which despite the name is the manual protected policy update run),
  `parity-receipts`, `verify-artifact-receipt`, and the docker and install smoke
  tests.

## Things that will bite you

**A green CI run does not mean parity was checked.** Receipt-gated tests call
`require_receipt()` from the root-level `receipt_test_support.py`, which
defaults to `../reflex_context/`. That directory does not exist in this repo,
it's private context. So unless `TETHER_RECEIPT_DIR` is set, or
`TETHER_REQUIRE_RECEIPTS=1`, or you're on a `release/*` branch, those tests skip
silently rather than failing. This is #292. Don't read a green run as proof of
numerical parity.

**The quality ratchet is protected and fails closed.** `quality-ratchet.yml`
runs via `pull_request_target` from protected `main` and compares ruff and mypy
findings between the base commit and the PR head. There's no checked-in
baseline. An ordinary PR cannot modify the ratchet script, the workflows,
CODEOWNERS, or the effective `[tool.ruff]` and `[tool.mypy]` sections of
`pyproject.toml`. Changing those needs the manual protected policy update run.
`tool.mypy.plugins` is forbidden outright, because mypy imports plugin code
during analysis and that would execute PR code. See `docs/ci_quality_ratchet.md`.

**mypy is non-strict, with one large blanket silence.**
`tether.exporters.monolithic` disables `assignment`, `attr-defined`,
`method-assign`, `misc` and `no-redef`. Don't widen that override. Fix the
annotation instead.

**The dependency pins in `pyproject.toml` carry load-bearing comments.** The
numpy bound is split by `platform_machine` because aarch64 and JetPack need
numpy <2 while lerobot needs >=2. `transformers` is capped below 5.4 because
5.4+ has a q_length regression that breaks the onnx-diagnostic patches. Each of
those comments records a specific production failure, so read it before touching
the bound.

**`uv.lock` is committed but CI installs with pip and floating deps** (#294), so
editing the lock does not change what CI actually resolves.

**Tests are Linux only.** macOS gets a `doctor` smoke test and nothing else
(#298). GPU and hardware paths skip via `@pytest.mark.skipif` or the `hardware`
marker, which you enable with `RUN_HARDWARE_TESTS=1`.

## Conventions

* The CLI is Typer plus Rich in `src/tether/cli.py`. Errors go to `err_console`
  (stderr) and normal output goes to `console` (stdout). Subprocess callers
  capture the two separately, so never print a failure to stdout. We've already
  shipped a bug from getting this wrong.
* Pytest markers are `asyncio` (strict mode, so decorate coroutine tests
  explicitly), `hardware`, and `receipt`.
* Bug fixes get a regression test, features get a test. One concern per PR,
  branched off `main`.
* `GOALS.yaml` is a running mission and status log. It's currently stale and
  still refers to `reflex-*` infra names (#296), so don't treat it as current
  truth.
* Licensing is inconsistent right now: `license = "BUSL-1.1"` alongside an
  Apache classifier (#297). Don't propagate either until that's settled.
