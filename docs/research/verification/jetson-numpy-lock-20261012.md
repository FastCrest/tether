# Jetson NumPy boundary and universal lock

Date: 2026-10-12. Checkout: `FastCrest/tether`, branch
`fix/uv-lock-numpy-aarch64`, base `e196b79` after PR #355. The initial
worktree was clean. Existing qualification in
`lerobot-06-export-port-20261012.md` covers Linux x86-64 and Mac arm64
`uv pip compile`, not a universal lock or Jetson NumPy 2 execution.

The customer is a Tether developer installing local evaluation on a desktop
GPU host, or deploying exported policies on Jetson. Done means a resolvable
universal lock, preservation of the Jetson deployment dependency boundary,
an actionable refusal before local-evaluation setup mutates a Jetson, a CI
freshness check, and the requested full suite with zero failures.

## Root cause and history

- VERIFIED: `uv lock --check` at the base fails because LeRobot 0.6.1 needs
  `numpy>=2.0,<2.3`, while base dependencies require `numpy>=1.24,<2` on
  `platform_machine == 'aarch64'`. Universal locking evaluates this branch
  even on the current Mac arm64 host. The committed lock lacked the current
  aarch64 split and optional dependency graph.
- VERIFIED: `git log -S 'numpy>=1.24.0,<2.0' -- pyproject.toml` identifies
  `543d229` on May 17 as the global cap's introduction. Its message names
  prevention of an ABI crash with Jetson Zoo wheels. `a72d054`, PR #143,
  on May 20 scopes it to aarch64. `install.sh` independently pins NumPy <2
  before installing Jetson torch and ORT.
- VERIFIED: `c6efffc` introduced `eval-local` with Python/Linux markers but
  no architecture boundary, before PR #355. LeRobot 0.5.1 already required
  NumPy >=2. PR #355 carries the conflict forward with 0.6.1. `native`,
  `monolithic`, `lerobot`, and `rtc` have the same incompatible branch.
- VERIFIED: the existing host contract in `docs/eval_local_linux.md` is
  Linux x86-64. Setup invokes uv after apt installation; readiness previously
  checked only Linux, module availability and CUDA before loading LeRobot.

## Current external evidence

Grades distinguish direct observations, externally reported behavior, and
unmeasured device compatibility. Package metadata alone does not establish
binary ABI compatibility.

| Grade | Claim and evidence |
| --- | --- |
| VERIFIED | Direct HTTP 200 reads of the [cu126 torch index](https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/torch/) list torch 2.8.0, 2.9.1, 2.10.0 and 2.11.0, all cp310 aarch64. The [cu129 torch index](https://pypi.jetson-ai-lab.io/jp6/cu129/+simple/torch/) lists 2.8.0 cp310/cp312, 2.9.0 and 2.9.1 cp310. Tether's Jetson installer leaves these versions unpinned, with base torch >=2.1. |
| VERIFIED | The [cu126 ORT index](https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/onnxruntime-gpu/) lists ORT 1.23.0 and 1.24.0 cp310. The [cu129 ORT index](https://pypi.jetson-ai-lab.io/jp6/cu129/+simple/onnxruntime-gpu/) lists 1.22.0 and 1.23.0 cp310/cp312, plus 1.24.0 cp310. Neither lists 1.25.1. Jetson uses these separate indexes, not the generic GPU extra's PyPI floor. |
| VERIFIED | Downloaded cu126 ORT 1.24.0 wheel has SHA-256 `d980b934b9a29c1a9d6f39751edd7662b69fadd75556a10ff363773a58ce0950`, matching both index listings. Its METADATA declares Python >=3.10 and NumPy >=1.21.6 with no upper bound. This is dependency metadata, not an execution result. |
| VERIFIED | NVIDIA's [JetPack 6.0 PyTorch redistributable directory](https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/) lists torch 2.4.0a0+nv24.05 cp310; the [6.1 directory](https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/) lists 2.5.0a0+nv24.08 cp310. This establishes availability, not NumPy 2 qualification. |
| ASSERTED(URL) | NVIDIA's [PyTorch installation guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) prescribes NumPy 1.26.1. Its examples do not certify the newer Jetson AI Lab wheels against NumPy 2. |
| ASSERTED(URL) | [ORT issue #27562](https://github.com/microsoft/onnxruntime/issues/27562) reports a NumPy 1.x ABI import failure with NumPy 2.0 on Jetson Orin NX using the cu126 ORT 1.23.0 cp310 wheel, hash `4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b`. The issue is closed and tagged stale; closure does not establish a corrected wheel. |
| UNKNOWN | NumPy 2 import, torch/NumPy tensor conversion, and CUDA inference compatibility across the listed torch and newer ORT wheels on each supported JetPack/L4T combination. No Jetson hardware execution occurred here. The old blanket comment about all bundled CUDA libraries and JAX was broader than the available evidence and was removed. |

Tether currently defaults its Dockerfiles to L4T r36.2.0/Python 3.10 and cu126.
`install.sh` selects cu126 by default and cu129 for its 6.2+ branch. These
existing mappings were inspected, not requalified or changed by this fix.
Newer JetPack/Thor stacks are not proof for these existing deployment paths.

## Decision and implementation

Retain the NumPy cap. There is no device evidence sufficient to lift it across
Tether's existing Jetson installations. Restrict every `eval-local` dependency
to Linux x86-64 and exclude aarch64 from all LeRobot dependencies. Other extras
must share that boundary or the universal lock still fails. Native export and
local evaluation run on the host; the device consumes exported policies.

The user identifies ADR 0002 as the device-agent-only decision. That ADR was
not found in this public checkout, so its contents are UNKNOWN here; the
implementation follows the user's explicit scope and the verified existing
x86-64 evaluator contract. It does not claim an independently inspected ADR.

The official `scripts/setup_libero_eval_linux.sh` refuses non-x86-64 Linux
with exit 2 and desktop/cloud guidance before driver checks, apt, venv or pip.
`check_local_readiness` also refuses that architecture before importing torch.
PEP 508 extras skip dependencies; they cannot make pip display a custom
refusal for an extra. Direct pip installs on aarch64 therefore skip
`eval-local` dependencies, and the runtime refuses evaluation. This limit is
explicit in the installation documentation.

## Additional stale-lock blockers

- VERIFIED: the repository had no uv version setting. Regeneration uses the
  installed `uv 0.11.2`; `required-version` and pytest CI now pin that same
  version.
- VERIFIED: TensorRT's PyPI `tensorrt-cu13-libs==10.16.1.11` stub cannot
  generate metadata on macOS because it requires a platform wheel. Read the
  published [NVIDIA x86-64 wheel](https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-10.16.1.11-py3-none-manylinux_2_28_x86_64.whl)
  METADATA using HTTP byte ranges. It has no Requires-Dist or Requires-Python.
  Version-specific `tool.uv.dependency-metadata` copies these facts, allowing
  universal resolution without downloading its 3.7 GB runtime on this Mac.
  This does not change TensorRT's dependency requirements or qualify GPU use.
- VERIFIED: PyPI returns 404 for `scservo-sdk` and `scservo_sdk`. LeRobot
  0.6.1's published metadata names `feetech-servo-sdk>=1,<2` for Feetech.
  Inspected that [distribution's source archive](https://pypi.org/project/feetech-servo-sdk/1.0.0/),
  SHA-256 `d4d3832e4b1b22a8222133a414db9f868224c2fb639426a1b11d96ddfe84e69c`.
  It packages `scservo_sdk` and declares pyserial in setup.py. Corrected the
  `[so100]` distribution name while preserving the existing import path.
- VERIFIED: [PyPI ORT GPU 1.25.1 metadata](https://pypi.org/pypi/onnxruntime-gpu/1.25.1/json)
  requires Python >=3.11, conflicting with Tether's >=3.10 universal range.
  Generic `[gpu]` and `[gpu-min]` dependencies now require Python >=3.11
  via markers; `install.sh` refuses these extras on 3.10 before pip with
  upgrade/Jetson guidance. Direct pip on 3.10 skips their GPU dependencies,
  documented in README. No ORT downgrade or Jetson Python floor change.

## Verification

Initial targeted regression run: 30 passed. Initial requested full-suite run:
4223 passed, 138 skipped, 14 warnings in 128.91 seconds, before the additional
lock blockers were fixed. Final dependency verification:

- VERIFIED: `uv lock` resolves 278 packages; `uv lock --check` passes using
  both the working cache and a fresh cache. No hand edits to `uv.lock`.
- VERIFIED: Linux x86-64/Python 3.12 `uv pip compile --extra eval-local`
  resolves LeRobot 0.6.1 and NumPy 2.2.6. Linux aarch64/Python 3.12 resolves
  NumPy 1.26.4 and skips LeRobot and the simulator dependencies. These are
  resolver checks, not Jetson GPU runtime qualification.
- VERIFIED: focused packaging, refusal and install smoke tests pass, 41
  passed and 1 skipped. Stubbed setup tests prove the Jetson refusal happens
  before any driver, uv or apt operation; stubbed bootstrap tests prove the
  Python 3.10 generic-GPU refusal happens before pip.
- VERIFIED: isolated installation of `feetech-servo-sdk==1.0.0` and pyserial
  succeeds. `scservo_sdk.PortHandler` and `PacketHandler` are callable in a
  fresh Python process. No physical arm was connected.
- VERIFIED: Ruff on the changed evaluator/tests, ShellCheck on both scripts,
  Bash syntax checks, and `git diff --check` pass.
- UNKNOWN: executing the dependency lock in a fresh GPU environment, device
  NumPy 2 compatibility, and actual LIBERO simulation on this Mac. The full
  suite uses the user-provided existing Python environment, not uv sync.


Final requested full-suite run: **VERIFIED, 4229 passed, 138 skipped,
14 warnings, zero failures in 125.46 seconds**, exit 0.

```bash
PYTHONPATH="$PWD/src:/Users/romirjain/dev/.sol-out/tether-mcpdeps:/Users/romirjain/dev/.sol-out/tether-asyncdeps" \
  /Users/romirjain/dev/studio-wt-s11/.sol-scratch/venv/bin/python -m pytest
```

The full output is retained at `/private/tmp/tether-numpy-pytest-final.log`.
SHA-256: `5592687e4a138918cd37d3da699be66caf8ecd474ff637e7e7624e82347323c8`.
Changes remain uncommitted; HEAD is still `e196b79`.

## Correction, 2026-10-26

VERIFIED: the change this note describes never reached `main`. PR #357 merged
as `40f892e`, and that commit, like its branch head `dcb3951`, adds only this
file (`git show --stat 40f892e`). `pyproject.toml` still caps NumPy below 2 on
aarch64 with no `eval-local` architecture marker. No eval-local refusal exists,
no workflow runs `uv lock --check`, and `uv.lock` was last changed in #326. At
`432923b`, `uv lock --check` still fails on the conflict named above, LeRobot
0.6.1's `numpy>=2.0,<2.3` against the aarch64 cap. The v0.13.0 changelog has no
#357 entry for this reason.

## Landed, 2026-10-26

The uncommitted change was recovered from the stash `WIP on uv-lock-numpy-aarch64`
and reapplied to `main` at `08b6b4f` (v0.13.0). Two differences from the note
above:

- `main` moved `[gpu]` to the pinned CUDA 12 cohort in #363. The Python 3.11
  markers are now added to those pins, and the `tensorrt-cu13-libs` metadata
  entry is dropped because nothing requires cu13 any more.
- `siglip-export` (#365, added after this note) needs NumPy >=2 through
  `onnx==1.23.2` -> `ml-dtypes>=0.5.4`. It now excludes aarch64 like the LeRobot
  extras, and `export_rtsm_siglip` refuses Linux aarch64 before importing.

VERIFIED: `uv lock` resolves 311 packages; `uv lock --check` passes with a fresh
cache. `uv pip compile` at Python 3.12:
- Linux x86-64: eval-local resolves lerobot 0.6.1 with NumPy 2.2.6;
  siglip-export resolves open-clip-torch 3.3.0 with NumPy 2.5.4.
- Linux aarch64: both extras resolve NumPy 1.26.4 without LeRobot or open-clip.
- macOS arm64 (deployment target 14.0, which onnxruntime 1.30 needs):
  siglip-export resolves.

UNKNOWN, as before: NumPy 2 compatibility on a Jetson.
