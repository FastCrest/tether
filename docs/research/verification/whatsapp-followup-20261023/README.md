# Tester follow-up, 2026-10-23

Customer job: install the corrected Studio desktop on the WSL2/5090 workstation,
run the requested simulation, serve the existing export and download usable
evidence. This source follow-up is based on public Tether `14c3a917` and retains
prior provider, dependency, Doctor and ScatterND fixes without rewriting their
historical device results.

## Corrections

- Prefer bounded Jetson device-tree board identity over generic GPU/tegrastats
  output. Nano/Super selects the Nano profile. Unidentified/unsupported Jetson
  identities set an explicit admission flag; `go` refuses before model resolution,
  download, cache creation or serving. Supported explicit overrides and ordinary
  CPU behavior retain their earlier contracts. Generic Orin-to-AGX guessing dates
  to `29494ad7`; it is not a safe memory-fit assumption. WSL's absolute
  `nvidia-smi` path is tried only when its PATH command is absent.
- Escape human-output characters a Windows cp1252/ASCII destination cannot encode.
  Normal UTF-8, JSON, stream error policy, chat whitespace and flushes are preserved.
  [Root cause, tests and the old PyPI wheel identity](console-and-distribution.md).
- Retain raw proof samples, quantiles, first/warm latency, jitter and control budgets
  before profile/certificate comparisons. A measured 50.04 ms fails 50 ms; 50.0 ms
  passes. Explicit rates and thresholds refuse nonfinite/boolean/invalid inputs,
  and receipt-derived periods cannot become infinite budgets.

## Verification and review

[344 affected cases pass](public-affected-tests.log.gz) after all source changes
and review corrections, in 8.38 seconds with six existing datetime deprecation
warnings. [Exact changed source hashes and review scope](source-checks.json).
Failing-first reproduction covered cp1252 output, Nano/AGX classification,
unsupported Jetson admission, threshold rounding and an infinite-rate admission
case. Independent cross-review accepted the final changes. No broad all-product,
native Windows/WSL, GPU, model-fit or physical qualification follows.

Doctor's ModelProto defect was already fixed in #363. The published 0.12.0 PyPI
wheel predates proof/realtime commands despite current source retaining that
version label. Studio must ship its exact new source identity; upgrading that old
wheel is not the tester delivery. No PyPI release, mutable update-feed change,
Actions/workflow change, paid GPU or robot operation occurred here.

The Orin full-model engine-build failure remains unresolved by these source
checks. Existing hardware certificates remain FAIL. Studio's synthetic HTTP
receipt does not establish consumed action horizons, physical execution or RTSM
policy-snapshot integration. Studio owns protected build/installation evidence and
replacement tester instructions before describing the download as updated.

## Protected quality follow-up

The first GitHub protected comparison reported one new mypy finding: the board
classifier's local `cls` was inferred as a string before assignment of an
optional GPU match. Naming the board result separately corrects the type without
changing behavior; all 58 hardware/go cases pass again. The initial local quality
check could not complete because this host's managed Python injects an unrelated
NumPy stub using syntax newer than the project's configured type target. That
operational failure is not a passing quality comparison. The normal protected
GitHub comparison will determine the corrected source's status; no gate or
workflow configuration is changed.
