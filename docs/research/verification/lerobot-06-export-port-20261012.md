# LeRobot 0.6.1 export port, 2026-10-12

The customer is a Tether user exporting a pi0, pi0.5 or SnapFlow policy with
LeRobot 0.6.1. The required outcome is that each patched prefix, denoise and
inference path executes the same math as that version's native forward.
The checkout started clean on `feat/lerobot-0.6.1` at
`b2d447f2cb6d1841c0bbfaa97c0891e5b95595a5`, PR #355. Changes are uncommitted.
The existing failed GPU run is recorded in Studio's
`docs/research/verification/lerobot-06-parity-20261012.md`; it is reused as
failure evidence, not as successful parity evidence.

## Source comparison and root cause

Compared the installed 0.5.1 and 0.6.1 wheel sources in
`/Users/romirjain/dev/studio-wt-lerobot06/.sol-scratch/lerobot06/venv-051d`
and `venv-061t`, including both `pi0/modeling_pi0.py` and
`pi05/modeling_pi05.py`. The version-pin commit `d7361f8` made the old
private-method calls fail. The tiny CPU pi0 regression reproduced the exact
`AttributeError` before the port.

0.5.1's `_prepare_attention_masks_4d` at pi0 line 623 and pi0.5 line 619
became `lerobot.policies.common.vla_utils.prepare_attention_masks_4d` at
0.6.1 line 93. Both modeling modules import and call that shared function.
This is extraction of duplicated VLA utilities. The default computation is
unchanged: add the singleton head dimension, then select `0.0` or
`OPENPI_ATTENTION_MASK_VALUE` with `torch.where`. The optional `dtype`
conversion in 0.6.1 only runs when requested. Tether uses the same default
as native LeRobot, including float32 additive masks.

Two other numerical changes matter:

- Both native denoise methods now use `clone_past_key_values`, a shared
  tensor-clone helper, instead of `copy.deepcopy`. The prefix cache remains
  unchanged between steps. Teacher export retains its equivalent frozen
  cache update. SnapFlow export now freezes updates too, because graph
  tracing can evaluate a single-step forward repeatedly. SnapFlow pi0
  training uses the tensor-clone helper to preserve both the prefix cache
  and gradients; pi0.5 retains its existing detached training-cache copy.
- Both native image embedding methods removed the 0.5.1 multiplication by
  `sqrt(hidden_size)`. Tether's pi0 export shim still multiplied, producing
  a maximum action difference of 0.5559507608 in the tiny three-step fixture.
  The shared exporter now uses the native image embedding unchanged, and
  the standalone export scripts no longer add that factor.

0.6.1 also extracts the sampling loop into
`lerobot.policies.common.flow_matching.euler_integrate`. Its time schedule
and arithmetic remain `dt=-1/num_steps`, `t=1+step*dt`, `x=x+dt*v`.
The decomposed baked wrapper already performs those same operations.

## Ported paths

- pi0 and pi0.5 monolithic teacher denoise patches.
- Decomposed pi0.5 prefix, per-step expert and baked Euler expert, including
  SnapFlow and state-out students. Student loading installs the same
  export-friendly denoise patch as monolithic export.
- SnapFlow pi0, pi0.5 and state-out denoise/prefix/inference overrides.
- SnapFlow monolithic export for both families, including explicit state
  forwarding for the state-out variant.
- Both SnapFlow training-prefix adapters.
- Standalone pi0/pi0.5 monolithic export scripts, layer-pruning profile,
  state-out smoke and state-out stage-two distillation. Their LeRobot pins
  now match the 0.6.1 helper they call. These Modal scripts were inspected
  and syntax-checked, not executed on GPU.

## Complete private LeRobot call audit

Searched `src/` and `scripts/` for private calls and imports, followed the
receiver types, and checked each owner against the installed 0.6.1 source.
The executable regression also checks the surviving signatures and rejects
removed-mask calls or new private LeRobot imports across those directories.

| Owner and private symbol | 0.6.1 status | Source and disposition |
| --- | --- | --- |
| `PI0Pytorch._prepare_attention_masks_4d` | Removed; all callers ported | Native shared-helper calls at `modeling_pi0.py:616,677,727` |
| `PI05Pytorch._prepare_attention_masks_4d` | Removed; all callers ported | Native shared-helper calls at `modeling_pi05.py:586,644,692` |
| `PI0Pytorch._apply_checkpoint` | Present | `modeling_pi0.py:466`, `(self, func, *args, **kwargs)` |
| `PI05Pytorch._apply_checkpoint` | Present | `modeling_pi05.py:462`, same signature |
| `PI0Policy._preprocess_images` | Present | `modeling_pi0.py:983`, `(self, batch)` returning image and mask lists |
| `PI05Policy._preprocess_images` | Present | `modeling_pi05.py:952`, same signature |

There are no underscore-prefixed imports from LeRobot in these directories.
`config._attn_implementation` belongs to Transformers, is present, and is
assigned by LeRobot 0.6.1 itself. `lerobot.policies.pi_gemma` and its
`create_causal_mask` still exist. Python `__class__` and cache `__dict__`
accesses are not private LeRobot callable symbols.

The original broken pi0 call came from `768a536d`, pi0.5 from `6ffeeff5`,
decomposed prefix from `bd0f7738`, and SnapFlow subclass from `0be7388f`.
The separate legacy import `lerobot.common.policies...` in
`scripts/modal_test_export.py` predates this migration and remains outside
these patched export paths.

## CPU verification

`tests/test_lerobot06_patched_forward.py` creates real random LeRobot models
with two language/expert layers, width 32, eight attention heads, a two-layer
SigLIP vision encoder, vocabulary 64, 16x16 images and a three-action chunk.
Only constructor configuration sizes are replaced. Embeddings, attention,
normalization, projections, caches, losses and integration use real library
code. No checkpoint download or large weights are involved.

Native reference outputs are captured before installing export patches.
Tests compare native full training-forward losses, denoise velocities and
three-step teacher inference. SnapFlow checks zero initialization and a
nonzero learned target-time correction against unpatched LeRobot forward
methods with the student's intended suffix embedding. State-out includes
its explicit state token. Both training adapters are exercised. The
real decomposed teacher and student wrappers compare prefix caches,
per-step velocities and baked integration against native outputs. Their
exported prefix, per-step and baked graphs are executed too. Torch graphs for steps
and full image/language-to-action inference are executed and compared too.
Every float32 comparison uses `rtol=1e-6`, `atol=1e-6`, with masked cameras
and padded language tokens in a batch of two. Subprocesses isolate global
export patches from the rest of pytest.

The requested 0.6.1 environment's actual interpreter is
`/Users/romirjain/dev/studio-wt-lerobot06/.sol-scratch/lerobot06/venv-061t/bin/python`.
It has Python 3.12.12, LeRobot 0.6.1, Transformers 5.3.0 and Torch 2.7.1.
The CPU Torch graphs are diagnostic evidence; production monolithic export
still correctly requires Torch >=2.8. No new GPU/ONNX Runtime acceptance
receipt was produced, and existing qualification verdicts are unchanged.

```bash
PYTHONPATH="$PWD/src" \
  /Users/romirjain/dev/studio-wt-lerobot06/.sol-scratch/lerobot06/venv-061t/bin/python \
  -m pytest -q tests/test_lerobot06_patched_forward.py \
  tests/test_snapflow_backend.py tests/test_monolithic_causal_mask_shim.py

PYTHONPATH="$PWD/src:/Users/romirjain/dev/.sol-out/tether-mcpdeps:/Users/romirjain/dev/.sol-out/tether-asyncdeps" \
  /Users/romirjain/dev/studio-wt-s11/.sol-scratch/venv/bin/python -m pytest
```

The final focused run in `venv-061t` passed: **48 passed in 71.20 seconds**,
including all seven new native/graph/private-API cases, the existing SnapFlow
backend tests and causal-mask tests. Raw output is
`/tmp/tether-lerobot06-final-focused.log`.

The full requested suite ran in 206.12 seconds: **4,169 passed, 133 skipped,
21 failed and 10 errors**. All 31 failures/errors are sandbox-denied TCP
socket binds: two fake HTTP servers and 29 ZeroMQ cases. Each reports
`Operation not permitted`; there were no other failures. The zero-failure
full-suite target therefore requires an environment permitted to bind local
TCP sockets. No test was weakened or skipped to conceal those results.
Raw output is `/tmp/tether-lerobot06-full-suite.log`.

After the final loader and test changes, the affected tests in the S11
interpreter passed: **56 passed, 11 skipped**. LeRobot-specific tests skip
there because that environment lacks LeRobot; they execute in `venv-061t`.
The affected core exporter/distillation files and new regression pass Ruff,
and all edited Python files pass Ruff's undefined-name check. `git diff
--check` is clean.

## Adversarial review (Claude, 2026-10-12)

- Line-by-line against the LeRobot 0.6.1 wheel (`venv-061t`): `vla_utils.prepare_attention_masks_4d`
  (`vla_utils.py:93-102`) is called with no `dtype`, as native does at `modeling_pi0.py:677,727`
  and `modeling_pi05.py:692`. The patched pi0/pi0.5 denoise steps (`monolithic.py` pi0 patch,
  `_apply_pi05_denoise_step_patch`) match native `denoise_step` (`modeling_pi0.py:705-743`,
  `modeling_pi05.py:671-708`) term by term; the two deliberate differences are (a) `F.pad`+AND
  instead of `torch.cat` for the 2D mask, equal by construction when no pad deficit, and (b) a
  frozen `DynamicLayer.update` instead of `clone_past_key_values`; both feed attention
  `cat(prefix_kv, suffix_kv)` and leave the prefix cache unchanged. The decomposed prefix wrapper
  matches native `sample_actions` prefix (`modeling_pi05.py:640-656`). 0.6.1 `embed_image`
  (`modeling_pi0.py:287-296`) has no `sqrt(hidden_size)` factor (0.5.1 had it at `:446`), so
  dropping Tether's shim is correct. `euler_integrate` (`flow_matching.py:93-120`) is
  `dt=-1/n, x+=dt*v`. VERIFIED.
- The CPU references are native: `native_step = cls.denoise_step`, `expected_*` from a
  `copy.deepcopy` of the model, all computed before `apply_export_patches()` and the SnapFlow
  installers run. SnapFlow references call the unpatched class `denoise_step` with only the
  student's suffix embedding substituted. VERIFIED (`tests/test_lerobot06_patched_forward.py:136-232`).
- Mutation checks (file restored by copy, shasum `7d653f5b...aef49` equal before and after each):
  M1 disable the denoise-phase cache freeze in the pi0 patch: 6 of 7 cases fail (pi0: 3-step
  actions max abs 0.118 at line 249). M2 make the pi0.5 patch ignore the prefix pad mask: pi05
  and decomposed fail (velocity max abs 0.012 at line 244). Both mutations killed. VERIFIED.
- Re-run after review: 48 passed in 77.7 s (`venv-061t`). VERIFIED.
