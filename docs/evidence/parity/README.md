# Export-parity receipts cited by `src/tether/registry/qualification.py`

Byte-for-byte copies. Check them with `shasum -a 256 -c SHA256SUMS` inside each campaign
directory. `tests/test_registry_qualification.py` opens every receipt the registry cites
and checks its hash, its family, verdict and scope, its producer commit and its model
against the registry entry.

| Campaign | Receipt | Producer commit | Source |
|---|---|---|---|
| `2026-09-18` | `openvla/openvla-export-parity-receipt.json` | `6e84101` | Tether Studio `tether_private/qualification/parity_evidence/2026-09-18/` |
| `2026-09-18` | `groot/groot-export-parity-receipt-strict-tf32.json` (passes only with `NVIDIA_TF32_OVERRIDE=0`) | `4cf9854` | same |
| `2026-10-12` | `pi0/pi0-export-parity-receipt.json`, `pi05/pi05-export-parity-receipt.json` | `b722306` | Studio `.../parity_evidence/2026-10-12/` |
| `smolvla-c83c3163-lang48-20261028` | `smolvla/smolvla-export-parity-receipt.json` | `0b85acd`, tag `evidence/smolvla-parity-0b85acd` | Modal volume `parity-gpu-receipts`, app `ap-9dJx0Pu4syh3rhA2uaKlqt` |
| `smolvla-c83c3163-lang48-20261028` | `smolvla/smolvla-export-parity-receipt-tf32-override-0.json` (diagnostic, TF32 off, does not qualify) | `ba4d020` | app `ap-JubVOROLFNj1HGND6f6enU` |

Producer commits on a PR branch do not survive a squash merge into main. A receipt's
`tether_commit` is kept reachable by an `evidence/*` tag. The SmolVLA tag exists. The
2026-09-18 and 2026-10-12 producers (`6e84101`, `4cf9854`, `b722306`) are not ancestors
of main and have no tag yet.

Every receipt is scoped to its exact hashed ONNX artifact, model revision, Tether commit,
provider and shared input. None of them shows task success.
