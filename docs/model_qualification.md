# Model qualification

The model registry describes available checkpoints. It does not, by itself, prove that export, training, task evaluation or physical deployment works.

`src/tether/registry/qualification.py` records the checked state of each model family. Tether Studio reads that contract and keeps unsupported operations blocked.

The current qualified Studio path is SmolVLA LoRA with pinned parent and dataset revisions, followed by native LIBERO evaluation on Linux with CUDA. The retained Studio evidence covers one real five-step adapter and matched development and held-out evaluation.

Other families remain visible for inspection. Their qualification records say whether checkpoint loading or export code exists and whether training or task evaluation still lacks acceptance evidence. A benchmark proves timing for the measured artifact and host. It does not prove task success or device readiness.

Each family record also lists its acceptance requirements. pi0 and pi0.5 need pinned checkpoints, export parity and matched development and held-out task evidence. GR00T needs the same evidence for its own exporter and task adapter. OpenVLA additionally needs tokenized-action parity. Code presence or a passing fixture does not satisfy these requirements.

RTC is not currently qualified as an end-to-end serving path. The server records chunk carry state, but its ordinary request path does not yet inject RTC guidance into the policy denoising loop. Strict CLI configuration rejects policies that do not accept the RTC keyword contract instead of silently returning plain inference. Qualification still requires a per-step expert path plus Linux GPU execution evidence.

When adding a family or registry entry:

1. Add its family to `FAMILY_QUALIFICATIONS`.
2. Pin the upstream checkpoint revision before calling an import reproducible.
3. Record export parity separately from task evaluation.
4. Require matched development and held-out evidence before candidate selection.
5. Keep physical target checks untested until a real enrolled device supplies them.
