# Local policy recipes

The local LeRobot v3 input contract admits `diffusion`, `vqbet` and `tdmpc`
with `mode=full`, `phase=train`, an immutable dataset content digest and explicit
recipe settings. ACT and pretrained SmolVLA retain their existing contracts.
Admission does not establish task quality, export parity or robot support.
Use `skip_export` for families without an exporter.

Diffusion cameras must share their RGB shape. Each side must be at least 32
pixels, with at least one side larger than 32 for ResNet18 initialization.
VQ-BeT and TDMPC accept exactly one RGB camera of at least 64 pixels per side;
TDMPC requires square images. State and action vectors come from the dataset.
Unsupported observation fields are refused explicitly.

TDMPC additionally needs `next.reward`, logical shape `[1]`, stored as a finite
scalar float in every transition. Singleton list storage is refused because
native TDMPC expects a scalar reward batch. This recipe performs offline
world/reward/Q learning; it does not implement SAC's online actor/learner loop.
VQ-BeT needs a completed VQ codebook phase before inference.

Local resume for these three families verifies the dataset again and selects a
complete numeric checkpoint. Saved LeRobot 0.6.1 config, trainer step, optimizer,
RNG and scheduler where applicable are required. Family, dataset root, step
budget, batch, seed and explicit LeRobot overrides must match. Changed settings,
old 0.5.1 configs or incomplete state return named refusals. The caller must
retain the checkpoint directory and dataset. Exact numerical equality with an
uninterrupted run is unverified.

Studio's adapter adds atomic 500-step saves, latest-two retention, checkpoint
hash review and CPU/MPS settings. Those are Studio responsibilities rather than
new Tether defaults. Local provider charges are $0. No hardware or paid-service
qualification follows from these recipes.

Verified in simulation on 10/15/2026: native local dataset VQ-BeT and TDMPC CPU
four-step plumbing and saved checkpoints; TDMPC uses explicitly synthetic zero
rewards for that probe. Full Tether suite: 4,222 passed, 140 skipped with the
S-11 interpreter and supplied MCP/async dependencies. Hardware, MPS numerical
qualification for VQ-BeT/TDMPC and learned task quality remain unverified.
