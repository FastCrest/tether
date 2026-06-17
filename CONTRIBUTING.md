# Contributing to tether

thanks for looking at tether. this is a deploy layer for VLAs (SmolVLA, pi0, pi0.5, GR00T N1.6) that gets you from `git clone` to a running policy server in minutes, with numerical parity to the reference checkpoint.

## quick start

```bash
# install from source
git clone https://github.com/FastCrest/tether && cd tether
uv pip install -e .

# sanity check your box
tether doctor

# run a test policy
tether serve --policy smolvla --device cuda
```

full setup details in `examples/01-chat-quickstart.md`.

## how to contribute

two paths depending on how involved you want to be:

### path 1: open a PR (no approval needed to start)

1. fork the repo
2. pick an issue tagged `good first issue` or `help wanted` (or propose your own, open an issue first to check fit)
3. branch off `main`, keep the change scoped, one concern per PR
4. run the tests: `pytest tests/` (or the relevant subset)
5. open a PR with a short description: what it fixes, how you tested it
6. maintainer reviews + merges

### path 2: become a direct collaborator

if you're shipping a VLA in production or in a research lab and want push access + design-partner context:

- discord: `romirj`
- email: playindus@gmail.com

briefly mention your stack (model, hardware, where you're stuck) and i'll add you as a collaborator + set up a call.

## what we need help with most

- **examples for new embodiments**: SO-100, unitree G1, aloha variants, custom arms
- **exporter coverage**: quantization paths (int8, fp8 where supported), new model families
- **docs + getting-started improvements**: anywhere you got stuck, that's a PR
- **ros2 bridge edge cases**: tf tree quirks, multi-node launch patterns
- **hardware probe expansions**: new jetson skus, amd rocm paths
- **test coverage**: the `tests/` dir always wants more integration coverage

## code style

- python 3.10+, type hints where they help readability
- `ruff` for linting, runs on pre-commit
- prefer clear over clever, this is infrastructure people depend on
- if you're adding a feature, add a test; if you're fixing a bug, add a regression test

## questions

open an issue with the `question` label or ping me on discord. fastest reply there.
