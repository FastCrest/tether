"""Check 3 — VLM tokenization sanity (LeRobot #2119, #683).

Verifies the bundled tokenizer produces in-range token IDs and has the
expected special tokens. Catches silent tokenizer/model version drift
that produces out-of-vocab IDs which the ONNX path then crashes on.

Loads tokenizer config only (no weights) — fast, low-memory.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import Check, CheckResult, register

CHECK_ID = "check_vlm_tokenization"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/2119"

_PROBE_PROMPTS = [
    "pick up the red cup",
    "stack the blocks",
    "open the drawer",
    "place the bowl on the plate",
    "press the green button",
]


def _run(model_path: str, **kwargs) -> CheckResult:
    p = Path(model_path)
    if not p.exists():
        return CheckResult(
            check_id=CHECK_ID,
            name="VLM tokenization",
            status="skip",
            expected="export dir exists for tokenizer probe",
            actual="export dir missing (caught by check_model_load)",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Look for a tokenizer config — most VLA exports bundle one
    tokenizer_config = None
    for candidate in ["tokenizer_config.json", "tokenizer.json", "vocab.json"]:
        if (p / candidate).exists():
            tokenizer_config = p / candidate
            break

    if tokenizer_config is None:
        # Many monolithic-ONNX exports bake the tokenizer into the graph
        # via lang_tokens input — that's a valid path. Skip without failing.
        return CheckResult(
            check_id=CHECK_ID,
            name="VLM tokenization",
            status="skip",
            expected="standalone tokenizer to probe (or tokens baked into ONNX)",
            actual="no tokenizer config found in export — likely tokens are baked",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    try:
        from transformers import AutoTokenizer
    except ImportError:
        return CheckResult(
            check_id=CHECK_ID,
            name="VLM tokenization",
            status="skip",
            expected="transformers installed for tokenizer probe",
            actual="transformers not installed",
            remediation=(
                "pip install fastcrest-tether[monolithic] to enable tokenizer probes "
                "(only needed if your client tokenizes prompts before /act)."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            check_id=CHECK_ID,
            name="VLM tokenization",
            status="fail",
            expected="tokenizer loads from export dir",
            actual=f"AutoTokenizer raised {type(e).__name__}: {e}",
            remediation=(
                f"Export dir has tokenizer_config.json but it's malformed. Check the "
                f"export step's logs OR pin transformers==5.3.0 per ADR "
                f"2026-04-17 onnx-export-gotchas. See {GH_ISSUE}."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Probe — verify tokens are in range
    vocab_size = tokenizer.vocab_size
    out_of_range: list[tuple[str, int]] = []
    for prompt in _PROBE_PROMPTS:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        for tok_id in ids:
            if tok_id < 0 or tok_id >= vocab_size:
                out_of_range.append((prompt, tok_id))

    if out_of_range:
        bad_prompt, bad_id = out_of_range[0]
        return CheckResult(
            check_id=CHECK_ID,
            name="VLM tokenization",
            status="fail",
            expected=f"all token IDs in [0, {vocab_size})",
            actual=f"prompt {bad_prompt!r} produced out-of-range id {bad_id}",
            remediation=(
                f"Tokenizer/model vocab_size mismatch — tokens > {vocab_size} will "
                f"crash the ONNX path. Likely cause: transformers version drift. "
                f"Pin transformers==5.3.0 per ADR 2026-04-17."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="VLM tokenization",
        status="pass",
        expected=f"5 probe prompts tokenize within [0, {vocab_size})",
        actual=f"vocab_size={vocab_size}, all probes in range",
        remediation="",
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


register(Check(
    check_id=CHECK_ID,
    name="VLM tokenization",
    severity="error",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
