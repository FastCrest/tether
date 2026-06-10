"""Auto-written VERIFICATION.md — customer-facing trust receipt.

Writes a manifest of every file in an export directory (sha256, size) plus
metadata (model, target, opset, denoise steps) and — after `tether validate`
has been run — the parity numbers (max_abs_diff per fixture + summary).

Called automatically by `tether export` with `parity=None` (produces the
skeleton manifest), and by `tether validate` with the full parity result
dict to overwrite with verified numbers.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

REPORT_FILENAME = "VERIFICATION.md"
_HASH_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _tether_version() -> str:
    try:
        from tether import __version__
        return __version__
    except Exception:
        return "unknown"


def _collect_files(export_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(export_dir.iterdir()):
        if not p.is_file() or p.name == REPORT_FILENAME:
            continue
        out.append({
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "sha256": _sha256(p),
        })
    return out


def _format_parity(parity: dict[str, Any] | None) -> str:
    if not parity:
        return (
            "## Parity\n\n"
            "_Not yet verified._ Run `tether validate export <export_dir>` to populate.\n"
        )
    summary = parity.get("summary", {}) or {}
    results = parity.get("results", []) or []
    max_abs = float(summary.get("max_abs_diff_across_all", 0))
    passed = bool(summary.get("passed", False))
    lines = [
        "## Parity",
        "",
        f"**Verdict:** {'PASS' if passed else 'FAIL'}",
        f"**Threshold:** {parity.get('threshold')}",
        f"**Fixtures:** {parity.get('num_test_cases')}",
        f"**Seed:** {parity.get('seed')}",
        f"**max_abs_diff across all fixtures:** {max_abs:.3e}",
        "",
        "| Fixture | max_abs_diff | mean_abs_diff | Passed |",
        "|---|---|---|---|",
    ]
    for r in results:
        ok = "PASS" if r.get("passed") else "FAIL"
        lines.append(
            f"| {r.get('fixture_idx', '?')} | "
            f"{float(r.get('max_abs_diff', 0)):.3e} | "
            f"{float(r.get('mean_abs_diff', 0)):.3e} | "
            f"{ok} |"
        )
    return "\n".join(lines) + "\n"


def write_verification_report(
    export_dir: str | Path,
    parity: dict[str, Any] | None = None,
    pro_customer_id: str | None = None,
) -> Path:
    """Write or overwrite ``<export_dir>/VERIFICATION.md``.

    Parameters
    ----------
    export_dir : Path
        The directory produced by `tether export`.
    parity : dict or None
        The result dict from `ValidateRoundTrip.run()`. If None, the parity
        section will say "not yet verified". Pass the full dict from the
        validate CLI to fill in the numbers.
    pro_customer_id : str or None
        If provided (Pro license active), embeds an HMAC fingerprint of
        the report content as a trailing markdown comment. Used for
        commercial-redistribution traceability. Free-tier callers
        (``pro_customer_id=None``) get an unfingerprinted report.

    Returns
    -------
    Path to the written VERIFICATION.md file.
    """
    export_dir = Path(export_dir)
    if not export_dir.exists():
        raise FileNotFoundError(f"Export directory not found: {export_dir}")

    cfg_path = export_dir / "tether_config.json"
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}

    model_id = cfg.get("model_id") or cfg.get("source_model") or "unknown"
    model_type = cfg.get("model_type") or "unknown"
    target = cfg.get("target") or "unknown"
    opset = cfg.get("opset") or cfg.get("onnx_opset") or "unknown"
    num_steps = cfg.get("num_denoising_steps") or "unknown"
    chunk_size = cfg.get("chunk_size") or cfg.get("action_chunk_size") or "unknown"
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    files = _collect_files(export_dir)
    total_bytes = sum(f["size_bytes"] for f in files)

    lines: list[str] = [
        "# Tether Export Verification",
        "",
        f"Generated: {now}",
        "",
        "## Export metadata",
        "",
        f"- **Model:** `{model_id}`",
        f"- **Model type:** {model_type}",
        f"- **Target:** {target}",
        f"- **ONNX opset:** {opset}",
        f"- **Denoising steps (baked in):** {num_steps}",
        f"- **Action chunk size:** {chunk_size}",
        f"- **Tether version:** {_tether_version()}",
        f"- **Platform:** {platform.platform()}",
        "",
        "## Files",
        "",
        f"Total: **{len(files)} files, {_human_size(total_bytes)}**",
        "",
        "| File | Size | SHA256 |",
        "|---|---|---|",
    ]
    for f in files:
        lines.append(
            f"| `{f['name']}` | {_human_size(f['size_bytes'])} | `{f['sha256']}` |"
        )
    lines.append("")
    lines.append(_format_parity(parity))
    lines.append("## Reproducer")
    lines.append("")
    if model_id != "unknown":
        lines.append("```bash")
        lines.append(f"tether export {model_id} --target {target} --output <dir>")
        lines.append("tether validate export <dir>")
        lines.append("```")
        lines.append("")
    lines.append(
        "_Auto-generated by `tether export` and `tether validate`. "
        "Re-run either command to refresh this file._"
    )

    body = "\n".join(lines) + "\n"

    # Pro-only fingerprint: HMAC over the canonical body, embedded as a
    # trailing HTML-style markdown comment so it survives copy-paste but
    # doesn't render in viewers. Free-tier reports omit this entirely.
    if pro_customer_id:
        try:
            from tether.pro.fingerprint import compute_fingerprint
            fp = compute_fingerprint(body.encode("utf-8"), pro_customer_id)
            body += (
                "\n<!-- tether_fingerprint: "
                + json.dumps(fp.to_dict(), separators=(",", ":"), sort_keys=True)
                + " -->\n"
            )
        except Exception:  # noqa: BLE001 — never block export on fingerprint failure
            pass

    out = export_dir / REPORT_FILENAME
    out.write_text(body)
    return out


__all__ = ["write_verification_report", "REPORT_FILENAME"]
