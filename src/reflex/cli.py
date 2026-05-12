"""Reflex CLI — deploy VLA models to edge hardware."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from reflex import __version__
from reflex.config import ExportConfig, get_hardware_profile, HARDWARE_PROFILES
from reflex.exporters._export_mode import ExportMode, InsufficientVRAMError

app = typer.Typer(
    name="reflex",
    help="Deploy any VLA model to any edge hardware. One command.",
    no_args_is_help=False,  # we render our own action-first summary in main()
)
console = Console()


_NOARGS_SUMMARY = """[bold]reflex[/bold] — deploy any VLA model to any edge hardware.

[bold cyan]Most-used:[/bold cyan]
  [green]reflex chat[/green]                start the natural-language assistant
  [green]reflex chat --tui[/green]          ↳ full-screen TUI (needs [dim]pip install 'reflex-vla\\[tui]'[/dim])
  [green]reflex go --model X[/green]        one-command deploy: probe → pull → export → serve
  [green]reflex doctor[/green]              diagnose install + GPU issues
  [green]reflex models list[/green]         browse the curated model registry

[dim]All commands:[/dim]  reflex --help
[dim]Examples:[/dim]      https://github.com/FastCrest/reflex-vla/tree/main/examples
[dim]Docs:[/dim]          https://fastcrest.com  ·  https://pypi.org/project/reflex-vla/
"""


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _looks_like_pi05_model_ref(model: str) -> bool:
    """Best-effort fast path for the HF/local refs handled by decomposed.py."""
    normalized = model.lower().replace("-", "").replace("_", "")
    return "pi05" in normalized or "pi0.5" in model.lower()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(
            f"reflex {__version__}\n"
            f"Reflex VLA — Copyright (c) 2026 FastCrest. "
            f"Source-available under BSL 1.1 (auto-converts to Apache 2.0 on 2030-04-28).\n"
            f"https://github.com/FastCrest/reflex-vla"
        )
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None, "--version", help="Show version and exit",
        callback=_version_callback, is_eager=True,
    ),
):
    # First-run onboarding prompt — fires once, before any command.
    # Shows telemetry (opt-out) and data contribution (opt-in) choices.
    # Skips in non-interactive contexts (CI, pipes). Ctrl+C safe.
    try:
        from reflex.onboarding import maybe_onboard
        maybe_onboard()
    except Exception:  # noqa: BLE001
        pass  # never block the CLI on onboarding issues

    # Once-per-day PyPI check for a newer reflex-vla; silent if up-to-date.
    # Honors REFLEX_NO_UPGRADE_CHECK=1; skipped on dev installs.
    try:
        from reflex.upgrade_check import maybe_nag
        maybe_nag(__version__)
    except Exception:  # noqa: BLE001
        pass  # never block the CLI on a network/cache hiccup

    # No subcommand → show the curated action-first summary (not typer's
    # alphabetical command dump). Beats burying `chat` and `go` for new users.
    if ctx.invoked_subcommand is None:
        console.print(_NOARGS_SUMMARY)
        raise typer.Exit()


@app.command(hidden=True)
def export(
    model: str = typer.Argument(help="HuggingFace model ID or local checkpoint path"),
    target: str = typer.Option("desktop", help="Target hardware: orin-nano, orin, orin-64, thor, desktop"),
    output: str = typer.Option("./reflex_export", help="Output directory"),
    precision: str = typer.Option("fp16", help="Precision: fp16, fp8, int8"),
    opset: int = typer.Option(19, help="ONNX opset version"),
    chunk_size: int = typer.Option(50, help="Action chunk size"),
    no_validate: bool = typer.Option(False, help="Skip ONNX validation"),
    dry_run: bool = typer.Option(False, help="Check exportability without building engines"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
    monolithic: bool = typer.Option(
        True,
        "--monolithic/--decomposed",
        help="Export path selector. Default: --monolithic (the cos=+1.000000 verified "
             "path, one ONNX file). Opt into --decomposed only if you specifically need "
             "the 5-stage export for debugging; --decomposed is the older path with "
             "known correctness gaps. Monolithic requires `pip install "
             "'reflex-vla[monolithic]'` (pins transformers==5.3.0).",
    ),
    export_mode: str = typer.Option(
        "auto",
        "--export-mode",
        help="Decomposed pi0.5 export scheduling: auto, sequential, or parallel. "
             "Only applies to --decomposed exports handled by the pi0.5 "
             "vlm_prefix/expert_denoise exporter.",
    ),
    num_steps: int = typer.Option(
        10,
        help="Denoise steps baked into the monolithic ONNX. "
             "Canonical flow-matching = 10 (SmolVLA, pi0, pi0.5); use 1 for exact "
             "one-shot Euler. GR00T (DDPM) uses 4 as its runtime default. "
             "For pi0.5 --decomposed, this is baked into expert_denoise.onnx.",
    ),
    from_distilled: bool = typer.Option(
        False,
        "--from-distilled",
        help="Treat MODEL as a reflex-saved SnapFlow student checkpoint dir "
             "(contains model.safetensors with target_time_embed_mlp.* keys). "
             "Auto-detects pi0 vs pi0.5 from config.json, exports at 1-NFE "
             "with target_time=1 baked in. Output ONNX has the same I/O "
             "signature as the matching teacher family's monolithic export, "
             "so `reflex serve` loads it through the standard path.",
    ),
):
    """Export a VLA model to ONNX + TensorRT for edge deployment."""
    _setup_logging(verbose)
    hardware = get_hardware_profile(target)
    try:
        requested_export_mode = ExportMode(export_mode)
    except ValueError:
        valid = ", ".join(mode.value for mode in ExportMode)
        console.print(f"[red]Invalid --export-mode {export_mode!r}. Valid: {valid}[/red]")
        raise typer.Exit(2)

    if monolithic:
        if requested_export_mode != ExportMode.AUTO:
            console.print(
                "[red]--export-mode only applies to --decomposed pi0.5 exports; "
                "monolithic export has no parallel prefix/expert split.[/red]"
            )
            raise typer.Exit(2)
        label = "SnapFlow student (1-NFE)" if from_distilled else "monolithic, cos=1.0 verified path"
        console.print(f"\n[bold]Reflex Export ({label})[/bold]")
        console.print(f"  Model:      {model}")
        console.print(f"  Output:     {output}")
        if not from_distilled:
            console.print(f"  num_steps:  {num_steps}")
        console.print()

        if dry_run:
            console.print("[yellow]--dry-run not supported with --monolithic yet (v0.3 item). "
                          "Re-run without --dry-run to export.[/yellow]")
            raise typer.Exit()

        try:
            if from_distilled:
                from reflex.exporters.monolithic import export_snapflow_student_monolithic
            else:
                from reflex.exporters.monolithic import export_monolithic
        except ImportError as exc:
            console.print(f"[red]{exc}[/red]", markup=False)
            console.print(
                "\nFix: pip install 'reflex-vla[monolithic]' "
                "(pins transformers==5.3.0; use a clean venv to avoid "
                "the base transformers<5.0 conflict)",
                style="cyan", markup=False,
            )
            raise typer.Exit(2)

        import time
        start = time.perf_counter()
        try:
            if from_distilled:
                result = export_snapflow_student_monolithic(model, output, target=target)
            else:
                result = export_monolithic(model, output, num_steps=num_steps, target=target)
        except ImportError as exc:
            console.print(f"Missing monolithic dep: {exc}", style="red", markup=False)
            raise typer.Exit(2)
        elapsed = time.perf_counter() - start
        console.print(f"\n[bold green]Monolithic export complete in {elapsed:.1f}s[/bold green]")
        console.print(f"  ONNX: {result['onnx_path']}")
        console.print(f"  Size: {result['size_mb']:.1f} MB")

        try:
            from reflex.verification_report import write_verification_report
            report_path = write_verification_report(output, parity=None)
            console.print(f"  Verification manifest: {report_path}")
        except Exception as exc:
            console.print(f"[yellow]Verification manifest skipped: {exc}[/yellow]")

        console.print(f"\n  [dim]Next:[/dim] [cyan]reflex serve {output}[/cyan]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Reflex Export[/bold]")
    console.print(f"  Model:     {model}")
    console.print(f"  Target:    {hardware.name} ({hardware.memory_gb}GB, {hardware.trt_precision})")
    console.print(f"  Precision: {precision}")
    console.print(f"  Output:    {output}")
    console.print()

    if dry_run:
        console.print("[dim]Checking exportability...[/dim]")
        from reflex.checkpoint import load_checkpoint, detect_model_type, validate_checkpoint

        state_dict, config = load_checkpoint(model)
        model_type = detect_model_type(state_dict)
        console.print(f"  Detected: {model_type or 'unknown'}")
        total_params = sum(v.numel() for v in state_dict.values())
        console.print(f"  Params:   {total_params / 1e6:.1f}M")

        warnings = validate_checkpoint(state_dict, model_type or "unknown")
        for w in warnings:
            console.print(f"  [yellow]Warning: {w}[/yellow]")

        # Check memory fit
        weight_gb = total_params * 2 / 1e9  # FP16
        if weight_gb > hardware.memory_gb * 0.7:
            console.print(f"  [red]Model ({weight_gb:.1f}GB) may not fit on {hardware.name} ({hardware.memory_gb}GB)[/red]")
        else:
            console.print(f"  [green]Model ({weight_gb:.1f}GB) fits on {hardware.name} ({hardware.memory_gb}GB)[/green]")

        if model_type is None:
            console.print("\n[yellow]Unknown model type — export may fail. Supported: smolvla, pi0, pi05.[/yellow]")
        else:
            console.print("\n[green]Dry run complete. Export should work.[/green]")
        raise typer.Exit()

    if _looks_like_pi05_model_ref(model):
        console.print("\n[bold]Reflex Export (pi0.5 decomposed)[/bold]")
        console.print(f"  Model:       {model}")
        console.print(f"  Target:      {hardware.name} ({hardware.memory_gb}GB, {hardware.trt_precision})")
        console.print(f"  Output:      {output}")
        console.print(f"  Export mode: {requested_export_mode.value}")
        console.print()

        try:
            from reflex.exporters.decomposed import export_pi05_decomposed
        except ImportError as exc:
            console.print(f"[red]{exc}[/red]", markup=False)
            console.print(
                "\nFix: pip install 'reflex-vla[monolithic]' "
                "(pins transformers==5.3.0; use a clean venv to avoid "
                "the base transformers<5.0 conflict)",
                style="cyan", markup=False,
            )
            raise typer.Exit(2)

        import time
        start = time.perf_counter()
        decomposed_steps = 1 if from_distilled else num_steps
        try:
            result = export_pi05_decomposed(
                model_id=model,
                output_dir=output,
                num_steps=decomposed_steps,
                target=target,
                student_checkpoint=model if from_distilled else None,
                export_mode=requested_export_mode,
            )
        except InsufficientVRAMError as exc:
            console.print(f"{exc}", style="red", markup=False)
            raise typer.Exit(2)
        except ImportError as exc:
            console.print(f"Missing decomposed export dep: {exc}", style="red", markup=False)
            raise typer.Exit(2)
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"Decomposed export failed: {type(exc).__name__}: {exc}",
                style="red",
                markup=False,
            )
            raise typer.Exit(1)

        elapsed = time.perf_counter() - start
        console.print(f"\n[bold green]Decomposed export complete in {elapsed:.1f}s[/bold green]")
        console.print(f"  Mode:   {result.get('export_mode', requested_export_mode.value)}")
        console.print(f"  Prefix: {result['vlm_prefix_onnx']} ({result['vlm_prefix_mb']:.1f} MB)")
        console.print(f"  Expert: {result['expert_denoise_onnx']} ({result['expert_denoise_mb']:.1f} MB)")
        console.print(f"\n  [dim]Next:[/dim] [cyan]reflex serve {output}[/cyan]")
        raise typer.Exit(0)

    if requested_export_mode != ExportMode.AUTO:
        console.print(
            "[red]--export-mode is only implemented for pi0.5 decomposed exports "
            "handled by the vlm_prefix/expert_denoise exporter. Re-run with "
            "--export-mode auto, or use a pi0.5 model ref.[/red]"
        )
        raise typer.Exit(2)

    # Full export — auto-dispatch to the right exporter based on model type
    from reflex.checkpoint import load_checkpoint, detect_model_type
    from reflex.exporters.smolvla_exporter import export_smolvla
    from reflex.exporters.pi0_exporter import export_pi0, export_pi05
    from reflex.exporters.gr00t_exporter import export_gr00t

    # Load once, detect, then pass state_dict to the exporter (avoids double-load)
    console.print("[dim]Loading checkpoint...[/dim]")
    state_dict, _ = load_checkpoint(model)
    model_type = detect_model_type(state_dict) or "smolvla"
    console.print(f"  Detected: [bold]{model_type}[/bold]")

    export_config = ExportConfig(
        model_id=model,
        target=target,
        output_dir=output,
        precision=precision,
        opset=opset,
        action_chunk_size=chunk_size,
        validate=not no_validate,
    )

    import time
    start = time.perf_counter()
    if model_type == "gr00t":
        # Use the full-stack exporter (wraps action_encoder + DiT + action_decoder)
        # so `reflex serve` can run the standard denoising loop.
        from reflex.exporters.gr00t_exporter import export_gr00t_full
        result = export_gr00t_full(export_config, state_dict=state_dict)
    elif model_type == "openvla":
        from reflex.exporters.openvla_exporter import export_openvla
        result = export_openvla(export_config, state_dict=state_dict)
    elif model_type == "pi05":
        result = export_pi05(export_config, state_dict=state_dict)
    elif model_type == "pi0":
        result = export_pi0(export_config, state_dict=state_dict)
    else:
        result = export_smolvla(export_config, state_dict=state_dict)
    elapsed_expert = time.perf_counter() - start

    # Print expert results
    console.print(f"\n[bold green]Expert export complete in {elapsed_expert:.1f}s[/bold green]")

    if "files" in result:
        for name, path in result["files"].items():
            size = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
            console.print(f"  {name}: {path} ({size:.1f}MB)")

    if "metadata" in result and "onnx_validation" in result["metadata"]:
        val = result["metadata"]["onnx_validation"]
        status = "[green]PASS[/green]" if val["passed"] else "[red]FAIL[/red]"
        console.print(f"  Validation: {status} (max_diff={val['max_diff']:.2e})")

    if "metadata" in result and "expert" in result["metadata"]:
        meta = result["metadata"]["expert"]
        console.print(f"  Expert: {meta['num_layers']} layers, {meta['total_params_m']:.1f}M params")

    # For SmolVLA: also export the VLM pipeline (vision_encoder + text_embedder + decoder_prefill)
    # so `reflex serve` can run with real task-conditioned actions instead of noise.
    # Note: VLM weights come from the base SmolVLM2-500M (not the SmolVLA checkpoint's
    # fine-tuned VLM). Fine-tuned VLM weight transfer is tracked as a v0.3 item.
    if model_type == "smolvla":
        console.print("\n[dim]Exporting VLM pipeline (vision + text + decoder)...[/dim]")
        from reflex.exporters.vlm_prefix_exporter import export_vlm_prefix
        vlm_start = time.perf_counter()
        try:
            # Pass the loaded state_dict so the VLM exporter can overlay the
            # fine-tuned vision/text weights instead of using BASE SmolVLM2.
            vlm_path = export_vlm_prefix(
                output_dir=output, opset=opset, state_dict=state_dict
            )
            elapsed_vlm = time.perf_counter() - vlm_start
            console.print(f"[bold green]VLM export complete in {elapsed_vlm:.1f}s[/bold green]")
            # Show VLM output files
            for fname in ("vision_encoder.onnx", "text_embedder.onnx", "decoder_prefill.onnx"):
                fpath = Path(output) / fname
                if fpath.exists():
                    data_path = fpath.with_suffix(".onnx.data")
                    size = fpath.stat().st_size / 1e6
                    if data_path.exists():
                        size += data_path.stat().st_size / 1e6
                    console.print(f"  {fname}: {size:.1f}MB")
            console.print(
                "  [dim]Note: VLM uses base SmolVLM2-500M weights. "
                "Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item).[/dim]"
            )
        except Exception as exc:
            console.print(f"[yellow]VLM export skipped: {exc}[/yellow]")
            console.print(
                "[yellow]Server will use dummy VLM conditioning (v0.1 fallback).[/yellow]"
            )

        # Save state_proj weights from checkpoint so the VLM orchestrator can
        # project robot state through the REAL trained matrix instead of the
        # random init we were falling back to (that was silently destroying
        # state information in every prefix — the ONE bug hiding behind all
        # the others, found by the PyTorch-vs-ONNX diff on 2026-04-17).
        try:
            import numpy as np
            sp_w_keys = [k for k in state_dict if k.endswith("state_proj.weight")]
            sp_b_keys = [k for k in state_dict if k.endswith("state_proj.bias")]
            if sp_w_keys:
                sp_w = state_dict[sp_w_keys[0]].detach().cpu().numpy().astype(
                    np.float32
                )
                np.save(Path(output) / "state_proj_weight.npy", sp_w)
                console.print(
                    f"  state_proj weight: {sp_w.shape} → state_proj_weight.npy"
                )
            if sp_b_keys:
                sp_b = state_dict[sp_b_keys[0]].detach().cpu().numpy().astype(
                    np.float32
                )
                np.save(Path(output) / "state_proj_bias.npy", sp_b)
                console.print(
                    f"  state_proj bias: {sp_b.shape} → state_proj_bias.npy"
                )
            if not sp_w_keys:
                console.print(
                    "  [yellow]WARNING: no state_proj weight in checkpoint — "
                    "orchestrator will fall back to random init and state "
                    "conditioning will be garbage.[/yellow]"
                )
        except Exception as exc:
            console.print(f"[yellow]state_proj save failed: {exc}[/yellow]")

        # Copy LeRobot policy normalizer/unnormalizer from the HF repo into the
        # export dir. Without these, the model receives un-normalized state and
        # returns actions in normalized space — producing garbage trajectories
        # in sim. Critical for LIBERO / real-robot eval success.
        if "/" in model and not Path(model).exists():
            try:
                from huggingface_hub import hf_hub_download

                console.print(
                    "\n[dim]Copying policy preprocessor/postprocessor stats...[/dim]"
                )
                import shutil

                stats_files = [
                    "policy_preprocessor.json",
                    "policy_postprocessor.json",
                    "policy_preprocessor_step_5_normalizer_processor.safetensors",
                    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                ]
                copied = 0
                for fname in stats_files:
                    try:
                        src = hf_hub_download(repo_id=model, filename=fname)
                        shutil.copy(src, Path(output) / fname)
                        copied += 1
                    except Exception:
                        # Not all SmolVLA checkpoints ship these (e.g. base)
                        pass
                if copied:
                    console.print(
                        f"  Copied {copied}/{len(stats_files)} normalizer files "
                        f"→ {output}"
                    )
                else:
                    console.print(
                        "  [dim]No normalizer files found in checkpoint "
                        "(base model or older format) — adapter will skip "
                        "normalization.[/dim]"
                    )
            except Exception as exc:
                console.print(
                    f"[yellow]Normalizer copy skipped: {exc}[/yellow]"
                )

    total_elapsed = time.perf_counter() - start
    console.print(f"\n[bold]Total export: {total_elapsed:.1f}s[/bold]")
    console.print(f"  Output: {output}")

    try:
        from reflex.verification_report import write_verification_report
        report_path = write_verification_report(output, parity=None)
        console.print(f"  [dim]Verification manifest: {report_path}[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Verification manifest skipped: {exc}[/yellow]")

    console.print(f"\n  [dim]Run on target hardware:[/dim]")
    console.print(f"  [cyan]reflex bench {output}[/cyan]")


@app.command(name="validate-legacy", hidden=True)
def validate(
    target: str = typer.Argument("", help="Export directory OR HuggingFace model ID (with --pre-export)"),
    model: str = typer.Option("", help="HuggingFace model ID for PyTorch reference (auto-detect from reflex_config.json if empty)"),
    threshold: float = typer.Option(
        1e-4,
        help="Max acceptable L2 abs diff per action dim. Default 1e-4.",
    ),
    num_cases: int = typer.Option(5, help="Number of seeded fixtures"),
    seed: int = typer.Option(0, help="RNG seed for fixtures + initial noise"),
    device: str = typer.Option("cpu", help="Device for PyTorch reference: cpu or cuda"),
    output_json: bool = typer.Option(False, "--output-json", help="Emit pure JSON instead of Rich tables"),
    init_ci: bool = typer.Option(False, "--init-ci", help="Emit .github/workflows/reflex-validate.yml and exit"),
    quick: bool = typer.Option(
        False, "--quick",
        help="Fast static checks only (file exists, ONNX loadable, no NaN). Skip parity harness.",
    ),
    pre_export: bool = typer.Option(
        False, "--pre-export",
        help="Check a raw checkpoint before exporting. Takes model ID, not export dir.",
    ),
    hardware: str = typer.Option("desktop", help="Hardware target for --pre-export memory check"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Validate an export: full parity (default), static checks (--quick), or pre-export checkpoint health (--pre-export)."""
    _setup_logging(verbose)

    if init_ci:
        from reflex.ci_template import emit_ci_template
        out = Path(".github/workflows/reflex-validate.yml")
        try:
            emit_ci_template(out, reflex_version=__version__)
        except FileExistsError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2)
        except Exception as exc:
            console.print(f"[red]Failed to emit CI template: {exc}[/red]")
            raise typer.Exit(2)
        console.print(f"[green]Wrote CI template:[/green] {out}")
        raise typer.Exit(0)

    if not target:
        console.print("[red]Export directory or model ID is required (unless --init-ci).[/red]")
        raise typer.Exit(2)

    # --pre-export: check a raw checkpoint (replaces old `reflex check`)
    if pre_export:
        from reflex.validate_training import run_all_checks
        console.print(f"\n[bold]Reflex Validate (pre-export)[/bold]")
        console.print(f"  Checkpoint: {target}")
        console.print(f"  Target:     {hardware}\n")

        results = run_all_checks(target, target=hardware)
        table = Table(title="Pre-export checks")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Detail")
        n_pass = 0
        for r in results:
            status = "[green]PASS[/green]" if r.passed else (
                "[yellow]WARN[/yellow]" if r.severity == "warning" else "[red]FAIL[/red]"
            )
            if r.passed:
                n_pass += 1
            table.add_row(r.name, status, r.detail[:80])
        console.print(table)
        console.print(f"\n  Passed: [bold]{n_pass}/{len(results)}[/bold]")
        raise typer.Exit(0 if n_pass == len(results) else 1)

    # --quick: static checks on an export directory (faster than full parity)
    if quick:
        export_path = Path(target)
        console.print(f"\n[bold]Reflex Validate (--quick)[/bold]")
        console.print(f"  Export: {export_path}\n")

        table = Table(title="Static export checks")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Detail")
        n_pass = n_total = 0

        def _check(name: str, ok: bool, detail: str) -> None:
            nonlocal n_pass, n_total
            n_total += 1
            if ok:
                n_pass += 1
            status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            table.add_row(name, status, detail[:80])

        _check("export_dir exists", export_path.exists(), str(export_path))
        config_path = export_path / "reflex_config.json"
        _check("reflex_config.json", config_path.exists(), str(config_path))

        config: dict = {}
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                _check("config parses", True, f"{len(config)} keys")
            except Exception as e:
                _check("config parses", False, str(e))

        # Check each expected ONNX file
        import onnxruntime as ort
        import numpy as np
        for fname in ("expert_stack.onnx", "vision_encoder.onnx", "text_embedder.onnx", "decoder_prefill.onnx"):
            fpath = export_path / fname
            if fpath.exists():
                try:
                    sess = ort.InferenceSession(str(fpath), providers=["CPUExecutionProvider"])
                    inputs = [inp.name for inp in sess.get_inputs()]
                    _check(f"{fname} loads", True, f"inputs={inputs}")
                except Exception as e:
                    _check(f"{fname} loads", False, str(e)[:80])
            else:
                # Only the expert_stack is required; VLM files are optional for non-SmolVLA
                if fname == "expert_stack.onnx":
                    _check(f"{fname} present", False, "missing (required)")
                else:
                    table.add_row(fname, "[dim]skipped[/dim]", "not present")

        console.print(table)
        console.print(f"\n  Passed: [bold]{n_pass}/{n_total}[/bold]")
        raise typer.Exit(0 if n_pass == n_total else 1)

    # Default: full ONNX-vs-PyTorch parity harness
    export_dir = target  # rename for legacy code paths below

    if device not in ("cpu", "cuda"):
        console.print(f"[red]--device must be 'cpu' or 'cuda', got: {device}[/red]")
        raise typer.Exit(2)

    from reflex.validate_roundtrip import ValidateRoundTrip

    try:
        runner = ValidateRoundTrip(
            export_dir=Path(export_dir),
            model_id=model or None,
            threshold=threshold,
            num_test_cases=num_cases,
            seed=seed,
            device=device,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    try:
        result = runner.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Validation interrupted by user.[/yellow]")
        raise typer.Exit(130)
    except FileNotFoundError as exc:
        console.print(f"[red]Missing required file: {exc}[/red]")
        raise typer.Exit(2)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    except Exception as exc:
        if verbose:
            import traceback
            traceback.print_exc()
        console.print(f"[red]Validation failed with unexpected error: {exc}[/red]")
        console.print("[yellow]Re-run with --verbose for the full traceback.[/yellow]")
        raise typer.Exit(2)

    summary = result.get("summary", {})
    passed = bool(summary.get("passed", False))

    if output_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        console.print("\n[bold]Reflex Validate[/bold]")
        console.print(f"  Export: {export_dir}")
        console.print(f"  Model type: {result.get('model_type')}")
        console.print(f"  Threshold: {result.get('threshold')}")

        per_table = Table(title="Per-fixture results", show_header=True, header_style="bold")
        per_table.add_column("fixture_idx", justify="right")
        per_table.add_column("max_abs_diff", justify="right")
        per_table.add_column("mean_abs_diff", justify="right")
        per_table.add_column("passed", justify="center")
        for r in result.get("results", []):
            ok = bool(r.get("passed"))
            per_table.add_row(
                str(r.get("fixture_idx", "")),
                f"{float(r.get('max_abs_diff', 0)):.2e}",
                f"{float(r.get('mean_abs_diff', 0)):.2e}",
                "[green]PASS[/green]" if ok else "[red]FAIL[/red]",
            )
        console.print(per_table)

        sum_table = Table(title="Summary", show_header=True, header_style="bold")
        sum_table.add_column("metric")
        sum_table.add_column("value")
        sum_table.add_row("max_abs_diff_across_all", f"{float(summary.get('max_abs_diff_across_all', 0)):.2e}")
        sum_table.add_row("passed", "[green]PASS[/green]" if passed else "[red]FAIL[/red]")
        sum_table.add_row("num_cases", str(result.get("num_test_cases")))
        sum_table.add_row("seed", str(result.get("seed")))
        sum_table.add_row("threshold", str(result.get("threshold")))
        console.print(sum_table)

    try:
        from reflex.verification_report import write_verification_report
        report_path = write_verification_report(export_dir, parity=result)
        if not output_json:
            console.print(f"  [dim]Updated verification receipt: {report_path}[/dim]")
    except Exception as exc:
        if not output_json:
            console.print(f"[yellow]Verification receipt update skipped: {exc}[/yellow]")

    raise typer.Exit(0 if passed else 1)


@app.command(name="bench", hidden=True)
def benchmark_cmd(
    export_dir: str = typer.Argument(help="Path to exported model directory"),
    iterations: int = typer.Option(100, help="Number of benchmark iterations"),
    warmup: int = typer.Option(20, help="Warmup iterations (excluded from stats)"),
    device: str = typer.Option("cuda", help="Device: cuda or cpu"),
    benchmark: str = typer.Option(
        "",
        "--benchmark",
        help="Also run task-success eval: simpler, maniskill (requires pip install 'reflex-vla[eval]'). LIBERO archived 2026-04-17 — see archive/scripts/.",
    ),
    episodes_per_task: int = typer.Option(
        10, help="Episodes per task for --benchmark (full suites use 50)"
    ),
    report: str = typer.Option(
        "",
        "--report",
        help="When set, write a methodology-rich Markdown bench report to this path. "
             "Includes p50/p95/p99 + p99.9 + jitter + 95%% CI + reproducibility envelope "
             "(git SHA, GPU, ORT/CUDA versions, ONNX file hashes, seed). Lifts ISB-1 "
             "methodology — see reference/NOTES.md sibling project section.",
    ),
    report_json: str = typer.Option(
        "",
        "--report-json",
        help="Same data as --report but as machine-readable JSON. Stable schema; "
             "CI can grep results without parsing markdown.",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="RNG seed pinned in the reproducibility envelope. Inference is "
             "deterministic at the noise initialization layer; pinning here lets "
             "you cite a number that re-runs identically.",
    ),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Benchmark exported model — latency (default) and optional task success.

    Default: loads the export, warms up, runs N iterations of the denoising loop,
    reports mean/p50/p95/p99 latency.

    With --benchmark <suite>: also runs task-success evaluation on the named
    simulation benchmark (SimplerEnv, ManiSkill). Requires the [eval] extra —
    sim dependencies are not in the base install.

    LIBERO was archived on 2026-04-17 — reflex's product wedge is deployment
    parity + latency, not sim benchmarking. Archived scripts live at
    archive/scripts/ if you want to resurrect them.
    """
    _setup_logging(verbose)
    import time as _t
    import numpy as np

    export_path = Path(export_dir)
    if not export_path.exists():
        console.print(f"[red]Export directory not found: {export_dir}[/red]")
        raise typer.Exit(1)

    onnx_files = list(export_path.glob("*.onnx"))
    if not onnx_files:
        console.print(f"[red]No ONNX file in {export_dir}[/red]")
        raise typer.Exit(1)

    # If --benchmark was requested, gate on the eval extra being installed
    if benchmark:
        try:
            import vla_eval  # noqa: F401
        except ImportError:
            console.print(
                f"--benchmark {benchmark} requires the eval extra.\n"
                f"  Install with: pip install 'reflex-vla[eval]'\n"
                f"  Or run without --benchmark for latency-only.",
                style="red", markup=False,
            )
            raise typer.Exit(2)
        valid = ("simpler", "maniskill")
        if benchmark not in valid:
            console.print(f"[red]Unknown benchmark '{benchmark}'. Try one of: {', '.join(valid)}[/red]")
            raise typer.Exit(2)

    console.print(f"\n[bold]Reflex Benchmark[/bold]")
    console.print(f"  Export:    {export_dir}")
    console.print(f"  Device:    {device}")
    console.print(f"  Warmup:    {warmup}")
    console.print(f"  Iterations: {iterations}")
    if benchmark:
        console.print(f"  Benchmark: [cyan]{benchmark}[/cyan] ({episodes_per_task} eps/task)")

    from reflex.runtime.server import ReflexServer
    server = ReflexServer(export_dir, device=device, strict_providers=False)
    console.print("[dim]Loading model...[/dim]")
    t0 = _t.perf_counter()
    server.load()
    load_s = _t.perf_counter() - t0
    if not server.ready:
        console.print("[red]Model failed to load.[/red]")
        raise typer.Exit(1)
    console.print(
        f"  Loaded:    {load_s:.1f}s  (mode={server._inference_mode})"
    )

    # Warmup
    console.print(f"[dim]Warming up ({warmup} iterations)...[/dim]")
    for _ in range(warmup):
        server.predict()

    # Bench
    console.print(f"[dim]Benchmarking ({iterations} iterations)...[/dim]")
    latencies: list[float] = []
    for _ in range(iterations):
        t0 = _t.perf_counter()
        server.predict()
        latencies.append((_t.perf_counter() - t0) * 1000)
    latencies.sort()

    mean = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    minv = latencies[0]
    maxv = latencies[-1]

    console.print(f"\n[bold]Per-chunk latency (10-step denoise loop):[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(justify="right")
    table.add_row("min",  f"{minv:7.2f} ms")
    table.add_row("mean", f"{mean:7.2f} ms")
    table.add_row("p50",  f"{p50:7.2f} ms")
    table.add_row("p95",  f"{p95:7.2f} ms")
    table.add_row("p99",  f"{p99:7.2f} ms")
    table.add_row("max",  f"{maxv:7.2f} ms")
    table.add_row("hz",   f"{1000.0/mean:7.1f}")
    console.print(table)

    console.print(
        f"\n  [dim]Inference mode:[/dim] [bold]{server._inference_mode}[/bold]"
    )
    if server._inference_mode == "onnx_cpu" and device == "cuda":
        console.print(
            "  [yellow]Note: requested device=cuda but ended up on CPU. "
            "Install onnxruntime-gpu and CUDA 12 + cuDNN 9 for GPU performance.[/yellow]"
        )

    # Methodology-rich report (Phase 1 bench-revamp). Backward-compat: only
    # writes a report when --report or --report-json is set; the printed
    # table above is the existing one-shot UX.
    if report or report_json:
        from reflex.bench import (
            BenchReport,
            capture_environment,
            compute_stats,
        )
        # Re-include the warmup samples so methodology.compute_stats can
        # discard them for documentation symmetry. The existing latencies
        # list contains ONLY post-warmup samples, so warmup_n=0 here.
        stats = compute_stats(latencies, warmup_n=0)
        env = capture_environment(
            export_dir=export_dir,
            device=device,
            inference_mode=server._inference_mode,
            seed=seed,
        )
        bench_report = BenchReport(
            stats=stats,
            environment=env,
            notes=[f"warmup={warmup} discarded BEFORE the recorded latencies "
                   f"(see iterations loop in cli.benchmark_cmd)"],
        )
        if report:
            bench_report.write_markdown(report)
            console.print(f"\n  [dim]Markdown report:[/dim] {report}")
        if report_json:
            bench_report.write_json(report_json)
            console.print(f"  [dim]JSON report:[/dim] {report_json}")

    # Task-success evaluation (optional, gated on --benchmark flag + [eval] extra)
    if benchmark:
        console.print(f"\n[bold]Task-success eval: {benchmark}[/bold]")
        try:
            from reflex.eval import run_task_benchmark
        except ImportError as exc:
            console.print(
                f"[red]reflex.eval module missing: {exc}[/red]\n"
                f"  The benchmark-plugin framework ships in v0.2 — see GOALS.yaml."
            )
            raise typer.Exit(2)

        eval_result = run_task_benchmark(
            benchmark,
            export_dir=export_dir,
            episodes_per_task=episodes_per_task,
            device=device,
        )
        success_rate = eval_result.get("success_rate", 0.0)
        console.print(f"\n  Task success: [bold]{success_rate * 100:.1f}%[/bold] "
                      f"({eval_result.get('episodes_completed', 0)} episodes)")


# ---------------------------------------------------------------------------
# `reflex eval` — task-success evaluation (correctness, not latency)
# Per ADR 2026-04-25-eval-as-a-service-architecture: top-level verb that
# wraps the existing Modal image + osmesa/MuJoCo recipe + vla-eval adapter.
# ---------------------------------------------------------------------------


@app.command(name="eval")
def eval_cmd(
    export_dir: str = typer.Argument(
        help="Path to exported model directory (output of `reflex export`)",
    ),
    suite: str = typer.Option(
        "libero", "--suite",
        help="Eval suite. Phase 1 ships LIBERO only; SimplerEnv is Phase 2.",
    ),
    num_episodes: int = typer.Option(
        3, "--num-episodes",
        help="Episodes per task. Default 3 = smoke; researchers reproducing "
             "published numbers pass 50-100. Wall-clock scales linearly.",
    ),
    tasks: str = typer.Option(
        "", "--tasks",
        help="Comma-separated task list. Empty (default) = all suite tasks "
             "(LIBERO ships 4 task families: spatial / object / goal / 10).",
    ),
    runtime: str = typer.Option(
        "modal", "--runtime",
        help="modal | local. Modal uses the bundled debian_slim+osmesa "
             "image (turnkey). local needs Linux x86_64 + the [eval-local] "
             "extra; NEVER silently falls back to Modal.",
    ),
    seed: int = typer.Option(
        0, "--seed",
        help="RNG seed. Default 0 matches `reflex bench`. Pass --seed 7 to "
             "reproduce prior modal_libero_*.py published results.",
    ),
    max_parallel: int = typer.Option(
        1, "--max-parallel",
        help="Max concurrent tasks. Phase 1 honors only when the runtime "
             "supports it (Modal yes, local no).",
    ),
    cost_preview: bool = typer.Option(
        False, "--cost-preview",
        help="Dry-run: estimate $ cost without invoking the suite. Useful "
             "before kicking off a 100-ep × 90-task run.",
    ),
    video: bool = typer.Option(
        False, "--video",
        help="Emit per-episode MP4 to <output>/videos/. Encoded at quality "
             "cap of ~10MB/episode. Phase 1 local-only; HF Hub upload Phase 2.",
    ),
    output: str = typer.Option(
        "./eval_output", "--output",
        help="Directory for JSON envelope + (optional) videos. Created if "
             "missing.",
    ),
    preflight_timeout: float = typer.Option(
        300.0, "--preflight-timeout",
        help="Seconds for the LIBERO smoke test. Cold osmesa scene-compile "
             "can take 60-180s on first run; raise on cold containers.",
    ),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Run task-success eval (LIBERO success rate + per-task numbers + optional video).

    Wraps the existing Modal image + osmesa/MuJoCo recipe + vla-eval adapter.
    Pre-flight smoke test catches 4-of-5 documented LIBERO failure modes
    before the expensive run starts.

    Examples:
        reflex eval ./my-export --suite libero --num-episodes 3
        reflex eval ./my-export --suite libero --num-episodes 50 --video
        reflex eval ./my-export --runtime local --tasks libero_spatial
        reflex eval ./my-export --cost-preview --num-episodes 100
    """
    _setup_logging(verbose)

    from reflex.eval.cost_model import (
        COST_PREVIEW_GUARDRAIL_USD,
        estimate_cost,
    )
    from reflex.eval.libero import (
        ALL_RUNTIMES,
        LiberoSuite,
        LiberoSuiteConfig,
    )
    from reflex.eval.preflight import PreflightSmokeTest
    from reflex.eval.report import build_envelope, capture_environment
    from reflex.eval.runner_dispatch import (
        default_libero_tasks,
        resolve_suite_runner,
        resolve_task_runner,
    )

    # ---- Validate inputs at the CLI layer (fail loud) ----
    export_path = Path(export_dir)
    if not export_path.exists():
        console.print(f"[red]Export directory not found: {export_dir}[/red]")
        raise typer.Exit(1)

    if suite != "libero":
        console.print(
            f"[red]Unknown suite: {suite!r}. Phase 1 ships LIBERO only.[/red]\n"
            f"  Phase 2 will add: simpler, customer."
        )
        raise typer.Exit(2)

    if runtime not in ALL_RUNTIMES:
        console.print(
            f"[red]Unknown runtime: {runtime!r}. "
            f"Choose one of: {', '.join(ALL_RUNTIMES)}[/red]"
        )
        raise typer.Exit(2)

    # Parse comma-separated tasks (empty → use suite defaults)
    parsed_tasks: tuple[str, ...] = tuple(
        t.strip() for t in tasks.split(",") if t.strip()
    ) if tasks else ()

    # Build config — validates num_episodes, max_parallel, episode_timeout_s
    try:
        config = LiberoSuiteConfig(
            num_episodes=num_episodes,
            tasks=parsed_tasks,
            runtime=runtime,
            video=video,
            output_dir=output,
            seed=seed,
            max_parallel=max_parallel,
            cost_preview=cost_preview,
        )
    except ValueError as exc:
        console.print(f"[red]Invalid configuration: {exc}[/red]")
        raise typer.Exit(2)

    # ---- Banner echo ----
    console.print("\n[bold]Reflex Eval[/bold]")
    console.print(f"  Export:      {export_dir}")
    console.print(f"  Suite:       [cyan]{suite}[/cyan]")
    console.print(f"  Runtime:     [cyan]{runtime}[/cyan]")
    console.print(f"  Episodes:    {num_episodes} per task")
    if parsed_tasks:
        console.print(f"  Tasks:       {', '.join(parsed_tasks)}")
    else:
        console.print(f"  Tasks:       [dim](suite defaults)[/dim]")
    console.print(f"  Seed:        {seed}")
    console.print(f"  Output:      {output}")
    if video:
        console.print(
            f"  Video:       [yellow]requested but Phase 2[/yellow] "
            f"[dim](encoder ready in src/reflex/eval/video.py; modal_libero "
            f"frame-capture wires Phase 2)[/dim]"
        )
    if cost_preview:
        console.print(f"  Mode:        [yellow]COST PREVIEW (no run)[/yellow]")

    # ---- Cost preview short-circuit (uses cost_model.estimate_cost) ----
    resolved_tasks = list(parsed_tasks) if parsed_tasks else default_libero_tasks()
    if cost_preview:
        cost_estimate = estimate_cost(
            suite=suite, runtime=runtime,
            tasks=resolved_tasks,
            num_episodes_per_task=num_episodes,
        )
        console.print(
            f"\n[yellow]Cost preview ({len(resolved_tasks)} tasks × "
            f"{num_episodes} eps):[/yellow]"
        )
        console.print(
            f"  Total estimate: [bold]${cost_estimate.total_usd:.2f}[/bold] USD"
        )
        console.print(f"  {cost_estimate.notes}")
        if cost_estimate.exceeds_guardrail:
            console.print(
                f"\n[red]Estimate exceeds ${COST_PREVIEW_GUARDRAIL_USD:.0f} "
                f"guardrail.[/red] Run with smaller --num-episodes "
                f"or fewer --tasks to lower the cost."
            )
        raise typer.Exit(0)

    # ---- Pre-flight smoke test (catches 4-of-5 LIBERO failure modes) ----
    console.print(
        f"\n[dim]Pre-flight smoke test "
        f"(timeout: {preflight_timeout:.0f}s)...[/dim]"
    )
    preflight_result = PreflightSmokeTest.run(timeout_s=preflight_timeout)
    if not preflight_result.passed:
        console.print(
            f"\n[red]Pre-flight FAILED[/red] "
            f"({preflight_result.failure_mode}, "
            f"{preflight_result.elapsed_s:.1f}s)\n"
            f"\n[bold]Remediation:[/bold]\n  {preflight_result.remediation}\n"
        )
        if preflight_result.stderr:
            console.print(
                f"[dim]Subprocess stderr (last 500 chars):[/dim]\n"
                f"{preflight_result.stderr[-500:]}"
            )
        raise typer.Exit(4)
    console.print(
        f"  [green]Pre-flight OK[/green] ({preflight_result.elapsed_s:.1f}s)"
    )

    # ---- Resolve runner + dispatch ----
    # Modal: full-suite dispatch via reflex.eval.modal_runner (one Modal
    # call per suite; saves N cold-starts vs per-episode fan-out).
    # Local: per-(task, episode) dispatch via LiberoSuite.run loop.
    console.print(f"\n[dim]Running suite...[/dim]")
    # Resolve task list (modal_runner needs explicit tasks; LiberoSuite.run
    # accepts a tasks_provider fallback).
    runtime_config = config
    if not parsed_tasks:
        runtime_config = LiberoSuiteConfig(
            num_episodes=num_episodes,
            tasks=tuple(default_libero_tasks()),
            runtime=runtime, video=video, output_dir=output, seed=seed,
            max_parallel=max_parallel, cost_preview=cost_preview,
        )

    if runtime == "modal":
        from reflex.eval.modal_runner import ModalNotInstalledError
        suite_runner = resolve_suite_runner(
            runtime=runtime, export_dir=export_path,
        )
        try:
            report = suite_runner(runtime_config, export_path)
        except ModalNotInstalledError as exc:
            console.print(f"\n[red]{exc}[/red]")
            raise typer.Exit(6)
    else:
        # local
        task_runner = resolve_task_runner(
            runtime=runtime, export_dir=export_path,
        )
        tasks_provider = None if parsed_tasks else default_libero_tasks
        report = LiberoSuite.run(
            export_dir=export_path,
            config=runtime_config,
            task_runner=task_runner,
            tasks_provider=tasks_provider,
        )

    # ---- Render summary ----
    console.print(
        f"\n[bold]Eval complete[/bold] "
        f"({report.wall_clock_s:.1f}s wall-clock)"
    )
    console.print(
        f"  Aggregate success: [bold]"
        f"{report.aggregate_success_rate * 100:.1f}%[/bold] "
        f"({report.aggregate_n_success}/{report.aggregate_n_total} episodes)"
    )
    if report.results:
        console.print(f"\n[bold]Per-task results:[/bold]")
        per_task = Table(show_header=True, box=None, padding=(0, 2))
        per_task.add_column("Task", style="cyan")
        per_task.add_column("Success", justify="right")
        per_task.add_column("N", justify="right")
        for r in report.results:
            per_task.add_row(
                r.task_id,
                f"{r.success_rate * 100:.1f}%",
                f"{r.n_success}/{r.n_total}",
            )
        console.print(per_task)

    # ---- JSON envelope (schema v1 LOCKED per ADR decision #3) ----
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    actual_tasks = list(report.tasks) if report.tasks else resolved_tasks
    cost_estimate = estimate_cost(
        suite=suite, runtime=runtime,
        tasks=actual_tasks,
        num_episodes_per_task=num_episodes,
    )
    env_block = capture_environment(export_dir=export_path)
    modal_block: dict | None = None
    if runtime == "modal":
        modal_block = {"image_digest": "TBD", "provider": "modal.com"}

    envelope = build_envelope(
        report=report,
        cost=cost_estimate,
        env=env_block,
        num_episodes_per_task=num_episodes,
        modal_block=modal_block,
    )
    envelope_path = envelope.write_json(output_path / "report.json")
    console.print(f"\n  [dim]JSON envelope:[/dim] {envelope_path}")

    # Honest signaling: if EVERY episode is adapter_error (because Day 3 stub
    # runners always do that), exit non-zero so CI doesn't think it succeeded.
    all_adapter_error = (
        report.aggregate_n_total > 0
        and report.aggregate_n_success == 0
        and all(
            ep.terminal_reason == "adapter_error"
            for r in report.results
            for ep in r.episodes
        )
    )
    if all_adapter_error:
        console.print(
            "\n[yellow]All episodes returned adapter_error[/yellow] — Day 3 "
            "ships the CLI substrate; the runtime task runner wires Day 4-5. "
            "Use --cost-preview for a no-run cost estimate."
        )
        raise typer.Exit(5)


@app.command(hidden=True)
def guard(
    action: str = typer.Argument(help="Action to check: 'init' to create config, 'check' to validate"),
    urdf: str = typer.Option("", help="URDF file path to extract joint limits"),
    config: str = typer.Option("", help="Safety config JSON file path"),
    output: str = typer.Option("./safety_config.json", help="Output path for safety config"),
    num_joints: int = typer.Option(6, help="Number of joints (when no URDF)"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Configure and test safety guardrails for VLA actions."""
    _setup_logging(verbose)

    from reflex.safety import ActionGuard, SafetyLimits

    if action == "init":
        if urdf:
            limits = SafetyLimits.from_urdf(urdf)
            console.print(f"[green]Extracted limits from URDF: {urdf}[/green]")
        else:
            limits = SafetyLimits.default(num_joints)
            console.print(f"[yellow]Using default limits for {num_joints} joints[/yellow]")

        console.print(f"  Joints: {len(limits.joint_names)}")
        for i, name in enumerate(limits.joint_names):
            console.print(
                f"    {name}: pos=[{limits.position_min[i]:.2f}, {limits.position_max[i]:.2f}], "
                f"vel_max={limits.velocity_max[i]:.2f}"
            )

        limits.save(output)
        console.print(f"\n[bold green]Safety config saved: {output}[/bold green]")
        console.print(f"[dim]Use with: reflex serve --safety-config {output}[/dim]")

    elif action == "check":
        if config:
            limits = SafetyLimits.from_json(config)
        elif urdf:
            limits = SafetyLimits.from_urdf(urdf)
        else:
            limits = SafetyLimits.default(num_joints)

        guard_instance = ActionGuard(limits=limits, mode="clamp")
        import numpy as np

        test_actions = np.random.randn(5, num_joints).astype(np.float32) * 5
        safe_actions, results = guard_instance.check(test_actions)

        console.print(f"\n[bold]Safety Check (5 random actions, range [-5, 5]):[/bold]")
        for i, r in enumerate(results):
            status = "[green]SAFE[/green]" if r.safe else "[red]CLAMPED[/red]" if r.clamped else "[red]REJECTED[/red]"
            console.print(f"  Action {i}: {status} ({len(r.violations)} violations, {r.check_time_ms:.3f}ms)")
            for v in r.violations[:3]:
                console.print(f"    {v}")

    else:
        console.print(f"[red]Unknown action: {action}. Use 'init' or 'check'.[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    export_dir: str = typer.Argument(help="Path to exported model directory"),
    port: int = typer.Option(8000, help="Server port"),
    host: str = typer.Option("0.0.0.0", help="Server host"),
    device: str = typer.Option("cuda", help="Device: cuda or cpu"),
    providers: str = typer.Option(
        "",
        help="Comma-separated ORT execution providers (e.g. "
             "'CUDAExecutionProvider,CPUExecutionProvider'). Overrides --device "
             "for provider selection when set.",
    ),
    no_strict_providers: bool = typer.Option(
        False,
        "--no-strict-providers",
        help="Allow silent fallback to CPU if the requested GPU provider fails "
             "to load. OFF by default — by default the server raises a loud "
             "error instead of silently falling back. Set this only if you "
             "explicitly want best-effort fallback.",
    ),
    safety_config: str = typer.Option(
        "",
        help="Path to a SafetyLimits JSON (from `reflex guard init`). When set, "
             "every returned action is clamped to the configured joint limits "
             "and violation counts are logged.",
    ),
    adaptive_steps: bool = typer.Option(
        False,
        "--adaptive-steps",
        help="Use reflex turbo adaptive denoising — stops the denoise loop "
             "early when velocity norm converges. Saves latency on easy tasks.",
    ),
    cloud_fallback: str = typer.Option(
        "",
        help="URL of a remote reflex serve (e.g. http://cloud-host:8000). When "
             "set, a reflex split orchestrator is configured for cloud-edge "
             "routing. v0.1 stores config only; full dispatch lands in Phase VI.",
    ),
    deadline_ms: float = typer.Option(
        0.0,
        help="Per-request deadline in ms. 0 = disabled. When set, predict() "
             "returns the last-known-good action instead if inference exceeds "
             "the deadline. Deadline misses are logged and counted.",
    ),
    max_batch: int = typer.Option(
        1,
        help="DEPRECATED — superseded by --max-batch-cost-ms in Phase 1 "
             "chunk-budget-batching. Setting --max-batch > 1 still works "
             "(legacy fixed-count batching) but emits a one-time deprecation "
             "warning at startup. Migration: set --max-batch-cost-ms = "
             "max_batch × per_request_cold_start_ms (e.g., --max-batch 8 "
             "→ --max-batch-cost-ms 400 at the 50ms cold-start default).",
    ),
    batch_timeout_ms: float = typer.Option(
        5.0,
        help="Maximum wait per batch flush in ms. The PolicyRuntime worker "
             "flushes when --max-batch-cost-ms is reached OR this timeout "
             "fires (whichever first). Lower = lower per-request latency; "
             "higher = better batching efficiency under bursty load.",
    ),
    max_batch_cost_ms: float = typer.Option(
        100.0,
        "--max-batch-cost-ms",
        help="Per-policy chunk-budget scheduler: flush a batch when the "
             "estimated GPU-ms cost reaches this value. Default 100ms — fits "
             "comfortably under most p99 SLOs; bump higher to favor "
             "throughput, lower for tail latency. Bounded [10, 500] ms. "
             "Per ADR 2026-04-24-chunk-budget-batching-architecture.",
    ),
    api_key: str = typer.Option(
        "",
        help="If set, every /act and /config request must include a matching "
             "X-Reflex-Key header or it's rejected 401. /health stays "
             "unauthenticated so load balancers can probe readiness. For "
             "production use, pass via env var (e.g. --api-key $REFLEX_API_KEY) "
             "rather than hardcoding.",
    ),
    replan_hz: float = typer.Option(
        0.0,
        help="If >0, enable async replan-while-execute action buffering "
             "(the Physical Intelligence sliding_window pattern). Set with "
             "--execute-hz. Example: --execute-hz 100 --replan-hz 20 means "
             "the robot pops an action 100 times/sec while fresh chunks are "
             "generated 20 times/sec. Buffer capacity is auto-sized from "
             "the ratio. 0 = disabled (return full chunks, current default).",
    ),
    execute_hz: float = typer.Option(
        0.0,
        help="Execute frequency in Hz — the rate at which the robot pops "
             "an action from the buffer. Only used when --replan-hz > 0.",
    ),
    rtc: bool = typer.Option(
        False,
        "--rtc",
        help="Enable Real-Time Chunking (RTC) — wraps inference with "
             "lerobot's RTCProcessor so the robot keeps executing the tail "
             "of one chunk while the next chunk is being computed. 2-3× "
             "effective throughput on Jetson-class latency. Requires "
             "`pip install reflex-vla[rtc]` (pulls lerobot==0.5.1).",
    ),
    rtc_execution_horizon: int = typer.Option(
        10,
        "--rtc-execution-horizon",
        help="With --rtc: number of actions locked to the previous chunk "
             "while the next is computed. Higher = more guidance, smoother "
             "transitions; lower = more freedom for the new chunk. Default 10.",
    ),
    rtc_schedule: str = typer.Option(
        "LINEAR",
        "--rtc-schedule",
        help="With --rtc: prefix attention schedule. ZEROS | ONES | LINEAR | EXP. "
             "Default LINEAR (matches lerobot's RTCConfig default).",
    ),
    rtc_max_guidance_weight: float = typer.Option(
        10.0,
        "--rtc-max-guidance-weight",
        help="With --rtc: max guidance weight clamp. Higher = stronger pull "
             "toward previous chunk's prefix; lower = looser. Default 10.0.",
    ),
    rtc_debug: bool = typer.Option(
        False,
        "--rtc-debug",
        help="With --rtc: enable lerobot's debug Tracker for per-step state "
             "capture. Useful for replay forensics; small per-call overhead.",
    ),
    record: str = typer.Option(
        "",
        help="If set, write every /act request+response to a JSONL trace in "
             "this directory. One file per server session, named "
             "<YYYYMMDD-HHMMSS>-<model_hash>-<session_id>.jsonl[.gz]. "
             "Replay with `reflex replay <file> --model <export>`. See "
             "TECHNICAL_PLAN §D.1 for the schema.",
    ),
    record_images: str = typer.Option(
        "hash_only",
        "--record-images",
        help="Image redaction policy when --record is set: "
             "'full' (~40MB/1k calls gzipped, base64 JPEG kept) | "
             "'hash_only' (~0.9MB/1k calls, image_sha256 only — default; "
             "sufficient for replay against a fixed image corpus) | "
             "'none' (drop image entirely; minimal size).",
    ),
    record_no_gzip: bool = typer.Option(
        False,
        "--record-no-gzip",
        help="When --record is set, write plain .jsonl instead of .jsonl.gz. "
             "Useful for quick grep during dev; production should keep gzip on.",
    ),
    embodiment: str = typer.Option(
        "",
        help="Per-embodiment config preset name (franka, so100, ur5, etc.). "
             "Loads configs/embodiments/<name>.json. Empty = no embodiment "
             "config (current default behavior). See "
             "docs/embodiment_schema.md for the schema and adding new presets.",
    ),
    custom_embodiment_config: str = typer.Option(
        "",
        "--custom-embodiment-config",
        help="Path to a custom embodiment config JSON. Overrides --embodiment "
             "if both are set. Use this for robots not covered by the shipped "
             "presets.",
    ),
    inject_latency_ms: float = typer.Option(
        0.0,
        "--inject-latency-ms",
        help="Synthetic deployment-latency injection (B.4 A2C2 transfer-validation "
             "gate). Adds asyncio.sleep AFTER inference + JSONL recording so "
             "recorded latency_ms is true compute cost while client observes "
             "inference + injected delay. Range [0, 1000]. 0 = off (default). "
             "Used to simulate Jetson-class deployment latency on Modal A10G "
             "for the A2C2 transfer gate; see arxiv 2509.23224 §4 for "
             "matching paper methodology.",
    ),
    no_prewarm: bool = typer.Option(
        False,
        "--no-prewarm",
        help="Skip the synthetic warmup forward at lifespan startup. Default "
             "behavior: warmup runs, /health returns 503 until warmup succeeds, "
             "then 200. With --no-prewarm: /health returns 200 the moment "
             "server.load() completes; first /act bears the 30-90s engine-build "
             "cost. Use only for fast-start dev workflows; production behind a "
             "load balancer should leave prewarm ON.",
    ),
    max_consecutive_crashes: int = typer.Option(
        5,
        "--max-consecutive-crashes",
        help="Circuit breaker: after this many consecutive /act predict "
             "exceptions or error-result responses, server.health_state flips "
             "to 'degraded' — /health returns 503, /act returns 503 with "
             "Retry-After: 60. Successful /act resets the counter. Default 5. "
             "Set to 0 to disable.",
    ),
    ros2: bool = typer.Option(
        False,
        "--ros2",
        help="Run a ROS2 bridge instead of the HTTP server. Subscribes to "
             "image/state/task topics and publishes action chunks to an "
             "action topic. Requires rclpy (apt-installed, not pip) — see "
             "reflex ros2-serve for the standalone equivalent. Mutually "
             "exclusive with the HTTP flags above (port, host, api-key, "
             "max-batch, etc. are ignored in ROS2 mode).",
    ),
    max_concurrent: int = typer.Option(
        0,
        "--max-concurrent",
        help="Maximum concurrent /act requests. 0 = unlimited (default). When "
             "set to N, a semaphore bounds in-flight requests; overload returns "
             "HTTP 429 with structured {error, message, request_id, "
             "concurrent_requests, max_concurrent} body + Retry-After: 1 header. "
             "TGI's overload pattern: reject fast, let client retry, don't let "
             "queue depth explode. /health + /metrics are exempt.",
    ),
    slo: str = typer.Option(
        "",
        "--slo",
        help="Latency SLO spec (e.g. 'p99=150ms'). When set, the server tracks "
             "per-request /act latency in a rolling window, emits "
             "reflex_slo_violations_total Prometheus metric when the percentile "
             "exceeds threshold, and optionally returns HTTP 503 (see --slo-mode). "
             "Phase 1 supports a single global SLO on /act; per-endpoint SLO is "
             "Phase 1.5.",
    ),
    slo_mode: str = typer.Option(
        "degrade",
        "--slo-mode",
        help="SLO violation behavior: 'log_only' (metric only), '503' (return "
             "HTTP 503 with measured p99 in body; client can fail over), or "
             "'degrade' (Phase 1: same as log_only. Phase 1.5: drops NFE + "
             "skips RTC eval to recover). Default 'degrade'.",
    ),
    mcp: bool = typer.Option(
        False,
        "--mcp",
        help="Expose the server as a Model Context Protocol surface so MCP-"
             "compatible agents (Claude Desktop, Cursor, custom) can discover "
             "Reflex in the mcp.so catalog and call /act as a tool. Additive "
             "to the HTTP API on stdio/HTTP transports. With --mcp-transport "
             "stdio (default), the MCP server owns stdin/stdout and FastAPI "
             "is NOT started (use for Claude Desktop / Cursor integration). "
             "With --mcp-transport http, both MCP (on --mcp-port) and FastAPI "
             "(on --port) run concurrently. Requires `pip install reflex-vla[mcp]`.",
    ),
    mcp_transport: str = typer.Option(
        "stdio",
        "--mcp-transport",
        help="MCP transport: 'stdio' (default; for Claude Desktop / Cursor) or "
             "'http' (streamable-http on --mcp-port). Only used when --mcp is set.",
    ),
    mcp_port: int = typer.Option(
        8001,
        "--mcp-port",
        help="MCP HTTP port (only when --mcp --mcp-transport http). Separate from "
             "--port which is the FastAPI port.",
    ),
    otel_endpoint: str = typer.Option(
        "",
        "--otel-endpoint",
        help="OTLP gRPC endpoint for trace export (e.g. 'localhost:4317' for "
             "Phoenix, or an OTel Collector). Requires `pip install reflex-vla"
             "[tracing]`. When unset, falls back to $OTEL_EXPORTER_OTLP_ENDPOINT "
             "and then 'localhost:4317'. Traces include gen_ai.operation.name, "
             "gen_ai.request.model, gen_ai.action.embodiment, gen_ai.action."
             "chunk_size, gen_ai.action.denoise_steps per OTel GenAI SemConv.",
    ),
    otel_sample: float = typer.Option(
        1.0,
        "--otel-sample",
        help="Trace sampling ratio [0.0, 1.0]. 1.0 = sample every /act (default, "
             "safe for dev/staging). 0.1 = 10% sampling (OTel SemConv starting "
             "point for high-traffic production). Uses parent-based sampler so "
             "child spans inherit the root decision (avoids partial traces).",
    ),
    robot_id: str = typer.Option(
        "",
        "--robot-id",
        help="Fleet-telemetry identifier for this process. When set, publishes "
             "`reflex_robot_info{robot_id=...}` Prometheus gauge and echoes "
             "robot_id on /health + /config responses. Customers deploying one "
             "Reflex process per robot join this against hot metrics via "
             "`instance` in Grafana (see dashboards/grafana/reflex-fleet.json). "
             "When unset, no extra cardinality is added — backward compatible.",
    ),
    auto_calibrate: bool = typer.Option(
        False,
        "--auto-calibrate",
        help="Run hardware-fit calibration at startup. Probes the GPU + "
             "embodiment + model_hash and SELECTS the right pre-shipped "
             "(variant × provider × NFE × chunk_size) configuration; "
             "passively learns latency_compensation_ms during the first 30s "
             "of /act traffic. Persists to --calibration-cache. Cache hit on "
             "matching hardware fingerprint = instant; miss = ~5-7s "
             "measurement. Per ADR 2026-04-25-auto-calibration-architecture.",
    ),
    calibration_cache: str = typer.Option(
        "~/.reflex/calibration.json",
        "--calibration-cache",
        help="Path to the calibration JSON cache. Default lives in the user's "
             "home dir; override to ship a frozen cache inside a container. "
             "Validated early — parent dir must be writable.",
    ),
    calibrate_force: bool = typer.Option(
        False,
        "--calibrate-force",
        help="Re-run calibration even on cache hit (useful after hardware swap). "
             "Requires --auto-calibrate.",
    ),
    a2c2_checkpoint: str = typer.Option(
        "",
        "--a2c2-checkpoint",
        help="Path to a trained A2C2 correction-head checkpoint (.npz). When "
             "set, the server applies per-step A2C2 corrections on the action "
             "chunk after the policy returns it. Auto-skipped at low latency "
             "(p95 < 40ms) or high success rate (>90%) — no overhead when not "
             "needed. Per a2c2-correction execution plan B.5. Train with "
             "scripts/train_a2c2_lerobot.py (Modal A100; user-authorized).",
    ),
    a2c2_latency_threshold_ms: float = typer.Option(
        40.0,
        "--a2c2-latency-threshold-ms",
        help="A2C2 hook auto-skips when latency_p95 < this (ms). Default 40 "
             "matches Orin Nano deployment expectations. Lower to force the "
             "hook to apply at lower latency (e.g., for paper-methodology "
             "measurement under --inject-latency-ms).",
    ),
    a2c2_success_threshold: float = typer.Option(
        0.90,
        "--a2c2-success-threshold",
        help="A2C2 hook auto-skips when /act success rate > this (0..1). "
             "NOTE: 'success' here is /act error-rate (server crash) NOT "
             "task-success — without task feedback wired in, leave at 0.90 "
             "for default behavior or set to 1.01 to disable success-skip "
             "for measurement runs.",
    ),
    bid_n_candidates: int = typer.Option(
        0,
        "--bid-num-candidates",
        help="Enable Bidirectional Decoding (BID) chunk selection per arxiv "
             "2408.17355. Sample N candidate chunks per /act + pick the one "
             "most coherent with the previously-emitted chunk. 0 (default) "
             "= disabled (single-sample inference). Recommended: 8 for "
             "balance between selection quality + Nx denoise cost. Mutually "
             "exclusive with --a2c2-checkpoint in Phase 1; both set = BID wins.",
    ),
    bid_coherence_window: int = typer.Option(
        5,
        "--bid-coherence-window",
        help="Window size K: BID scores first K actions of new chunk vs "
             "last K actions of previous chunk. Default 5 per the paper.",
    ),
    bid_coherence_metric: str = typer.Option(
        "l2",
        "--bid-coherence-metric",
        help="BID coherence scoring metric: 'l2' (default; lower L2 distance "
             "wins) or 'cos' (higher cosine similarity wins).",
    ),
    cuda_graphs: bool = typer.Option(
        False,
        "--cuda-graphs",
        help="Enable ORT CUDA-graph capture + replay on the decomposed "
             "vlm_prefix + expert_denoise ONNX sessions. A100+: full capture "
             "(~4.4x vlm_prefix + ~3.0x expert_denoise per Modal A/B "
             "2026-04-25). A10G: expert-only capture (~3.8x); vlm_prefix "
             "graceful-degrades to eager at session init with an INFO log + "
             "`reflex_cuda_graph_capture_failed_at_init_total` metric (A10G "
             "vlm_prefix OOMs at capture time per vLLM #5517 memory "
             "overhead pattern). Default off — opt-in for Phase 1 per ADR "
             "2026-04-24-cuda-graphs-architecture.",
    ),
    action_similarity_threshold: float = typer.Option(
        0.0,
        "--action-similarity-threshold",
        help="Action-similarity fast path (FlashVLA, arxiv 2505.21200). When "
             ">0, the inference path skips the expert + reuses the prior "
             "action chunk if its L2 distance to the new chunk is below this "
             "value. Paper default 0.05; 0.0 = disabled (default). Caps "
             "consecutive skips via --max-similar-skips. Decomposed pi0.5 "
             "only; ignored on monolithic exports. Per Phase 1.5 spec "
             "features/01_serve/subfeatures/_perf_compound/"
             "action-similarity-fast-path.",
    ),
    max_similar_skips: int = typer.Option(
        3,
        "--max-similar-skips",
        help="Cap on consecutive cached-action returns from the action-"
             "similarity fast path. Prevents drift on slow-changing scenes. "
             "Paper default 3. Only used when --action-similarity-threshold "
             ">0.",
    ),
    policy_a: str = typer.Option(
        "", "--policy-a",
        help="2-policy A/B mode: path to policy A export. When set, --policy-b "
             "must also be set + --no-rtc enforced (RTC carry-over is per-policy). "
             "See docs/policy_versioning.md.",
    ),
    policy_b: str = typer.Option(
        "", "--policy-b",
        help="2-policy A/B mode: path to policy B export. Must be paired with "
             "--policy-a + --no-rtc. Mutually exclusive with --shadow-policy.",
    ),
    split: int = typer.Option(
        50, "--split",
        help="2-policy mode: percent of traffic routed to policy A in [0, 100]. "
             "Sticky-per-episode (router uses SHA-256 hash of episode_id). "
             "Edge cases: 0 = all to B, 100 = all to A (shadow-staging mode).",
    ),
    shadow_policy: str = typer.Option(
        "", "--shadow-policy",
        help="(Phase 1.5) shadow inference: path to a policy that runs alongside "
             "the primary on a sample of traffic. Phase 1: shipped INERT (logs "
             "warning when set, no shadow execution). Mutually exclusive with "
             "--policy-b.",
    ),
    shadow_sample: float = typer.Option(
        0.0, "--shadow-sample",
        help="(Phase 1.5) fraction of /act requests to mirror to --shadow-policy "
             "in [0, 1]. Phase 1: ignored.",
    ),
    no_rtc: bool = typer.Option(
        False, "--no-rtc",
        help="Disable RTC even when --rtc was previously enabled. REQUIRED in "
             "2-policy mode (--policy-b set) per ADR "
             "2026-04-25-policy-versioning-architecture.",
    ),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Start a VLA inference server. POST /act with image + instruction → actions.

    Composable wedges: --safety-config (guard), --adaptive-steps (turbo),
    --cloud-fallback (split), --deadline-ms (WCET).
    """
    _setup_logging(verbose)

    export_path = Path(export_dir)
    if not export_path.exists():
        console.print(f"[red]Export directory not found: {export_dir}[/red]")
        console.print(f"[dim]Run 'reflex export' first to create an export.[/dim]")
        raise typer.Exit(1)

    onnx_files = list(export_path.glob("*.onnx"))
    if not onnx_files:
        console.print(f"[red]No ONNX files found in {export_dir}[/red]")
        raise typer.Exit(1)

    # ---- Policy-versioning Day 5 validation ----
    # 2-policy mode requires both --policy-a + --policy-b. Mutually
    # exclusive with --shadow-policy. --no-rtc enforced in 2-policy mode.
    two_policy_mode = bool(policy_a or policy_b)
    if two_policy_mode and not (policy_a and policy_b):
        console.print(
            "[red]--policy-a and --policy-b must be set together for "
            "2-policy mode.[/red]\n"
            "[dim]To roll out a single policy, drop both flags and pass "
            "the model path as the positional argument.[/dim]"
        )
        raise typer.Exit(1)
    if two_policy_mode and shadow_policy:
        console.print(
            "[red]--policy-b and --shadow-policy are mutually exclusive.[/red]\n"
            "[dim]Pick A/B for production rollout, OR shadow for risk-free "
            "comparison.[/dim]"
        )
        raise typer.Exit(1)
    if two_policy_mode:
        from reflex.runtime.policy import validate_split_and_no_rtc
        try:
            validate_split_and_no_rtc(split_a_percent=split, no_rtc=no_rtc)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        # Verify both policy paths exist before attempting to load
        for label, path in (("--policy-a", policy_a), ("--policy-b", policy_b)):
            if not Path(path).exists():
                console.print(
                    f"[red]{label} export not found: {path}[/red]"
                )
                raise typer.Exit(1)
        # 2-policy mode active. The actual 2-instance load + dispatcher
        # wiring happens in create_app's lifespan (per ADR Days 9-10):
        # setup_two_policy_serving builds server B alongside server A,
        # composes a TwoPolicyDispatcher, stores state on server.two_policy_state.
        # /act handler routes via dispatcher when state is set.
        console.print(
            f"\n[bold]2-policy mode active[/bold] "
            f"(--policy-a={policy_a}, --policy-b={policy_b}, "
            f"--split={split}, --no-rtc enforced)."
        )

    if shadow_policy:
        console.print(
            f"\n[yellow]--shadow-policy={shadow_policy} (Phase 1.5; "
            f"shipped inert in Phase 1).[/yellow] "
            f"[dim]Shadow execution lands when "
            f"shadow-inference primitive ships.[/dim]\n"
        )

    # Resolve --embodiment / --custom-embodiment-config (B.1). Validate
    # early — before any compute or runtime checks — so a bad config fails
    # loud at the CLI layer, not at first /act.
    embodiment_cfg = None
    if custom_embodiment_config or embodiment:
        from reflex.embodiments import EmbodimentConfig, list_presets
        from reflex.embodiments.validate import (
            format_errors,
            validate_embodiment_config,
        )
        try:
            if custom_embodiment_config:
                if embodiment:
                    console.print(
                        f"[yellow]--custom-embodiment-config overrides "
                        f"--embodiment {embodiment}[/yellow]"
                    )
                embodiment_cfg = EmbodimentConfig.load_custom(custom_embodiment_config)
            else:
                embodiment_cfg = EmbodimentConfig.load_preset(embodiment)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Failed to load embodiment config: {exc}[/red]")
            console.print(
                f"[dim]Available presets: {list_presets() or '(none)'}[/dim]"
            )
            raise typer.Exit(1)

        ok, errs = validate_embodiment_config(embodiment_cfg)
        if not ok:
            console.print(
                f"[red]Embodiment config '{embodiment_cfg.embodiment}' failed "
                f"validation:[/red]"
            )
            console.print(format_errors(errs))
            raise typer.Exit(1)
        warnings = [e for e in errs if e["severity"] == "warn"]
        if warnings:
            console.print(
                f"[yellow]Embodiment config '{embodiment_cfg.embodiment}' "
                f"has warnings:[/yellow]"
            )
            console.print(format_errors(warnings))

    # Build RtcAdapterConfig if --rtc was passed (B.3 Day 1). Validates at
    # the CLI layer — fail loud before runtime imports (same pattern as
    # embodiment validation above).
    rtc_cfg = None
    if rtc:
        from reflex.runtime.rtc_adapter import RtcAdapterConfig
        try:
            rtc_cfg = RtcAdapterConfig(
                enabled=True,
                replan_hz=replan_hz if replan_hz > 0 else 20.0,
                execute_hz=execute_hz if execute_hz > 0 else 100.0,
                rtc_execution_horizon=rtc_execution_horizon,
                prefix_attention_schedule=rtc_schedule,
                max_guidance_weight=rtc_max_guidance_weight,
                debug=rtc_debug,
            )
        except ValueError as exc:
            console.print(f"[red]Invalid RTC config: {exc}[/red]")
            raise typer.Exit(1)

    # ROS2 mode short-circuits the HTTP path — hand off to the bridge.
    if ros2:
        try:
            from reflex.runtime.ros2_bridge import run_ros2_bridge
        except ImportError as exc:
            console.print(f"[red]ros2 bridge unavailable: {exc}[/red]")
            raise typer.Exit(2)
        console.print(f"[bold green]reflex serve --ros2[/bold green]")
        console.print(f"  export:   {export_dir}")
        console.print(f"  device:   {device}")
        console.print(
            f"  [dim]HTTP flags ignored in ROS2 mode. Use `reflex ros2-serve` "
            f"for full topic/rate customization.[/dim]"
        )
        try:
            run_ros2_bridge(
                export_dir,
                device=device,
                safety_config=safety_config or None,
            )
        except KeyboardInterrupt:
            console.print("[yellow]ros2 bridge stopped.[/yellow]")
        return

    # Parse providers
    provider_list: list[str] | None = None
    if providers:
        provider_list = [p.strip() for p in providers.split(",") if p.strip()]

    # Detect the common "I pip installed onnxruntime instead of onnxruntime-gpu"
    # footgun before we spin up the server.
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        console.print(
            "[red]onnxruntime is not installed.[/red]\n"
            "For GPU: [cyan]pip install onnxruntime-gpu[/cyan]\n"
            "For CPU: [cyan]pip install onnxruntime[/cyan]"
        )
        raise typer.Exit(1)

    cuda_requested = (
        device == "cuda"
        or (provider_list and "CUDAExecutionProvider" in provider_list)
    )
    cuda_available_in_ort = "CUDAExecutionProvider" in available

    console.print(f"\n[bold]Reflex Serve[/bold]")
    console.print(f"  Export:  {export_dir}")
    console.print(f"  Device:  {device}")
    if provider_list:
        console.print(f"  Providers: {provider_list}")
    console.print(f"  Strict:  {not no_strict_providers}")
    console.print(f"  Server:  http://{host}:{port}")
    console.print(f"  [dim]ORT available providers: {available}[/dim]")

    # Composed wedges summary
    composed = []
    if safety_config:
        composed.append(f"[cyan]safety[/cyan]={safety_config}")
    if adaptive_steps:
        composed.append("[cyan]adaptive-steps[/cyan]")
    if cloud_fallback:
        composed.append(f"[cyan]cloud-fallback[/cyan]={cloud_fallback}")
    if deadline_ms > 0:
        composed.append(f"[cyan]deadline[/cyan]={deadline_ms:.0f}ms")
    if max_batch > 1:
        composed.append(f"[cyan]batch[/cyan]={max_batch}@{batch_timeout_ms:.0f}ms")
    if embodiment_cfg is not None:
        composed.append(f"[cyan]embodiment[/cyan]={embodiment_cfg.embodiment}")
    if record:
        composed.append(
            f"[cyan]record[/cyan]={record} ({record_images}"
            f"{', no-gzip' if record_no_gzip else ''})"
        )
    if rtc:
        composed.append(
            f"[cyan]rtc[/cyan]=horizon{rtc_execution_horizon}/{rtc_schedule}"
        )
    if composed:
        console.print(f"  Wedges:  {' · '.join(composed)}")

    if cuda_requested and not cuda_available_in_ort:
        console.print(
            "\n[red]⚠ CUDAExecutionProvider not available in this ORT install.[/red]\n"
            "  Likely cause: you installed `onnxruntime` (CPU-only).\n"
            "  Fix:   [cyan]pip uninstall onnxruntime && pip install onnxruntime-gpu[/cyan]\n"
            "  Also:  ORT 1.20+ requires CUDA 12.x + cuDNN 9.x on the library path.\n"
            "  Or:    pass [cyan]--device cpu[/cyan] to explicitly use CPU.\n"
            "  Or:    pass [cyan]--no-strict-providers[/cyan] to allow CPU fallback anyway.\n"
        )
        if not no_strict_providers:
            raise typer.Exit(1)

    console.print()
    console.print(f"  [dim]Endpoints:[/dim]")
    console.print(f"  [cyan]POST /act[/cyan]     — send image + instruction, get actions")
    console.print(f"  [cyan]GET  /health[/cyan]  — check server status")
    console.print(f"  [cyan]GET  /config[/cyan]  — view model config")
    console.print()

    try:
        from reflex.runtime.server import create_app
        import uvicorn
    except ImportError:
        console.print("Install serve dependencies: pip install 'reflex-vla[serve]'", style="red", markup=False)
        raise typer.Exit(1)

    if replan_hz > 0 and execute_hz <= 0:
        console.print(
            "[red]--replan-hz requires --execute-hz > 0 (the robot's pop rate).[/red]"
        )
        raise typer.Exit(1)

    if not (0.0 <= otel_sample <= 1.0):
        console.print(
            f"[red]--otel-sample must be in [0.0, 1.0], got {otel_sample}[/red]"
        )
        raise typer.Exit(1)

    # chunk-budget-batching CLI validation + --max-batch deprecation
    if not (10.0 <= max_batch_cost_ms <= 500.0):
        console.print(
            f"[red]--max-batch-cost-ms must be in [10, 500], got {max_batch_cost_ms}[/red]"
        )
        raise typer.Exit(1)
    if max_batch > 1:
        console.print(
            "[yellow]--max-batch > 1 is DEPRECATED in Phase 1 chunk-budget-"
            f"batching. Use --max-batch-cost-ms instead. Current value "
            f"--max-batch={max_batch} is ignored at the runtime layer; "
            f"PolicyRuntime always uses --max-batch-cost-ms (default 100).[/yellow]"
        )

    # auto-calibration mutual-exclusion validation
    if calibrate_force and not auto_calibrate:
        console.print(
            "[red]--calibrate-force requires --auto-calibrate.[/red]"
        )
        raise typer.Exit(1)
    _calib_cache_path = Path(calibration_cache).expanduser() if calibration_cache else None
    if auto_calibrate and _calib_cache_path is not None:
        # Fail loud at CLI layer if parent dir isn't writable. Mirrors the
        # embodiment-config validation pattern at cli.py:1128-1162.
        try:
            _calib_cache_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            console.print(
                f"[red]--calibration-cache parent dir not writable: "
                f"{_calib_cache_path.parent} — {exc}[/red]"
            )
            raise typer.Exit(1)

    # SLO enforcement (Phase 1 latency-slo-enforcement feature).
    # --slo required to enable; default mode is "degrade".
    slo_tracker = None
    if slo:
        try:
            from reflex.runtime.slo import SLOTracker, parse_slo_spec, validate_slo_mode
            _slo_spec = parse_slo_spec(slo)
            _slo_mode_validated = validate_slo_mode(slo_mode)
            slo_tracker = SLOTracker(_slo_spec)
        except ValueError as exc:
            console.print(f"[red]SLO config invalid: {exc}[/red]")
            raise typer.Exit(1)
        composed.append(f"[cyan]slo={slo}/{slo_mode}[/cyan]")
    else:
        _slo_mode_validated = "degrade"  # ignored when slo_tracker is None

    app_instance = create_app(
        export_dir,
        device=device,
        providers=provider_list,
        strict_providers=not no_strict_providers,
        safety_config=safety_config or None,
        adaptive_steps=adaptive_steps,
        cloud_fallback_url=cloud_fallback,
        deadline_ms=deadline_ms if deadline_ms > 0 else None,
        max_batch=max_batch,
        batch_timeout_ms=batch_timeout_ms,
        api_key=api_key or None,
        replan_hz=replan_hz if replan_hz > 0 else None,
        execute_hz=execute_hz if execute_hz > 0 else None,
        embodiment_config=embodiment_cfg,
        record_dir=record or None,
        record_image_redaction=record_images,
        record_gzip=not record_no_gzip,
        rtc_config=rtc_cfg,
        inject_latency_ms=inject_latency_ms,
        prewarm=not no_prewarm,
        max_consecutive_crashes=max_consecutive_crashes,
        slo_tracker=slo_tracker,
        slo_mode=_slo_mode_validated,
        max_concurrent=max_concurrent if max_concurrent > 0 else None,
        otel_endpoint=otel_endpoint or None,
        otel_sample=otel_sample,
        robot_id=robot_id or None,
        cuda_graphs_enabled=cuda_graphs,
        action_similarity_threshold=action_similarity_threshold,
        max_similar_skips=max_similar_skips,
        max_batch_cost_ms=max_batch_cost_ms,
        a2c2_checkpoint=a2c2_checkpoint or None,
        a2c2_latency_threshold_ms=a2c2_latency_threshold_ms,
        a2c2_success_threshold=a2c2_success_threshold,
        bid_n_candidates=bid_n_candidates,
        bid_coherence_window=bid_coherence_window,
        bid_coherence_metric=bid_coherence_metric,
        auto_calibrate=auto_calibrate,
        calibration_cache_path=str(_calib_cache_path) if _calib_cache_path else None,
        calibrate_force=calibrate_force,
        # Policy-versioning Days 9-10: 2-policy mode wiring. Per ADR
        # 2026-04-25-policy-versioning-architecture.
        policy_b_export_dir=policy_b or None,
        policy_split_a_percent=split,
        policy_crash_threshold=max_consecutive_crashes,
    )
    if api_key:
        composed.append("[cyan]api-key-auth[/cyan]")
    if replan_hz > 0:
        composed.append(
            f"[cyan]replan[/cyan]={replan_hz:g}Hz/execute={execute_hz:g}Hz"
        )
    if otel_endpoint:
        composed.append(
            f"[cyan]otel[/cyan]={otel_endpoint}@{otel_sample:g}"
        )
    if robot_id:
        composed.append(f"[cyan]robot[/cyan]={robot_id}")
    if cuda_graphs:
        composed.append("[cyan]cuda-graphs[/cyan]")
    if action_similarity_threshold > 0:
        composed.append(
            f"[cyan]action-fast-path[/cyan]"
            f"=L2<{action_similarity_threshold:g}/max-skip={max_similar_skips}"
        )
    if a2c2_checkpoint:
        composed.append(f"[cyan]a2c2[/cyan]={Path(a2c2_checkpoint).name}")
    if auto_calibrate:
        composed.append("[cyan]auto-calibrate[/cyan]" + ("[force]" if calibrate_force else ""))
    composed.append(f"[cyan]batch-budget[/cyan]={max_batch_cost_ms:g}ms")
    # MCP server integration (Phase 1 mcp-server feature).
    # --mcp --mcp-transport stdio: MCP-only mode (FastAPI NOT started — stdio
    #   needs to own stdin/stdout; used for Claude Desktop / Cursor).
    # --mcp --mcp-transport http: both MCP (on --mcp-port) AND FastAPI run.
    # no --mcp: FastAPI only (legacy behavior).
    if mcp:
        if mcp_transport not in ("stdio", "http"):
            console.print(
                f"[red]Invalid --mcp-transport {mcp_transport!r}; expected 'stdio' or 'http'.[/red]"
            )
            raise typer.Exit(1)
        try:
            from reflex.mcp import create_mcp_server
        except ImportError:
            console.print(
                "MCP dependency not installed. Run:\n"
                "  pip install 'reflex-vla[mcp]'",
                style="red", markup=False,
            )
            raise typer.Exit(1)
        # Pull the live ReflexServer out of the FastAPI app's state
        reflex_srv = getattr(app_instance.state, "reflex_server", None)
        if reflex_srv is None:
            console.print(
                "[red]Could not find ReflexServer on the app state; MCP needs a live "
                "inference engine. Report this at github.com/FastCrest/reflex-vla/issues.[/red]"
            )
            raise typer.Exit(1)
        mcp_srv = create_mcp_server(reflex_srv)
        composed.append(f"[cyan]mcp={mcp_transport}[/cyan]")

        if mcp_transport == "stdio":
            console.print("[bold green]Starting MCP server (stdio)...[/bold green]")
            console.print("[dim]FastAPI NOT started — stdio owns stdin/stdout.[/dim]")
            # mcp.run() blocks until client disconnects
            mcp_srv.run(transport="stdio")
            return
        # HTTP mode: run MCP in a background thread, FastAPI on main thread
        import threading
        def _run_mcp_http():
            mcp_srv.run(transport="streamable-http", host="127.0.0.1", port=mcp_port)
        mcp_thread = threading.Thread(target=_run_mcp_http, daemon=True, name="mcp-http")
        mcp_thread.start()
        console.print(
            f"[bold green]MCP server running on http://127.0.0.1:{mcp_port} "
            f"(streamable-http)[/bold green]"
        )

    console.print("[bold green]Starting server...[/bold green]")
    uvicorn.run(app_instance, host=host, port=port, log_level="info" if verbose else "warning")


@app.command(name="ros2-serve", hidden=True)
def ros2_serve(
    export_dir: str = typer.Argument(help="Path to exported model directory"),
    device: str = typer.Option("cuda", help="ORT execution device: cuda or cpu"),
    image_topic: str = typer.Option(
        "/camera/image_raw",
        help="sensor_msgs/Image topic for observation frames",
    ),
    state_topic: str = typer.Option(
        "/joint_states",
        help="sensor_msgs/JointState topic — .position field becomes the state vector",
    ),
    task_topic: str = typer.Option(
        "/reflex/task",
        help="std_msgs/String topic for the text instruction",
    ),
    action_topic: str = typer.Option(
        "/reflex/actions",
        help="std_msgs/Float32MultiArray topic — published chunk, flattened",
    ),
    rate_hz: float = typer.Option(20.0, help="Inference rate (Hz)"),
    safety_config: str = typer.Option("", help="Path to SafetyLimits JSON"),
    node_name: str = typer.Option("reflex_vla", help="ROS2 node name"),
    mcp: bool = typer.Option(
        False,
        "--mcp",
        help="Also expose the live ROS2 bridge as MCP tools (4 read/actuation "
             "tools + robot://status resource per ros2-mcp-bridge.md). Pairs "
             "with --mcp-transport stdio for Claude Desktop / Cursor or http "
             "for cross-process agent loops.",
    ),
    mcp_transport: str = typer.Option(
        "stdio",
        "--mcp-transport",
        help="MCP transport when --mcp is set. 'stdio' (Claude Desktop default; "
             "MCP owns stdin/stdout, rclpy spins in background thread) or 'http' "
             "(MCP runs on --mcp-port background thread; rclpy.spin owns main).",
    ),
    mcp_port: int = typer.Option(
        8001,
        "--mcp-port",
        help="MCP HTTP port (only used with --mcp-transport http).",
    ),
):
    """Run a ROS2 node wrapping reflex inference; optionally expose as MCP.

    Requires ROS2 installed via apt or robostack (rclpy is NOT pip-installable).
    Source your ROS2 environment before running:

        source /opt/ros/humble/setup.bash   # or iron / jazzy
        reflex ros2-serve ./my_export/

    With --mcp, exposes the live ROS2 bridge as MCP tools so coding agents
    (Claude, Cursor, custom) can introspect AND drive the robot through one
    connection. Per ros2-mcp-bridge.md (Phase 1.5).

        reflex ros2-serve ./my_export/ --mcp                     # stdio
        reflex ros2-serve ./my_export/ --mcp --mcp-transport http --mcp-port 8001
    """
    try:
        from reflex.runtime.ros2_bridge import run_ros2_bridge
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    if mcp and mcp_transport not in ("stdio", "http"):
        console.print(
            f"[red]Invalid --mcp-transport {mcp_transport!r}; "
            f"expected 'stdio' or 'http'.[/red]"
        )
        raise typer.Exit(1)

    console.print(f"[bold green]Starting reflex ros2 bridge[/bold green]")
    console.print(f"  export_dir: {export_dir}")
    console.print(f"  node_name: {node_name}")
    console.print(f"  rate_hz: {rate_hz}")
    console.print(f"  subs: {image_topic}, {state_topic}, {task_topic}")
    console.print(f"  pub:  {action_topic}")
    if mcp:
        console.print(f"  mcp:  {mcp_transport}" + (f" (port {mcp_port})" if mcp_transport == "http" else ""))
    try:
        run_ros2_bridge(
            export_dir,
            device=device,
            safety_config=safety_config or None,
            image_topic=image_topic,
            state_topic=state_topic,
            task_topic=task_topic,
            action_topic=action_topic,
            rate_hz=rate_hz,
            node_name=node_name,
            mcp=mcp,
            mcp_transport=mcp_transport,
            mcp_port=mcp_port,
        )
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)


@app.command(hidden=True)
def replay(
    trace_file: str = typer.Argument(help="Path to recorded JSONL trace (.jsonl or .jsonl.gz)"),
    model: str = typer.Option(
        "",
        "--model",
        help="Path to target export dir for replay. Required unless --no-replay.",
    ),
    diff: str = typer.Option(
        "actions",
        "--diff",
        help="Diff mode (Day 2 ships actions only; latency/cache/all in Day 3).",
    ),
    n: int = typer.Option(
        0,
        "--n",
        help="Replay first N records only. 0 = all.",
    ),
    output: str = typer.Option(
        "",
        "--output",
        help="Write machine-readable diff report to this JSON path.",
    ),
    fail_on: str = typer.Option(
        "",
        "--fail-on",
        help="Exit non-zero if any diff of this type fails (e.g. --fail-on actions).",
    ),
    no_replay: bool = typer.Option(
        False,
        "--no-replay",
        help="Parse the trace + print header/counts without loading the model. "
             "Useful for inspecting traces and validating their schema.",
    ),
):
    """Replay a recorded /act trace against a target model.

    Day 2 scope: load JSONL, replay each request, compute per-record actions
    diff (cosine + max_abs). Latency / cache / guard diff modes land Day 3.

    Trace format: TECHNICAL_PLAN §D.1 (schema v1).
    """
    from reflex.replay.cli import run_replay

    if not no_replay and not model:
        console.print(
            "[red]--model is required (or pass --no-replay to inspect the trace "
            "without loading a model).[/red]"
        )
        raise typer.Exit(1)
    code = run_replay(
        trace_file,
        model or None,
        diff_mode=diff,
        n=n,
        output_json=output,
        fail_on=fail_on,
        no_replay=no_replay,
    )
    if code != 0:
        raise typer.Exit(code)


@app.command(hidden=True)
def targets():
    """List supported hardware targets."""
    table = Table(title="Supported Hardware Targets")
    table.add_column("Target", style="cyan")
    table.add_column("Name")
    table.add_column("Memory")
    table.add_column("FP8")
    table.add_column("Precision")

    for key, hw in HARDWARE_PROFILES.items():
        table.add_row(
            key,
            hw.name,
            f"{hw.memory_gb} GB",
            "yes" if hw.fp8_support else "no",
            hw.trt_precision,
        )

    console.print(table)


# NOTE: this top-level `models` command was shadowed by the `models` typer
# subgroup added in the model-zoo-cli ship (2026-04-24). Decorator removed
# in the verb-noun refactor (same day) — function kept as dead code rather
# than deleted to preserve any imports of `from reflex.cli import models`.
def models():
    """[DEAD] Old top-level `reflex models` — shadowed by the typer subgroup."""
    from reflex.checkpoint import SUPPORTED_MODELS

    table = Table(title="Supported VLA Models")
    table.add_column("Type", style="cyan")
    table.add_column("HF ID")
    table.add_column("Params")
    table.add_column("Action head")
    table.add_column("Export")

    status_map = {
        "smolvla": "[green]✓ ONNX + validated[/green]",
        "pi0": "[green]✓ ONNX + validated[/green]",
        "pi05": "[green]✓ ONNX + AdaRMSNorm[/green]",
        "gr00t": "[green]✓ DiT + AdaLN + validated[/green]",
        "openvla": "[yellow]use optimum-onnx; Reflex only ships postprocess helpers[/yellow]",
    }

    for key, info in SUPPORTED_MODELS.items():
        table.add_row(
            key,
            info["hf_id"],
            f"{info['params_m']}M",
            info["action_head"],
            status_map.get(key, "[yellow]planned[/yellow]"),
        )

    console.print(table)
    console.print("\n[dim]Usage:[/dim] [cyan]reflex export <hf_id>[/cyan] — auto-detects model type.")


# `reflex distill` registered below via `app.command(name="distill")` on the
# finetune package. Kept out of this file so test collection doesn't pull
# lerobot + torch + SnapFlow deps just to load the CLI module.


@app.command(hidden=True)
def turbo(
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """[DEPRECATED] Adaptive denoising now lives on `reflex serve --adaptive-steps`."""
    console.print(
        "[yellow]`reflex turbo` is deprecated and will be removed in v0.3.[/yellow]\n"
        "[yellow]Adaptive denoising is now a flag on serve:[/yellow]\n"
        "  [cyan]reflex serve <export> --adaptive-steps[/cyan]\n\n"
        "[dim]Note: adaptive denoising only produces safe results on pi0.\n"
        "For pi0.5/SmolVLA/GR00T, use `reflex distill` instead (v0.2+).[/dim]"
    )
    raise typer.Exit(0)


@app.command(hidden=True)
def split(
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """[DEPRECATED] Cloud-edge orchestration is now a flag on `reflex serve`."""
    console.print(
        "[yellow]`reflex split` is deprecated and will be removed in v0.3.[/yellow]\n"
        "[yellow]Cloud-edge fallback is now a flag on serve:[/yellow]\n"
        "  [cyan]reflex serve <export> --cloud-fallback <url>[/cyan]\n\n"
        "[dim]Fewer than 10% of production deployments use cloud-edge split,\n"
        "so a dedicated command was removed in favor of a flag.[/dim]"
    )
    raise typer.Exit(0)


@app.command(hidden=True)
def adapt(
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """[DEPRECATED] Velocity clamping folded into `reflex guard`. Cross-embodiment archived."""
    console.print(
        "[yellow]`reflex adapt` is deprecated and will be removed in v0.3.[/yellow]\n"
        "[yellow]Velocity/torque limits are now part of `reflex guard`:[/yellow]\n"
        "  [cyan]reflex guard init --urdf <file> --output ./safety.json[/cyan]\n\n"
        "[dim]Cross-embodiment action remapping had no users; archived.\n"
        "Open an issue if you need it back.[/dim]"
    )
    raise typer.Exit(0)


@app.command(hidden=True)
def check(
    checkpoint: str = typer.Argument(help="HuggingFace ID or local path"),
    target: str = typer.Option("desktop", help="Target hardware: orin-nano, orin, orin-64, thor, desktop"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """[DEPRECATED] Replaced by `reflex validate --pre-export`. Forwards for compat."""
    console.print(
        "[yellow]`reflex check` is deprecated and will be removed in v0.3.[/yellow]\n"
        "[yellow]Use:[/yellow] [cyan]reflex validate "
        f"{checkpoint} --pre-export --hardware {target}[/cyan]\n"
    )
    _setup_logging(verbose)
    from reflex.validate_training import run_all_checks

    results = run_all_checks(checkpoint, target=target)
    table = Table(title="Pre-Deployment Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")
    n_pass = 0
    for r in results:
        status = "[green]PASS[/green]" if r.passed else (
            "[yellow]WARN[/yellow]" if r.severity == "warning" else "[red]FAIL[/red]"
        )
        if r.passed:
            n_pass += 1
        table.add_row(r.name, status, r.detail[:80])
    console.print(table)
    console.print(f"\n  Passed: [bold]{n_pass}/{len(results)}[/bold]")
    if n_pass < len(results):
        raise typer.Exit(1)


def _check_trt_ep_load_chain(add) -> None:
    """Validate that ORT-TRT EP can actually load + activate.

    Adds 4 checks to the doctor table via the `add(name, ok, detail)` callback:
    1. libnvinfer.so.10 loadable (the TRT runtime; from `tensorrt` pip pkg)
    2. libcublas.so.12 loadable (CUDA BLAS; from `nvidia-cublas-cu12` pip pkg)
    3. libcudnn.so.9 loadable (CUDA NN; from `nvidia-cudnn-cu12` pip pkg)
    4. ort.InferenceSession with TRT EP succeeds + active providers includes it
       (gold standard — proves the entire load chain works end-to-end)

    Each check has a remediation hint inline. Per ADR
    2026-04-29-ort-trt-ep-first-class-support.md.
    """
    import ctypes
    import os

    # Check 1-3: shared library loadability. Resolve full paths from the
    # candidate dirs (same logic as reflex/__init__.py:_candidate_lib_dirs)
    # because bare `ctypes.CDLL("libnvinfer.so.10")` only works if the
    # dynamic loader's path includes the lib dir — and modifying
    # LD_LIBRARY_PATH after process start does NOT update the loader.
    # The reflex import already eagerly dlopen's these libs with RTLD_GLOBAL;
    # we re-do the lookup here to surface the per-lib status to the user.
    from reflex import _candidate_lib_dirs

    def _find_lib(libname: str) -> str | None:
        for libdir in _candidate_lib_dirs():
            full = os.path.join(libdir, libname)
            if os.path.exists(full):
                return full
        return None

    # Use \[ to escape Rich markup brackets so they render as literal text
    # in the doctor table.
    libs = [
        ("libnvinfer.so.10", "TensorRT runtime",
         r"pip install 'reflex-vla\[serve,gpu]' (brings tensorrt>=10)"),
        ("libcublas.so.12", "CUDA cuBLAS",
         r"pip install nvidia-cublas-cu12 (auto-included in \[serve,gpu])"),
        ("libcudnn.so.9", "CUDA cuDNN",
         r"pip install nvidia-cudnn-cu12 (auto-included in \[serve,gpu])"),
    ]
    all_libs_loadable = True
    for libname, friendly, fix_hint in libs:
        full_path = _find_lib(libname)
        if full_path is None:
            all_libs_loadable = False
            add(
                f"{friendly} ({libname})",
                False,
                f"NOT installed (not found in pip site-packages). "
                f"Fix: {fix_hint}",
            )
            continue
        try:
            ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
            add(f"{friendly} ({libname})", True, f"loadable at {full_path}")
        except OSError as exc:
            all_libs_loadable = False
            add(
                f"{friendly} ({libname})",
                False,
                f"installed at {full_path} but not loadable: {exc}. "
                f"Fix: {fix_hint}",
            )

    # Check 4: gold standard — actually create an ORT session with TRT EP
    # and verify the provider becomes active. Skip if libs failed (would
    # produce a less-informative error).
    if not all_libs_loadable:
        add(
            "ORT-TRT EP active",
            False,
            "skipped — fix the missing libs above first",
        )
        return

    try:
        import onnxruntime as ort
    except ImportError:
        add("ORT-TRT EP active", False,
            "onnxruntime not installed — pip install 'reflex-vla[serve,gpu]'")
        return

    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        add(
            "ORT-TRT EP active",
            False,
            "TensorrtExecutionProvider not in onnxruntime's available "
            "providers list. Either onnxruntime-gpu isn't installed (use "
            "'reflex-vla[serve,gpu]') or you're on a CPU-only ORT build.",
        )
        return

    # Build a tiny dummy ONNX in-memory and try to load with TRT EP.
    # If the session creates AND active providers includes TRT EP, we're golden.
    try:
        import onnx
        from onnx import helper, TensorProto

        # Trivial 1-add graph: y = x + x. Smallest valid ONNX.
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        node = helper.make_node("Add", ["x", "x"], ["y"])
        graph = helper.make_graph([node], "doctor-probe", [x], [y])
        model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
        model_proto.ir_version = 9
        model_bytes = model_proto.SerializeToString()
    except Exception as exc:  # noqa: BLE001
        add("ORT-TRT EP active", False,
            f"could not build probe ONNX: {type(exc).__name__}: {exc}")
        return

    try:
        sess = ort.InferenceSession(
            model_bytes,
            providers=["TensorrtExecutionProvider", "CPUExecutionProvider"],
        )
        active = sess.get_providers()
        if "TensorrtExecutionProvider" in active:
            add("ORT-TRT EP active", True,
                f"session created, active providers: {active}")
        else:
            add(
                "ORT-TRT EP active",
                False,
                f"session created but TRT EP fell back. Active: {active}. "
                f"Likely: libnvinfer or CUDA libs not findable at runtime "
                f"despite ctypes load succeeding (try setting "
                f"LD_LIBRARY_PATH manually).",
            )
    except Exception as exc:  # noqa: BLE001
        add(
            "ORT-TRT EP active",
            False,
            f"session creation failed: {type(exc).__name__}: {str(exc)[:200]}",
        )


@app.command()
def doctor(
    model: str = typer.Option(
        "",
        "--model",
        help="Optional path to an exported model directory. When passed, runs "
             "deploy diagnostics (5 falsifiable checks for known LeRobot async "
             "issues + systemic VLA deploy failures) AFTER the system probe. "
             "Without --model, runs system probe only.",
    ),
    embodiment: str = typer.Option(
        "custom",
        "--embodiment",
        help="Embodiment preset (franka/so100/ur5) for deploy-diagnostic cross-checks. "
             "Only used when --model is also passed.",
    ),
    rtc: bool = typer.Option(
        False,
        "--rtc",
        help="Validate RTC chunk-boundary alignment in deploy diagnostics. "
             "Only used when --model is also passed.",
    ),
    output_format: str = typer.Option(
        "human",
        "--format",
        help="Output format for deploy diagnostics: 'human' (table) or 'json' "
             "(machine-readable, schema_version=1). System probe is always human-readable.",
    ),
    skip: list[str] = typer.Option(
        [],
        "--skip",
        help="Deploy-diagnostic check IDs to skip. Repeatable.",
    ),
    show_calibration: bool = typer.Option(
        False,
        "--show-calibration",
        help="Pretty-print the auto-calibration cache from "
             "--calibration-cache (default ~/.reflex/calibration.json). "
             "When combined with --format json, emits a machine-readable "
             "snapshot for CI / scripts. Per a2u-calibration plan B.5 Day 4.",
    ),
    calibration_cache: str = typer.Option(
        "~/.reflex/calibration.json",
        "--calibration-cache",
        help="Path to the auto-calibration cache JSON. Used by "
             "--show-calibration.",
    ),
):
    """Diagnose Reflex install + GPU issues + (optionally) per-deploy issues.

    Two modes:
      reflex doctor                              # system probe (Python, CUDA,
                                                 # ORT providers, fastapi, etc.)
      reflex doctor --model ./export/pi05 \\     # system probe + 5 deploy checks
                    --embodiment franka          # against your specific export

    Exit codes: 0 all pass, 1 at least one deploy-check fail, 2 invocation error,
    3 environment error.

    Plan: features/01_serve/subfeatures/_dx_gaps/reflex-doctor_plan.md
    """
    import platform
    import shutil
    import sys

    if output_format not in ("human", "json"):
        console.print(
            f"[red]--format must be 'human' or 'json', got {output_format!r}[/red]"
        )
        raise typer.Exit(2)

    # --show-calibration short-circuits the full doctor flow: just print the
    # cache + exit. Used by operators to quickly inspect what the auto-calibrate
    # cache holds + how stale it is.
    if show_calibration:
        from reflex.runtime.calibration import (
            CalibrationCache,
            HardwareFingerprint,
        )
        cache_path = Path(calibration_cache).expanduser()
        if not cache_path.exists():
            if output_format == "json":
                import json as _json
                _json.dump({"error": "cache_not_found", "path": str(cache_path)},
                           sys.stdout, indent=2)
                print()
            else:
                console.print(
                    f"[yellow]No calibration cache found at {cache_path}. "
                    f"Run `reflex serve --auto-calibrate` first.[/yellow]"
                )
            raise typer.Exit(0)
        try:
            cache = CalibrationCache.load(cache_path)
        except Exception as exc:
            console.print(f"[red]Failed to load cache: {exc}[/red]")
            raise typer.Exit(1)

        current_fp = HardwareFingerprint.current()
        is_stale = cache.is_stale(current_fp)

        if output_format == "json":
            import json as _json
            payload = {
                "path": str(cache_path),
                "current_fingerprint": current_fp.to_dict(),
                "is_stale": is_stale,
                "cache": cache.to_dict(),
            }
            _json.dump(payload, sys.stdout, indent=2)
            print()
            raise typer.Exit(0)

        # Human-readable
        console.print(f"[bold]Calibration cache:[/bold] {cache_path}")
        console.print(f"  schema_version: {cache.schema_version}")
        console.print(f"  reflex_version: {cache.reflex_version}")
        console.print(f"  calibration_date: {cache.calibration_date}")
        console.print(
            f"  hardware_fingerprint: " +
            ("[green]matches current host[/green]" if not is_stale else
             "[yellow]STALE — hardware/version mismatch or > 30d old[/yellow]")
        )
        console.print(f"\n[bold]Entries ({len(cache.entries)}):[/bold]")
        if not cache.entries:
            console.print("  (none)")
        else:
            for k, e in cache.entries.items():
                console.print(
                    f"  {k}: chunk={e.chunk_size} nfe={e.nfe} "
                    f"latency_comp={e.latency_compensation_ms:g}ms "
                    f"provider={e.provider} variant={e.variant} "
                    f"quality={e.measurement_quality.quality_score:.2f}"
                )
        raise typer.Exit(0)

    table = Table(title="Reflex Doctor")
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail")

    def add(name: str, ok: bool, detail: str):
        symbol = "[green]✓[/green]" if ok else "[yellow]⚠[/yellow]"
        table.add_row(name, symbol, detail)

    # Python
    py = sys.version_info
    add(
        "Python version",
        py >= (3, 10),
        f"{py.major}.{py.minor}.{py.micro} (need ≥3.10)",
    )

    # OS / architecture
    add("Platform", True, f"{platform.system()} {platform.machine()}")

    # torch + CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        cuda_detail = (
            f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
            f"available={cuda_ok}"
        )
        if cuda_ok:
            cuda_detail += f", devices={torch.cuda.device_count()}, "
            cuda_detail += f"name={torch.cuda.get_device_name(0)}"
        add("torch + CUDA", cuda_ok, cuda_detail)
    except ImportError as e:
        add("torch + CUDA", False, f"torch not installed: {e}")

    # ─── Multi-GPU mixed-arch guard ─────────────────────────────────────
    # If the customer has 2+ NVIDIA GPUs of different architectures (e.g.
    # 1× A100 + 1× RTX 5090), ORT only uses CUDA_VISIBLE_DEVICES[0] by
    # default. Surface the mix loudly so they don't get a working ORT
    # session on GPU 0 + silent kernel-image failure if they ever target
    # GPU 1. Quiet on single-GPU + uniform-multi-GPU systems.
    try:
        import subprocess as _sub
        proc = _sub.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3.0,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            lines = [
                ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()
            ]
            if len(lines) >= 2:
                # Crude arch detection — same compute-cap if same generation.
                # We don't have nvidia-ml-py; use name-substring heuristics.
                def _arch_from_name(name: str) -> str:
                    n = name.lower()
                    if any(p in n for p in ("rtx 50", "rtx pro 60", "blackwell", "b200", "gb200")):
                        return "blackwell"
                    if any(p in n for p in ("h100", "h200", "hopper")):
                        return "hopper"
                    if any(p in n for p in ("rtx 40", "l4", "l40", "ada")):
                        return "ada"
                    if any(p in n for p in ("a100", "a10g", "a40", "ampere", "rtx 30")):
                        return "ampere"
                    if "orin" in n or "tegra" in n:
                        return "orin"
                    return "unknown"

                names = [ln.split(",", 1)[1].strip() for ln in lines]
                archs = {_arch_from_name(n) for n in names}
                archs.discard("unknown")
                if len(archs) > 1:
                    add(
                        "  → multi-GPU arch consistency",
                        False,
                        f"⚠ Mixed GPU architectures: {sorted(archs)} ({len(lines)} GPUs). "
                        f"ORT uses CUDA_VISIBLE_DEVICES[0] only by default. "
                        f"If you target GPU 1 via CUDA_VISIBLE_DEVICES, kernels "
                        f"compiled for GPU 0's arch will silently fail.",
                    )
    except (FileNotFoundError, ImportError, OSError):
        pass

    # ─── Jetson JetPack guard ───────────────────────────────────────────
    # Jetson devices ship CUDA + cuDNN baked into JetPack at the OS level.
    # Customers running ORT 1.25+ on JetPack 5.x (CUDA 11.4) will silently
    # fall to CPU because ORT's bundled CUDA 12 EP can't find compatible
    # libs. Surface JetPack version + ORT compatibility loudly.
    try:
        from pathlib import Path as _P
        jetson_release = _P("/etc/nv_tegra_release")
        if jetson_release.exists():
            content = jetson_release.read_text(errors="ignore")
            # Format example: "# R36 (release), REVISION: 4.0, GCID: ..."
            jetpack_major = "unknown"
            for line in content.splitlines():
                if line.startswith("# R"):
                    parts = line.split()
                    if len(parts) >= 2:
                        jetpack_major = parts[1].lstrip("R")
                    break
            # JetPack R36+ ships CUDA 12.x; R35 ships CUDA 11.4
            # ORT 1.20+ requires CUDA 12.x → R36+ is required for GPU EP.
            try:
                jp_int = int(jetpack_major)
            except (TypeError, ValueError):
                jp_int = 0
            if jp_int and jp_int < 36:
                add(
                    "  → Jetson JetPack target",
                    False,
                    f"❌ JetPack R{jetpack_major} ships CUDA 11.4. ORT 1.20+ "
                    f"requires CUDA 12.x → CUDAExecutionProvider will silently "
                    f"fall to CPU. Upgrade to JetPack R36+ (Orin) or use "
                    f"reflex-vla[serve,onnx] for CPU-only inference.",
                )
            elif jp_int >= 36:
                add(
                    "  → Jetson JetPack target",
                    True,
                    f"JetPack R{jetpack_major} (CUDA 12.x compatible).",
                )
    except (OSError, ImportError):
        pass

    # ─── CUDA driver vs cuDNN version skew guard ────────────────────────
    # cuDNN minor versions have driver minimum requirements (cuDNN 9.5
    # needs NVIDIA driver R555+; cuDNN 9.0 needs R550+). Mismatch causes
    # silent kernel failures at first invocation, not at session-init.
    # Catches: customer pinned old driver via apt-hold, used Reflex's
    # bundled cuDNN at the wrong system driver level.
    try:
        import subprocess as _sub
        nv_proc = _sub.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3.0,
        )
        if nv_proc.returncode == 0:
            driver_str = (nv_proc.stdout or "").strip().split("\n")[0]
            try:
                driver_major = int(driver_str.split(".")[0])
            except (ValueError, IndexError):
                driver_major = 0
            try:
                from importlib.metadata import version as _v
                cudnn_v = _v("nvidia-cudnn-cu12")
                cudnn_minor = int(cudnn_v.split(".")[1]) if cudnn_v else 0
                # cuDNN 9.5 needs driver R555+; cuDNN 9.0 needs R550+
                min_driver = 555 if cudnn_minor >= 5 else 550
                if driver_major and driver_major < min_driver:
                    add(
                        "  → CUDA driver vs cuDNN",
                        False,
                        f"❌ NVIDIA driver R{driver_major} predates cuDNN "
                        f"{cudnn_v} requirement (needs R{min_driver}+). "
                        f"Kernels will silently fail at first inference call. "
                        f"Upgrade driver: `sudo apt install nvidia-driver-{min_driver}` "
                        f"or use cuDNN <9.5 (`pip install 'nvidia-cudnn-cu12<9.5'`).",
                    )
                elif driver_major:
                    add(
                        "  → CUDA driver vs cuDNN",
                        True,
                        f"driver R{driver_major} OK for cuDNN {cudnn_v}",
                    )
            except Exception:  # noqa: BLE001 — best-effort probe
                pass
    except (FileNotFoundError, ImportError, OSError):
        pass

    # ONNX Runtime + execution providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_trt = "TensorrtExecutionProvider" in providers
        has_cuda = "CUDAExecutionProvider" in providers
        ort_detail = f"ort {ort.__version__}, providers={providers}"
        add(
            "ONNX Runtime",
            True,
            ort_detail,
        )
        add(
            "  → CUDAExecutionProvider",
            has_cuda,
            "available" if has_cuda else (
                "NOT available — install onnxruntime-gpu or check CUDA 12 + cuDNN 9 system libs"
            ),
        )
        add(
            "  → TensorrtExecutionProvider",
            has_trt,
            "available — reflex serve will auto-prefer this" if has_trt else
            "NOT available — TRT FP16 disabled, will use CUDA EP",
        )

        # ─── ORT-TRT EP empirical session test ───────────────────────────
        # `available_providers` says the lib is loaded — it does NOT confirm
        # session-init will succeed with that provider. The v0.7 install gap
        # (caught 2026-04-29 by the v07-install-validation experiment + ADR
        # 2026-04-29-ort-trt-ep-first-class-support) is exactly this: TRT EP
        # registered but `InferenceSession(providers=[TRT EP])` falls back to
        # CUDA because libnvinfer.so.10 isn't in the dlopen path. Customers
        # silently lose the 5.55× perf win.
        # Empirical fix: try creating a session with TRT EP forced + check
        # active providers. Only runs when TRT EP is "available" per above.
        if has_trt:
            try:
                import os as _os
                import tempfile as _tmp
                import onnx as _onnx
                from onnx import TensorProto, helper

                # Build a tiny stub model (1x1 add) — small enough that
                # ORT session creation is the bottleneck, not graph compile.
                node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])
                graph = helper.make_graph(
                    [node],
                    "trt_ep_smoke",
                    [
                        helper.make_tensor_value_info("A", TensorProto.FLOAT, [1]),
                        helper.make_tensor_value_info("B", TensorProto.FLOAT, [1]),
                    ],
                    [helper.make_tensor_value_info("C", TensorProto.FLOAT, [1])],
                )
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
                model.ir_version = 9

                with _tmp.NamedTemporaryFile(suffix=".onnx", delete=False) as _f:
                    _f.write(model.SerializeToString())
                    stub_path = _f.name

                try:
                    # Force-prefer TRT EP. If session creation succeeds AND
                    # active providers include TRT EP, the loadchain is fine.
                    sess = ort.InferenceSession(
                        stub_path,
                        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
                    )
                    active = sess.get_providers()
                    if "TensorrtExecutionProvider" in active:
                        add(
                            "  → TRT EP empirical load",
                            True,
                            "session-init with TRT EP succeeds; production "
                            "serve will get the 5.55× win.",
                        )
                    else:
                        add(
                            "  → TRT EP empirical load",
                            False,
                            f"❌ TRT EP listed available but session falls back "
                            f"to {active}. Likely libnvinfer.so.10 not on dlopen "
                            f"path. Fix: `pip install -U 'tensorrt>=10.0,<11'` + "
                            f"restart shell. Customer silently loses ~5× perf "
                            f"otherwise. See ADR 2026-04-29-ort-trt-ep-first-class.",
                        )
                finally:
                    _os.unlink(stub_path)
            except Exception as _exc:  # noqa: BLE001
                add(
                    "  → TRT EP empirical load",
                    False,
                    f"⚠ session-init test failed: {type(_exc).__name__}: {_exc}. "
                    f"Customer's TRT EP is likely broken; serve will silently "
                    f"fall to CUDA EP.",
                )

        # ─── Blackwell guard ──────────────────────────────────────────
        # Background: ORT 1.25.0 (2026-04-20) shipped Blackwell sm_120
        # kernels via PR #27278; 1.25.1 (2026-04-27) is current stable.
        # Earlier 1.23/1.24 regressed sm_120 (cudaErrorNoKernelImageForDevice).
        # Customers running Blackwell hardware on ORT < 1.25.1 hit a hard
        # segfault at session-init that is NOT a reflex bug. Surfaced
        # 2026-04-28 by tester (rob, RTX 5090); cost 2 weeks of his time
        # before we tracked the upstream fix.
        # This check fails LOUD per CLAUDE.md "no silent fallbacks" so
        # future Blackwell customers see the upgrade path immediately.
        from reflex.runtime.server import _gpu_is_blackwell
        if _gpu_is_blackwell():
            from packaging.version import Version
            installed = Version(ort.__version__)
            min_blackwell_safe = Version("1.25.1")
            if installed < min_blackwell_safe:
                add(
                    "  → Blackwell sm_120 support",
                    False,
                    f"❌ ORT {ort.__version__} predates Blackwell support. "
                    f"Upgrade to >=1.25.1 (1.25.0 added sm_120 kernels via "
                    f"PR #27278). Reflex on this hardware will SEGFAULT at "
                    f"session-init until upgraded. Run: "
                    f"`pip install -U 'onnxruntime-gpu>=1.25.1'`",
                )
            else:
                add(
                    "  → Blackwell sm_120 support",
                    True,
                    f"ORT {ort.__version__} ≥ 1.25.1 — Blackwell sm_120 "
                    f"kernels available. Live caveat: open ORT issue #27621 "
                    f"(silent threading deadlock on sm_120 with PTX JIT + "
                    f"GIL); reflex's single-thread inference path doesn't "
                    f"trigger it, but multi-threaded customers should monitor.",
                )
    except ImportError:
        add(
            "ONNX Runtime",
            False,
            "not installed — run `pip install onnxruntime-gpu` (or [onnx] for CPU)",
        )

    # ONNX (the format library)
    try:
        import onnx
        add("onnx (graph format)", True, f"version {onnx.__version__}")
    except ImportError:
        add("onnx (graph format)", False, "not installed — included in core deps now")

    # onnxscript (needed for torch.onnx.export new path)
    try:
        import onnxscript
        add("onnxscript", True, f"version {onnxscript.__version__}")
    except ImportError:
        add("onnxscript", False, "not installed — needed by torch.onnx.export")

    # transformers + huggingface_hub
    try:
        import transformers
        add("transformers", True, f"version {transformers.__version__}")
    except ImportError:
        add("transformers", False, "not installed — needed for some exporters")
    try:
        import huggingface_hub
        add("huggingface_hub", True, f"version {huggingface_hub.__version__}")
    except ImportError:
        add("huggingface_hub", False, "not installed — needed to download checkpoints")

    # FastAPI + uvicorn (for serve)
    try:
        import fastapi
        import uvicorn
        add("fastapi + uvicorn", True, f"fastapi {fastapi.__version__} / uvicorn {uvicorn.__version__}")
    except ImportError:
        # Recommend the right extras for this platform: NVIDIA-GPU users want gpu,
        # everyone else (Mac, CPU-only Linux, ARM Jetson) wants onnx (CPU runtime).
        import platform as _plat
        is_apple_silicon = _plat.system() == "Darwin"
        cpu_only = is_apple_silicon  # extend later: detect non-NVIDIA Linux too
        extra = "serve,onnx" if cpu_only else "serve,gpu"
        add(
            "fastapi + uvicorn",
            False,
            f"not installed — run `pip install 'reflex-vla\\[{extra}]'` for the server",
        )

    # safetensors
    try:
        import safetensors
        add("safetensors", True, f"version {safetensors.__version__}")
    except ImportError:
        add("safetensors", False, "not installed — needed to load checkpoints")

    # trtexec (for building .trt engines via reflex export)
    trtexec_path = shutil.which("trtexec")
    add(
        "trtexec (TensorRT)",
        bool(trtexec_path),
        trtexec_path or "not on PATH — TRT engine build skipped during reflex export "
                         "(install Jetpack on Jetson, or use nvcr.io/nvidia/tensorrt container)",
    )

    # ─── ORT-TRT EP load chain validation (v0.7) ─────────────────────
    # ORT-TRT EP gives ~5.55x speedup vs ORT-CUDA EP on transformer
    # workloads (Modal A10G spike 2026-04-29, SmolVLA monolithic).
    # Most users silently get ORT-CUDA fallback because libnvinfer.so.10
    # / libcublas.so.12 / libcudnn.so.9 aren't on LD_LIBRARY_PATH.
    # These checks make the actual problem visible.
    # Per ADR 2026-04-29-ort-trt-ep-first-class-support.md.
    _check_trt_ep_load_chain(add)
    # ──────────────────────────────────────────────────────────────────

    # Disk space at /tmp (used for transient export intermediates only — the
    # actual model export cache lives at ~/.cache/reflex/exports). On many
    # Linux distros /tmp is tmpfs (RAM-backed), in which case "free disk"
    # really means "free RAM", so label it explicitly to avoid confusion.
    try:
        usage = shutil.disk_usage("/tmp")
        free_gb = usage.free / 1e9
        is_tmpfs = False
        try:
            with open("/proc/mounts", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == "/tmp" and parts[2] == "tmpfs":
                        is_tmpfs = True
                        break
        except Exception:
            pass
        suffix = " (tmpfs/RAM-backed — model exports use ~/.cache/reflex/exports instead)" if is_tmpfs else " (transient ONNX/TRT scratch only — exports land in ~/.cache/reflex/exports)"
        add(
            "Free space in /tmp",
            free_gb > 2,
            f"{free_gb:.1f} GB free{suffix}",
        )
    except Exception as e:
        add("Free space in /tmp", False, str(e))

    # HuggingFace cache
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(hf_home):
        try:
            usage = shutil.disk_usage(hf_home)
            add("HF cache disk", usage.free > 10e9, f"{hf_home} ({usage.free / 1e9:.1f} GB free)")
        except Exception:
            pass

    # Reflex itself
    try:
        from reflex import __version__ as reflex_version
        add("reflex-vla", True, f"version {reflex_version}")
    except Exception as e:
        add("reflex-vla", False, str(e))

    # Curate data-contribution status (informational; no pass/fail).
    try:
        from reflex.curate import nudge_engine as _curate_nudge
        _status_line = _curate_nudge.doctor_status()
        add("Data contribution", True, _status_line)
    except Exception as _curate_exc:  # noqa: BLE001
        add("Data contribution", False, f"unavailable: {_curate_exc}")

    # Curate queue disk usage. Warns when queue exceeds 500 MB
    # (signals stuck uploads or disk-fill protection nearing the 1 GB limit).
    try:
        from reflex.curate.uploader import (
            DEFAULT_QUEUE_DIR as _Q,
            DEFAULT_REJECTED_DIR as _R,
            DEFAULT_UPLOADED_DIR as _U,
        )
        _q_path = Path(_Q).expanduser()
        _r_path = Path(_R).expanduser()
        _u_path = Path(_U).expanduser()

        def _bytes_under(p: Path) -> tuple[int, int]:
            if not p.exists():
                return 0, 0
            n = 0
            total = 0
            for f in p.glob("*.jsonl"):
                n += 1
                try:
                    total += f.stat().st_size
                except OSError:
                    continue
            return n, total

        _qn, _qb = _bytes_under(_q_path)
        _un, _ub = _bytes_under(_u_path)
        _rn, _rb = _bytes_under(_r_path)
        _q_mb = _qb / (1024 * 1024)
        _detail = (
            f"queue {_qn} files / {_qb / (1024 * 1024):.1f} MB · "
            f"uploaded {_un} / {_ub / (1024 * 1024):.1f} MB · "
            f"rejected {_rn} / {_rb / (1024 * 1024):.1f} MB"
        )
        # 500 MB warning threshold (1 GB hard limit per FreeContributorCollector spec).
        add("Contribute queue", _q_mb < 500, _detail)
    except Exception as _q_exc:  # noqa: BLE001
        add("Contribute queue", False, f"unavailable: {_q_exc}")

    console.print(table)
    console.print(
        "\n[dim]If something here is unexpected, see "
        "[cyan]docs/getting_started.md → Troubleshooting[/cyan] before "
        "opening an issue.[/dim]"
    )

    # Deploy diagnostics — only when --model is passed (B.4 Day 1 + future)
    if model:
        from reflex.diagnostics import (
            exit_code as _exit_code,
            format_human,
            format_json,
            run_all_checks,
        )

        console.print()
        console.print("[bold]Deploy diagnostics:[/bold]")
        console.print()

        results = run_all_checks(
            model_path=model,
            embodiment_name=embodiment,
            rtc=rtc,
            skip=skip,
        )
        if output_format == "json":
            console.print(format_json(
                results,
                model_path=model,
                embodiment_name=embodiment,
            ))
        else:
            console.print(format_human(results))

        code = _exit_code(results)
        if code != 0:
            raise typer.Exit(code)


@app.command(name="validate-dataset", hidden=True)
def validate_dataset(
    path: str = typer.Argument(help="Path to LeRobot v3.0 dataset root (contains meta/info.json)"),
    embodiment: str = typer.Option(
        "",
        "--embodiment",
        help="Embodiment preset (franka/so100/ur5) for cross-checking action_dim. "
             "When set, loads configs/embodiments/<name>.json and compares its "
             "action_dim against the dataset's declared action shape. Optional.",
    ),
    custom_embodiment_config: str = typer.Option(
        "",
        "--custom-embodiment-config",
        help="Path to a custom embodiment config JSON. Overrides --embodiment.",
    ),
    output_format: str = typer.Option(
        "human",
        "--format",
        help="Report format: 'human' (default, plain-text) or 'json' (machine-readable).",
    ),
    output: str = typer.Option(
        "",
        "--output",
        help="Write the report to this file instead of stdout. Useful in CI.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Treat WARN findings as BLOCKERs. Use in CI when any deviation should fail.",
    ),
):
    """Validate a LeRobot training dataset against the model + embodiment expectations.

    Pre-flight check before spending Modal credits on a training/distillation run that
    would crash mid-way due to action-dim mismatch, NaN actions, or schema drift. Pairs
    with `reflex doctor` (which validates model + runtime).

    Exit codes: 0 ok, 1 warnings, 2 blockers (or warnings under --strict).

    Examples:
      reflex validate-dataset ~/datasets/aloha_sim
      reflex validate-dataset ~/datasets/aloha_sim --embodiment franka
      reflex validate-dataset ~/datasets/aloha_sim --format json --output report.json
      reflex validate-dataset ~/datasets/aloha_sim --strict   # CI gating
    """
    from reflex.validation import (
        Decision,
        format_human,
        format_json,
        overall_decision,
        run_all_checks,
    )

    if output_format not in ("human", "json"):
        console.print(f"[red]--format must be 'human' or 'json', got {output_format!r}[/red]")
        raise typer.Exit(2)

    dataset_path = Path(path)
    if not dataset_path.exists():
        console.print(f"[red]Dataset path does not exist: {dataset_path}[/red]")
        raise typer.Exit(2)

    embodiment_cfg = None
    if custom_embodiment_config or embodiment:
        try:
            from reflex.embodiments import EmbodimentConfig
            if custom_embodiment_config:
                embodiment_cfg = EmbodimentConfig.load_custom(custom_embodiment_config)
            else:
                embodiment_cfg = EmbodimentConfig.load_preset(embodiment)
        except Exception as e:
            console.print(f"[yellow]Could not load embodiment config: {e}[/yellow]")

    results = run_all_checks(dataset_path, embodiment_config=embodiment_cfg, strict=strict)
    decision = overall_decision(results, strict=strict)

    if output_format == "json":
        report = format_json(results, dataset_root=str(dataset_path), decision=decision)
    else:
        report = format_human(results, dataset_root=str(dataset_path))

    if output:
        Path(output).write_text(report)
        console.print(f"Report written to {output} — decision: [bold]{decision.value.upper()}[/bold]")
    else:
        if output_format == "json":
            console.print(report)
        else:
            console.print(report)
            console.print(f"\nOverall decision: [bold]{decision.value.upper()}[/bold]")

    exit_code = {Decision.OK: 0, Decision.WARN: 1, Decision.BLOCKER: 2, Decision.SKIPPED: 0}[decision]
    if exit_code != 0:
        raise typer.Exit(exit_code)


# `reflex models {list, pull, info}` — curated VLA registry browser/downloader.
# Defined inline so the typer subgroup wiring stays visible at the CLI surface.
models_app = typer.Typer(help="Browse + download Reflex-compatible VLA models from HuggingFace.")


@models_app.command("list")
def models_list(
    family: str = typer.Option("", "--family", help="Filter by family: pi0/pi05/smolvla/openvla/groot"),
    device: str = typer.Option("", "--device",
                                help="Filter by supported device (orin_nano, agx_orin, thor, a10g, a100, h100, h200)"),
    embodiment: str = typer.Option("", "--embodiment", help="Filter by supported embodiment (franka, so100, ur5)"),
    output_format: str = typer.Option("human", "--format", help="'human' (table) or 'json'"),
):
    """List Reflex-compatible models from the curated registry.

    Examples:
      reflex models list
      reflex models list --family pi05
      reflex models list --device orin_nano
      reflex models list --device a10g --embodiment franka
    """
    from reflex.registry import REGISTRY, filter_models

    if output_format not in ("human", "json"):
        console.print(f"[red]--format must be 'human' or 'json', got {output_format!r}[/red]")
        raise typer.Exit(2)

    entries = filter_models(
        family=family or None, device=device or None, embodiment=embodiment or None,
    )

    if output_format == "json":
        import json
        rows = [
            {
                "model_id": e.model_id,
                "hf_repo": e.hf_repo,
                "family": e.family,
                "action_dim": e.action_dim,
                "size_mb": e.size_mb,
                "supported_embodiments": list(e.supported_embodiments),
                "supported_devices": list(e.supported_devices),
                "requires_export": e.requires_export,
                "license": e.license,
                "description": e.description,
            }
            for e in entries
        ]
        typer.echo(json.dumps({"n": len(rows), "models": rows}, indent=2))
        return

    if not entries:
        console.print(
            f"[yellow]No models match filters (family={family or 'any'}, "
            f"device={device or 'any'}, embodiment={embodiment or 'any'}).[/yellow]"
        )
        console.print(f"Registry has {len(REGISTRY)} entries total — drop filters to see all.")
        return

    table = Table(title=f"Reflex Model Registry ({len(entries)} of {len(REGISTRY)})")
    table.add_column("model_id", style="cyan", no_wrap=True)
    table.add_column("family", no_wrap=True)
    table.add_column("a_dim", justify="right")
    table.add_column("size", justify="right")
    table.add_column("embodiments")
    table.add_column("devices")
    table.add_column("description")
    for e in entries:
        size_str = f"{e.size_mb / 1000:.1f}GB" if e.size_mb >= 1000 else f"{e.size_mb}MB"
        table.add_row(
            e.model_id, e.family, str(e.action_dim), size_str,
            ", ".join(e.supported_embodiments), ", ".join(e.supported_devices),
            e.description[:60] + ("..." if len(e.description) > 60 else ""),
        )
    console.print(table)
    console.print(
        "\n[dim]reflex models pull <model_id>   # download to ~/.cache/reflex/models/<id>[/dim]"
        "\n[dim]reflex models info <model_id>   # see benchmarks + per-device support[/dim]"
    )


@models_app.command("pull")
def models_pull(
    model_id: str = typer.Argument(help="Registry id from `reflex models list`"),
    target_dir: str = typer.Option("", "--target-dir",
                                    help="Where to write weights. Default: ~/.cache/reflex/models/<model_id>/"),
    no_verify: bool = typer.Option(False, "--no-verify",
                                    help="Skip the post-download structure check"),
    revision: str = typer.Option("", "--revision",
                                  help="Override the registry's pinned hf_revision (advanced)"),
):
    """Download a model's weights from HuggingFace into the local cache.

    Example:
      reflex models pull pi05-libero
      reflex models pull smolvla-base --target-dir /data/models/smolvla
    """
    from reflex.registry import by_id, REGISTRY

    entry = by_id(model_id)
    if entry is None:
        # Accept the HF repo id too (`lerobot/smolvla_base` → `smolvla-base`).
        # Saves users from having to know the registry alias.
        for e in REGISTRY:
            if e.hf_repo == model_id:
                entry = e
                break
    if entry is None:
        available = sorted(e.model_id for e in REGISTRY)
        console.print(f"[red]Unknown model_id: {model_id!r}[/red]")
        console.print(f"Available registry ids: {', '.join(available)}")
        console.print("Tip: you can also pass the HuggingFace repo id (e.g. lerobot/smolvla_base).")
        raise typer.Exit(2)

    target = Path(target_dir) if target_dir else (Path.home() / ".cache" / "reflex" / "models" / entry.model_id)
    target.mkdir(parents=True, exist_ok=True)

    rev = revision or entry.hf_revision
    rev_str = rev if rev else "HEAD (unpinned — consider --revision for reproducibility)"

    console.print(f"Pulling [cyan]{entry.model_id}[/cyan]")
    console.print(f"  hf_repo:  {entry.hf_repo}")
    console.print(f"  revision: {rev_str}")
    console.print(f"  size:     ~{entry.size_mb}MB")
    console.print(f"  target:   {target}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[red]huggingface_hub not installed. pip install reflex-vla[/red]")
        raise typer.Exit(2)

    try:
        snapshot_download(
            repo_id=entry.hf_repo,
            revision=rev or None,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        console.print(f"[red]Download failed: {type(e).__name__}: {e}[/red]")
        raise typer.Exit(1)

    if not no_verify:
        contents = sorted(p.name for p in target.iterdir())
        console.print(f"[green]Pulled.[/green] {len(contents)} top-level entries: {contents[:10]}")
        if entry.requires_export:
            console.print(
                f"\n[yellow]Next: this model ships as raw weights. Run "
                f"[cyan]reflex export {target}[/cyan] to produce ONNX, then "
                f"[cyan]reflex serve <export-dir>[/cyan].[/yellow]"
            )
        else:
            console.print(
                f"\n[green]Ready to serve:[/green] [cyan]reflex serve {target}[/cyan]"
            )


@models_app.command("info")
def models_info(
    model_id: str = typer.Argument(help="Registry id from `reflex models list`"),
    output_format: str = typer.Option("human", "--format", help="'human' or 'json'"),
):
    """Show benchmarks + per-device support for a single model.

    Example:
      reflex models info pi05-libero
    """
    from reflex.registry import by_id

    entry = by_id(model_id)
    if entry is None:
        console.print(f"[red]Unknown model_id: {model_id!r}[/red]")
        raise typer.Exit(2)

    if output_format == "json":
        import json
        body = {
            "model_id": entry.model_id,
            "hf_repo": entry.hf_repo,
            "hf_revision": entry.hf_revision,
            "family": entry.family,
            "action_dim": entry.action_dim,
            "size_mb": entry.size_mb,
            "supported_embodiments": list(entry.supported_embodiments),
            "supported_devices": list(entry.supported_devices),
            "requires_export": entry.requires_export,
            "license": entry.license,
            "description": entry.description,
            "benchmarks": [
                {"device": b.device, "p50_ms": b.p50_ms, "p99_ms": b.p99_ms,
                 "vram_mb": b.vram_mb, "measured_at": b.measured_at}
                for b in entry.benchmarks
            ],
        }
        typer.echo(json.dumps(body, indent=2))
        return

    console.print(f"[bold cyan]{entry.model_id}[/bold cyan] ([dim]{entry.hf_repo}[/dim])")
    console.print(f"  family:        {entry.family}")
    console.print(f"  action_dim:    {entry.action_dim}")
    console.print(f"  size:          {entry.size_mb}MB")
    console.print(f"  license:       {entry.license}")
    console.print(f"  embodiments:   {', '.join(entry.supported_embodiments) or '(none)'}")
    console.print(f"  devices:       {', '.join(entry.supported_devices) or '(none)'}")
    console.print(f"  needs export:  {'YES — run reflex export after pull' if entry.requires_export else 'NO — Reflex-ready'}")
    console.print(f"\n{entry.description}")

    if entry.benchmarks:
        bt = Table(title="Benchmarks")
        bt.add_column("device")
        bt.add_column("p50 (ms)", justify="right")
        bt.add_column("p99 (ms)", justify="right")
        bt.add_column("VRAM (MB)", justify="right")
        bt.add_column("measured")
        for b in entry.benchmarks:
            bt.add_row(b.device, f"{b.p50_ms:.1f}", f"{b.p99_ms:.1f}",
                       str(b.vram_mb), b.measured_at)
        console.print()
        console.print(bt)
    else:
        console.print("\n[dim]No benchmarks yet. Run [cyan]reflex bench <export>[/cyan] after pull.[/dim]")


@app.command()
def go(
    model: str = typer.Option(
        "",
        "--model",
        help="Registry id (e.g. pi05-libero) OR family name (pi05/smolvla/pi0). "
             "Run `reflex models list` to browse.",
    ),
    embodiment: str = typer.Option(
        "",
        "--embodiment",
        help="Embodiment preset (franka/so100/ur5). Optional but recommended — "
             "cross-checks dataset/action shapes.",
    ),
    device_class: str = typer.Option(
        "",
        "--device-class",
        help="Override hardware probe (h200/h100/a100/a10g/thor/agx_orin/orin_nano/cpu). "
             "Use when probe misclassifies.",
    ),
    target_dir: str = typer.Option(
        "",
        "--target-dir",
        help="Where to cache weights. Default: ~/.cache/reflex/models/<id>/",
    ),
    port: int = typer.Option(8000, "--port", help="HTTP port for /act + /health"),
    host: str = typer.Option("0.0.0.0", "--host"),
    api_key: str = typer.Option("", "--api-key", help="If set, /act requires X-Reflex-Key header"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Probe + resolve + print plan; do not pull or serve.",
    ),
):
    """One-command deploy: probe hardware → pick model → pull → export → serve.

    Examples:
      reflex go --model pi05 --embodiment franka
      reflex go --model smolvla-base --device-class orin_nano --port 8001
      reflex go --model pi05-libero --dry-run

    For models that ship as raw PyTorch (requires_export=True in registry),
    this command pulls + exports inline (5-15 min) + serves. The export step
    requires the [monolithic] extra:
      pip install 'reflex-vla[monolithic]'
    Without it, `reflex go` errors with the install command.

    Exported artifacts cache at ~/.cache/reflex/exports/<model_id>/ — re-runs
    skip the export on cache hit.

    Plan ref: features/01_serve/subfeatures/_dx_gaps/one-command-deploy.md
    """
    from reflex.runtime.hardware_probe import (
        CANONICAL_DEVICE_CLASSES,
        probe_device_class,
    )
    from reflex.runtime.model_resolver import (
        ModelResolverError,
        resolve_model,
    )

    if not model:
        console.print("[red]--model is required (e.g. --model pi05-libero).[/red]")
        console.print("Run [cyan]reflex models list[/cyan] to browse.")
        raise typer.Exit(2)

    if device_class and device_class not in CANONICAL_DEVICE_CLASSES:
        console.print(
            f"[red]--device-class {device_class!r} not in {CANONICAL_DEVICE_CLASSES}[/red]"
        )
        raise typer.Exit(2)

    # Step 1: probe hardware
    probe = probe_device_class(override=device_class or None)
    console.print(f"[bold cyan]device:[/bold cyan]   {probe.device_class} "
                  f"(via {probe.detection_method}{f', GPU={probe.raw_gpu_name}' if probe.raw_gpu_name else ''})")
    for note in probe.notes:
        console.print(f"  [yellow]note:[/yellow] {note}")

    # Step 2: resolve model
    try:
        resolution = resolve_model(model=model, device_class=probe.device_class, embodiment=embodiment)
    except ModelResolverError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
    entry = resolution.entry
    console.print(f"[bold cyan]model:[/bold cyan]    {entry.model_id} "
                  f"({entry.hf_repo}, {entry.size_mb}MB, action_dim={entry.action_dim})")
    console.print(f"  strategy: {resolution.matched_strategy}")
    for note in resolution.notes:
        console.print(f"  [yellow]note:[/yellow] {note}")

    # Step 3: target dir
    target = Path(target_dir) if target_dir else (Path.home() / ".cache" / "reflex" / "models" / entry.model_id)

    if dry_run:
        console.print(f"[bold cyan]target:[/bold cyan]   {target}")
        next_step = "export inline then serve" if entry.requires_export else f"start serve on port {port}"
        console.print(f"\n[bold green]DRY RUN[/bold green] — would pull weights and {next_step}.")
        return

    # Step 4: pull (skip if already cached + non-empty)
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        console.print(f"[bold cyan]cache hit:[/bold cyan] {target} already populated; skipping pull.")
    else:
        console.print(f"[bold cyan]pulling:[/bold cyan]  {entry.hf_repo} → {target}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            console.print("[red]huggingface_hub not installed.[/red]")
            raise typer.Exit(2)
        try:
            snapshot_download(
                repo_id=entry.hf_repo,
                revision=entry.hf_revision or None,
                local_dir=str(target),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            console.print(f"[red]Download failed: {type(e).__name__}: {e}[/red]")
            raise typer.Exit(1)

    # Step 5: if model ships as raw weights, export inline before serving.
    # device_class (probe namespace) → target (HARDWARE_PROFILES namespace).
    _DEVICE_CLASS_TO_TARGET = {
        "orin_nano": "orin-nano",
        "agx_orin": "orin",
        "thor": "thor",
        "h200": "desktop", "h100": "desktop", "a100": "desktop",
        "a10g": "desktop", "cpu": "desktop",
    }
    if entry.requires_export:
        export_target = _DEVICE_CLASS_TO_TARGET.get(probe.device_class, "desktop")
        # Respect REFLEX_HOME for cache root so tests + custom installs can override.
        reflex_home = Path(os.environ.get("REFLEX_HOME", Path.home() / ".cache" / "reflex"))
        export_dir = reflex_home / "exports" / entry.model_id
        export_marker = export_dir / "VERIFICATION.md"
        meta_marker = export_dir / "_reflex_meta.json"

        # Validate cache: VERIFICATION.md presence + version-pinned _reflex_meta.json.
        # Stale caches (built by older reflex versions, mismatched export target, or
        # incomplete writes) are silently corrupting if reused — auto-invalidate
        # rather than letting the server crash mysteriously at startup.
        cache_valid = False
        if export_marker.exists():
            try:
                from reflex import __version__ as _current_reflex_version
            except Exception:  # noqa: BLE001
                _current_reflex_version = "unknown"
            if meta_marker.exists():
                try:
                    import json as _json_cache
                    meta = _json_cache.loads(meta_marker.read_text())
                    cached_version = meta.get("reflex_version", "?")
                    cached_target = meta.get("export_target", "?")
                    if cached_version != _current_reflex_version:
                        console.print(
                            f"[yellow]⚠ Cache stale[/yellow]: built by reflex {cached_version}, "
                            f"you're on {_current_reflex_version}. Rebuilding."
                        )
                    elif cached_target != export_target:
                        console.print(
                            f"[yellow]⚠ Cache target mismatch[/yellow]: built for "
                            f"{cached_target}, you need {export_target}. Rebuilding."
                        )
                    else:
                        cache_valid = True
                except Exception as _exc:  # noqa: BLE001
                    console.print(
                        f"[yellow]⚠ Cache metadata unreadable ({_exc.__class__.__name__})[/yellow]: "
                        f"rebuilding to be safe."
                    )
            else:
                console.print(
                    f"[yellow]⚠ Legacy cache detected[/yellow] (no version metadata, "
                    f"likely built by reflex ≤0.5.3). Rebuilding."
                )
            if not cache_valid:
                import shutil as _shutil_cache
                _shutil_cache.rmtree(export_dir, ignore_errors=True)

        if cache_valid:
            console.print(f"[bold cyan]export hit:[/bold cyan] {export_dir} already populated; skipping export.")
        else:
            console.print(
                f"[bold cyan]exporting:[/bold cyan] {target} → {export_dir} "
                f"(target={export_target}, monolithic, 5-15 min depending on hardware)"
            )
            from reflex.exporters.monolithic import export_monolithic  # module always importable
            export_dir.mkdir(parents=True, exist_ok=True)
            import time as _time
            _t0 = _time.perf_counter()
            try:
                result = export_monolithic(
                    str(target), str(export_dir),
                    num_steps=10, target=export_target,
                )
            except ImportError as exc:
                # export_monolithic does its own runtime dep check (lerobot, onnx-diagnostic, scipy);
                # these aren't in the base install. Surface the same hint reflex export uses.
                console.print(f"{exc}", style="red", markup=False)
                console.print(
                    "\n`reflex go` needs the monolithic export extras to deploy this model.\n"
                    "Fix: pip install 'reflex-vla[monolithic]'\n"
                    "(pins transformers==5.3.0; use a clean venv to avoid the "
                    "base transformers<5.0 conflict)",
                    style="cyan", markup=False,
                )
                raise typer.Exit(2)
            except Exception as exc:  # noqa: BLE001
                console.print(f"Export failed: {type(exc).__name__}: {exc}", style="red", markup=False)
                raise typer.Exit(1)
            elapsed = _time.perf_counter() - _t0
            console.print(f"[bold green]export complete in {elapsed:.1f}s[/bold green]  ONNX={result.get('onnx_path','?')} ({result.get('size_mb',0):.0f} MB)")

            try:
                from reflex.verification_report import write_verification_report
                write_verification_report(str(export_dir), parity=None)
            except Exception:  # noqa: BLE001
                pass  # Verification manifest is informational, not load-bearing

            # Write the version-pinned cache marker so future runs can detect
            # stale caches and rebuild instead of silently using mismatched bytes.
            try:
                import json as _json_meta
                from reflex import __version__ as _current_reflex_version
                from datetime import datetime as _dt
                meta = {
                    "reflex_version": _current_reflex_version,
                    "model_id": entry.model_id,
                    "export_target": export_target,
                    "export_mode": "monolithic",
                    "completed_at": _dt.utcnow().isoformat() + "Z",
                }
                (export_dir / "_reflex_meta.json").write_text(
                    _json_meta.dumps(meta, indent=2)
                )
            except Exception:  # noqa: BLE001
                pass  # Cache marker is best-effort; legacy fallback handles missing meta

        # Hand off serve to the exported dir, not the raw weights dir.
        target = export_dir

    # requires_export=False (or just-exported) → start serve directly
    console.print(f"\n[bold green]Starting serve on http://{host}:{port}[/bold green]")
    from reflex.runtime.server import create_app

    embodiment_cfg = None
    if embodiment:
        try:
            from reflex.embodiments import EmbodimentConfig
            embodiment_cfg = EmbodimentConfig.load_preset(embodiment)
        except Exception as e:
            console.print(f"[yellow]Could not load embodiment config: {e}[/yellow]")

    app_instance = create_app(
        export_dir=str(target),
        device="cuda" if probe.device_class != "cpu" else "cpu",
        embodiment_config=embodiment_cfg,
        api_key=api_key or None,
    )
    import uvicorn
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Verb-noun subgroups (2026-04-24 refactor — see ADR
# 01_decisions/2026-04-24-cli-verb-noun-now-config-later-dashboard-eventually.md).
#
# Visible top-level: serve, doctor, models, train, validate, inspect, go (= 7).
# Old top-level commands stay registered under hidden=True so existing scripts
# don't break; they will be removed in v0.2.
# ---------------------------------------------------------------------------

train_app = typer.Typer(
    help="Train models — finetune existing checkpoints, distill teachers into 1-NFE students."
)
validate_app = typer.Typer(
    help="Pre-flight validation — datasets before training, exports before serving."
)
inspect_app = typer.Typer(
    help="Diagnostic + forensic tools — bench, replay traces, hardware targets, guard state."
)

# Cross-register existing functions under the new verb-noun paths.
# Same callable, two surface names: old hidden, new visible.
#
# v0.9.5 (2026-05-07) CLI cut pass: hidden=True on cluttered/redundant
# inspect commands per the surface-audit. Each stays callable directly
# (`reflex inspect bench`, etc. still works); just removed from --help.
# Reduces customer cognitive load on `reflex inspect --help` from 5 → 2.
models_app.command("export")(export)
validate_app.command("dataset")(validate_dataset)
validate_app.command("export")(validate)
# inspect bench: internal-only latency microbench (`customer_signal: internal`
# per spec). Customers don't run benches. Hidden 2026-05-07.
inspect_app.command("bench", hidden=True)(benchmark_cmd)
inspect_app.command("replay")(replay)  # legitimate trace replay tool
# inspect targets: lists hardware profiles. Used once during install,
# never after. Hidden 2026-05-07.
inspect_app.command("targets", hidden=True)(targets)
# inspect guard: dumps shipped safety config. Niche diagnostic;
# fired ~twice in entire experiment history. Hidden 2026-05-07.
inspect_app.command("guard", hidden=True)(guard)
# inspect doctor: pure duplicate of top-level `reflex doctor`.
# Cross-registration was for "completeness" but adds a redundant entry
# to --help. Hidden 2026-05-07; top-level `doctor` is the canonical path.
inspect_app.command("doctor", hidden=True)(doctor)


@inspect_app.command("traces")
def inspect_traces(
    dir: Optional[str] = typer.Option(
        None, "--dir",
        help="Trace directory to scan. Defaults to ~/.cache/reflex/traces and /tmp/traces.",
    ),
    since: Optional[str] = typer.Option(
        None, "--since",
        help="Only show traces newer than this window: e.g. '7d', '24h', '1h'.",
    ),
    task: Optional[str] = typer.Option(
        None, "--task",
        help="Filter by task name substring (matched against first record's instruction).",
    ),
    status: Optional[str] = typer.Option(
        None, "--status",
        help="Filter by trace status. Currently 'any' (no filter); episode-success not yet recorded.",
    ),
    limit: int = typer.Option(50, "--limit", help="Max rows to show."),
) -> None:
    """List recorded /act traces (JSONL files written by `reflex serve --record <dir>`)."""
    import gzip
    import json
    import time
    from pathlib import Path as _Path

    candidates: list[_Path] = []
    if dir:
        candidates.append(_Path(dir))
    else:
        candidates.append(_Path.home() / ".cache" / "reflex" / "traces")
        candidates.append(_Path("/tmp/traces"))

    files: list[_Path] = []
    for d in candidates:
        if d.exists() and d.is_dir():
            files.extend(sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True))
            files.extend(sorted(d.glob("*.jsonl.gz"), key=lambda p: p.stat().st_mtime, reverse=True))

    if not files:
        console.print(
            "No trace files found. Enable recording with: reflex serve <export> --record /tmp/traces",
            markup=False,
        )
        return

    cutoff_ts: float = 0.0
    if since:
        unit = since[-1]
        try:
            n = int(since[:-1])
            mult = {"h": 3600, "d": 86400, "m": 60}.get(unit, 0)
            if mult:
                cutoff_ts = time.time() - n * mult
        except ValueError:
            pass

    rows: list[tuple[str, str, str, str, str]] = []
    for f in files:
        if cutoff_ts and f.stat().st_mtime < cutoff_ts:
            continue
        first_record_task = "?"
        n_records = 0
        try:
            opener = gzip.open if f.suffix == ".gz" else open
            with opener(f, "rt") as fh:  # type: ignore[arg-type]
                for i, line in enumerate(fh):
                    n_records += 1
                    if i == 0 or first_record_task == "?":
                        try:
                            rec = json.loads(line)
                            instr = rec.get("instruction") or rec.get("request", {}).get("instruction") or ""
                            if instr and instr != "?":
                                first_record_task = instr[:40]
                        except json.JSONDecodeError:
                            pass
                    if i >= 200:
                        break
        except Exception:
            continue
        if task and task.lower() not in first_record_task.lower():
            continue
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
        size_kb = f.stat().st_size // 1024
        rows.append((mtime, f.name[:40], first_record_task, str(n_records), f"{size_kb} KB"))
        if len(rows) >= limit:
            break

    if not rows:
        console.print("No traces match the filter.", markup=False)
        return

    table = Table(title=f"Recorded traces ({len(rows)} of {len(files)} total)")
    table.add_column("Modified")
    table.add_column("File")
    table.add_column("Task")
    table.add_column("Records")
    table.add_column("Size")
    for r in rows:
        table.add_row(*r)
    console.print(table)

app.add_typer(models_app, name="models")
app.add_typer(train_app, name="train")
app.add_typer(validate_app, name="validate")
app.add_typer(inspect_app, name="inspect")


# ─── reflex traces {query, summary} ─────────────────────────────────────────
# Customer trace archive (Phase 1.5 v1) per spec
# features/01_serve/subfeatures/_ecosystem/customer-trace-archive/.
# Operates on JSONL traces written by `reflex serve --record <dir>`.
# Phase 1.5 ships: query (filter + export) + summary (aggregations).
# Phase 2 deferred: parquet+DuckDB index for fast filter on million-record
# archives + SQL surface for power users.
traces_app = typer.Typer(
    help="Searchable + summarizable view over recorded /act traces.",
    no_args_is_help=True,
)
app.add_typer(traces_app, name="traces")


def _default_trace_dirs() -> list[Path]:
    """Same default order as `reflex inspect traces`."""
    return [
        Path.home() / ".cache" / "reflex" / "traces",
        Path("/tmp/traces"),
    ]


def _resolve_output_format(output: Optional[str], explicit: Optional[str]) -> str:
    """Pick output format: explicit --format > suffix on --output > 'table'."""
    if explicit:
        if explicit not in ("table", "json", "csv"):
            raise typer.BadParameter(
                f"--format must be 'table' / 'json' / 'csv', got {explicit!r}"
            )
        return explicit
    if output:
        suffix = Path(output).suffix.lower().lstrip(".")
        if suffix in ("json", "csv"):
            return suffix
    return "table"


@traces_app.command("query")
def traces_query(
    dir: Optional[str] = typer.Option(
        None, "--dir",
        help="Trace directory to scan. Defaults to ~/.cache/reflex/traces "
             "and /tmp/traces.",
    ),
    since: Optional[str] = typer.Option(
        None, "--since",
        help="Only include records newer than this window: '7d', '24h', "
             "'30m'. File mtime is used as the cheap pre-filter.",
    ),
    task: Optional[str] = typer.Option(
        None, "--task",
        help="Case-insensitive substring match against request.instruction.",
    ),
    status: str = typer.Option(
        "any", "--status",
        help="Filter by request status: 'any' (default) / 'success' / 'failed'. "
             "'failed' = response had an `error` field (server-side error). "
             "Episode-level task-success is not yet recorded; that lands in v2.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Substring match against header.model_hash. Files with no match "
             "are skipped entirely (cheap, runs once per file).",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit",
        help="Max records to emit. Default unbounded.",
    ),
    output: Optional[str] = typer.Option(
        None, "--output",
        help="Output file. Format auto-detected from suffix (.json / .csv) "
             "unless --format is set. Default writes a Rich table to stdout.",
    ),
    format: Optional[str] = typer.Option(
        None, "--format",
        help="Output format override: 'table' / 'json' / 'csv'. Default "
             "auto-detect from --output suffix.",
    ),
) -> None:
    """Filter + export recorded /act traces. Composes with Pro-tier
    record-replay; runs locally so customer data stays on-prem."""
    from reflex.traces.archive import (
        TraceFilter,
        query_traces,
    )
    from typing import cast

    if status not in ("any", "success", "failed"):
        raise typer.BadParameter(
            f"--status must be 'any' / 'success' / 'failed', got {status!r}"
        )

    dirs = [Path(dir)] if dir else _default_trace_dirs()
    flt = TraceFilter(
        since=since,
        task=task,
        status=cast("Any", status),  # narrow after validation
        model=model,
        limit=limit,
    )
    try:
        records = query_traces(dirs, filter_=flt)
    except ValueError as exc:
        console.print(f"[red]Invalid filter: {exc}[/red]")
        raise typer.Exit(1)

    if not records:
        console.print("[dim]No traces match the filter.[/dim]")
        return

    fmt = _resolve_output_format(output, format)

    if fmt == "json":
        import json as _json
        rows = [
            {
                "file": str(r.file),
                "seq": r.seq,
                "timestamp": r.timestamp,
                "instruction": r.instruction,
                "latency_ms": r.latency_ms,
                "status": "failed" if r.is_failed else "success",
                "error": r.error,
            }
            for r in records
        ]
        text = _json.dumps(rows, indent=2)
        if output:
            Path(output).write_text(text + "\n", encoding="utf-8")
            console.print(f"Wrote {len(rows)} record(s) to {output}")
        else:
            print(text)
        return

    if fmt == "csv":
        import csv as _csv
        import io as _io
        buf = _io.StringIO()
        w = _csv.writer(buf)
        w.writerow([
            "file", "seq", "timestamp", "instruction",
            "latency_ms", "status", "error_reason",
        ])
        for r in records:
            err_reason = (
                (r.error or {}).get("reason", "") if r.error else ""
            )
            w.writerow([
                str(r.file), r.seq, r.timestamp, r.instruction,
                r.latency_ms,
                "failed" if r.is_failed else "success",
                err_reason,
            ])
        text = buf.getvalue()
        if output:
            Path(output).write_text(text, encoding="utf-8")
            console.print(f"Wrote {len(records)} record(s) to {output}")
        else:
            print(text, end="")
        return

    # Default: table to stdout
    table = Table(title=f"Trace records ({len(records)} match)")
    table.add_column("Timestamp")
    table.add_column("Status")
    table.add_column("Latency (ms)")
    table.add_column("Task")
    table.add_column("File")
    for r in records:
        table.add_row(
            r.timestamp,
            "[red]failed[/red]" if r.is_failed else "[green]success[/green]",
            f"{r.latency_ms:.1f}",
            (r.instruction[:50] + "...") if len(r.instruction) > 50 else r.instruction,
            r.file.name,
        )
    console.print(table)


@traces_app.command("summary")
def traces_summary(
    dir: Optional[str] = typer.Option(
        None, "--dir",
        help="Trace directory to scan. Defaults to ~/.cache/reflex/traces "
             "and /tmp/traces.",
    ),
    since: Optional[str] = typer.Option(
        None, "--since",
        help="Only include records newer than this window: '7d', '24h', '30m'.",
    ),
    task: Optional[str] = typer.Option(
        None, "--task",
        help="Pre-filter by task substring before grouping.",
    ),
    status: str = typer.Option(
        "any", "--status",
        help="Pre-filter by status before grouping.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Pre-filter by model_hash substring before grouping.",
    ),
    by: str = typer.Option(
        "task", "--by",
        help="Group records by this dimension: 'task' (default) / 'model' / "
             "'day'. 'model' uses model_hash from the trace filename.",
    ),
    output: Optional[str] = typer.Option(
        None, "--output",
        help="Output file. Format auto-detected from suffix (.json / .csv) "
             "unless --format is set. Default writes a Rich table.",
    ),
    format: Optional[str] = typer.Option(
        None, "--format",
        help="Output format override: 'table' / 'json' / 'csv'.",
    ),
) -> None:
    """Aggregate trace records by task, model, or day. Each bucket gets
    count, success_rate, latency p50/p95/p99/max."""
    from reflex.traces.archive import (
        TraceFilter,
        summarize_traces,
    )
    from typing import cast

    if status not in ("any", "success", "failed"):
        raise typer.BadParameter(
            f"--status must be 'any' / 'success' / 'failed', got {status!r}"
        )
    if by not in ("task", "model", "day"):
        raise typer.BadParameter(
            f"--by must be 'task' / 'model' / 'day', got {by!r}"
        )

    dirs = [Path(dir)] if dir else _default_trace_dirs()
    flt = TraceFilter(since=since, task=task, status=cast("Any", status), model=model)
    try:
        summaries = summarize_traces(
            dirs, filter_=flt, by=cast("Any", by),
        )
    except ValueError as exc:
        console.print(f"[red]Invalid filter: {exc}[/red]")
        raise typer.Exit(1)

    if not summaries:
        console.print("[dim]No traces match the filter.[/dim]")
        return

    fmt = _resolve_output_format(output, format)

    if fmt == "json":
        import json as _json
        rows = [s.as_dict() for s in summaries]
        text = _json.dumps(rows, indent=2)
        if output:
            Path(output).write_text(text + "\n", encoding="utf-8")
            console.print(f"Wrote {len(rows)} summary buckets to {output}")
        else:
            print(text)
        return

    if fmt == "csv":
        import csv as _csv
        import io as _io
        buf = _io.StringIO()
        w = _csv.DictWriter(buf, fieldnames=list(summaries[0].as_dict().keys()))
        w.writeheader()
        for s in summaries:
            w.writerow(s.as_dict())
        text = buf.getvalue()
        if output:
            Path(output).write_text(text, encoding="utf-8")
            console.print(f"Wrote {len(summaries)} summary buckets to {output}")
        else:
            print(text, end="")
        return

    # Default: table
    table = Table(title=f"Trace summary by {by} ({len(summaries)} bucket(s))")
    table.add_column(by.capitalize(), max_width=60)
    table.add_column("Count")
    table.add_column("Success rate")
    table.add_column("p50 ms")
    table.add_column("p95 ms")
    table.add_column("p99 ms")
    table.add_column("max ms")
    for s in summaries:
        table.add_row(
            s.bucket,
            str(s.count),
            f"{s.success_rate:.1%}",
            f"{s.latency_p50_ms:.1f}",
            f"{s.latency_p95_ms:.1f}",
            f"{s.latency_p99_ms:.1f}",
            f"{s.latency_max_ms:.1f}",
        )
    console.print(table)


# ─── reflex pro {activate, status, deactivate} ──────────────────────────────
# Customer-facing Pro tier commands. Activation flow lives in
# src/reflex/pro/activate.py; status reads ~/.reflex/pro.license; deactivate
# clears the local file (license remains valid server-side until revoked).
pro_app = typer.Typer(
    help="Reflex Pro — activate, check, or deactivate your Pro license.",
    no_args_is_help=True,
)
app.add_typer(pro_app, name="pro")


@pro_app.command("activate")
def pro_activate(
    code: str = typer.Argument(..., help="Activation code (REFLEX-XXXX-XXXX-XXXX) sent by the operator."),
    endpoint: Optional[str] = typer.Option(
        None, "--endpoint",
        help="Override license worker URL. Defaults to REFLEX_LICENSE_ENDPOINT env or production.",
    ),
) -> None:
    """Redeem an activation code: fetch + verify + write the Pro license."""
    from reflex.pro.activate import (
        ActivationCodeError,
        ActivationError,
        ActivationNetworkError,
        activate_license,
    )
    from reflex.pro.signature import LicenseSignatureError

    try:
        license = activate_license(code, endpoint=endpoint)
    except ActivationCodeError as exc:
        console.print(f"[red]Activation failed:[/red] {exc}")
        raise typer.Exit(1)
    except ActivationNetworkError as exc:
        console.print(f"[red]Network error:[/red] {exc}")
        raise typer.Exit(2)
    except LicenseSignatureError as exc:
        console.print(f"[red]Signature verification failed:[/red] {exc}")
        console.print("[red]Refusing to write a license that didn't verify.[/red]")
        raise typer.Exit(3)
    except ActivationError as exc:
        console.print(f"[red]Activation error:[/red] {exc}")
        raise typer.Exit(4)

    console.print("[green]✓[/green] License fetched, signature verified, written to ~/.reflex/pro.license")
    console.print(f"[green]✓[/green] Hardware bound: gpu_uuid={license.get('hardware_binding', {}).get('gpu_uuid', 'unknown')}")
    console.print(f"[green]✓[/green] Telemetry: ON by default — disable with [cyan]REFLEX_NO_TELEMETRY=1[/cyan]")
    console.print()
    console.print(f"Welcome to Reflex Pro. License [cyan]{license.get('license_id')}[/cyan] expires [cyan]{license.get('expires_at')}[/cyan].")
    console.print("Run [cyan]reflex pro status[/cyan] anytime to check.")


@pro_app.command("status")
def pro_status(
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON for scripts."),
) -> None:
    """Show the current Pro license status (expiry, days remaining, last heartbeat)."""
    import json as _json
    from datetime import datetime, timezone
    from pathlib import Path

    from reflex.pro.activate import DEFAULT_LICENSE_PATH

    path = Path(DEFAULT_LICENSE_PATH).expanduser()
    if not path.exists():
        if json_output:
            print(_json.dumps({"present": False, "path": str(path)}))
            raise typer.Exit(1)
        console.print(f"[yellow]No Pro license found at {path}.[/yellow]")
        console.print("Activate with: [cyan]reflex pro activate <activation_code>[/cyan]")
        raise typer.Exit(1)

    try:
        data = _json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]License file is corrupt:[/red] {exc}")
        raise typer.Exit(2)

    expires_at = data.get("expires_at", "")
    last_hb = data.get("last_heartbeat_at", "")
    days_remaining = "?"
    if expires_at:
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            delta = exp - datetime.now(timezone.utc)
            days_remaining = max(0, int(delta.total_seconds() // 86400))
        except Exception:  # noqa: BLE001
            days_remaining = "?"

    summary = {
        "present": True,
        "path": str(path),
        "license_id": data.get("license_id"),
        "customer_id": data.get("customer_id"),
        "tier": data.get("tier"),
        "license_version": data.get("license_version"),
        "expires_at": expires_at,
        "days_remaining": days_remaining,
        "max_seats": data.get("max_seats"),
        "last_heartbeat_at": last_hb,
        "hardware_bound": data.get("hardware_binding") is not None,
        "signed": bool(data.get("signature")),
    }

    if json_output:
        print(_json.dumps(summary, indent=2))
        return

    console.print(f"[bold]Reflex Pro license:[/bold] {path}")
    console.print(f"  license_id:    [cyan]{summary['license_id']}[/cyan]")
    console.print(f"  customer:      {summary['customer_id']}")
    console.print(f"  tier:          {summary['tier']}")
    console.print(f"  version:       v{summary['license_version']} ({'signed' if summary['signed'] else 'UNSIGNED — legacy'})")
    console.print(f"  expires:       {expires_at} ([cyan]{days_remaining} days remaining[/cyan])")
    console.print(f"  max_seats:     {summary['max_seats']}")
    console.print(f"  last heartbeat: {last_hb or '[yellow](never)[/yellow]'}")
    console.print(f"  hardware:      {'bound ✓' if summary['hardware_bound'] else '[yellow]unbound[/yellow]'}")


@pro_app.command("deactivate")
def pro_deactivate(
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Remove the local Pro license file. Server-side license remains until revoked."""
    from pathlib import Path

    from reflex.pro.activate import DEFAULT_LICENSE_PATH

    path = Path(DEFAULT_LICENSE_PATH).expanduser()
    if not path.exists():
        console.print("[yellow]No Pro license file to remove.[/yellow]")
        return

    if not yes:
        confirm = typer.confirm(f"Remove {path}? (server-side license stays valid until you ask the operator to revoke it)")
        if not confirm:
            console.print("Aborted.")
            raise typer.Exit(1)

    path.unlink()
    console.print(f"[green]✓[/green] Removed {path}")
    console.print("Server-side license remains valid. Ask the operator to revoke if needed:")
    console.print("  python -m reflex.admin.revoke_license --license-id <yours>")


# Register `reflex {finetune,distill}` (legacy hidden) AND `reflex train
# {finetune,distill}` (new). Same callable; old scripts still work.
# Lazy-import protects users who don't have training deps installed — they
# only break if they run the commands themselves.
try:
    from reflex.finetune.cli import finetune_command
    app.command(name="finetune", hidden=True)(finetune_command)
    train_app.command("finetune")(finetune_command)
except Exception as _finetune_import_exc:  # pragma: no cover - defensive
    pass

try:
    from reflex.finetune.cli_distill import distill_command
    app.command(name="distill", hidden=True)(distill_command)
    train_app.command("distill")(distill_command)
except Exception as _distill_import_exc:  # pragma: no cover - defensive
    pass


@app.command()
def chat(
    proxy_url: Optional[str] = typer.Option(
        None, "--proxy-url",
        help="FastCrest proxy URL. Defaults to https://chat.fastcrest.com or $FASTCREST_PROXY_URL.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Don't execute tool calls; just print the commands the agent would run.",
    ),
    no_stream: bool = typer.Option(
        False, "--no-stream",
        help="Disable token streaming (default streams live). Use for scripts that pipe output.",
    ),
    tui: bool = typer.Option(
        False, "--tui",
        help="Use the Textual full-screen TUI (multi-panel layout, scroll-back, mouse). "
             "Requires `pip install 'reflex-vla[tui]'`. Falls back to the Rich REPL if "
             "textual isn't installed.",
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Resume the most-recent saved chat session from ~/.cache/reflex/chat_history/.",
    ),
) -> None:
    """Natural-language chat that can run reflex commands for you."""
    if tui:
        from reflex.chat.tui import run_tui
        run_tui(proxy_url=proxy_url, dry_run=dry_run)
    else:
        from reflex.chat.console import run_repl
        run_repl(proxy_url=proxy_url, dry_run=dry_run, no_stream=no_stream, resume=resume)


config_app = typer.Typer(name="config", help="Show + manage reflex configuration.", no_args_is_help=True)
# Hidden 2026-05-07: config schema is currently a stub (no real
# config knobs surfaced through this CLI yet — the verb-noun ADR
# scopes config-driven workflows for Phase 2). Premature top-level
# verb. Keep callable for any power-user scripts.
app.add_typer(config_app, name="config", hidden=True)


# ─── reflex contribute {opt-in,opt-out,revoke,status,info} ──────────────────
# Curate wedge data-contribution program. Implementation lives in
# src/reflex/curate/opt_in_cli.py; this is the wiring.
from reflex.curate.opt_in_cli import contribute_app  # noqa: E402
app.add_typer(contribute_app, name="contribute")


# ─── reflex calibrate <embodiment> ───────────────────────────────────────────
# Physical-arm calibration. SO-ARM 100 substrate vendored from auto_soarm
# (MIT) per ADR 2026-05-06-vendor-auto-soarm.md. Other embodiments add their
# own subcommand groups under this.
calibrate_app = typer.Typer(
    name="calibrate",
    help="Calibrate a physical robot arm (joint zeroing + workspace bounds).",
    no_args_is_help=True,
)
# Hidden 2026-05-07: SO-ARM 100 hardware-specific (corners/surface/tap).
# Customers without SO-100 hardware (~95% of the user base) don't need
# to see this. Stays callable directly for SO-100 customers.
app.add_typer(calibrate_app, name="calibrate", hidden=True)

so100_calibrate_app = typer.Typer(
    name="so100",
    help="SO-ARM 100 calibration (corners + surface + tap model).",
    no_args_is_help=True,
)
calibrate_app.add_typer(so100_calibrate_app, name="so100")


@so100_calibrate_app.command("corners")
def calibrate_so100_corners() -> None:
    """Hand-guide the arm to the 4 numbered tablet corners; record joint poses."""
    import subprocess as _sp
    from reflex.embodiments.so100.calibration import calibrate_corners as _mod
    # The upstream module is invoked as a script. Re-exec via -m for now;
    # Phase 1.5 can swap to in-process invocation when we audit the script's
    # __main__ block for safe-as-library use.
    _sp.run([sys.executable, "-m", "reflex.embodiments.so100.calibration.calibrate_corners"], check=False)


@so100_calibrate_app.command("surface")
def calibrate_so100_surface() -> None:
    """Probe the tablet surface for tap depth; fit the tap model."""
    import subprocess as _sp
    _sp.run([sys.executable, "-m", "reflex.embodiments.so100.calibration.calibrate_surface"], check=False)


@so100_calibrate_app.command("all")
def calibrate_so100_all() -> None:
    """Run the full SO-ARM 100 calibration sequence: corners → surface."""
    import subprocess as _sp
    _sp.run([sys.executable, "-m", "reflex.embodiments.so100.calibration.calibrate_all"], check=False)


@so100_calibrate_app.command("preflight")
def calibrate_so100_preflight(
    skip_camera: bool = typer.Option(
        False, "--skip-camera", help="Skip the camera-availability check.",
    ),
) -> None:
    """Non-invasive checks: arm serial / motor IDs / camera / ADB tablet."""
    import subprocess as _sp
    cmd = [sys.executable, "-m", "reflex.embodiments.so100.calibration.preflight"]
    if skip_camera:
        cmd.append("--skip-camera")
    _sp.run(cmd, check=False)


# ─── reflex bench-game <game> {collect, eval} ───────────────────────────────
# Real-arm bench games. Vendored from auto_soarm (MIT) per ADR 2026-05-06.
# Separate typer group from the existing `reflex bench` (hardware-level) so
# the existing surface stays untouched.
bench_game_app = typer.Typer(
    name="bench-game",
    help="Reflex bench games — real-arm regression-test rigs (tablet tap, etc.).",
    no_args_is_help=True,
)
# Hidden 2026-05-07: real-arm bench games vendored from auto_soarm.
# SO-100 + tablet-specific rig; ~3 customers globally. Hidden from
# default --help to reduce surface clutter; SO-100 customers can still
# invoke directly.
app.add_typer(bench_game_app, name="bench-game", hidden=True)

circle_lr_app = typer.Typer(
    name="circle_lr",
    help="Canonical SO-ARM 100 + tablet circle-tap benchmark.",
    no_args_is_help=True,
)
bench_game_app.add_typer(circle_lr_app, name="circle_lr")


@circle_lr_app.command("collect")
def bench_game_circle_lr_collect(
    episodes: int = typer.Option(20, "--episodes", help="Number of episodes to collect."),
) -> None:
    """Collect a circle-tap dataset on real SO-ARM 100 + Android tablet."""
    import subprocess as _sp
    cmd = [
        sys.executable, "-m", "reflex.bench.games.circle_lr.circle_collect",
        "--episodes", str(episodes),
    ]
    _sp.run(cmd, check=False)


@circle_lr_app.command("eval")
def bench_game_circle_lr_eval(
    ckpt: str = typer.Option(..., "--ckpt", help="Path to a trained ACT checkpoint."),
    episodes: int = typer.Option(8, "--episodes", help="Number of eval episodes."),
    remote_host: Optional[str] = typer.Option(
        None, "--remote-host", help="Remote inference host (use with --remote-port).",
    ),
    remote_port: Optional[int] = typer.Option(
        None, "--remote-port", help="Remote inference port.",
    ),
) -> None:
    """Run the trained policy against the live tablet + score hits."""
    import subprocess as _sp
    cmd = [
        sys.executable, "-m", "reflex.bench.games.circle_lr.circle_eval",
        "--ckpt", ckpt, "--episodes", str(episodes),
    ]
    if remote_host:
        cmd += ["--remote-host", remote_host]
    if remote_port:
        cmd += ["--remote-port", str(remote_port)]
    _sp.run(cmd, check=False)


# ─── reflex curate {convert} ─────────────────────────────────────────────────
# Curate dataset-conversion subcommand. `reflex curate convert <input> --format X`
curate_app = typer.Typer(
    name="curate",
    help="Reflex Curate — convert recorded traces to published dataset formats.",
    no_args_is_help=True,
)
app.add_typer(curate_app, name="curate")


@curate_app.command("convert")
def curate_convert(
    input_jsonl: str = typer.Argument(
        ..., help="Path to a JSONL trace file or directory of JSONL files.",
    ),
    format: str = typer.Option(
        ..., "--format", "-f",
        help="Target format: lerobot-v3 | hdf5 | rlds | openx-embodiment",
    ),
    output: str = typer.Option(
        ..., "--output", "-o",
        help="Output directory for the converted dataset.",
    ),
    min_quality: Optional[float] = typer.Option(
        None, "--min-quality",
        help="Drop episodes with quality_score below this threshold.",
    ),
    canonical_only: bool = typer.Option(
        False, "--canonical-only",
        help="Drop non-canonical episodes (dedup cluster non-canonicals filtered out).",
    ),
    robot_type: str = typer.Option(
        "unknown", "--robot-type",
        help="Embodiment slug (franka / so100 / ur5 / aloha / ...). Used by lerobot-v3 + openx-embodiment.",
    ),
    fps: int = typer.Option(
        30, "--fps",
        help="Target frame rate for the dataset (used by lerobot-v3).",
    ),
) -> None:
    """Convert Reflex JSONL traces to a published dataset format."""
    import json as _json
    from reflex.curate.format_converters import (
        CONVERTER_REGISTRY,
        HDF5Converter,
        LeRobotV3Converter,
        OpenXEmbodimentConverter,
        RLDSConverter,
    )

    if format not in CONVERTER_REGISTRY:
        console.print(
            f"[red]Unknown format[/red] [cyan]{format}[/cyan]; available: "
            f"[cyan]{', '.join(sorted(CONVERTER_REGISTRY.keys()))}[/cyan]"
        )
        raise typer.Exit(2)

    try:
        if format == "lerobot-v3":
            converter = LeRobotV3Converter(robot_type=robot_type, fps=fps)
        elif format == "hdf5":
            converter = HDF5Converter()
        elif format == "openx-embodiment":
            converter = OpenXEmbodimentConverter(embodiment=robot_type)
        else:
            converter = CONVERTER_REGISTRY[format]()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to construct converter:[/red] {exc}")
        raise typer.Exit(2)

    try:
        result = converter.convert(
            input_jsonl=input_jsonl,
            output_dir=output,
            min_quality=min_quality,
            canonical_only=canonical_only,
        )
    except ImportError as exc:
        console.print(f"[red]Missing dependency:[/red] {exc}")
        raise typer.Exit(3)
    except NotImplementedError as exc:
        console.print(f"[yellow]Not yet implemented:[/yellow] {exc}")
        raise typer.Exit(4)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Conversion failed:[/red] {exc}")
        raise typer.Exit(5)

    # Render summary table
    table = Table(title=f"Conversion: {format} → {output}")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Format", format)
    table.add_row("Episodes", str(result.episode_count))
    table.add_row("Steps", str(result.step_count))
    table.add_row("Bytes written", f"{result.bytes_written:,}")
    table.add_row("Skipped", str(result.skipped_episodes))
    if result.skipped_reasons:
        table.add_row(
            "Skip reasons",
            ", ".join(f"{k}={v}" for k, v in result.skipped_reasons.items()),
        )
    console.print(table)

    if result.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for w in result.warnings:
            console.print(f"  - {w}")


@app.command(hidden=True)
def status() -> None:
    """List running reflex serve processes (PID, port, command).

    Hidden 2026-05-07: niche diagnostic; `ps aux | grep reflex`
    accomplishes the same thing. Power-users can still invoke directly.
    """
    import re
    import subprocess as _sp
    try:
        out = _sp.run(["ps", "-eo", "pid,etime,command"], capture_output=True, text=True, timeout=5)
    except Exception as e:
        console.print(f"ps failed: {e}", style="red", markup=False)
        raise typer.Exit(1)
    rows: list[tuple[str, str, str, str]] = []
    for line in out.stdout.splitlines():
        # Match `reflex serve` or `python -m reflex serve` or `python -m reflex.cli serve`
        if not re.search(r"(?:^|/)reflex\s+serve\b|reflex\.cli\s+serve\b|-m\s+reflex\s+serve\b", line):
            continue
        if "grep" in line.lower() or "/status" in line or "ros2-serve" in line:
            continue
        m = re.match(r"\s*(\d+)\s+([\d:\-]+)\s+(.*)", line)
        if not m:
            continue
        pid, etime, cmd = m.group(1), m.group(2), m.group(3)
        port = ""
        pm = re.search(r"--port\s+(\d+)", cmd)
        if pm:
            port = pm.group(1)
        rows.append((pid, etime, port or "?", cmd[:120]))
    if not rows:
        console.print("No reflex serve processes detected.", markup=False)
        return
    table = Table(title="Reflex serve — running")
    table.add_column("PID")
    table.add_column("Uptime")
    table.add_column("Port")
    table.add_column("Command")
    for r in rows:
        table.add_row(*r)
    console.print(table)


@config_app.command("show")
def config_show() -> None:
    """Show effective reflex configuration (paths, defaults, env vars)."""
    import os as _os
    from pathlib import Path as _Path
    from reflex import __version__ as _v
    home = _Path(_os.environ.get("REFLEX_HOME", _Path.home() / ".cache" / "reflex"))
    table = Table(title=f"Reflex config (v{_v})")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("reflex_home", str(home))
    table.add_row("model_cache", str(home / "models"))
    table.add_row("export_default", str(_Path.cwd() / "reflex_export"))
    table.add_row("hf_cache", _os.environ.get("HF_HOME", str(_Path.home() / ".cache" / "huggingface")))
    table.add_row("FASTCREST_PROXY_URL", _os.environ.get("FASTCREST_PROXY_URL", "(default: https://chat.fastcrest.com)"))
    table.add_row("OPENAI_API_KEY", "set" if _os.environ.get("OPENAI_API_KEY") else "(unset — chat uses hosted proxy)")
    # Telemetry and data contribution status
    try:
        from reflex.onboarding import get_onboarding_state
        state = get_onboarding_state()
        table.add_row("telemetry", "on" if state.telemetry_enabled else "off")
        table.add_row("contribute_data", "on" if state.contribute_data else "off")
        table.add_row("dont_ask_again", "yes" if state.dont_ask_again else "no")
    except Exception:
        table.add_row("telemetry", "(unknown)")
        table.add_row("contribute_data", "(unknown)")
    console.print(table)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(help="Config key: telemetry, contribute-data"),
    value: str = typer.Argument(help="Value: on, off"),
) -> None:
    """Set a configuration value (telemetry on/off, contribute-data on/off)."""
    from reflex.onboarding import set_telemetry_enabled, set_contribute_data

    key_lower = key.lower().replace("_", "-")
    value_lower = value.lower()

    if value_lower not in ("on", "off", "true", "false", "1", "0"):
        console.print(f"Invalid value: {value}. Use on/off.", style="red")
        raise typer.Exit(1)
    enabled = value_lower in ("on", "true", "1")

    if key_lower == "telemetry":
        state = set_telemetry_enabled(enabled)
        console.print(f"Telemetry set to: {'on' if state.telemetry_enabled else 'off'}")
    elif key_lower == "contribute-data":
        state = set_contribute_data(enabled)
        console.print(f"Data contribution set to: {'on' if state.contribute_data else 'off'}")
    else:
        console.print(
            f"Unknown config key: {key}. Valid keys: telemetry, contribute-data",
            style="red",
        )
        raise typer.Exit(1)


# ── Data subcommands ──────────────────────────────────────────────────

data_app = typer.Typer(name="data", help="Manage episode data uploads and contributions.", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("review")
def data_review(
    pending: bool = typer.Option(False, "--pending", help="Show only pending uploads"),
) -> None:
    """Review queued and completed episode data uploads."""
    from reflex.pro.upload import UploadClient

    client = UploadClient()
    if pending:
        manifests = client.pending_manifests()
        if not manifests:
            console.print("No pending uploads.")
            return
        table = Table(title="Pending Uploads")
        table.add_column("Episode ID")
        table.add_column("Size")
        table.add_column("Queued At")
        table.add_column("Attempts")
        for m in manifests:
            table.add_row(m.episode_id, str(m.file_size), m.queued_at, str(m.attempts))
        console.print(table)
    else:
        pending_m = client.pending_manifests()
        completed_m = client.completed_manifests()
        console.print(f"Pending: {len(pending_m)}, Completed: {len(completed_m)}")
        if pending_m:
            table = Table(title="Pending")
            table.add_column("Episode ID")
            table.add_column("Size")
            table.add_column("Queued At")
            for m in pending_m:
                table.add_row(m.episode_id, str(m.file_size), m.queued_at)
            console.print(table)
        if completed_m:
            table = Table(title="Completed")
            table.add_column("Episode ID")
            table.add_column("Size")
            table.add_column("Completed At")
            for m in completed_m:
                table.add_row(m.episode_id, str(m.file_size), m.completed_at or "?")
            console.print(table)


@data_app.command("stats")
def data_stats() -> None:
    """Show episode data upload statistics."""
    from reflex.pro.upload import UploadClient

    client = UploadClient()
    s = client.stats()
    table = Table(title="Upload Stats")
    table.add_column("Key")
    table.add_column("Value")
    for k, v in s.items():
        table.add_row(str(k), str(v))
    console.print(table)


@data_app.command("revoke")
def data_revoke() -> None:
    """Delete ALL queued and completed episode data. GDPR/CCPA compliance."""
    from reflex.onboarding import set_contribute_data
    from reflex.pro.upload import UploadClient

    client = UploadClient()
    removed = client.revoke_all()
    set_contribute_data(False)
    console.print(f"Revoked: {removed} files deleted. Data contribution disabled.")


if __name__ == "__main__":
    app()
