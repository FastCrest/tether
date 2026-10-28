"""Local SnapFlow runner; requires the dependency gates in the plan first."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import random
import time
import shutil
import hashlib
import subprocess

ROOT = Path('/data/tether')
os.environ.setdefault('HF_HOME', str(ROOT / 'hf'))
os.environ.setdefault('HF_LEROBOT_HOME', str(ROOT / 'lerobot'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def prepare_teacher():
    """Keep upstream files immutable; override only the tokenizer location."""
    override = os.environ.get('TETHER_TEACHER_CHECKPOINT')
    if override:
        target = Path(override)
        for name in ['model.safetensors', 'config.json', 'policy_preprocessor.json', 'policy_postprocessor.json']:
            if not (target/name).is_file():
                raise RuntimeError(f'Missing teacher file: {target/name}')
        return target
    source = ROOT / 'checkpoints/pi05_libero_finetuned_v044'
    target = ROOT / 'checkpoints/pi05_libero_finetuned_v044-local'
    tokenizer = ROOT / 'tokenizers/paligemma-3b-pt-224'
    if not (source / 'model.safetensors').is_file() or not (tokenizer / 'tokenizer.json').is_file():
        raise RuntimeError('Run download_pi05_distill_assets.py to completion first.')
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.name not in ('policy_preprocessor.json', 'config.json'):
            link = target / item.name
            if not link.exists():
                link.symlink_to(item)
    processor = json.loads((source / 'policy_preprocessor.json').read_text())
    for step in processor['steps']:
        if step['registry_name'] == 'tokenizer_processor':
            step['config']['tokenizer_name'] = str(tokenizer)
    (target / 'policy_preprocessor.json').write_text(json.dumps(processor, indent=2))
    config = json.loads((source / 'config.json').read_text())
    config.update(compile_model=False, push_to_hub=False)
    (target / 'config.json').write_text(json.dumps(config, indent=2))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--checkpoint-every', type=int, default=10000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--heartbeat-every', type=int, default=10)
    args = parser.parse_args()
    teacher = prepare_teacher()
    print(f'Local teacher: {teacher}', flush=True)
    if args.prepare_only:
        return
    import torch
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('A ROCm PyTorch build with an available GPU is required.')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    from tether.finetune.config import FinetuneConfig
    from tether.finetune.run import run_finetune
    from tether.finetune.hooks import HookRegistry
    import numpy as np
    import psutil
    from torch.utils.tensorboard import SummaryWriter
    from plot_pi05_distill import render
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = FinetuneConfig(
        base=str(teacher), dataset='lerobot/libero', teacher_export=str(teacher),
        output=args.output, phase='distill', mode='full', variant='default',
        num_steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed, precision='bf16',
        skip_export=True, dry_run=args.dry_run,
        dataset_revision='a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4',
        extra_lerobot_args={'consistency_alpha': 1.0, 'libero_gate_skip': True,
                            'checkpoint_every': args.checkpoint_every,
                            'heartbeat_every': args.heartbeat_every,
                            'distill_dataset_root': str(ROOT / 'lerobot/lerobot/libero'),
                            'gradient_check_every': 10,
                            'video_backend': 'pyav'},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'reproduction_inputs.json').write_text(json.dumps({
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'inputs': json.loads((ROOT / 'download_manifest.json').read_text()),
        'torch': torch.__version__, 'hip': torch.version.hip,
        'teacher_processor_override': str(teacher / 'policy_preprocessor.json'),
        'actual_teacher_checkpoint': str(teacher),
        'actual_teacher_conversion': json.loads((teacher/'conversion.json').read_text())
            if (teacher/'conversion.json').exists() else None,
        'task_success_evaluated': False,
        'tokenizer_identity': json.loads((ROOT / 'tokenizer_source_identity.json').read_text()),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'code_sha256': {str(p.relative_to(Path(__file__).resolve().parents[1])):
                        hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                            Path(__file__).resolve(),
                            Path(__file__).resolve().parents[1] / 'src/tether/finetune/backends/snapflow_backend.py',
                            Path(__file__).resolve().parents[1] / 'src/tether/distill/snapflow.py',
                            Path(__file__).resolve().parents[1] / 'src/tether/distill/snapflow_pi0_model.py']},
    }, indent=2))
    hooks = HookRegistry()
    # A fresh run at the same path must not retain the stopped run's cached
    # TensorBoard points after its event files have been archived.
    writer = SummaryWriter(str(args.output / 'tensorboard'), purge_step=0, flush_secs=5)
    started = time.time()
    last_step_time = started
    telemetry = (args.output / 'telemetry.jsonl').open('a', buffering=1)
    process = psutil.Process()
    def on_start(ctx, **row):
        nonlocal started, last_step_time
        started = last_step_time = time.time()
        (args.output / 'training_started.json').write_text(json.dumps({'unix_time': started}))
    def on_checkpoint(ctx, **row):
        for item in teacher.iterdir():
            if item.is_file() and item.name.startswith('policy_'):
                shutil.copy2(item, row['ckpt_path'] / item.name)
        render(args.output)
    def observe(ctx, **row):
        nonlocal last_step_time
        now = time.time()
        step = row['step']
        metrics = dict(step=step, wall_time_s=now-started, step_time_s=now-last_step_time,
                       rss_gib=process.memory_info().rss/2**30,
                       system_available_gib=psutil.virtual_memory().available/2**30,
                       gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
                       gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       loss=row['loss'], flow_matching=row['flow_matching'], consistency=row['consistency'])
        if not all(np.isfinite(metrics[k]) for k in ['loss', 'flow_matching', 'consistency']):
            raise RuntimeError(f'Non-finite training loss at step {step}')
        if row.get('gradient_norm') is not None:
            metrics['gradient_norm'] = row['gradient_norm']
        telemetry.write(json.dumps(metrics)+'\n')
        for key, val in metrics.items():
            if key != 'step': writer.add_scalar(key, val, step)
        if step % args.heartbeat_every == 0: writer.flush()
        if step == 1 or step % 100 == 0: render(args.output)
        last_step_time = now
    hooks.register('on_start', on_start)
    hooks.register('on_step', observe)
    hooks.register('on_checkpoint', on_checkpoint)
    try:
        result = run_finetune(cfg, hooks=hooks)
    finally:
        writer.close()
        telemetry.close()
        if (args.output / 'telemetry.jsonl').stat().st_size: render(args.output)
    (args.output / 'result.json').write_text(json.dumps({
        'status': result.status, 'error': result.error,
        'completed_steps': result.training_steps_completed,
        'checkpoint': str(result.final_checkpoint_path),
        'ended_unix_time': time.time(),
    }, indent=2))
    print(result)
    if result.error or result.status != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
