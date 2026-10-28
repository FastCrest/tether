"""Render training loss and throughput from the actual JSONL telemetry."""
import argparse
import json
from pathlib import Path


def render(run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    run_dir = Path(run_dir)
    rows = []
    for line in (run_dir / 'telemetry.jsonl').read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A live writer may not have completed the last line.
    if not rows:
        return
    steps = np.array([r['step'] for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for key, label in [('loss', 'Total'), ('flow_matching', 'Flow matching'), ('consistency', 'Consistency')]:
        vals = np.array([r[key] for r in rows])
        line, = axes[0].plot(steps, vals, alpha=.2, linewidth=.6)
        window = min(50, max(1, len(vals)//5))
        smooth = np.convolve(vals, np.ones(window)/window, mode='valid')
        axes[0].plot(steps[window-1:], smooth, color=line.get_color(), label=f'{label} ({window}-step mean)')
    axes[0].set(ylabel='Loss', title='pi0.5 SnapFlow distillation on ROCm')
    axes[0].legend()
    axes[1].plot(steps[1:], [r['step_time_s'] for r in rows[1:]], label='Step duration')
    axes[1].set(xlabel='Training step', ylabel='Seconds / step')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png', 'svg']:
        temp = run_dir / f'.loss.tmp.{ext}'
        fig.savefig(temp, dpi=150)
        temp.replace(run_dir / f'loss.{ext}')
    plt.close(fig)
    summary = dict(completed_steps=int(steps[-1]),
                   first_50_mean_loss=float(np.mean([r['loss'] for r in rows[:50]])),
                   last_50_mean_loss=float(np.mean([r['loss'] for r in rows[-50:]])),
                   last_loss=rows[-1]['loss'],
                   training_wall_time_s=rows[-1]['wall_time_s'],
                   steady_step_median_s=float(np.median([r['step_time_s'] for r in rows[1:]])) if len(rows)>1 else None,
                   peak_gpu_allocated_gib=max(r['gpu_peak_allocated_gib'] for r in rows),
                   peak_rss_gib=max(r['rss_gib'] for r in rows))
    (run_dir / 'training_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    render(p.parse_args().run_dir)
