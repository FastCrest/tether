"""Create the management report only after training and evaluation are complete."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile


def summarize_libero90(evidence_dir, checkpoint):
    """Require all 90 tasks and five real student rollouts per task."""
    import numpy as np
    all_outcomes=[]
    per_task=[]
    for task in range(90):
        path=Path(evidence_dir)/f'libero_90-task-{task}.json'
        raw=json.loads(path.read_text())
        if (raw.get('suite')!='libero_90' or raw.get('errors') or raw.get('total_eps')!=5 or raw.get('requested_seed')!=7001
                or not raw.get('student') or raw.get('inference_steps')!=1
                or not raw.get('denoise_calls_per_chunk_verified')
                or Path(raw['checkpoint']).resolve()!=Path(checkpoint).resolve()):
            raise RuntimeError(f'Incomplete or mismatched LIBERO-90 student evidence: task {task}')
        if len(raw['per_task'])!=1 or raw['per_task'][0]['task_idx']!=task:
            raise RuntimeError(f'Wrong LIBERO-90 task identity: {task}')
        episodes=sorted(raw['per_task'][0]['episodes'],key=lambda e:e['ep'])
        if [e['ep'] for e in episodes]!=list(range(5)):
            raise RuntimeError(f'Missing or duplicate LIBERO-90 episodes: task {task}')
        outcomes=[int(e['success']) for e in episodes]
        if sum(outcomes)!=raw['total_success']:
            raise RuntimeError(f'Inconsistent LIBERO-90 success count: task {task}')
        all_outcomes.append(outcomes)
        per_task.append({'task':task,'description':raw['per_task'][0]['task_description'],
                         'success':sum(outcomes),'episodes':5})
    values=np.array(all_outcomes)
    rng=np.random.default_rng(7001)
    bootstrap=[]
    for _ in range(10000):
        sampled=values[rng.integers(0,90,90)]
        sampled=np.take_along_axis(sampled,rng.integers(0,5,(90,5)),axis=1)
        bootstrap.append(float(sampled.mean()*100))
    return {'suite':'libero_90','tasks':90,'episodes':450,'success':int(values.sum()),
            'success_rate_pct':float(values.mean()*100),
            'success_rate_95pct_cluster_bootstrap_ci_pct':np.percentile(bootstrap,[2.5,97.5]).tolist(),
            'per_task':per_task,'checkpoint':str(checkpoint),'seed':7001,
            'included_in_original_40_task_aggregate':False}


def main():
    import numpy as np
    from plot_pi05_distill import render
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('/data/tether/runs/pi05-snapflow-10k'))
    p.add_argument('--evidence',type=Path,default=Path('/data/tether/evaluation'))
    p.add_argument('--output',type=Path,default=Path('/data/tether/delivery'))
    args=p.parse_args()
    result=json.loads((args.run/'result.json').read_text())
    if result['status']!='ok' or result['completed_steps']!=10000:
        raise RuntimeError('The requested 10,000-step training has not completed.')
    checkpoint=Path(result['checkpoint']).resolve()
    expected=args.run/'training/checkpoints/00010000/pretrained_model'
    if checkpoint!=expected.resolve() or not (checkpoint/'model.safetensors').is_file():
        raise RuntimeError('The requested final checkpoint is missing or differs from the training result.')
    render(args.run)
    training=json.loads((args.run/'training_summary.json').read_text())
    if training['completed_steps']!=10000:
        raise RuntimeError('Training telemetry does not prove exact completion.')
    latency=json.loads((args.evidence/'latency/latency.json').read_text())
    if Path(latency['student_1step']['checkpoint']).resolve()!=checkpoint or latency['measured_samples']<50:
        raise RuntimeError('Latency does not measure the final student with the requested sample count.')
    suites=['libero_spatial','libero_object','libero_goal','libero_10']
    pairs=[]
    per_suite=[]
    outcomes={}
    for suite in suites:
        counts={'teacher':0,'student':0}
        for task in range(10):
            arm_outcomes={}
            for arm in ['teacher','student']:
                raw=json.loads((args.evidence/arm/f'{suite}-task-{task}.json').read_text())
                if raw['errors'] or raw['total_eps']!=5 or raw['requested_seed']!=7001 or not raw.get('denoise_calls_per_chunk_verified'):
                    raise RuntimeError(f'Incomplete or mismatched {arm} evaluation: {suite} task {task}')
                if raw['inference_steps']!=(1 if arm=='student' else 10):
                    raise RuntimeError('Evaluation used the wrong inference step count.')
                if arm=='student' and Path(raw['checkpoint']).resolve()!=checkpoint:
                    raise RuntimeError('Evaluation does not use the final student checkpoint.')
                if raw['per_task'][0]['task_idx']!=task:
                    raise RuntimeError('Evaluation task identity does not match its filename.')
                eps=raw['per_task'][0]['episodes']
                if sorted(e['ep'] for e in eps)!=list(range(5)):
                    raise RuntimeError('Missing or duplicate evaluation episodes.')
                arm_outcomes[arm]=[int(e['success']) for e in sorted(eps,key=lambda e:e['ep'])]
                counts[arm]+=sum(arm_outcomes[arm])
            pairs.append(np.array(arm_outcomes['student'])-np.array(arm_outcomes['teacher']))
        per_suite.append({'suite':suite, **counts, 'episodes_per_arm':50})
    total={arm:sum(s[arm] for s in per_suite) for arm in ['teacher','student']}
    delta=(total['student']-total['teacher'])/200*100
    rng=np.random.default_rng(7001)
    paired=np.array(pairs)
    boot=[]
    for _ in range(10000):
        sampled=paired[rng.integers(0,40,40)]
        sampled=np.take_along_axis(sampled,rng.integers(0,5,(40,5)),axis=1)
        boot.append(float(sampled.mean()*100))
    ci=np.percentile(boot,[2.5,97.5]).tolist()
    libero90=summarize_libero90(args.evidence/'student-libero90',checkpoint)
    official90=json.loads(Path('/data/tether/libero90_download_manifest.json').read_text())
    if official90.get('status')!='complete' or official90.get('verified_tasks')!=90:
        raise RuntimeError('The official LIBERO-90 dataset download is incomplete.')
    metrics={'training':training,'latency':latency,'per_suite':per_suite,
             'libero90_student':libero90,
             'libero90_dataset':official90,
             'experiment':json.loads((args.run/'reproduction_inputs.json').read_text()),
             'environment':json.loads(Path('/data/tether/environment.json').read_text()),
             'teacher_success':total['teacher'],'student_success':total['student'],
             'episodes_per_arm':200,'paired_success_delta_pp':delta,
             'paired_delta_95pct_cluster_bootstrap_ci_pp':ci,
             'bootstrap':'10,000 paired two-level resamples: task then episode, seed 7001'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    shutil.copy2(args.run/'loss.png',args.output/'loss.png')
    shutil.copy2('/data/tether/logs/rocm-reproduction-environment.txt',args.output/'environment.txt')
    t=latency['teacher_10step']; s=latency['student_1step']
    conclusion=(f'The 1-step student achieved {total["student"]/2:.1f}% LIBERO success versus '
                f'{total["teacher"]/2:.1f}% for the 10-step teacher ({delta:+.1f} percentage points), '
                f'with {latency["policy_speedup"]:.2f}× lower median policy inference time. '
                f'On the separate LIBERO-90 suite, the student achieved {libero90["success_rate_pct"]:.1f}% '
                f'({libero90["success"]}/450).')
    rows='\n'.join(f'| {x["suite"].replace("libero_", "").title()} | {x["teacher"]}/50 ({x["teacher"]*2:.1f}%) | {x["student"]}/50 ({x["student"]*2:.1f}%) |' for x in per_suite)
    if delta < -5:
        recommendation='Retain the teacher for deployment. The student trades too much task success for latency; further distillation work is needed.'
    elif ci[0] < -5:
        recommendation='Expand the paired evaluation before replacing the teacher; the current confidence interval does not rule out a material task-success regression.'
    else:
        recommendation='Proceed to a controlled deployment validation; physical-robot performance and the target control-loop budget remain to be measured.'
    report=f'''# pi0.5 Single-Step Distillation on AMD ROCm

{conclusion}

**Recommendation for the original 40-task comparison:** {recommendation}

## Work delivered

Delivered AMD environment adaptation, a **10,000-step distillation run**, loss monitoring,
**400 matched simulation rollouts plus 450 student LIBERO-90 rollouts**, and synchronized inference benchmarks.
The teacher is `{Path(os.environ.get('TETHER_TEACHER_CHECKPOINT', 'pi05_libero_finetuned_v044')).name}`;
the distillation dataset contains 1,693 LIBERO demonstrations (273,465 frames, 40 tasks).
Default SnapFlow trains the action expert and time/action projections, with the VLM frozen.

## Environment and training

- **Environment:** AMD Radeon 8060S, ROCm 10.1 nightly PyTorch, Python 3.12,
  LeRobot 0.6.1, Transformers 5.5.4; isolated dependencies reuse `rocm-nightly`.
- **Training:** batch 4, bf16; {training['training_wall_time_s']/3600:.2f} hours;
  {training['steady_step_median_s']:.2f} s/step median;
  {training['peak_gpu_allocated_gib']:.1f} GiB peak PyTorch GPU allocation on shared system memory.
- **Loss:** mean total loss over the first/last 50 steps:
  {training['first_50_mean_loss']:.3f} → {training['last_50_mean_loss']:.3f}.

## Measured results

| Metric | Teacher (10 steps) | Student (1 step) |
|---|---:|---:|
| LIBERO success, 200 episodes each | {total['teacher']}/200 ({total['teacher']/2:.1f}%) | {total['student']}/200 ({total['student']/2:.1f}%) |
| Policy latency, median / p95 | {t['policy']['p50_ms']:.1f} / {t['policy']['p95_ms']:.1f} ms | {s['policy']['p50_ms']:.1f} / {s['policy']['p95_ms']:.1f} ms |
| Preprocess + policy + postprocess, median | {t['pipeline']['p50_ms']:.1f} ms | {s['pipeline']['p50_ms']:.1f} ms |

| LIBERO suite | Teacher | Student |
|---|---:|---:|
{rows}

**Additional LIBERO-90 evaluation:** student **{libero90['success']}/450
({libero90['success_rate_pct']:.1f}%)** across all 90 tasks, five initial states each;
95% task/episode bootstrap interval:
{libero90['success_rate_95pct_cluster_bootstrap_ci_pct'][0]:.1f}–{libero90['success_rate_95pct_cluster_bootstrap_ci_pct'][1]:.1f}%.
This score is reported separately from the 40-task aggregate. Distillation uses the
original 40-task dataset; official LIBERO-90 demonstrations were downloaded separately
({official90['bytes']/2**30:.2f} GiB) and were not added to training.

All 40 tasks use five matched initial states each. The paired success delta is
{delta:+.1f} pp (95% task/episode bootstrap interval: {ci[0]:+.1f} to {ci[1]:+.1f} pp).
Latency: batch 1, real observations, {latency['warmup']} warmups,
{latency['measured_samples']} samples per arm, GPU synchronization, one VLM prefix per chunk.

## Interpretation and limits

The full preprocessing/inference/postprocessing speedup is **{latency['pipeline_speedup']:.2f}×**;
reducing denoising steps leaves VLM compute in place. Results cover the evaluated LIBERO
simulation suites, with limited per-task sample size; physical robots remain unmeasured.
The LIBERO-90 result has no matched teacher baseline in this run.
Native teacher projections retain fp32, while the student uses bf16.
This reproduces the public implementation; incomplete original provenance prevents a
bit-exact replica claim.

![Training loss](loss.png)

Detailed measurements and package versions: `metrics.json`, `environment.txt`.
'''
    (args.output/'report.md').write_text(report)
    with tarfile.open(args.output.parent/'pi05-distillation-report.tar.gz','w:gz') as archive:
        for name in ['report.md','loss.png','metrics.json','environment.txt']:
            archive.add(args.output/name,arcname=name)
    print(args.output/'report.md')


if __name__=='__main__': main()
