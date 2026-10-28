"""A short management report from complete, matched LIBERO-90 rollouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def collect_arm(directory,checkpoint,student):
    outcomes=[]
    tasks=[]
    protocol=None
    for task in range(90):
        raw=json.loads((Path(directory)/f'libero_90-task-{task}.json').read_text())
        if (raw.get('suite')!='libero_90' or raw.get('errors') or raw.get('total_eps')!=5
                or raw.get('requested_seed')!=7001 or raw.get('student')!=student
                or raw.get('inference_steps')!=(1 if student else 10)
                or not raw.get('denoise_calls_per_chunk_verified')
                or Path(raw['checkpoint']).resolve()!=Path(checkpoint).resolve()):
            raise RuntimeError(f'Invalid LIBERO-90 evidence: task {task}, student={student}')
        if len(raw['per_task'])!=1 or raw['per_task'][0]['task_idx']!=task:
            raise RuntimeError('Wrong task identity')
        episodes=sorted(raw['per_task'][0]['episodes'],key=lambda e:e['ep'])
        if [e['ep'] for e in episodes]!=list(range(5)):
            raise RuntimeError('Missing or duplicate episodes')
        values=[int(e['success']) for e in episodes]
        if sum(values)!=raw['total_success']:
            raise RuntimeError('Inconsistent success count')
        current={key:raw[key] for key in ['max_steps','resize_size','replan_steps','num_steps_wait']}
        if protocol is None: protocol=current
        if current!=protocol: raise RuntimeError('Protocol changed between tasks')
        outcomes.append(values)
        tasks.append({'task':task,'description':raw['per_task'][0]['task_description'],
                      'episodes':episodes,'successes':sum(values)})
    return {'successes':sum(map(sum,outcomes)),'attempts':450,
            'success_rate_pct':sum(map(sum,outcomes))/450*100,
            'outcomes':outcomes,'tasks':tasks,'protocol':protocol,
            'checkpoint':str(checkpoint),'student':student}


def paired_delta(teacher,student):
    import numpy as np
    if teacher['protocol']!=student['protocol']:
        raise RuntimeError('Teacher and student evaluation protocols differ')
    if [t['description'] for t in teacher['tasks']]!=[t['description'] for t in student['tasks']]:
        raise RuntimeError('Teacher and student task descriptions differ')
    values=np.array(student['outcomes'])-np.array(teacher['outcomes'])
    rng=np.random.default_rng(7001)
    bootstrap=[]
    for _ in range(10000):
        sampled=values[rng.integers(0,90,90)]
        sampled=np.take_along_axis(sampled,rng.integers(0,5,(90,5)),axis=1)
        bootstrap.append(float(sampled.mean()*100))
    return float(values.mean()*100),np.percentile(bootstrap,[2.5,97.5]).tolist()


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--student-checkpoint',type=Path,required=True)
    p.add_argument('--student-step',type=int,required=True)
    p.add_argument('--evidence',type=Path,default=Path('/data/tether/evaluation/libero90-interim'))
    args=p.parse_args()
    teacher_path=Path(os.environ.get('TETHER_TEACHER_CHECKPOINT','/data/tether/checkpoints/pi05_libero_finetuned_v044-local'))
    if args.student_checkpoint.parent.name!=f'{args.student_step:08d}':
        raise RuntimeError('The report step does not match the evaluated checkpoint path')
    teacher=collect_arm(args.evidence/'teacher',teacher_path,False)
    student=collect_arm(args.evidence/'student',args.student_checkpoint,True)
    delta,ci=paired_delta(teacher,student)
    metrics={'teacher':teacher,'student':student,'student_training_step':args.student_step,
             'paired_success_delta_pp':delta,'paired_delta_95pct_ci_pp':ci,
             'seed':7001,'evaluation_concurrent_with_training':True,
             'teacher_weights_sha256':file_hash(teacher_path/'model.safetensors'),
             'student_weights_sha256':file_hash(args.student_checkpoint/'model.safetensors')}
    (args.evidence/'metrics.json').write_text(json.dumps(metrics,indent=2))
    if teacher['successes']==student['successes']==0:
        conclusion='Both models failed all evaluated cases. This experiment does not demonstrate LIBERO-90 capability.'
    elif ci[1]<0:
        conclusion='The teacher performs better. This student checkpoint does not yet support replacing it.'
    elif ci[0]>0:
        conclusion='The student performs better in this paired test. Confirm the result with the final checkpoint.'
    else:
        conclusion='The difference is inconclusive at this sample size. Use the final checkpoint evaluation for the decision.'
    report=f'''# pi0.5 Distillation — LIBERO-90 Check

## Setup

| Item | Setting |
|---|---|
| Teacher | `{teacher_path.name}`, 10 denoising steps |
| Student | SnapFlow checkpoint at **{args.student_step:,} training steps**, 1 denoising step |
| Training | LIBERO 40-task data; batch 4; bf16; LR 1e-4; 10,000-step run retained |
| Evaluation | 90 tasks × 5 matched initial states; 450 episodes/model; seed 7001; {teacher['protocol']['max_steps']}-step limit |
| Hardware | AMD Radeon 8060S; ROCm 10.1; PyTorch nightly |

## Results

| Model | Successes / episodes | Success rate |
|---|---:|---:|
| Teacher, 10-step | {teacher['successes']}/450 | {teacher['success_rate_pct']:.1f}% |
| Student, 1-step | {student['successes']}/450 | {student['success_rate_pct']:.1f}% |

Student minus teacher: **{delta:+.1f} percentage points**
(95% paired task/episode bootstrap interval: {ci[0]:+.1f} to {ci[1]:+.1f} pp).

## Conclusion

{conclusion}

This evaluates an intermediate checkpoint, not the final 10,000-step model.
LIBERO-90 demonstrations were not used in this distillation run; the teacher was previously trained on LIBERO-90.

Measurements and checkpoint hashes: `metrics.json`.
'''
    (args.evidence/'report.md').write_text(report)
    print(args.evidence/'report.md')


if __name__=='__main__': main()
