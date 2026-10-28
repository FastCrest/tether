"""Start a fresh distillation run and matched LIBERO-90 evaluation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/data/tether')
REPO=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'rocm-venv/bin/python'
TEACHER=ROOT/'checkpoints/pi05-libero90mix-torch'


def launch(command,log,env):
    with log.open('w') as out:
        return subprocess.Popen(command,cwd=REPO,env=env,stdout=out,
                                stderr=subprocess.STDOUT,start_new_session=True)


def main():
    smoke=json.loads((TEACHER/'runtime_smoke.json').read_text())
    if not smoke['finite_actions'] or smoke['denoise_calls']!=10:
        raise RuntimeError('Converted teacher must pass the ROCm inference gate')
    names={'run_pi05_distill_rocm.py','evaluate_pi05_distill_rocm.py',
           'deliver_pi05_distill_rocm.py','run_libero90_interim.py'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: argv=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        if any(Path(a.decode(errors='replace')).name in names for a in argv if a):
            raise RuntimeError(f'Existing pipeline process {proc.name}; do not overwrite a live run')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=ROOT/'archives'/f'old-teacher-stopped-2398-{stamp}'
    archive.mkdir(parents=True)
    run=ROOT/'runs/pi05-snapflow-10k'
    if run.exists(): run.rename(ROOT/'runs'/f'pi05-snapflow-old-teacher-stopped-2398-{stamp}')
    paths=[ROOT/'evaluation'/name for name in ['libero90-interim','teacher','student','student-libero90','latency']]
    paths += [ROOT/name for name in ['delivery_pipeline.json','restart_status.json']]
    paths += [ROOT/'logs'/name for name in ['training_10k.log','teacher_evaluation.log',
        'student_evaluation.log','student_libero90_evaluation.log','latency_final.log',
        'libero90-interim-teacher.log','libero90-interim-student.log','libero90-interim-supervisor.log']]
    for path in paths:
        if path.exists(): path.rename(archive/path.name)
    env=os.environ.copy()
    env.update(TETHER_TEACHER_CHECKPOINT=str(TEACHER),HF_HOME=str(ROOT/'hf'),
        HF_LEROBOT_HOME=str(ROOT/'lerobot'),LIBERO_CONFIG_PATH=str(ROOT/'libero-config'),
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',TMPDIR=str(ROOT/'tmp'),TOKENIZERS_PARALLELISM='false')
    delivery=launch([str(PYTHON),str(REPO/'scripts/deliver_pi05_distill_rocm.py'),'--skip-smoke'],
                    ROOT/'logs/libero90-teacher-delivery.log',env)
    evidence=ROOT/'evaluation/libero90-interim'
    evidence.mkdir(parents=True)
    teacher=launch([str(PYTHON),str(REPO/'scripts/evaluate_pi05_distill_rocm.py'),'eval',
        '--checkpoint',str(TEACHER),'--suites','libero_90','--episodes','5','--seed','7001',
        '--output',str(evidence/'teacher')],ROOT/'logs/libero90-interim-teacher.log',env)
    (evidence/'teacher_job.json').write_text(json.dumps({'pid':teacher.pid,'checkpoint':str(TEACHER)}))
    (evidence/'student_selection.json').write_text(json.dumps({'step':5000,
        'checkpoint':str(ROOT/'runs/pi05-snapflow-10k/training/checkpoints/00005000/pretrained_model'),
        'reason':'First formal checkpoint from the LIBERO-90 teacher distillation run'}))
    supervisor=launch([str(PYTHON),str(REPO/'scripts/run_libero90_interim.py')],
                      ROOT/'logs/libero90-interim-supervisor.log',env)
    state={'status':'started','started_unix_time':time.time(),'teacher':str(TEACHER),
        'delivery_pid':delivery.pid,'teacher_evaluation_pid':teacher.pid,
        'interim_supervisor_pid':supervisor.pid,'archive':str(archive)}
    (ROOT/'libero90_teacher_restart.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()
