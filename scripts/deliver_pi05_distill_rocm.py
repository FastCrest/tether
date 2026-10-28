"""Sequential local delivery pipeline; each phase must succeed before the next."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import argparse

ROOT=Path('/data/tether')
REPO=Path(__file__).resolve().parents[1]
PYTHON=Path('/data/tether/rocm-venv/bin/python')
TEACHER=Path(os.environ.get('TETHER_TEACHER_CHECKPOINT',str(ROOT/'checkpoints/pi05_libero_finetuned_v044-local')))
SMOKE=ROOT/'runs/pi05-snapflow-smoke/training/checkpoints/00000020/pretrained_model'
FINAL=ROOT/'runs/pi05-snapflow-10k/training/checkpoints/00010000/pretrained_model'
STATE=ROOT/'delivery_pipeline.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-smoke',action='store_true',help='Reuse the already-passed local rollout/latency smoke checks.')
    args=parser.parse_args()
    os.environ.update(HF_HOME=str(ROOT/'hf'),HF_LEROBOT_HOME=str(ROOT/'lerobot'),
                      LIBERO_CONFIG_PATH=str(ROOT/'libero-config'), MUJOCO_GL='egl',
                      PYOPENGL_PLATFORM='egl',TMPDIR=str(ROOT/'tmp'),TOKENIZERS_PARALLELISM='false')
    os.chdir(REPO)
    if (ROOT/'runs/pi05-snapflow-10k').exists():
        raise RuntimeError('Formal output directory already exists; inspect its live process/state before rerunning.')
    eval_script=str(REPO/'scripts/evaluate_pi05_distill_rocm.py')
    download=json.loads((ROOT/'libero90_download_manifest.json').read_text())
    if download.get('status')!='complete' or download.get('verified_tasks')!=90:
        raise RuntimeError('Complete and verify the official LIBERO-90 download first.')
    phases=[
        ('student_rollout_smoke',[str(PYTHON),eval_script,'eval','--checkpoint',str(SMOKE),'--student',
          '--suites','libero_10','--tasks','0','--episodes','1','--output',str(ROOT/'evaluation/smoke-student')]),
        ('latency_smoke',[str(PYTHON),eval_script,'benchmark','--checkpoint',str(SMOKE),
          '--warmup','2','--samples','5','--output',str(ROOT/'evaluation/smoke-latency')]),
        ('training_10k',[str(PYTHON),str(REPO/'scripts/run_pi05_distill_rocm.py'),
          '--steps','10000','--batch-size','4','--checkpoint-every','5000','--heartbeat-every','25',
          '--learning-rate','1e-4','--seed','42','--output',str(ROOT/'runs/pi05-snapflow-10k')]),
        ('latency_final',[str(PYTHON),eval_script,'benchmark','--checkpoint',str(FINAL),
          '--output',str(ROOT/'evaluation/latency')]),
        ('teacher_evaluation',[str(PYTHON),eval_script,'eval','--checkpoint',str(TEACHER),
          '--episodes','5','--seed','7001','--output',str(ROOT/'evaluation/teacher')]),
        ('student_evaluation',[str(PYTHON),eval_script,'eval','--checkpoint',str(FINAL),'--student',
          '--episodes','5','--seed','7001','--output',str(ROOT/'evaluation/student')]),
        ('student_libero90_evaluation',[str(PYTHON),eval_script,'eval','--checkpoint',str(FINAL),'--student',
          '--suites','libero_90','--episodes','5','--seed','7001',
          '--output',str(ROOT/'evaluation/student-libero90')]),
        ('report',[str(PYTHON),str(REPO/'scripts/report_pi05_distill.py')]),
    ]
    if args.skip_smoke:
        if os.environ.get('TETHER_TEACHER_CHECKPOINT'):
            smoke=json.loads((TEACHER/'runtime_smoke.json').read_text())
            if not smoke.get('finite_actions') or smoke.get('denoise_calls')!=10:
                raise RuntimeError('New converted teacher must pass GPU inference before training.')
        else:
            smoke=json.loads((ROOT/'evaluation/smoke-student/libero_10-task-0.json').read_text())
            latency=json.loads((ROOT/'evaluation/smoke-latency/latency.json').read_text())
            if (smoke.get('errors') or smoke.get('total_eps')!=1
                    or latency['teacher_10step']['denoise_calls_per_chunk']!=10
                    or latency['student_1step']['denoise_calls_per_chunk']!=1):
                raise RuntimeError('Existing smoke checks are incomplete or failed.')
        phases=phases[2:]
    state={'pid':os.getpid(),'started_unix_time':time.time(),'status':'running','phases':[]}
    for name,command in phases:
        log=ROOT/'logs'/f'{name}.log'
        row={'name':name,'command':command,'log':str(log),'started_unix_time':time.time(),'status':'running'}
        state['phases'].append(row)
        with log.open('w') as output:
            process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT)
            row['pid']=process.pid
            STATE.write_text(json.dumps(state,indent=2))
            code=process.wait()
        row.update(exit_code=code,ended_unix_time=time.time(),status='complete' if code==0 else 'failed')
        if code:
            state['status']='failed'
            STATE.write_text(json.dumps(state,indent=2))
            raise SystemExit(code)
        STATE.write_text(json.dumps(state,indent=2))
    state.update(status='complete',ended_unix_time=time.time())
    STATE.write_text(json.dumps(state,indent=2))


if __name__=='__main__': main()
