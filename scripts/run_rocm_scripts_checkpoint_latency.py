"""Temporarily pause JAX workers, benchmark checkpoints, always resume workers."""
import json
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/data/tether')
REPO=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms',default='teacher,student')
    args=parser.parse_args()
    arms=args.arms.split(',')
    if any(x not in ['teacher','student'] for x in arms):parser.error('Invalid arm')
    out=ROOT/'evaluation/rocm-scripts-checkpoint-latency'
    out.mkdir(parents=True,exist_ok=True)
    paused=[]
    state={'status':'starting','pid':os.getpid(),'started':time.time()}
    status=out/'status.json'
    try:
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():continue
            try:argv=(process/'cmdline').read_bytes().split(b'\0')
            except OSError:continue
            if any(Path(arg.decode(errors='replace')).name in {'run_parallel_jax_libero90.py','evaluate_pi05_jax_rocm.py'} for arg in argv if arg):
                pid=int(process.name);os.kill(pid,signal.SIGSTOP);paused.append(pid)
        state['paused_jax_pids']=paused;status.write_text(json.dumps(state,indent=2))
        for arm in arms:
            state.update(status='benchmarking',arm=arm);status.write_text(json.dumps(state,indent=2))
            command=[str(ROOT/'rocm-venv/bin/python'),str(REPO/'scripts/benchmark_rocm_scripts_pi05.py'),
                     '--arm',arm,'--json',str(out/f'{arm}.json')]
            with (ROOT/'logs'/f'rocm-scripts-latency-{arm}.log').open('w') as log:
                subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        teacher=json.loads((out/'teacher.json').read_text())
        student=json.loads((out/'student.json').read_text())
        rows=[]
        for arm,result in [('teacher',teacher),('student',student)]:
            row=dict(result);row['entry_test_name']=row['test_name'];row['test_name']='openpi_pi05_'+arm;rows.append(row)
        (out/'benchmarks.json').write_text(json.dumps({'suite':'pytorch_models','results':rows},indent=2))
        state.update(status='complete',ended=time.time(),
                     speedup=teacher['latency_p50_ms']/student['latency_p50_ms'])
        (out/'report.md').write_text(f'# ROCm-scripts checkpoint latency\n\n| Model | Mean (ms) | P50 (ms) | Samples | Pipeline mean (ms) |\n|---|---:|---:|---:|---:|\n| Teacher, 10-step | {teacher["latency_mean_ms"]:.2f} | {teacher["latency_p50_ms"]:.2f} | {teacher["timed_iterations"]} | {teacher["pipeline_mean_ms"]:.2f} |\n| Student 10k, 1-step | {student["latency_mean_ms"]:.2f} | {student["latency_p50_ms"]:.2f} | {student["timed_iterations"]} | {student["pipeline_mean_ms"]:.2f} |\n\nP50 speedup: {state["speedup"]:.2f}×. rocm-scripts openpi_pi05.py entry/harness with a LIBERO checkpoint adapter; max-autotune, batched SigLIP, batch 1, 50×7 actions, real LIBERO observations. JAX workers paused during measurement and resumed afterwards. Teacher uses native mixed precision including FP32 vision; student uses all BF16, so speedup includes precision effects. Historical 228 ms is pi0_droid, not pi05_droid.\n')
    except BaseException as exc:
        state.update(status='failed',error=str(exc));raise
    finally:
        for pid in paused:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
        state['jax_resumed']=True;status.write_text(json.dumps(state,indent=2))
        print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
