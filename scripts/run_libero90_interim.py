"""Finish the teacher baseline, evaluate the selected checkpoint, and report."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/data/tether')
EVIDENCE=ROOT/'evaluation/libero90-interim'
REPO=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'rocm-venv/bin/python'


def alive(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat[stat.rfind(')')+2:].split()[0]!='Z'
    except FileNotFoundError: return False


def main():
    from report_libero90_interim import collect_arm
    state={'status':'waiting_for_teacher','started_unix_time':time.time()}
    status=EVIDENCE/'status.json'
    teacher_job=json.loads((EVIDENCE/'teacher_job.json').read_text())
    state['teacher_pid']=teacher_job['pid']
    status.write_text(json.dumps(state,indent=2))
    while alive(teacher_job['pid']): time.sleep(10)
    collect_arm(EVIDENCE/'teacher',teacher_job['checkpoint'],False)
    state['status']='waiting_for_student_checkpoint'
    status.write_text(json.dumps(state,indent=2))
    while True:
        selection=json.loads((EVIDENCE/'student_selection.json').read_text())
        checkpoint=Path(selection['checkpoint'])
        model=checkpoint/'model.safetensors'
        if model.is_file() and (checkpoint/'config.json').is_file():
            if selection['step']!=5000 or str(checkpoint) in (ROOT/'logs/training_10k.log').read_text():
                break
        delivery=json.loads((ROOT/'delivery_pipeline.json').read_text())
        training=next((phase for phase in delivery['phases'] if phase['name']=='training_10k'),None)
        if training is None or not alive(training['pid']):
            raise RuntimeError('Training ended before the selected student checkpoint was available')
        time.sleep(10)
    state.update(status='student_evaluation',student_checkpoint=str(checkpoint),student_step=selection['step'])
    command=[str(PYTHON),str(REPO/'scripts/evaluate_pi05_distill_rocm.py'),'eval',
             '--checkpoint',str(checkpoint),'--student','--suites','libero_90',
             '--episodes','5','--seed','7001','--output',str(EVIDENCE/'student')]
    with (ROOT/'logs/libero90-interim-student.log').open('w') as out:
        process=subprocess.Popen(command,cwd=REPO,stdout=out,stderr=subprocess.STDOUT)
        state['student_pid']=process.pid
        status.write_text(json.dumps(state,indent=2))
        code=process.wait()
    if code:
        state.update(status='failed',exit_code=code)
        status.write_text(json.dumps(state,indent=2))
        raise SystemExit(code)
    state['status']='generating_report'
    status.write_text(json.dumps(state,indent=2))
    subprocess.run([str(PYTHON),str(REPO/'scripts/report_libero90_interim.py'),
        '--student-checkpoint',str(checkpoint),'--student-step',str(selection['step'])],cwd=REPO,check=True)
    state.update(status='complete',ended_unix_time=time.time(),report=str(EVIDENCE/'report.md'))
    status.write_text(json.dumps(state,indent=2))


if __name__=='__main__':
    try: main()
    except Exception as exc:
        status=EVIDENCE/'status.json'
        state=json.loads(status.read_text()) if status.exists() else {}
        state.update(status='failed',error=str(exc),ended_unix_time=time.time())
        status.write_text(json.dumps(state,indent=2))
        raise
