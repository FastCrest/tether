"""Wait for the verified official dataset, preserve the stopped run, then restart delivery."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/data/tether')
REPO=Path(__file__).resolve().parents[1]


def process_running(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat[stat.rfind(')')+2:].split()[0]!='Z'
    except FileNotFoundError:
        return False


def require_training_stopped():
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        try: argv=(process/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if any(arg.endswith(b'run_pi05_distill_rocm.py') for arg in argv):
            raise RuntimeError(f'A training process is already running: {process.name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-pid',type=int,required=True)
    args=parser.parse_args()
    status_path=ROOT/'restart_status.json'
    state={'pid':os.getpid(),'download_pid':args.download_pid,'status':'waiting_for_verified_libero90',
           'started_unix_time':time.time()}
    status_path.write_text(json.dumps(state,indent=2))
    try:
        while process_running(args.download_pid): time.sleep(5)
        manifest=json.loads((ROOT/'libero90_download_manifest.json').read_text())
        if manifest.get('status')!='complete' or manifest.get('verified_tasks')!=90:
            raise RuntimeError('Download ended without a verified 90-task dataset.')
        require_training_stopped()
        stopped=ROOT/'runs/pi05-snapflow-10k'
        if stopped.exists():
            result=json.loads((stopped/'result.json').read_text())
            if result.get('status')!='interrupted_by_user':
                raise RuntimeError('The existing formal run is not a confirmed user-stopped run.')
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            archive=ROOT/'runs'/f'pi05-snapflow-10k-stopped-{result["completed_steps"]}-{stamp}'
            stopped.rename(archive)
            state['previous_run_archive']=str(archive)
            for name in ['training_10k.log','delivery_pipeline.json']:
                path=ROOT/'logs'/name if name.endswith('.log') else ROOT/name
                if path.exists(): shutil.copy2(path,archive/name)
        state['status']='delivery_running'
        with (ROOT/'logs/restarted-delivery-pipeline.log').open('a') as output:
            process=subprocess.Popen([str(ROOT/'rocm-venv/bin/python'),
                str(REPO/'scripts/deliver_pi05_distill_rocm.py'),'--skip-smoke'],
                cwd=REPO,stdout=output,stderr=subprocess.STDOUT)
            state['delivery_pid']=process.pid
            status_path.write_text(json.dumps(state,indent=2))
            code=process.wait()
        state.update(status='complete' if code==0 else 'failed',exit_code=code,ended_unix_time=time.time())
        status_path.write_text(json.dumps(state,indent=2))
        if code: raise SystemExit(code)
    except Exception as exc:
        state.update(status='failed',error=str(exc),ended_unix_time=time.time())
        status_path.write_text(json.dumps(state,indent=2))
        raise


if __name__=='__main__': main()
