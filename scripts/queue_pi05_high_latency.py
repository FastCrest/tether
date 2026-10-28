"""Start fast benchmark only after GPU HIGH and CPU performance are observed."""
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/tether');REPO=Path(__file__).resolve().parents[1];OUT=ROOT/'evaluation/pi05-fast-high-latency';OUT.mkdir(parents=True,exist_ok=True)
state={'status':'waiting_for_gpu_high','pid':os.getpid(),'started':time.time()}
try:
 while True:
  gpu=Path('/sys/class/drm/card1/device/power_dpm_force_performance_level').read_text().strip()
  cpus={p.parent.name:p.read_text().strip() for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor')}
  state.update(gpu=gpu,cpu_governors=sorted(set(cpus.values())))
  (OUT/'queue_status.json').write_text(json.dumps(state,indent=2))
  if gpu=='high' and cpus and set(cpus.values())=={'performance'}:break
  time.sleep(5)
 state.update(status='benchmarking');(OUT/'queue_status.json').write_text(json.dumps(state,indent=2))
 env=os.environ.copy();env['PI05_FAST_OUTPUT']=str(OUT)
 with (OUT/'supervisor.log').open('w') as stream:
  subprocess.run([str(ROOT/'rocm-venv/bin/python'),str(REPO/'scripts/run_pi05_fast_checkpoint_latency.py')],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
 state.update(status='complete',ended=time.time(),gpu_after=Path('/sys/class/drm/card1/device/power_dpm_force_performance_level').read_text().strip())
except BaseException as exc:
 state.update(status='failed',error=str(exc));raise
finally:(OUT/'queue_status.json').write_text(json.dumps(state,indent=2))
