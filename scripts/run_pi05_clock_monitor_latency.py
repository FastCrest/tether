"""Serial monitored benchmarks, preserving previous HIGH results."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/tether');REPO=Path(__file__).resolve().parents[1];OUT=ROOT/'evaluation/pi05-fast-high-clock-monitor';OUT.mkdir(parents=True,exist_ok=True)
state={'status':'starting','pid':os.getpid(),'started':time.time()}
def save():(OUT/'status.json').write_text(json.dumps(state,indent=2))
save()
try:
 with (ROOT/'evaluation_gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  env=os.environ.copy();env.update(PI05_FAST_OUTPUT=str(OUT),PI05_BENCH_ITERS='400')
  for arm in ['teacher','student']:
   if Path('/sys/class/drm/card1/device/power_dpm_force_performance_level').read_text().strip()!='high':raise RuntimeError('GPU no longer HIGH')
   state.update(status='benchmarking',arm=arm);save()
   cmd=[str(ROOT/'rocm-venv/bin/python'),str(REPO/'scripts/monitor_pi05_fast_clocks.py'),'--interval','0.2','--csv',str(OUT/f'{arm}-clocks.csv'),'--',str(ROOT/'rocm-venv/bin/python'),str(REPO/'scripts/benchmark_pi05_fast_checkpoints.py'),'--arm',arm]
   with (OUT/f'{arm}.log').open('w') as stream:subprocess.run(cmd,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
  state.update(status='complete',ended=time.time());save()
except BaseException as exc:
 state.update(status='failed',error=str(exc));save();raise
