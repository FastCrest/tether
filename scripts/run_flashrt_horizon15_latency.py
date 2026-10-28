"""BF16 FlashRT horizon15 long benchmark with stage-separated telemetry."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/tether');REPO=Path(__file__).resolve().parents[1];OUT=ROOT/'evaluation/flashrt-horizon15-latency';OUT.mkdir(parents=True,exist_ok=True)
state={'status':'starting','pid':os.getpid(),'started':time.time()}
def save():(OUT/'status.json').write_text(json.dumps(state,indent=2))
save()
try:
 with (ROOT/'evaluation_gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for arm in ['teacher','student']:
   state.update(status='benchmarking',arm=arm);save()
   env=os.environ.copy();env.update(FLASHRT_BENCH_HORIZON='15',FLASHRT_BENCH_ITERS='400',FLASHRT_BENCH_EAGER_ONLY='1',FLASHRT_BENCH_OUTPUT=str(OUT))
   cmd=[str(ROOT/'rocm-venv/bin/python'),str(REPO/'scripts/monitor_pi05_fast_clocks.py'),'--interval','0.2','--start-marker','^Timed loop eager','--end-marker','^eager ','--csv',str(OUT/f'{arm}-clocks.csv'),'--',str(ROOT/'rocm-venv/bin/python'),str(REPO/'scripts/benchmark_flashrt_pi05_checkpoints.py'),'--arm',arm]
   with (OUT/f'{arm}.log').open('w') as stream:subprocess.run(cmd,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
  state.update(status='complete',ended=time.time());save()
except BaseException as exc:
 state.update(status='failed',error=str(exc));save();raise
