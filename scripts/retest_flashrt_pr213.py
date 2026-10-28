import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/tether');REPO=Path(__file__).resolve().parents[1];OUT=ROOT/'evaluation/flashrt-pr213-retest';OUT.mkdir(parents=True,exist_ok=True)
state={'status':'starting','pid':os.getpid(),'started':time.time()}
def save():(OUT/'status.json').write_text(json.dumps(state,indent=2))
save()
try:
 with (ROOT/'evaluation_gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  cases=[('final-teacher50','final','teacher',50),('final-teacher15','final','teacher',15),('original-teacher15','original','teacher',15),('original-teacher50','original','teacher',50),('final-student50','final','student',50)]
  for tag,version,arm,horizon in cases:
   state.update(status='benchmarking',case=tag);save()
   env=os.environ.copy();env.update(FLASHRT_BENCH_ROOT='/data/tether/deps/flashrt-pr213-original' if version=='original' else '/home/haojiema/code/amd_robot/ref_arch/inference_engine/flashrt',FLASHRT_BENCH_OUTPUT=str(OUT),FLASHRT_BENCH_TAG=tag,FLASHRT_BENCH_HORIZON=str(horizon))
   with (OUT/f'{tag}.log').open('w') as stream:subprocess.run([str(ROOT/'rocm-venv/bin/python'),str(REPO/'scripts/benchmark_flashrt_pi05_checkpoints.py'),'--arm',arm],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
  state.update(status='complete',ended=time.time());save()
except BaseException as exc:
 state.update(status='failed',error=str(exc));save();raise
