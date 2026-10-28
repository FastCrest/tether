import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/tether');OUT=ROOT/'evaluation/flashrt-checkpoint-latency';REPO=Path(__file__).resolve().parents[1];OUT.mkdir(parents=True,exist_ok=True)
state={'status':'starting','pid':os.getpid(),'started':time.time()}
def save():(OUT/'status.json').write_text(json.dumps(state,indent=2))
save()
try:
 with (ROOT/'evaluation_gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for arm in ['teacher','student']:
   state.update(status='benchmarking',arm=arm);save()
   with (OUT/f'{arm}.log').open('w') as stream:
    subprocess.run([str(ROOT/'rocm-venv/bin/python'),str(REPO/'scripts/benchmark_flashrt_pi05_checkpoints.py'),'--arm',arm],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True)
  rows=[json.loads((OUT/f'{a}.json').read_text()) for a in ['teacher','student']]
  lines=['# FlashRT BF16 Pi0.5 latency','','| Model | Mode | Mean ms | P50 ms | n |','|---|---|---:|---:|---:|']
  for row in rows:
   for mode,r in row['results'].items():lines.append(f'| {row["arm"]} | {mode} | {r["mean_ms"]:.2f} | {r["p50_ms"]:.2f} | {r["n"]} |')
  lines+=['','Native gfx1151 BF16 pipeline, two views, fixed preprocessed images/prompt/noise, 50x32 internal actions. Teacher10steps; student1step with target-time MLP folded into constant time embedding. No W4A4. Closed-loop parity/success not yet validated.']
  (OUT/'report.md').write_text('\n'.join(lines)+'\n');state.update(status='complete',ended=time.time());save()
except BaseException as exc:
 state.update(status='failed',error=str(exc));save();raise
