"""Pack per-checkpoint INT4 weights and run fast-library benchmarks serially."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/tether');REPO=Path(__file__).resolve().parents[1];FAST=Path('/home/haojiema/code/amd_robot/models/pi05-pytorch-fast');PY=str(ROOT/'rocm-venv/bin/python');OUT=Path(os.environ.get('PI05_FAST_OUTPUT',str(ROOT/'evaluation/pi05-fast-checkpoint-latency')))
OUT.mkdir(parents=True,exist_ok=True)
state={'status':'starting','pid':os.getpid(),'started':time.time()}
def save():(OUT/'status.json').write_text(json.dumps(state,indent=2))
def run(cmd,log):
 with (OUT/log).open('w') as stream:subprocess.run(cmd,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True)
try:
 with (ROOT/'evaluation_gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for arm in ['teacher','student']:
   path=ROOT/'checkpoints/pi05-libero90mix-torch' if arm=='teacher' else ROOT/'runs/pi05-snapflow-10k/training/checkpoints/00010000/pretrained_model'
   pack=ROOT/'checkpoints'/f'pi05-fast-{arm}-w4a4'
   state.update(status='packing',arm=arm);save()
   if not (pack/'meta.json').exists():run([PY,str(FAST/'scripts/pack_w4a4_prefix.py'),'--model',str(path),'--out',str(pack)],f'{arm}-pack.log')
   state.update(status='benchmarking');save()
   run([PY,str(REPO/'scripts/benchmark_pi05_fast_checkpoints.py'),'--arm',arm],f'{arm}.log')
  results=[json.loads((OUT/f'{a}.json').read_text()) for a in ['teacher','student']]
  lines=['# pi05-pytorch-fast checkpoint latency','','| Model | Mean ms | P50 ms | n |','|---|---:|---:|---:|']
  for r in results:lines.append(f'| {r["arm"]} | {r["mean_ms"]:.2f} | {r["p50_ms"]:.2f} | {r["n"]} |')
  lines+=['','Synthetic bench_latency.py with checkpoint-loader adapter. W4A4 prefix LM / FP16 prefix / BF16 vision / compile max-autotune-no-cudagraphs / ATEN. Teacher10steps, student1step. Language200/present52, cache off. No clock locking. Quantized success rate not tested.']
  (OUT/'report.md').write_text('\n'.join(lines)+'\n');state.update(status='complete',ended=time.time());save()
except BaseException as exc:
 state.update(status='failed',error=str(exc));save();raise
