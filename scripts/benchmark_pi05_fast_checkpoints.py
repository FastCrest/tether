"""Run pi05-pytorch-fast's synthetic bench with our checkpoint loader."""
import argparse
import json
import os
from pathlib import Path
import runpy
import statistics
import types
import sys

FAST=Path('/home/haojiema/code/amd_robot/models/pi05-pytorch-fast')
ROOT=Path('/data/tether')
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['teacher','student'],required=True);args=p.parse_args()
os.environ.update(PI05_COMPILE='1',PI05_COMPILE_MODE='max-autotune-no-cudagraphs',PI05_PREFIX_LM_FP16='1',PI05_INDUCTOR_GEMM_BACKENDS='ATEN',PI05_LANG_TOKENS='',PI05_CACHE_EMPTY_CAM='0',PI05_BENCH_ITERS=os.environ.get('PI05_BENCH_ITERS','50'),PI05_BENCH_WARMUP=os.environ.get('PI05_BENCH_WARMUP','10'))
os.environ['PI05_W4A4_PACK']=str(ROOT/'checkpoints'/f'pi05-fast-{args.arm}-w4a4')
sys.path.insert(0,str(FAST));sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark_rocm_scripts_pi05 import load_policy
import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
gpu_level_before=Path('/sys/class/drm/card1/device/power_dpm_force_performance_level').read_text().strip()
path=ROOT/'checkpoints/pi05-libero90mix-torch' if args.arm=='teacher' else ROOT/'runs/pi05-snapflow-10k/training/checkpoints/00010000/pretrained_model'
os.environ['POLICY_ID']=str(path);os.environ['POLICY_LOAD']=str(path)
# Prevent source checkpoint mutation and duplicate SnapFlow class installation.
import pi05_fast.checkpoint_compat as compat
import pi05_fast.snapflow_pi05 as snap
compat.sanitize_snapflow_checkpoint=lambda *a,**kw:[]
snap.install_pi05_snapflow_eval_hooks=lambda:None
@classmethod
def loader(cls,*a,**kw):
    policy=load_policy(path,args.arm=='student')
    from pi05_fast.snapflow_pi05 import _resolve_snapflow_pi05_class
    from lerobot.policies.pi05.modeling_pi05 import prepare_attention_masks_4d
    fast_class=_resolve_snapflow_pi05_class()
    policy.model._prepare_attention_masks_4d=prepare_attention_masks_4d
    if args.arm=='student':
        policy.model.__class__=fast_class
    else:
        # Use the fast library's dtype-aware suffix/cache implementation,
        # retaining native ten-step sample_actions and no target-time addition.
        def embed_suffix(self,noisy_actions,timestep,target_time=None):
            return fast_class.embed_suffix(self,noisy_actions.to(self.action_in_proj.weight.dtype),timestep,target_time=target_time)
        policy.model.embed_suffix=types.MethodType(embed_suffix,policy.model)
        policy.model.denoise_step=types.MethodType(fast_class.denoise_step,policy.model)
    policy.config.n_action_steps=10
    policy.config.num_inference_steps=1 if args.arm=='student' else 10
    return policy
PI05Policy.from_pretrained=loader
original=statistics.median
samples=[]
def median(values):
    samples.extend(values)
    return original(values)
statistics.median=median
runpy.run_path(str(FAST/'scripts/bench_latency.py'),run_name='__main__')
if len(samples)!=int(os.environ['PI05_BENCH_ITERS']):raise RuntimeError(f'Unexpected sample count {len(samples)}')
result={'gpu_level_before':gpu_level_before,'gpu_level_after':Path('/sys/class/drm/card1/device/power_dpm_force_performance_level').read_text().strip(),'cpu_governors':sorted({p.read_text().strip() for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor')}),'arm':args.arm,'checkpoint':str(path),'samples_ms':samples,'mean_ms':statistics.mean(samples),'p50_ms':original(samples),'n':len(samples),'torch':torch.__version__,'hip':torch.version.hip,'script':str(FAST/'scripts/bench_latency.py'),'adapter':str(Path(__file__).resolve()),'num_inference_steps':1 if args.arm=='student' else 10,'input':'script synthetic 3 cameras 224x224, language200/present52','environment':{k:v for k,v in os.environ.items() if k.startswith('PI05_')}}
out=Path(os.environ.get('PI05_FAST_OUTPUT',str(ROOT/'evaluation/pi05-fast-checkpoint-latency')));out.mkdir(parents=True,exist_ok=True);(out/f'{args.arm}.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['samples_ms','environment']},indent=2),flush=True)
