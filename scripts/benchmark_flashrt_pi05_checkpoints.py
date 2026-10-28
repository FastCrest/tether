"""Native FlashRT RDNA BF16 latency with SnapFlow time-conditioning adapter."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path('/data/tether');FLASH=Path(os.environ.get('FLASHRT_BENCH_ROOT','/home/haojiema/code/amd_robot/ref_arch/inference_engine/flashrt'));HORIZON=int(os.environ.get('FLASHRT_BENCH_HORIZON','50'));ITERS=int(os.environ.get('FLASHRT_BENCH_ITERS','50'))
sys.path.insert(0,str(FLASH))
os.environ.setdefault('FLASH_RT_PALIGEMMA_TOKENIZER',str(ROOT/'tokenizers/paligemma-3b-pt-224/tokenizer.model'))
os.environ['PYTORCH_HIP_ALLOC_CONF']='expandable_segments:False'
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
import flash_rt.amd.frontends.torch.pi05_rdna35 as frontend
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['teacher','student'],required=True);args=p.parse_args()
path=ROOT/'checkpoints/pi05-libero90mix-torch' if args.arm=='teacher' else ROOT/'runs/pi05-snapflow-10k/training/checkpoints/00010000/pretrained_model'
# SnapFlow target_time=1 is constant during 1-NFE inference. Fold its MLP
# into the static sinusoid table BEFORE the existing time MLP and modulation
# precomputation, preserving native FlashRT runtime kernels.
base=frontend._build_time_embeddings
adapter_diff=None
if args.arm=='student':
    with safe_open(str(path/'model.safetensors'),framework='pt') as sf:
        target={k:sf.get_tensor('model.target_time_embed_mlp.'+k).to('cuda',dtype=torch.bfloat16) for k in ['0.weight','0.bias','2.weight','2.bias']}
    def time_table(steps):
        assert steps==1
        table=base(1).to('cuda')
        with torch.inference_mode():
            delta=F.linear(F.silu(F.linear(table,target['0.weight'],target['0.bias'])),target['2.weight'],target['2.bias'])
            return (table.float()+delta.float()).to(torch.bfloat16).cpu()
    frontend._build_time_embeddings=time_table
print('Loading FlashRT',args.arm,flush=True)
import inspect
kwargs={'num_views':2,'num_steps':1 if args.arm=='student' else 10,'chunk_size':HORIZON,'use_fp8':False}
if 'action_dim' in inspect.signature(frontend.Pi05TorchFrontendAmdRdna35).parameters:kwargs['action_dim']=7
fe=frontend.Pi05TorchFrontendAmdRdna35(path,**kwargs)
fe.set_prompt('pick up the object',state=np.zeros(8,dtype=np.float32))
torch.manual_seed(7001)
images=torch.rand((2,224,224,3),device='cuda',dtype=torch.bfloat16)*2-1
noise=torch.randn((1,HORIZON,32),device='cuda',dtype=torch.bfloat16)
results={}
with torch.inference_mode():
    eager=fe.forward_with_fixed_noise(images,noise,use_graph=False).clone()
    if not torch.isfinite(eager).all():raise RuntimeError('Nonfinite native BF16 output')
    for graph in ([False] if os.environ.get('FLASHRT_BENCH_EAGER_ONLY')=='1' else [False,True]):
        label='graph' if graph else 'eager'
        print('Warmup',label,flush=True)
        for _ in range(10):fe.forward_with_fixed_noise(images,noise,use_graph=graph)
        torch.cuda.synchronize();times=[]
        print('Timed loop',label,flush=True)
        for _ in range(ITERS):
            torch.cuda.synchronize();t=time.perf_counter();out=fe.forward_with_fixed_noise(images,noise,use_graph=graph);torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
        if not torch.isfinite(out).all():raise RuntimeError('Nonfinite measured output')
        results[label]={'mean_ms':statistics.mean(times),'p50_ms':statistics.median(times),'samples_ms':times,'n':ITERS,'max_abs_difference_vs_eager':float((out-eager).abs().max())}
        print(label,results[label]['mean_ms'],results[label]['p50_ms'],flush=True)
result={'arm':args.arm,'checkpoint':str(path),'precision':'BF16 native FlashRT RDNA35','num_steps':fe.num_steps,'views':2,'prompt_len':fe._prompt_len,'flashrt_root':str(FLASH),'chunk_size':HORIZON,'action_dim':7,'target_time_adapter':args.arm=='student','timing':'GPU-synchronized host wall time; fixed preprocessed images/prompt/noise; excludes external pre/post processing','results':results,'torch':torch.__version__,'hip':torch.version.hip,'gpu_performance_level':Path('/sys/class/drm/card1/device/power_dpm_force_performance_level').read_text().strip()}
OUT=Path(os.environ.get('FLASHRT_BENCH_OUTPUT',str(ROOT/'evaluation/flashrt-checkpoint-latency')));OUT.mkdir(parents=True,exist_ok=True);(OUT/(os.environ.get('FLASHRT_BENCH_TAG',args.arm)+'.json')).write_text(json.dumps(result,indent=2))
print('COMPLETE',args.arm,flush=True)
