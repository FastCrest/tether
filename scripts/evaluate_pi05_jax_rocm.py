"""Evaluate the original Orbax teacher using native OpenPI JAX on ROCm."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path('/data/tether')
OPENPI=ROOT/'deps/openpi-jax'
for path in [OPENPI/'src',OPENPI/'packages/openpi-client/src',ROOT/'deps/LIBERO']:
    sys.path.insert(0,str(path))
# Reuse simulator dependencies only as fallback; conda's JAX remains first.
sys.path.append(str(ROOT/'rocm-venv/lib/python3.12/site-packages'))
os.environ.setdefault('JAX_PLATFORMS','rocm')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('XLA_FLAGS','--xla_disable_hlo_passes=triton-softmax-rewriter --xla_gpu_enable_triton_gemm=false')
os.environ.setdefault('OPENPI_DATA_HOME',str(ROOT/'openpi-cache'))
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'libero-config'))
SOURCE=ROOT/'checkpoints/pi05-libero90mix/pi05_l90mix_pretrain_bs64_lr2.5e-5_10k/9999'
if os.environ.get('TETHER_JAX_CPUS'):
    os.sched_setaffinity(0,{int(x) for x in os.environ['TETHER_JAX_CPUS'].split(',')})


def load_initial(task):
    import torch
    from libero.libero import get_libero_path
    # These are the official LIBERO simulator state files already validated
    # locally, rather than model weights downloaded from an arbitrary repo.
    path=Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
    return torch.load(path,weights_only=False)


def build_policy():
    import jax
    import jax.numpy as jnp
    import numpy as np
    import orbax.checkpoint as ocp
    from flax import traverse_util
    from openpi.models.pi0_config import Pi0Config
    from openpi.models import model as model_module
    from openpi.policies.policy import Policy
    from openpi.policies.libero_policy import LiberoInputs,LiberoOutputs
    from openpi import transforms
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared.normalize import NormStats
    assert any(d.platform=='gpu' for d in jax.devices()), 'ROCm GPU is required'
    print('JAX devices:',jax.devices(),flush=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        meta=checkpointer.metadata(SOURCE/'params')
        meta=getattr(meta,'item_metadata',meta)
        item={'params':meta['params']}
        raw=checkpointer.restore(SOURCE/'params',ocp.args.PyTreeRestore(item=item,
            restore_args=jax.tree.map(lambda _:ocp.ArrayRestoreArgs(restore_type=np.ndarray,dtype=np.float32),item)))['params']
    flat=traverse_util.flatten_dict(raw)
    if all(k[-1]=='value' for k in flat):flat={k[:-1]:v for k,v in flat.items()}
    params=traverse_util.unflatten_dict(flat)
    params=jax.tree.map(lambda x:jnp.asarray(x,dtype=jnp.bfloat16),params)
    model=Pi0Config(pi05=True).load(params)
    rawstats=json.loads((SOURCE/'assets/libero90mix/norm_stats.json').read_text())['norm_stats']
    stats={k:NormStats(**{name:np.array(value) for name,value in v.items()}) for k,v in rawstats.items()}
    return Policy(model,transforms=[LiberoInputs(model_module.ModelType.PI05),
        transforms.Normalize(stats,use_quantiles=True),transforms.ResizeImages(224,224),
        transforms.TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=True),
        transforms.PadStatesAndActions(32)],output_transforms=[
        transforms.Unnormalize(stats,use_quantiles=True),LiberoOutputs()],sample_kwargs={'num_steps':10})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--tasks',default='',help='Comma-separated task indices; default all 90')
    parser.add_argument('--worker',action='store_true',help='Parent owns the GPU evaluation lock')
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/libero90-jax')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    status=args.output/'status.json'
    state={'status':'initializing','pid':os.getpid(),'checkpoint':str(SOURCE),'started':time.time()}
    def save():status.write_text(json.dumps(state,indent=2))
    save()
    try:
        import fcntl
        # Full evaluations cooperate with the existing PyTorch GPU queue.
        lock=(ROOT/'evaluation_gpu.lock').open('a')
        if not args.smoke and not args.worker:
            state['status']='waiting_for_gpu_slot';save()
            print('Waiting for existing evaluation to release GPU slot',flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
        policy=build_policy()
        import numpy as np
        import torch
        import jax
        def seed_all(seed):
            random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        def noise(seed):
            seed_all(seed)
            return torch.randn((1,50,32),device='cuda',dtype=torch.float32).cpu().numpy()
        obs={'observation/image':np.zeros((224,224,3),dtype=np.uint8),
             'observation/wrist_image':np.zeros((224,224,3),dtype=np.uint8),
             'observation/state':np.zeros(8,dtype=np.float32),'prompt':'close the top drawer of the cabinet'}
        action=policy.infer(obs,noise=noise(7001))['actions']
        assert action.shape==(50,7) and np.isfinite(action).all()
        print('JAX_POLICY_SMOKE_PASS',action.shape,flush=True)
        if args.smoke:
            from libero.libero import benchmark,get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
            suite=benchmark.get_benchmark_dict()['libero_90']()
            task=suite.get_task(0)
            env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
            try:
                env.reset();env.set_init_state(load_initial(task)[0])
                env.step([0,0,0,0,0,0,-1])
            finally:env.close()
            state.update(status='smoke_pass',jax=jax.__version__,device=str(jax.devices()[0]));save();return
        import cv2
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        suite=benchmark.get_benchmark_dict()['libero_90']()
        selected=[int(x) for x in args.tasks.split(',')] if args.tasks else list(range(90))
        if len(set(selected))!=len(selected) or any(x<0 or x>=90 for x in selected):
            raise ValueError('Invalid task selection')
        state.update(status='evaluating',episodes=0,successes=0,jax=jax.__version__,device=str(jax.devices()[0]),
            protocol='Local PyTorch comparison: seed 7001, 5 initial states, INTER_AREA, replan 5, wait 10, max 400; native OpenPI JAX; matched torch initial noise',
            total_episodes=len(selected)*5);save()
        outcomes=[]
        def call_seed(task,ep,call):
            return int.from_bytes(hashlib.sha256(f'7001:{task}:{ep}:{call}'.encode()).digest()[:8],'big')%(2**31)
        for task_idx in selected:
            path=args.output/f'libero_90-task-{task_idx}.json'
            task=suite.get_task(task_idx)
            seed_all(7001)
            env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
            initial=load_initial(task)
            result={'suite':'libero_90','task_idx':task_idx,'description':task.language,'episodes':[],
                    'checkpoint':str(SOURCE),'backend':'JAX ROCm','seed':7001,'max_steps':400,'replan_steps':5}
            print(f'TASK {task_idx}/89: {task.language}',flush=True)
            try:
                for ep in range(5):
                    env.reset();observation=env.set_init_state(initial[ep]);plan=collections.deque()
                    seed_all(call_seed(task_idx,ep,-1));done=False;calls=0
                    for step in range(410):
                        if step<10:
                            observation,_,done,_=env.step([0,0,0,0,0,0,-1]);continue
                        if not plan:
                            q=np.asarray(observation['robot0_eef_quat']).copy();q[3]=np.clip(q[3],-1,1)
                            den=np.sqrt(1-q[3]**2)
                            aa=np.zeros(3) if np.isclose(den,0) else q[:3]*2*np.arccos(q[3])/den
                            data={'observation/image':cv2.resize(observation['agentview_image'][::-1,::-1],(224,224),interpolation=cv2.INTER_AREA),
                                'observation/wrist_image':cv2.resize(observation['robot0_eye_in_hand_image'][::-1,::-1],(224,224),interpolation=cv2.INTER_AREA),
                                'observation/state':np.concatenate([observation['robot0_eef_pos'],aa,observation['robot0_gripper_qpos']]).astype(np.float32),
                                'prompt':task.language}
                            chunk=policy.infer(data,noise=noise(call_seed(task_idx,ep,calls)))['actions'];calls+=1
                            if not np.isfinite(chunk).all():raise RuntimeError('Non-finite JAX actions')
                            plan.extend(chunk[:5])
                        observation,_,done,_=env.step(plan.popleft().tolist())
                        if done:break
                    result['episodes'].append({'ep':ep,'success':bool(done),'steps':step+1,'inference_calls':calls})
                    state['episodes']+=1;state['successes']+=int(done);state.update(task=task_idx,episode=ep);save()
                    print(f'  ep {ep}: {"✓" if done else "✗"} steps={step+1}; total {state["successes"]}/{state["episodes"]}',flush=True)
                path.write_text(json.dumps(result,indent=2));outcomes.append(result)
            finally:env.close()
        state.update(status='complete',success_rate_pct=state['successes']/state['total_episodes']*100,ended=time.time());save()
        (args.output/'summary.json').write_text(json.dumps(state,indent=2))
        (args.output/'report.md').write_text(f'# Native JAX LIBERO-90 evaluation\n\nSuccess: **{state["successes"]}/{state["total_episodes"]} ({state["success_rate_pct"]:.1f}%)**.\n\nSame source checkpoint and local evaluation protocol as the 76.2% PyTorch teacher. JAX {jax.__version__}, ROCm GPU.\n')
        print('COMPLETE',state,flush=True)
    except Exception as exc:
        state.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
