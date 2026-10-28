"""Measure actual policy latency and run matched LIBERO rollouts on ROCm."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import types
import copy
from contextlib import contextmanager
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path('/data/tether')
TEACHER = Path(os.environ.get('TETHER_TEACHER_CHECKPOINT', str(ROOT/'checkpoints/pi05_libero_finetuned_v044-local')))
os.environ.setdefault('HF_HOME', str(ROOT / 'hf'))
os.environ.setdefault('HF_LEROBOT_HOME', str(ROOT / 'lerobot'))
os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT / 'libero-config'))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('TMPDIR', str(ROOT / 'tmp'))
(ROOT / 'tmp').mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def concurrent_gpu_jobs():
    jobs=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid(): continue
        try: argv=(process/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        evaluator=any(arg.endswith(b'evaluate_pi05_distill_rocm.py') for arg in argv)
        trainer=any(arg.endswith(b'run_pi05_distill_rocm.py') for arg in argv)
        if trainer or (evaluator and b'eval' in argv): jobs.append(int(process.name))
    return jobs


@contextmanager
def evaluation_slot(mode):
    import fcntl
    if mode=='benchmark':
        while True:
            jobs=concurrent_gpu_jobs()
            if not jobs: break
            print(f'Waiting for GPU training/evaluation jobs before latency benchmark: {jobs}',flush=True)
            time.sleep(30)
    # Evaluation may overlap training, but policy evaluations and latency
    # measurements from this tool must not overlap each other.
    with (ROOT/'evaluation_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(lock,fcntl.LOCK_UN)


def load_policy(checkpoint, student=False):
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05 import processor_pi05  # register processors
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from tether.finetune.backends.snapflow_backend import _build_preprocessor
    if student:
        from tether.distill.snapflow_pi0_model import load_snapflow_student
        policy = load_snapflow_student(checkpoint).to('cuda').to(torch.bfloat16).eval()
        def predict(self, batch, noise=None, **kwargs):
            from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
            images, masks = self._preprocess_images(batch)
            out = self.model.sample_actions_1step(images, masks, batch[OBS_LANGUAGE_TOKENS],
                                                batch[OBS_LANGUAGE_ATTENTION_MASK], noise=noise)
            return out[:, :, :self.config.output_features['action'].shape[0]]
        policy.predict_action_chunk = types.MethodType(predict, policy)
    else:
        # Preserve the native checkpoint's bf16 backbone and fp32 auxiliary
        # projections. The native denoiser explicitly casts its output to fp32.
        policy = PI05Policy.from_pretrained(checkpoint).to('cuda').eval()
        policy.config.num_inference_steps = 10
    disable = getattr(policy.model, 'gradient_checkpointing_disable', None)
    if disable: disable()
    denoise = policy.model.denoise_step
    policy.model._measured_denoise_calls = 0
    def counted_denoise(self, *args, **kwargs):
        self._measured_denoise_calls += 1
        return denoise(*args, **kwargs)
    policy.model.denoise_step = types.MethodType(counted_denoise, policy.model)
    policy._expected_denoise_calls = 1 if student else 10
    pre = _build_preprocessor(teacher_path=TEACHER,
                              image_key_map=None, device='cuda')
    post = PolicyProcessorPipeline.from_pretrained(
        str(TEACHER),
        config_filename='policy_postprocessor.json',
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return policy, pre, post


def measured_predict(policy):
    import torch
    original = policy.predict_action_chunk
    def predict(self, *args, **kwargs):
        before = self.model._measured_denoise_calls
        torch.cuda.synchronize()
        with torch.inference_mode(): out = original(*args, **kwargs)
        torch.cuda.synchronize()
        calls = self.model._measured_denoise_calls-before
        if calls != self._expected_denoise_calls:
            raise RuntimeError(f'Expected {self._expected_denoise_calls} denoise calls, got {calls}')
        if not torch.isfinite(out).all(): raise RuntimeError('Non-finite policy actions')
        return out
    policy.predict_action_chunk = types.MethodType(predict, policy)


def benchmark(args):
    import torch
    import numpy as np
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset('lerobot/libero', root=ROOT / 'lerobot/lerobot/libero',
                              revision='a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4', video_backend='pyav')
    indices = np.random.default_rng(7001).choice(len(dataset), size=10, replace=False).tolist()
    batches = [default_collate([dataset[i]]) for i in indices]
    teacher = str(TEACHER)
    results = {}
    for arm, checkpoint, student in [('teacher_10step', teacher, False), ('student_1step', args.checkpoint, True)]:
        policy, pre, post = load_policy(checkpoint, student)
        measured_predict(policy)
        torch.cuda.reset_peak_memory_stats()
        observations = [pre(copy.deepcopy(b)) for b in batches]
        timings = []
        pipeline = []
        for i in range(args.warmup + args.samples):
            batch = observations[i % len(observations)]
            torch.manual_seed(7001 + i)
            noise = torch.randn(1, 50, 32, device='cuda', dtype=torch.float32)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode(): action = policy.predict_action_chunk(batch, noise=noise)
            torch.cuda.synchronize()
            latency = (time.perf_counter()-start)*1000
            if not torch.isfinite(action).all(): raise RuntimeError('Non-finite actions during benchmark')
            raw_batch = copy.deepcopy(batches[i % len(batches)])
            torch.cuda.synchronize()
            start = time.perf_counter()
            processed = pre(raw_batch)
            with torch.inference_mode(): out = policy.predict_action_chunk(processed, noise=noise)
            post(out.detach().cpu())
            torch.cuda.synchronize()
            pipeline_latency = (time.perf_counter()-start)*1000
            if i >= args.warmup:
                timings.append(latency)
                pipeline.append(pipeline_latency)
        def stats(values):
            return {'mean_ms': float(np.mean(values)), 'p50_ms':float(np.median(values)),
                    'p95_ms':float(np.percentile(values,95)), 'std_ms':float(np.std(values)), 'samples_ms':values}
        results[arm] = dict(checkpoint=checkpoint, policy=stats(timings),
                            pipeline=stats(pipeline), denoise_calls_per_chunk=policy._expected_denoise_calls,
                            peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        del policy, pre, post, observations
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    results.update(warmup=args.warmup, measured_samples=args.samples, batch_size=1,
                   input_frame_indices=indices, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                   policy_speedup=results['teacher_10step']['policy']['p50_ms']/results['student_1step']['policy']['p50_ms'],
                   pipeline_speedup=results['teacher_10step']['pipeline']['p50_ms']/results['student_1step']['pipeline']['p50_ms'],
                   timing='Host wall clock with GPU synchronization; includes VLM prefix once per chunk, no cross-chunk cache.')
    (args.output / 'latency.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({k:v for k,v in results.items() if k.endswith('speedup')}, indent=2))


def resolve_task_indices(task_count, selected):
    indices=[int(t) for t in selected.split(',')] if selected else list(range(task_count))
    if len(indices)!=len(set(indices)) or any(i<0 or i>=task_count for i in indices):
        raise ValueError(f'Task indices must be unique and in [0, {task_count})')
    return indices


def evaluate(args):
    from tether.eval.libero_rollout import run_libero_rollout
    from libero.libero import benchmark as libero_benchmark
    policy, pre, post = load_policy(args.checkpoint, args.student)
    measured_predict(policy)
    for suite in args.suites.split(','):
        task_count=libero_benchmark.get_benchmark_dict()[suite]().n_tasks
        task_indices = resolve_task_indices(task_count,args.tasks)
        # Persist one task at a time so a later failure cannot erase completed episodes.
        for task in task_indices:
            raw = run_libero_rollout(policy=policy, preprocessor=pre, postprocessor=post,
                task_suite_name=suite, task_indices=[task], num_episodes=args.episodes,
                seed=args.seed, use_native=True, verification_device='cuda',
                save_video_dir=str(args.output / 'videos') if args.video else '',
                label='student_1step' if args.student else 'teacher_10step')
            raw.update(checkpoint=args.checkpoint, student=args.student, inference_steps=1 if args.student else 10,
                       requested_seed=args.seed, requested_episodes=args.episodes,
                       denoise_calls_per_chunk_verified=not bool(raw['errors']))
            (args.output / f'{suite}-task-{task}.json').write_text(json.dumps(raw, indent=2))
            if raw['errors']: raise RuntimeError(f'LIBERO evaluation errors: {raw["errors"]}')
    if (args.student and args.suites=='libero_90' and not args.tasks
            and args.episodes==5 and args.seed==7001):
        from report_pi05_distill import summarize_libero90
        summary=summarize_libero90(args.output,args.checkpoint)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(f'LIBERO-90 student: {summary["success"]}/450 = {summary["success_rate_pct"]:.1f}%')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['benchmark','eval'])
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--student',action='store_true')
    p.add_argument('--suites',default='libero_spatial,libero_object,libero_goal,libero_10')
    p.add_argument('--tasks',default='')
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--seed',type=int,default=7001)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--samples',type=int,default=50)
    p.add_argument('--video',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with evaluation_slot(args.mode):
        benchmark(args) if args.mode=='benchmark' else evaluate(args)
