"""Use rocm-scripts' pi05 entry point/harness for our LIBERO checkpoints."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import types

REPO=Path('/home/haojiema/code/rocm/rocm-scripts')
ROOT=Path('/data/tether')
sys.path.insert(0,str(REPO/'test/pytorch'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ.setdefault('HF_HOME',str(ROOT/'hf'))
os.environ.setdefault('HF_LEROBOT_HOME',str(ROOT/'lerobot'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'torch'))
os.environ.setdefault('TORCH_BLAS_PREFER_HIPBLASLT','1')
os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL','1')


def load_policy(path,student):
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    config=PI05Config.from_pretrained(str(path))
    config.compile_model=False
    config.device='meta'
    with torch.device('meta'):
        policy=PI05Policy(config)
        if student:
            from tether.distill.snapflow_pi0_model import enable_snapflow
            enable_snapflow(policy.model)
            policy.to(torch.bfloat16)
    expected=policy.state_dict()
    state=load_file(str(path/'model.safetensors'))
    embedding='model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
    head='model.paligemma_with_expert.paligemma.lm_head.weight'
    if embedding not in state and head in state:state[embedding]=state[head]
    if head not in state and embedding in state:state[head]=state[embedding]
    if set(state)!=set(expected):
        raise RuntimeError(f'Checkpoint keys do not match: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}')
    state={k:v.to(dtype=expected[k].dtype) for k,v in state.items()}
    policy.load_state_dict(state,strict=True,assign=True)
    policy.model.paligemma_with_expert.paligemma.tie_weights()
    # Recreate deterministic nonpersistent buffers skipped by state_dict.
    for name,buffer in list(policy.named_buffers()):
        if buffer.device.type!='meta':continue
        module_name,key=name.rsplit('.',1)
        module=policy.get_submodule(module_name)
        if key=='position_ids':
            value=torch.arange(buffer.shape[-1]).expand(buffer.shape)
        elif key=='embed_scale':
            value=torch.tensor(math.sqrt(module.embedding_dim),dtype=buffer.dtype)
        elif key in ['inv_freq','original_inv_freq']:
            fresh=type(module)(module.config,device='cpu')
            value=fresh._buffers[key]
            module.attention_scaling=fresh.attention_scaling
        else:raise RuntimeError(f'Unknown meta buffer {name}')
        module._buffers[key]=value
    config.device='cuda'
    policy.to('cuda').eval()
    policy.model.gradient_checkpointing_disable()
    if student:
        def predict(self,batch,noise=None,**kwargs):
            from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK
            images,masks=self._preprocess_images(batch)
            out=self.model.sample_actions_1step(images,masks,batch[OBS_LANGUAGE_TOKENS],batch[OBS_LANGUAGE_ATTENTION_MASK],noise=noise)
            return out[:,:,:7]
        policy.predict_action_chunk=types.MethodType(predict,policy)
    return policy


def batch_images(model):
    # Port rocm-scripts' batched SigLIP optimization, retaining LeRobot's
    # language embedding scaling (its embedding module already scales it).
    import torch
    def embed_prefix(self,images,img_masks,tokens,masks):
        batch=images[0].shape[0]
        image_embeddings=self.paligemma_with_expert.embed_image(torch.cat(images,dim=0))
        image_embeddings=image_embeddings.reshape(len(images),batch,*image_embeddings.shape[1:])
        embs=[image_embeddings[i] for i in range(len(images))]
        pads=[mask[:,None].expand(batch,image_embeddings.shape[2]) for mask in img_masks]
        language=self.paligemma_with_expert.embed_language_tokens(tokens)
        embs.append(language);pads.append(masks)
        ar=torch.zeros((batch,sum(x.shape[1] for x in embs)),dtype=torch.bool,device=tokens.device)
        return torch.cat(embs,dim=1),torch.cat(pads,dim=1),ar
    model.embed_prefix=types.MethodType(embed_prefix,model)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['teacher','student'],required=True)
    parser.add_argument('--vision-precision',choices=['native','bf16'],default='native')
    parser.add_argument('--compile-mode',default='max-autotune',choices=['max-autotune','eager'])
    args,remaining=parser.parse_known_args()
    sys.argv=[str(REPO/'test/pytorch/openpi_pi05.py')]+remaining
    import _openpi_setup
    # ROCm 10.1's expandable allocator invalidated stream capture on this
    # workload. Keep normal caching allocation for the graph-enabled retry.
    os.environ['PYTORCH_HIP_ALLOC_CONF']='expandable_segments:False'
    from torch_test import benchmark,run
    def adapted_benchmark():
        import statistics
        import time
        import torch
        from torch.utils.data import default_collate
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition,transition_to_policy_action
        from tether.finetune.backends.snapflow_backend import _build_preprocessor
        torch.set_float32_matmul_precision('high')
        path=ROOT/'checkpoints/pi05-libero90mix-torch' if args.arm=='teacher' else ROOT/'runs/pi05-snapflow-10k/training/checkpoints/00010000/pretrained_model'
        started=time.perf_counter();policy=load_policy(path,args.arm=='student')
        if args.vision_precision=='bf16':
            # Controlled benchmark-only change: leave language/expert/projections
            # untouched, change only the vision encoder and multimodal projector.
            gemma=policy.model.paligemma_with_expert
            gemma.paligemma.model.vision_tower.to(torch.bfloat16)
            gemma.paligemma.model.multi_modal_projector.to(torch.bfloat16)
            def embed_image(self,image):
                dtype=self.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
                return self.paligemma.model.get_image_features(image.to(dtype)).pooler_output.to(image.dtype)
            gemma.embed_image=types.MethodType(embed_image,gemma)
        precision_counts={}
        for name,param in policy.named_parameters():
            key=str(param.dtype)
            precision_counts[key]=precision_counts.get(key,0)+param.numel()
        pre=_build_preprocessor(teacher_path=ROOT/'checkpoints/pi05-libero90mix-torch',image_key_map=None,device='cuda')
        post=PolicyProcessorPipeline.from_pretrained(str(ROOT/'checkpoints/pi05-libero90mix-torch'),config_filename='policy_postprocessor.json',to_transition=policy_action_to_transition,to_output=transition_to_policy_action)
        dataset=LeRobotDataset('lerobot/libero',root=ROOT/'lerobot/lerobot/libero',revision='a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4',video_backend='pyav')
        indices=[157619,169671,175046,50588,256441,269680,210952,267012,50997,202042]
        raw=[default_collate([dataset[i]]) for i in indices]
        prepared=[pre(copy.deepcopy(x)) for x in raw]
        torch.manual_seed(7001);noise=torch.randn((1,50,32),device='cuda',dtype=torch.float32)
        with torch.inference_mode():before=policy.predict_action_chunk(prepared[0],noise=noise).clone()
        batch_images(policy.model)
        with torch.inference_mode():after=policy.predict_action_chunk(prepared[0],noise=noise).clone()
        delta=float((before-after).abs().max())
        if not torch.isfinite(after).all():raise RuntimeError('Invalid batched image output')
        print('Batched image action max abs difference:',delta,flush=True)
        if args.compile_mode!='eager':
            method='sample_actions_1step' if args.arm=='student' else 'sample_actions'
            setattr(policy.model,method,torch.compile(getattr(policy.model,method),mode=args.compile_mode))
        last=[None];counter=[0]
        @torch.inference_mode()
        def core():
            torch.compiler.cudagraph_mark_step_begin()
            last[0]=policy.predict_action_chunk(prepared[counter[0]%10],noise=noise)
            counter[0]+=1
        @torch.inference_mode()
        def pipeline():
            torch.compiler.cudagraph_mark_step_begin()
            batch=pre(copy.deepcopy(raw[counter[0]%10]))
            action=policy.predict_action_chunk(batch,noise=noise)
            last[0]=post(action.detach().cpu());counter[0]+=1
        print('Benchmark pure model',args.arm,args.compile_mode,flush=True)
        # The stock time-based warmup may stop after only two calls when the
        # first call compiles. Warm every observation before sampling latency.
        for _ in range(20):core()
        torch.cuda.synchronize()
        metrics=benchmark(core,min_iter=50,max_iter=200)
        with torch.inference_mode():compiled=policy.predict_action_chunk(prepared[0],noise=noise).clone()
        if not torch.isfinite(compiled).all():raise RuntimeError('Invalid compiled output')
        compile_delta=float((after-compiled).abs().max())
        print('Benchmark full preprocessing/model/postprocessing',flush=True)
        full=benchmark(pipeline,min_iter=50,max_iter=200)
        return metrics,{'model':args.arm+'_pi05_libero','architecture':'VLA','checkpoint':str(path),
            'action_shape':[50,7],'batch_size':1,'compile_mode':args.compile_mode,'sanity_check':'PASS',
            'cudagraphs_enabled':'--no-cudagraphs' not in remaining,
            'allocator_config':os.environ['PYTORCH_HIP_ALLOC_CONF'],
            'denoise_steps':1 if args.arm=='student' else 10,'batched_siglip':True,
            'vision_precision_override':args.vision_precision,'parameter_dtype_counts':precision_counts,
            'weight_precision':'native_mixed_with_bf16_vision' if args.vision_precision=='bf16' and args.arm=='teacher' else 'all_bfloat16' if args.arm=='student' else 'native_mixed_bfloat16_float32_vision',
            'batch_image_max_abs_action_difference':delta,'compile_max_abs_action_difference':compile_delta,
            'pure_model_samples_ms':metrics.latency_ms,'pipeline_samples_ms':full.latency_ms,
            'pipeline_mean_ms':statistics.mean(full.latency_ms),'pipeline_p50_ms':statistics.median(full.latency_ms),
            'timing':'rocm-scripts torch_test.benchmark: GPU-synchronized wall time; pure model excludes external preprocessing, RNG and finite checks; pipeline includes preprocessing/model/postprocessing',
            'entry_script':str(REPO/'test/pytorch/openpi_pi05.py'),
            'harness_sha256':hashlib.sha256((REPO/'test/pytorch/torch_test.py').read_bytes()).hexdigest(),
            'adapter':str(Path(__file__).resolve()),'torch':torch.__version__,'hip':torch.version.hip,
            'input_frame_indices':indices,'total_setup_and_benchmark_s':time.perf_counter()-started}
    # Keep the repository entry point and its timing harness; replace only the
    # hardcoded DROID loader/workload with our supplied LIBERO checkpoints.
    _openpi_setup.main=lambda **kwargs:sys.exit(run(adapted_benchmark))
    runpy.run_path(str(REPO/'test/pytorch/openpi_pi05.py'),run_name='__main__')


if __name__=='__main__':main()
