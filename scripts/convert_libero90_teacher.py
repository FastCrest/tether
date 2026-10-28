"""Convert Orbax pi05 with official OpenPI slicing into LeRobot weights."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


def main():
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from flax import traverse_util
    import torch
    from safetensors.torch import save_file
    from safetensors import safe_open
    root = Path('/data/tether')
    source = root/'checkpoints/pi05-libero90mix/pi05_l90mix_pretrain_bs64_lr2.5e-5_10k/9999'
    output = root/'checkpoints/pi05-libero90mix-torch'
    converter = root/'deps/openpi-converter/convert_jax_model_to_pytorch.py'
    # Reuse upstream's two weight conversion functions unchanged; avoid its
    # OpenPI runtime imports and randomly initialized, unused expert LM head.
    tree = ast.parse(converter.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in ('slice_paligemma_state_dict', 'slice_gemma_state_dict')]
    if len(functions) != 2:
        raise RuntimeError('Upstream converter functions changed')
    namespace = {'np': np, 'torch': torch}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(converter),'exec'),namespace)
    print('Restoring Orbax arrays on CPU',flush=True)
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(source/'params')
        metadata = getattr(metadata, 'item_metadata', metadata)
        item = {'params':metadata['params']}
        raw = ckptr.restore(source/'params',ocp.args.PyTreeRestore(item=item,
            restore_args=jax.tree.map(lambda _:ocp.ArrayRestoreArgs(restore_type=np.ndarray,dtype=np.float32),item)))['params']
    flat = traverse_util.flatten_dict(raw)
    if all(k[-1]=='value' for k in flat):
        flat = {k[:-1]:v for k,v in flat.items()}
    params = traverse_util.unflatten_dict(flat)
    pg = SimpleNamespace(
        vision_config=SimpleNamespace(hidden_size=1152,num_hidden_layers=27,num_attention_heads=16,
                                      intermediate_size=4304,patch_size=14,projection_dim=2048),
        text_config=SimpleNamespace(hidden_size=2048,num_hidden_layers=18,num_attention_heads=8,
                                    head_dim=256,intermediate_size=16384))
    expert = SimpleNamespace(hidden_size=1024,num_hidden_layers=18,num_attention_heads=8,head_dim=256)
    p,e = namespace['slice_paligemma_state_dict'](traverse_util.flatten_dict(params['PaliGemma'],sep='/'),pg)
    e = namespace['slice_gemma_state_dict'](e,expert,num_expert=1,checkpoint_dir=str(source),pi05=True)
    weights = {**p,**e}
    for key in ['action_in_proj','action_out_proj','time_mlp_in','time_mlp_out']:
        weights[key+'.weight']=torch.from_numpy(params[key]['kernel']).T
        weights[key+'.bias']=torch.from_numpy(params[key]['bias'])
    weights = {'model.'+k:v.to(torch.bfloat16).contiguous() for k,v in weights.items()}
    template = root/'checkpoints/pi05_libero_finetuned_v044'
    with safe_open(template/'model.safetensors',framework='pt') as f:
        expected = {k:tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    # LeRobot saves the tied PaliGemma embedding under its LM-head name.
    embedding = 'model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
    if embedding in weights and embedding not in expected:
        weights['model.paligemma_with_expert.paligemma.lm_head.weight']=weights.pop(embedding)
    unused = 'model.paligemma_with_expert.gemma_expert.lm_head.weight'
    missing = set(expected)-set(weights)
    unexpected = set(weights)-set(expected)
    if missing != {unused} or unexpected:
        raise RuntimeError(f'Unexpected key mismatch: missing={missing}, extra={unexpected}')
    # This output head is never called by flow matching. Avoid copying old
    # trained weights: explicitly initialize it to zero and record that fact.
    weights[unused]=torch.zeros(expected[unused],dtype=torch.bfloat16)
    for key,value in weights.items():
        if tuple(value.shape)!=expected[key] or not torch.isfinite(value).all():
            raise RuntimeError(f'Invalid converted tensor {key}: {value.shape}, expected {expected[key]}')
    output.mkdir(exist_ok=True)
    save_file(weights,str(output/'model.safetensors'))
    stats=json.loads((source/'assets/libero90mix/norm_stats.json').read_text())['norm_stats']
    config=json.loads((template/'config.json').read_text())
    config.update(compile_model=False,push_to_hub=False,pretrained_path=str(output),repo_id=None)
    config['normalization_mapping']={'VISUAL':'IDENTITY','STATE':'QUANTILES','ACTION':'QUANTILES'}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    norm={f'{feature}.{key}':torch.tensor(value,dtype=torch.float32)
          for feature,group in [('observation.state','state'),('action','actions')]
          for key,value in stats[group].items()}
    for name in ['policy_preprocessor.json','policy_postprocessor.json']:
        processor=json.loads((template/name).read_text())
        for step in processor['steps']:
            if step['registry_name'] in ['normalizer_processor','unnormalizer_processor']:
                step['config']['norm_map'].update(STATE='QUANTILES',ACTION='QUANTILES')
                save_file(norm,str(output/step['state_file']))
            if step['registry_name']=='tokenizer_processor':
                step['config']['tokenizer_name']=str(root/'tokenizers/paligemma-3b-pt-224')
        (output/name).write_text(json.dumps(processor,indent=2))
    evidence={'source_repo':'aayushis196/pi05-libero90mix-taco-checkpoints',
        'source_revision':'af13ddd7bbd028f81cb52645ae676efba2a98f3b','source_checkpoint':str(source),
        'converter':str(converter),'converter_sha256':hashlib.sha256(converter.read_bytes()).hexdigest(),
        'openpi_revision':(root/'deps/openpi-converter/revision.txt').read_text().strip(),
        'unused_zero_initialized_parameter':unused,'tensor_count':len(weights),
        'normalization':'OpenPI pi05 quantiles from source checkpoint; no old teacher statistics',
        'output':str(output)}
    (output/'conversion.json').write_text(json.dumps(evidence,indent=2))
    print('CONVERSION_COMPLETE',output,flush=True)


if __name__=='__main__':main()
