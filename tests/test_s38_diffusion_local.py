"""Diffusion joins raw v3 admission; no relaxation for exports or other families."""
from pathlib import Path
import pytest
from tether.finetune.config import FinetuneConfig
from tether.finetune.local_dataset import validate_local_config, LocalDatasetError
from tether.finetune.run import _build_lerobot_command


def config(tmp_path, **changes):
    return FinetuneConfig(base='', policy='diffusion', mode='full', dataset='studio-local/own',
        dataset_root=tmp_path/'data', dataset_content_sha256='a'*64, output=tmp_path/'out', **changes)


def test_raw_diffusion_admission_and_no_act_flags(tmp_path):
    cfg = config(tmp_path)
    validate_local_config(cfg)
    # Command schema without local verification; full verified-byte path is covered by the S-38 real smoke.
    cfg.dataset_root = None; cfg.dataset_content_sha256 = None
    command = _build_lerobot_command(cfg)
    assert '--policy.type=diffusion' in command
    assert not any(arg.startswith('--policy.chunk_size=') for arg in command)

@pytest.mark.parametrize('family', ['sac', 'smolvla', 'wall_x', 'multi_task_dit', 'gaussian_actor'])
def test_unadmitted_family_stays_refused(tmp_path, family):
    cfg = config(tmp_path); cfg.policy = family
    with pytest.raises(LocalDatasetError, match='from scratch'):
        validate_local_config(cfg)


def test_diffusion_does_not_admit_the_smolvla_export_profile(tmp_path):
    cfg = config(tmp_path); cfg.dataset_content_sha256 = None; cfg.dataset_manifest_sha256 = 'b'*64
    with pytest.raises(LocalDatasetError):
        validate_local_config(cfg)


@pytest.mark.parametrize('family', ['vqbet', 'tdmpc'])
def test_local_offline_family_admitted_without_act_chunk(tmp_path, family):
    cfg = config(tmp_path); cfg.policy = family
    validate_local_config(cfg)
    cfg.dataset_root = None; cfg.dataset_content_sha256 = None
    command = _build_lerobot_command(cfg)
    assert f'--policy.type={family}' in command
    assert not any(arg.startswith('--policy.chunk_size=') for arg in command)


def test_local_resume_selects_complete_numeric_checkpoint_and_binds_input(tmp_path, monkeypatch):
    import json
    import tether.finetune.lerobot_v3 as v3
    import tether.finetune.preflight.schema as schema
    import tether.finetune.run as run
    cfg = config(tmp_path); cfg.resume = True
    saved = cfg.output/'training/checkpoints/000500'
    (saved/'pretrained_model').mkdir(parents=True)
    (saved/'training_state').mkdir()
    for name in ('training_step.json','optimizer_state.safetensors','optimizer_param_groups.json','rng_state.safetensors','scheduler_state.json'):
        (saved/'training_state'/name).write_text('{}')
    (saved/'training_state/training_step.json').write_text('{"step":500}')
    data = {'env_eval_freq':0,'policy': {'type':'diffusion'}, 'dataset': {'root':str(cfg.dataset_root)}, 'output_dir':str(cfg.output/'training'),
            'steps':cfg.num_steps,'batch_size':cfg.batch_size,'seed':cfg.seed}
    path = saved/'pretrained_model/train_config.json';path.write_text(json.dumps(data))
    monkeypatch.setattr(run,'verify_local_dataset', lambda cfg: {'dataset_root':str(cfg.dataset_root)})
    from tether.finetune.preflight.result import PreflightCheck
    monkeypatch.setattr(schema,'check_schema',lambda cfg:PreflightCheck('schema','ok','fixture'))
    command = _build_lerobot_command(cfg)
    assert command == ['lerobot-train', f'--config_path={path}', '--resume=true']
    data['batch_size'] += 1;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='resume.batch_size'):_build_lerobot_command(cfg)
    data['batch_size'] -= 1; data['policy']['horizon']=16;path.write_text(json.dumps(data))
    cfg.extra_lerobot_args={'policy.horizon':32}
    with pytest.raises(ValueError,match='resume.policy.horizon'):_build_lerobot_command(cfg)
    cfg.extra_lerobot_args={'policy.horizon':16}
    assert _build_lerobot_command(cfg)==['lerobot-train',f'--config_path={path}','--resume=true']
    data['output_dir']=str(tmp_path/'elsewhere');path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='resume.output_dir'):_build_lerobot_command(cfg)


@pytest.mark.parametrize('reward,ok', [(0.,True),([1.],False),(float('nan'),False)])
def test_tdmpc_refuses_non_scalar_or_nonfinite_stored_rewards(tmp_path,monkeypatch,reward,ok):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tether.finetune.preflight.schema as schema
    cfg=config(tmp_path);cfg.policy='tdmpc'
    data=cfg.dataset_root/'data/chunk-000/file-000.parquet';data.parent.mkdir(parents=True)
    pq.write_table(pa.table({'next.reward':[reward]}),data)
    monkeypatch.setattr(schema,'verify_local_dataset',lambda cfg:{'info':{'features':{
        'observation.state':{'dtype':'float32','shape':[6]},
        'action':{'dtype':'float32','shape':[6]},
        'observation.images.wrist':{'dtype':'image','shape':[96,96,3]},
        'next.reward':{'dtype':'float32','shape':[1]}}}})
    result=schema.check_schema(cfg)
    assert result.severity==('ok' if ok else 'fail')
    if not ok:assert 'features.next.reward' in result.summary


def test_local_vqbet_requires_explicit_cpu_before_native_factory(tmp_path,monkeypatch):
    import tether.finetune.preflight.schema as schema
    cfg=config(tmp_path);cfg.policy='vqbet';cfg.extra_lerobot_args={'policy.device':'mps'}
    monkeypatch.setattr(schema,'verify_local_dataset',lambda cfg:{'info':{'features':{
        'observation.state':{'dtype':'float32','shape':[6]},'action':{'dtype':'float32','shape':[6]},
        'observation.images.wrist':{'dtype':'image','shape':[96,96,3]}}}})
    result=schema.check_schema(cfg)
    assert result.is_blocking and 'policy.device' in result.summary and 'refuses MPS' in result.summary


def test_tdmpc_missing_stored_reward_is_named(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tether.finetune.preflight.schema as schema
    cfg=config(tmp_path);cfg.policy='tdmpc'
    path=cfg.dataset_root/'data/chunk-000/file-000.parquet';path.parent.mkdir(parents=True)
    pq.write_table(pa.table({'index':[0]}),path)
    monkeypatch.setattr(schema,'verify_local_dataset',lambda cfg:{'info':{'features':{
        'observation.state':{'dtype':'float32','shape':[6]},'action':{'dtype':'float32','shape':[6]},
        'observation.images.wrist':{'dtype':'image','shape':[96,96,3]},'next.reward':{'dtype':'float32','shape':[1]}}}})
    result=schema.check_schema(cfg)
    assert result.is_blocking and 'features.next.reward: missing stored column' in result.summary
