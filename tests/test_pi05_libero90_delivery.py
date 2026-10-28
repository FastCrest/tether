"""Coverage gates for full LIBERO-90 evaluation and its separate success metric."""
import importlib.util
import json
from pathlib import Path

import pytest


def load_script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_evidence(root,checkpoint):
    for task in range(90):
        successes=[task%2==0]*5
        result={'suite':'libero_90','errors':[],'total_eps':5,'requested_seed':7001,'student':True,
                'inference_steps':1,'denoise_calls_per_chunk_verified':True,
                'checkpoint':str(checkpoint),'total_success':sum(successes),
                'per_task':[{'task_idx':task,'task_description':f'task {task}',
                             'episodes':[{'ep':ep,'success':success} for ep,success in enumerate(successes)]}]}
        (root/f'libero_90-task-{task}.json').write_text(json.dumps(result))


def test_all_libero90_tasks_are_selected_by_default():
    script=load_script('evaluate_pi05_distill_rocm')
    assert script.resolve_task_indices(90,'')==list(range(90))
    assert script.resolve_task_indices(10,'')==list(range(10))
    with pytest.raises(ValueError): script.resolve_task_indices(90,'89,90')
    with pytest.raises(ValueError): script.resolve_task_indices(90,'0,0')


def test_libero90_success_is_separate_and_covers_450_episodes(tmp_path):
    script=load_script('report_pi05_distill')
    checkpoint=tmp_path/'model'
    write_evidence(tmp_path,checkpoint)
    result=script.summarize_libero90(tmp_path,checkpoint)
    assert result['tasks']==90
    assert result['episodes']==450
    assert result['success']==225
    assert result['success_rate_pct']==50
    assert result['included_in_original_40_task_aggregate'] is False
    assert len(result['per_task'])==90


def test_libero90_report_refuses_missing_task_89(tmp_path):
    script=load_script('report_pi05_distill')
    checkpoint=tmp_path/'model'
    write_evidence(tmp_path,checkpoint)
    (tmp_path/'libero_90-task-89.json').unlink()
    with pytest.raises(FileNotFoundError): script.summarize_libero90(tmp_path,checkpoint)


def test_libero90_report_refuses_another_checkpoint(tmp_path):
    script=load_script('report_pi05_distill')
    write_evidence(tmp_path,tmp_path/'wrong-model')
    with pytest.raises(RuntimeError,match='mismatched'):
        script.summarize_libero90(tmp_path,tmp_path/'expected-model')


def test_libero90_report_refuses_another_suite(tmp_path):
    script=load_script('report_pi05_distill')
    checkpoint=tmp_path/'model'
    write_evidence(tmp_path,checkpoint)
    path=tmp_path/'libero_90-task-0.json'
    wrong=json.loads(path.read_text())
    wrong['suite']='libero_10'
    path.write_text(json.dumps(wrong))
    with pytest.raises(RuntimeError,match='mismatched'):
        script.summarize_libero90(tmp_path,checkpoint)


def test_interim_comparison_requires_complete_matched_evidence(tmp_path):
    script=load_script('report_libero90_interim')
    checkpoint=tmp_path/'model'
    write_evidence(tmp_path,checkpoint)
    for path in tmp_path.glob('libero_90-task-*.json'):
        raw=json.loads(path.read_text())
        raw.update(max_steps=400,resize_size=224,replan_steps=10,num_steps_wait=10)
        path.write_text(json.dumps(raw))
    arm=script.collect_arm(tmp_path,checkpoint,True)
    assert arm['successes']==225
    assert script.paired_delta(arm,arm)==(0.0,[0.0,0.0])
    other={**arm,'protocol':{**arm['protocol'],'replan_steps':50}}
    with pytest.raises(RuntimeError,match='protocols differ'):
        script.paired_delta(arm,other)
    path=tmp_path/'libero_90-task-89.json'
    raw=json.loads(path.read_text())
    raw['per_task'][0]['episodes'][4]['ep']=3
    path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError,match='duplicate episodes'):
        script.collect_arm(tmp_path,checkpoint,True)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        script.collect_arm(tmp_path,checkpoint,True)
