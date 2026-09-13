import importlib.util
from pathlib import Path

import pytest

from tether.eval.evidence_capture import EpisodeEvidenceWriter


SCRIPT = Path(__file__).parents[1] / "scripts/retrieve_modal_evaluation_evidence.py"
spec = importlib.util.spec_from_file_location("retrieve_modal_evidence", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(module)


def test_retrieval_uses_safe_volume_path_and_validates(tmp_path):
    calls = []
    def runner(command, check):
        calls.append(command)
        root = Path(command[-1]) / "task-0/episode-0"
        writer = EpisodeEvidenceWriter(root, provenance={"seed": 7})
        writer.record_step(step_index=0, phase="settling", observation={}, action=None, policy_request=None, task_events=[])
        writer.finish("timeout")
    manifests = module.retrieve("evaluation-evidence/libero_10/parent/seed-7", tmp_path / "download", runner=runner)
    assert len(manifests) == 1
    assert calls[0][:4] == ["modal", "volume", "get", "pi0-onnx-outputs"]


@pytest.mark.parametrize("path", ["../secret", "/evaluation-evidence/run", "other/run"])
def test_retrieval_rejects_unsafe_paths(path):
    with pytest.raises(ValueError):
        module.safe_remote_path(path)
