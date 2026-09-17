from pathlib import Path

import yaml


def test_goals_tracker_is_valid_yaml_and_qualification_claims_have_receipts():
    root = Path(__file__).resolve().parents[1]
    tracker = yaml.safe_load((root / "GOALS.yaml").read_text())
    by_id = {item["id"]: item for item in tracker["goals"]}
    expected_receipts = {
        "pi0-onnx-parity": "tests/test_pi0_onnx_parity.py",
        "pi05-onnx-parity": "tests/test_pi05_onnx_parity.py",
        "gr00t-onnx-parity": "tests/test_gr00t_onnx_parity.py",
        "openvla-onnx-parity": "tests/test_openvla_onnx_parity.py",
        "multi-model-native-parity": "tests/test_native_parity.py",
    }
    for goal_id, receipt in expected_receipts.items():
        if by_id[goal_id].get("status") == "done":
            assert (root / receipt).is_file(), f"{goal_id} is done without {receipt}"
