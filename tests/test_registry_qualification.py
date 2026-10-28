import ast
import re
from pathlib import Path

from tether.registry.models import list_families
from tether.registry.qualification import (
    FAMILY_QUALIFICATIONS,
    qualification_for,
    qualification_gaps,
)


def test_every_registry_family_has_an_explicit_qualification():
    assert set(list_families()) == set(FAMILY_QUALIFICATIONS)


def test_qualification_is_conservative_and_copied():
    smolvla = qualification_for("smolvla")
    assert smolvla["training"] == "trains-and-loads-lora-task-gain-unmeasured"
    assert qualification_for("dreamzero")["checkpoint"] == "registry-only"
    smolvla["notes"].append("changed")
    assert "changed" not in qualification_for("smolvla")["notes"]


def test_acceptance_requirements_are_recorded_for_every_family():
    assert all(value.get("acceptance") for value in FAMILY_QUALIFICATIONS.values())
    assert qualification_gaps("pi0")
    assert qualification_gaps("pi05")


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_qualified_export_status_and_parity_evidence_agree():
    """A qualified export needs a passing Linux-CUDA receipt, and a family with one
    must not still claim verification-pending (the 2026-10-28 stale state)."""
    for family, facts in FAMILY_QUALIFICATIONS.items():
        export = (facts.get("evidence") or {}).get("export")
        qualified = str(facts["export"]).startswith("qualified")
        assert qualified == (export is not None), family
        if export is None:
            continue
        assert export["verdict"] == "passed" and export["scope"] == "linux-cuda", family
        assert export["receipt"].endswith(".json"), family
        assert _HEX64.fullmatch(export["receipt_sha256"]), family
        assert _HEX40.fullmatch(export["tether_commit"]), family
        assert _HEX40.fullmatch(export["model"].rsplit("@", 1)[1]), family
        if "condition" in export:
            assert facts["export"].endswith("strict-fp32"), family


def test_qualified_evaluation_needs_a_nonzero_matched_success():
    """SmolVLA was listed as qualified-libero-linux-cuda while every Tether arm
    scored 0/20; a qualified evaluation must name an arm that actually succeeded."""
    for family, facts in FAMILY_QUALIFICATIONS.items():
        evaluation = (facts.get("evidence") or {}).get("evaluation")
        if not str(facts["evaluation"]).startswith("qualified"):
            if evaluation is not None:
                assert evaluation["qualified_arm"] is None, family
            continue
        assert evaluation is not None, family
        successes, total = evaluation["arms"][evaluation["qualified_arm"]]
        assert 0 < successes <= total, family


def test_smolvla_evaluation_records_the_matched_libero_result():
    smolvla = qualification_for("smolvla")
    assert not smolvla["evaluation"].startswith("qualified")
    arms = smolvla["evidence"]["evaluation"]["arms"]
    assert sorted(arms.values()) == [[0, 20], [0, 20]]
    assert qualification_gaps("smolvla")


def test_studio_ast_reader_sees_every_value():
    """Studio reads FAMILY_QUALIFICATIONS with a literal-only AST walk, so every
    value must be a literal (no f-strings, names or calls)."""
    source = Path(__file__).resolve().parents[1] / "src" / "tether" / "registry" / "qualification.py"
    tree = ast.parse(source.read_text())
    node = next(
        item.value for item in tree.body
        if isinstance(item, ast.AnnAssign) and getattr(item.target, "id", "") == "FAMILY_QUALIFICATIONS"
    )
    assert ast.literal_eval(node) == FAMILY_QUALIFICATIONS


def test_nested_evidence_is_copied():
    pi0 = qualification_for("pi0")
    pi0["evidence"]["export"]["verdict"] = "changed"
    assert qualification_for("pi0")["evidence"]["export"]["verdict"] == "passed"


def test_training_status_follows_measured_task_gain():
    """Training is 'qualified' only when a trained arm beat its parent on matched
    cases; trains-and-loads evidence alone (SmolVLA LoRA: 0/20 vs 0/20) is not."""
    for family, facts in FAMILY_QUALIFICATIONS.items():
        training = (facts.get("evidence") or {}).get("training")
        if training is None:
            continue
        cand, parent = training["candidate_matched"], training["parent_matched"]
        gain = training["task_gain_measured"] and cand[0] / cand[1] > parent[0] / parent[1]
        assert str(facts["training"]).startswith("qualified") == gain, family
        if training["trains"] and training["loads"] and not gain:
            assert facts["training"].startswith("trains-and-loads"), family
