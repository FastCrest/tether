"""Unit tests for the protected-SHA quality ratchet."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci_quality_ratchet import (
    DiffHunk,
    Finding,
    RatchetError,
    changed_authorization_inputs,
    compare_findings,
    is_baseline_artifact,
    main,
    parse_mypy,
    parse_ruff_check,
    parse_ruff_format,
    relocate_base_line,
    run_ratchet,
    validate_tool_policy_text,
    verify_current_main_binding,
    verify_policy_environment,
    verify_policy_approval_evidence,
    verify_policy_artifacts_response,
    verify_policy_run_response,
    verify_policy_status_response,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ci_quality_ratchet"


def _finding(*, rule: str = "F401", line: int = 10, message: str = "unused") -> Finding:
    return Finding("ruff-check", rule, "src/tether/example.py", message, line, 1)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def test_tool_specific_parsers_produce_stable_fingerprints(tmp_path: Path) -> None:
    ruff_value = json.loads((FIXTURES / "ruff_check.json").read_text())
    ruff_value[0]["filename"] = str(tmp_path / "src/tether/example.py")
    ruff = parse_ruff_check(json.dumps(ruff_value), checkout=tmp_path)
    formatted = parse_ruff_format(
        (FIXTURES / "ruff_format.txt").read_text(),
        checkout=tmp_path,
    )
    mypy = parse_mypy((FIXTURES / "mypy.txt").read_text(), checkout=tmp_path)

    assert ruff[0].fingerprint == (
        "ruff-check",
        "F401",
        "src/tether/example.py",
        "`os` imported but unused",
    )
    assert formatted[0].fingerprint == (
        "ruff-format",
        "format",
        "src/tether/example.py",
        "File is not formatted",
    )
    assert mypy[0].fingerprint == (
        "mypy",
        "return-value",
        "src/tether/example.py",
        "Incompatible return value type",
    )
    assert len(mypy) == 1


def test_diff_relocation_handles_large_unrelated_insertions() -> None:
    hunk = DiffHunk(old_start=0, old_count=0, new_start=1, new_count=200)
    assert relocate_base_line(10, [hunk]) == 210
    comparison = compare_findings(
        [_finding(line=10)],
        [_finding(line=210)],
        diff_hunks=lambda _path: [hunk],
    )
    assert comparison.passed
    assert comparison.relocated_matches == 1
    assert not comparison.new_findings


def test_new_fingerprint_fails_even_when_total_count_decreases() -> None:
    comparison = compare_findings(
        [_finding(rule="F401"), _finding(rule="F841")],
        [_finding(rule="E722", message="bare except")],
        diff_hunks=lambda _path: [],
    )
    assert not comparison.passed
    assert [item.rule for item in comparison.new_findings] == ["E722"]
    assert comparison.head_count < comparison.base_count


def test_duplicate_fingerprint_increase_is_new_debt() -> None:
    comparison = compare_findings(
        [_finding(line=10)],
        [_finding(line=10), _finding(line=30)],
        diff_hunks=lambda _path: [],
    )
    assert not comparison.passed
    assert len(comparison.new_findings) == 1
    assert comparison.new_findings[0].line == 30


def test_removed_findings_are_allowed() -> None:
    comparison = compare_findings(
        [_finding(rule="F401"), _finding(rule="F841")],
        [_finding(rule="F401")],
        diff_hunks=lambda _path: [],
    )
    assert comparison.passed
    assert [item.rule for item in comparison.removed_findings] == ["F841"]


def test_mutable_baseline_artifact_names_are_reserved() -> None:
    assert is_baseline_artifact(".ci/quality-baseline/findings.json")
    assert is_baseline_artifact("quality-baseline.json")
    assert is_baseline_artifact(".github/quality-baseline/ruff.json")
    assert not is_baseline_artifact(".github/workflows/quality-ratchet.yml")
    assert not is_baseline_artifact("scripts/ci_quality_ratchet.py")


def test_workflows_separate_ordinary_and_protected_policy_changes() -> None:
    workflows = Path(__file__).parents[1] / ".github" / "workflows"
    ordinary = (workflows / "quality-ratchet.yml").read_text()
    protected = (workflows / "quality-baseline-update.yml").read_text()
    judge = (Path(__file__).parents[1] / "scripts" / "ci_quality_ratchet.py").read_text()
    assert 'PROTECTED_POLICY="$PROTECTED_POLICY_DIR/pyproject.toml"' in protected
    assert ".protected-base-pyproject.toml" not in protected
    assert "merge-multiple: true" in protected

    assert "pull_request_target:" in ordinary
    assert "${BASE_SHA}:scripts/ci_quality_ratchet.py" in ordinary
    assert "quality-ratchet/policy-approved" in judge
    assert "--verify-policy-status-response" in ordinary
    assert "--verify-policy-run-response" in ordinary
    assert "--verify-policy-artifacts-response" in ordinary
    assert '.creator.login == "github-actions[bot]"' not in ordinary
    assert "quality-policy-approval-${HEAD_SHA}-${RUN_ID}" in ordinary
    assert '--expected-candidate-sha "$HEAD_SHA"' in ordinary
    assert '--expected-base-sha "$BASE_SHA"' in ordinary
    assert "artifact-ids: ${{ steps.authorization.outputs.artifact_id }}" in ordinary
    approval_download = ordinary.split("- name: Download exact approval evidence", 1)[1].split(
        "- name: Verify immutable approval evidence binding", 1
    )[0]
    assert "merge-multiple: true" in approval_download
    assert "Bind the event to the current protected-main tip" in ordinary
    assert "--verify-main-ref-response" in ordinary
    assert "--expected-current-main-sha" in ordinary
    assert ordinary.index("Bind the event to the current protected-main tip") < ordinary.index(
        "Verify protected policy authorization"
    )
    assert "Reconfirm current protected-main tip before publishing" in ordinary
    assert ordinary.index(
        "Reconfirm current protected-main tip before publishing"
    ) < ordinary.index("Publish protected result on the exact head SHA")
    assert "workflow_dispatch:" in protected
    assert "validate-candidate:" in protected
    assert "exercise-candidate:" in protected
    assert "authorize-policy-update:" in protected
    assert "needs: validate-candidate" in protected
    assert "needs: [validate-candidate, exercise-candidate]" in protected
    assert protected.index("validate-candidate:") < protected.index("exercise-candidate:")
    assert protected.index("exercise-candidate:") < protected.index("authorize-policy-update:")
    assert "name: quality-ratchet-policy" in protected
    assert "--verify-policy-environment" in protected
    assert "statuses: write" in protected.split("authorize-policy-update:", 1)[1]
    assert "statuses: write" not in protected.split("authorize-policy-update:", 1)[0]
    assert "Upload immutable validation evidence before candidate execution" in protected
    assert "quality-policy-approval-${CANDIDATE_SHA}-${GITHUB_RUN_ID}" in protected
    assert "artifact-ids: ${{ steps.validation.outputs.artifact_id }}" in protected
    pre_seal = protected.split("exercise-candidate:", 1)[0]
    untrusted = protected.split("exercise-candidate:", 1)[1].split("authorize-policy-update:", 1)[0]
    assert "--validate-policy-syntax" in pre_seal
    assert "--show-settings" not in pre_seal
    assert "--no-error-summary -c 'pass'" not in pre_seal
    assert "--show-settings" in untrusted
    assert "--no-error-summary -c 'pass'" in untrusted
    assert "Reconfirm current protected tip immediately before approval" in protected
    assert protected.index(
        "Reconfirm current protected tip immediately before approval"
    ) < protected.index("Build immutable approval evidence")
    assert "Reconfirm current protected tip immediately before status" in protected
    assert protected.index(
        "Reconfirm current protected tip immediately before status"
    ) < protected.index("Authorize the exact evidence-bound commit")


def test_policy_status_prefilter_accepts_live_shape_and_rejects_spoofs() -> None:
    candidate = "a" * 40
    value = json.loads((FIXTURES / "policy_status.json").read_text())
    assert (
        verify_policy_status_response(
            value,
            expected_candidate_sha=candidate,
            expected_repository="FastCrest/tether",
        )
        == 33163572947
    )

    status = value["statuses"][0]
    tampered = [
        {**status, "avatar_url": "https://avatars.githubusercontent.com/u/1?v=4"},
        {key: item for key, item in status.items() if key != "avatar_url"},
        {**status, "context": "quality-ratchet/protected"},
        {**status, "state": "pending"},
        {**status, "description": "sha:spoofed run:33163572947"},
        {
            **status,
            "target_url": "https://github.com/attacker/fork/actions/runs/33163572947",
        },
    ]
    for invalid_status in tampered:
        with pytest.raises(RatchetError):
            verify_policy_status_response(
                {"sha": candidate, "statuses": [invalid_status]},
                expected_candidate_sha=candidate,
                expected_repository="FastCrest/tether",
            )

    with pytest.raises(RatchetError):
        verify_policy_status_response(
            {"sha": "b" * 40, "statuses": value["statuses"]},
            expected_candidate_sha=candidate,
            expected_repository="FastCrest/tether",
        )


def test_same_actions_identity_needs_exact_run_and_artifact_chain() -> None:
    candidate = "a" * 40
    base = "b" * 40
    status = json.loads((FIXTURES / "policy_status.json").read_text())
    run_id = verify_policy_status_response(
        status,
        expected_candidate_sha=candidate,
        expected_repository="FastCrest/tether",
    )
    run = json.loads((FIXTURES / "policy_run.json").read_text())
    verify_policy_run_response(
        run,
        expected_base_sha=base,
        expected_run_id=run_id,
        expected_repository="FastCrest/tether",
    )
    artifact_name = f"quality-policy-approval-{candidate}-{run_id}"
    artifacts = json.loads((FIXTURES / "policy_artifacts.json").read_text())
    assert verify_policy_artifacts_response(artifacts, expected_name=artifact_name) == 9682669486

    run_tampering = [
        {**run, "id": run_id + 1},
        {**run, "head_sha": "c" * 40},
        {**run, "event": "push"},
        {**run, "head_branch": "feature"},
        {**run, "path": ".github/workflows/unrelated.yml"},
        {**run, "conclusion": "failure"},
        {**run, "repository": {"full_name": "attacker/fork"}},
    ]
    for invalid_run in run_tampering:
        with pytest.raises(RatchetError):
            verify_policy_run_response(
                invalid_run,
                expected_base_sha=base,
                expected_run_id=run_id,
                expected_repository="FastCrest/tether",
            )

    artifact = artifacts["artifacts"][0]
    artifact_tampering = [
        {**artifact, "name": "quality-policy-approval-spoofed"},
        {**artifact, "expired": True},
        {**artifact, "id": 0},
        {**artifact, "digest": "sha256:short"},
    ]
    for invalid_artifact in artifact_tampering:
        with pytest.raises(RatchetError):
            verify_policy_artifacts_response(
                {"artifacts": [invalid_artifact]},
                expected_name=artifact_name,
            )


def test_policy_approval_evidence_is_bound_to_exact_sha_base_and_run() -> None:
    candidate = "a" * 40
    base = "b" * 40
    value = {
        "base_sha": base,
        "candidate_sha": candidate,
        "repository": "FastCrest/tether",
        "run_id": 12345,
        "schema_version": 1,
        "validation_artifact_digest": "sha256:" + "c" * 64,
        "validation_artifact_id": 67890,
        "workflow_path": ".github/workflows/quality-baseline-update.yml",
    }
    verify_policy_approval_evidence(
        value,
        expected_candidate_sha=candidate,
        expected_base_sha=base,
        expected_run_id=12345,
        expected_repository="FastCrest/tether",
    )

    tampered_values = [
        {**value, "candidate_sha": "d" * 40},
        {**value, "base_sha": "d" * 40},
        {**value, "run_id": 54321},
        {**value, "repository": "attacker/fork"},
        {**value, "workflow_path": ".github/workflows/unrelated.yml"},
        {**value, "validation_artifact_digest": "sha256:short"},
        {**value, "extra": "ambiguous"},
    ]
    for tampered in tampered_values:
        with pytest.raises(RatchetError):
            verify_policy_approval_evidence(
                tampered,
                expected_candidate_sha=candidate,
                expected_base_sha=base,
                expected_run_id=12345,
                expected_repository="FastCrest/tether",
            )


def test_current_main_binding_rejects_stale_event_reruns() -> None:
    base = "a" * 40
    head = "b" * 40
    current = "c" * 40
    ref_value = {
        "ref": "refs/heads/main",
        "object": {"type": "commit", "sha": base},
    }
    pull_request_value = {
        "state": "open",
        "base": {"ref": "main", "sha": base},
        "head": {"sha": head},
    }
    verify_current_main_binding(
        ref_value,
        expected_main_sha=base,
        pull_request_value=pull_request_value,
        expected_head_sha=head,
    )

    stale_cases = [
        (
            {"ref": "refs/heads/main", "object": {"type": "commit", "sha": current}},
            pull_request_value,
        ),
        (ref_value, {**pull_request_value, "base": {"ref": "main", "sha": current}}),
        (ref_value, {**pull_request_value, "head": {"sha": current}}),
    ]
    for stale_ref, stale_pull_request in stale_cases:
        with pytest.raises(RatchetError):
            verify_current_main_binding(
                stale_ref,
                expected_main_sha=base,
                pull_request_value=stale_pull_request,
                expected_head_sha=head,
            )


def test_policy_environment_requires_exact_reviewer_allowlist() -> None:
    trusted = {"type": "User", "reviewer": {"login": "rylinjames"}}
    valid = {
        "protection_rules": [
            {"type": "required_reviewers", "reviewers": [trusted]},
        ],
        "deployment_branch_policy": {
            "protected_branches": True,
            "custom_branch_policies": False,
        },
    }
    verify_policy_environment(valid)

    untrusted = {"type": "User", "reviewer": {"login": "untrusted-reviewer"}}
    invalid_values = [
        {**valid, "protection_rules": []},
        {
            **valid,
            "protection_rules": [{"type": "required_reviewers", "reviewers": [trusted, untrusted]}],
        },
        {
            **valid,
            "protection_rules": [{"type": "required_reviewers", "reviewers": [trusted, trusted]}],
        },
        {
            **valid,
            "protection_rules": [
                {
                    "type": "required_reviewers",
                    "reviewers": [{"type": "Team", "reviewer": {"login": "rylinjames"}}],
                }
            ],
        },
    ]
    for invalid in invalid_values:
        with pytest.raises(RatchetError):
            verify_policy_environment(invalid)


def test_candidate_policy_rejects_mypy_plugin_without_executing_or_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "plugin-executed"
    evidence = tmp_path / "protected-evidence.json"
    evidence.write_text('{"sealed": true}\n')
    (tmp_path / "malicious_plugin.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        f"Path({str(evidence)!r}).write_text('tampered')\n"
        "def plugin(version):\n    raise RuntimeError(version)\n"
    )
    policy = tmp_path / "candidate-pyproject.toml"
    policy.write_text(
        '[tool.ruff]\nline-length = 88\n[tool.mypy]\nplugins = ["malicious_plugin.py"]\n'
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RatchetError):
        validate_tool_policy_text(policy.read_text())
    assert main(["--validate-policy-syntax", str(policy)]) == 2
    assert not marker.exists()
    assert evidence.read_text() == '{"sealed": true}\n'

    # Control: the same config really does execute the plugin when mypy loads it.
    control = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "mypy",
            f"--config-file={policy}",
            "--no-error-summary",
            "-c",
            "pass",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert control.returncode != 0
    assert marker.read_text() == "executed"
    assert evidence.read_text() == "tampered"


def test_pr_modified_mypy_plugin_never_executes_in_ratchet(tmp_path: Path) -> None:
    repo = tmp_path / "plugin-head"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "ratchet@example.test")
    _git(repo, "config", "user.name", "Ratchet Test")
    (repo / "pyproject.toml").write_text(
        '[project]\nname="plugin-head"\nversion="1"\n'
        '[tool.ruff]\ntarget-version="py310"\n'
        '[tool.mypy]\npython_version="3.10"\nplugins=["local_plugin.py"]\n'
    )
    (repo / "local_plugin.py").write_text(
        "from mypy.plugin import Plugin\n"
        "class LocalPlugin(Plugin):\n    pass\n"
        "def plugin(version):\n    return LocalPlugin\n"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base with forbidden plugin policy")
    base = _git(repo, "rev-parse", "HEAD")

    marker = tmp_path / "head-plugin-executed"
    (repo / "local_plugin.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "from mypy.plugin import Plugin\n"
        "class LocalPlugin(Plugin):\n    pass\n"
        "def plugin(version):\n    return LocalPlugin\n"
    )
    _git(repo, "commit", "-am", "modify local plugin in pull request")
    head = _git(repo, "rev-parse", "HEAD")

    report = tmp_path / "plugin-head-report.json"
    with pytest.raises(RatchetError, match="mypy plugins are forbidden"):
        run_ratchet(repo=repo, base_sha=base, head_sha=head, report_path=report)
    assert not marker.exists()
    assert not report.exists()


def test_authorization_inputs_distinguish_dependency_and_tool_policy_changes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "ratchet@example.test")
    _git(repo, "config", "user.name", "Ratchet Test")
    (repo / "pyproject.toml").write_text(
        '[project]\nname="example"\nversion="1"\n'
        '[tool.ruff]\nline-length=100\n[tool.mypy]\npython_version="3.10"\n'
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")

    (repo / "pyproject.toml").write_text(
        '[project]\nname="example"\nversion="2"\n'
        '[tool.ruff]\nline-length=100\n[tool.mypy]\npython_version="3.10"\n'
    )
    _git(repo, "commit", "-am", "dependency metadata only")
    metadata_head = _git(repo, "rev-parse", "HEAD")
    assert changed_authorization_inputs(repo, base, metadata_head) == []

    with (repo / "pyproject.toml").open("a") as handle:
        handle.write("ignore_errors=true\n")
    _git(repo, "commit", "-am", "weaken mypy")
    policy_head = _git(repo, "rev-parse", "HEAD")
    assert changed_authorization_inputs(repo, metadata_head, policy_head) == [
        "pyproject.toml#[tool.mypy]"
    ]


def test_adversarial_head_cannot_hide_new_debt_or_replace_judge(tmp_path: Path) -> None:
    repo = tmp_path / "adversarial"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "ratchet@example.test")
    _git(repo, "config", "user.name", "Ratchet Test")
    (repo / "src").mkdir()
    (repo / "scripts").mkdir()
    (repo / ".github/workflows").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        '[project]\nname="adversarial"\nversion="1"\n'
        '[tool.ruff]\ntarget-version="py310"\nexclude=["sitecustomize.py"]\n'
        '[tool.mypy]\npython_version="3.10"\n'
    )
    (repo / "src/example.py").write_text("import os\n\n\ndef existing() -> str:\n    return 1\n")
    (repo / "scripts/ci_quality_ratchet.py").write_text("PROTECTED = True\n")
    (repo / ".github/workflows/quality-ratchet.yml").write_text("name: protected\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base debt")
    base = _git(repo, "rev-parse", "HEAD")

    (repo / "pyproject.toml").write_text(
        '[project]\nname="adversarial"\nversion="1"\n'
        '[tool.ruff]\ntarget-version="py310"\nexclude=["src"]\n'
        '[tool.mypy]\npython_version="3.10"\nignore_errors=true\n'
    )
    (repo / "scripts/ci_quality_ratchet.py").write_text("raise SystemExit(0)\n")
    (repo / ".github/workflows/quality-ratchet.yml").write_text("name: bypass\n")
    marker = tmp_path / "sitecustomize-executed"
    (repo / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "attempt policy bypass")
    policy_head = _git(repo, "rev-parse", "HEAD")

    policy_report = tmp_path / "policy-report.json"
    assert not run_ratchet(
        repo=repo,
        base_sha=base,
        head_sha=policy_head,
        report_path=policy_report,
    )
    policy_value = json.loads(policy_report.read_text())
    assert policy_value["base_count"] == policy_value["head_count"] == 2
    assert policy_value["new_findings"] == []
    assert not policy_value["authorization_approved"]

    (repo / "src/example.py").write_text(
        "import os\nimport sys\n\n\ndef existing() -> str:\n    return 1\n"
        "\n\ndef added() -> str:\n    return 2\n"
    )
    _git(repo, "add", "src/example.py")
    _git(repo, "commit", "-m", "add hidden debt")
    head = _git(repo, "rev-parse", "HEAD")

    report = tmp_path / "adversarial-report.json"
    assert not run_ratchet(
        repo=repo,
        base_sha=base,
        head_sha=head,
        report_path=report,
        authorization_approved=True,
    )
    value = json.loads(report.read_text())
    assert value["base_count"] == 2
    assert value["head_count"] == 4
    assert len(value["new_findings"]) == 2
    assert not marker.exists()
    assert value["authorization_input_changes"] == [
        ".github/workflows/quality-ratchet.yml",
        "pyproject.toml#[tool.mypy]",
        "pyproject.toml#[tool.ruff]",
        "scripts/ci_quality_ratchet.py",
    ]
