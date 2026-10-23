"""User-selected providers survive chat and CLI boundaries without a GPU run."""

import json

import pytest
from typer.testing import CliRunner

from tether.chat.executor import _argv_for
from tether.chat.schema import by_name
from tether.cli import app


@pytest.mark.parametrize(
    "tool,params",
    [
        (
            "prove_deployment",
            {"export_dir": "./export", "device": "cuda", "startup_timeout_s": 180},
        ),
        ("serve_model", {"export_dir": "./export", "device": "cuda"}),
        ("deploy_one_command", {"model": "smolvla-base"}),
    ],
)
def test_chat_exposes_and_preserves_explicit_provider_choice(tool, params):
    providers = "TensorrtExecutionProvider,CUDAExecutionProvider"
    properties = by_name()[tool]["function"]["parameters"]["properties"]
    assert "providers" in properties
    argv = _argv_for(tool, params | {"providers": providers})
    assert argv[argv.index("--providers") + 1] == providers
    if tool == "prove_deployment":
        assert argv[argv.index("--startup-timeout-s") + 1] == "180"
        assert argv[argv.index("--device") + 1] == "cuda"


def test_prove_cli_keeps_startup_and_request_budgets_separate(monkeypatch):
    import tether.deploy_proof as proof

    seen = {}

    def run(**kwargs):
        seen.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(proof, "run_deploy_proof", run)
    result = CliRunner().invoke(
        app,
        [
            "prove",
            "./export",
            "--device",
            "cuda",
            "--providers",
            "TensorrtExecutionProvider,CUDAExecutionProvider",
            "--startup-timeout-s",
            "180",
            "--timeout-s",
            "2",
            "--control-hz",
            "20",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["passed"]
    assert seen["startup_timeout_s"] == 180
    assert seen["timeout_s"] == 2
    assert seen["control_hz"] == 20
    assert seen["providers"] == "TensorrtExecutionProvider,CUDAExecutionProvider"


def test_go_exposes_provider_strictness_without_starting_server():
    result = CliRunner().invoke(app, ["go", "--help"])
    assert result.exit_code == 0
    assert "--providers" in result.output
    assert "--no-strict-providers" in result.output
