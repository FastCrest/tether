from types import SimpleNamespace

import pytest

from tests.export_config_factory import write_test_export_config


@pytest.mark.parametrize("model_type", ["smolvla", "pi0", "pi05"])
def test_http_server_refuses_rtc_for_baked_loop_exports(tmp_path, monkeypatch, model_type):
    """Monolithic exports bake the Euler loop; RTC kwargs would be ignored."""
    from tether.runtime.server import create_app

    monkeypatch.delenv("TETHER_NATIVE", raising=False)
    write_test_export_config(tmp_path, model_type=model_type, export_kind="monolithic_onnx")
    with pytest.raises(RuntimeError, match="needs per-step velocity access"):
        create_app(str(tmp_path), device="cpu", rtc_config=SimpleNamespace(enabled=True))


def test_rtc_conflicts_are_refused_before_loading_export(tmp_path):
    from tether.runtime.server import create_app

    with pytest.raises(RuntimeError, match="--max-batch"):
        create_app(
            str(tmp_path / "missing-export"),
            device="cpu",
            max_batch=4,
            rtc_config=SimpleNamespace(enabled=True),
        )
