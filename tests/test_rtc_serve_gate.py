from types import SimpleNamespace

import pytest


def test_http_server_rejects_unconnected_rtc_before_loading_export(tmp_path):
    from tether.runtime.server import create_app

    with pytest.raises(RuntimeError, match="not connected to the HTTP /act"):
        create_app(
            str(tmp_path / "missing-export"),
            device="cpu",
            rtc_config=SimpleNamespace(enabled=True),
        )
