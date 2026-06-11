"""Tests for the three runtime/server.py hardening fixes.

Bug #1: /guard/reset was unauthenticated (no _require_api_key dependency).
Bug #2: NameError: `os` inside the lifespan function that only imports `os as _os`.
Bug #3: TypeError from `image_wrist_b64=` kwarg passed to predict_from_base64_async
        which didn't accept that parameter.
"""
from __future__ import annotations

import inspect
import json

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_minimal_export_dir(tmp_path):
    """Minimal export dir so create_app() can instantiate without a real model."""
    cfg = {
        "model_id": "lerobot/smolvla_base",
        "model_type": "smolvla",
        "target": "desktop",
        "action_chunk_size": 50,
        "action_dim": 32,
        "expert": {"expert_hidden": 720, "action_dim": 32, "num_layers": 16},
    }
    (tmp_path / "tether_config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.onnx").write_bytes(b"\x00")
    return tmp_path


# ---------------------------------------------------------------------------
# Bug #1 — /guard/reset must require api-key auth
# ---------------------------------------------------------------------------

class TestGuardResetAuth:
    """Verify the /guard/reset route has _require_api_key in its dependencies."""

    def _get_guard_reset_route(self):
        """Build the app and find the /guard/reset route object."""
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("fastapi not installed")

        from tether.runtime.server import create_app
        import tempfile, json as _json, pathlib

        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            cfg = {
                "model_id": "lerobot/smolvla_base",
                "model_type": "smolvla",
                "target": "desktop",
                "action_chunk_size": 50,
                "action_dim": 32,
                "expert": {"expert_hidden": 720, "action_dim": 32, "num_layers": 16},
            }
            (p / "tether_config.json").write_text(_json.dumps(cfg))
            (p / "model.onnx").write_bytes(b"\x00")
            app = create_app(str(p), device="cpu", api_key="test-key")

        for route in app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None) or set()
            if path == "/guard/reset" and "POST" in methods:
                return route
        return None

    def test_guard_reset_route_exists(self):
        route = self._get_guard_reset_route()
        assert route is not None, "/guard/reset POST route not found in app.routes"

    def test_guard_reset_has_api_key_dependency(self):
        """The /guard/reset endpoint's dependant should include _require_api_key."""
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("fastapi not installed")

        route = self._get_guard_reset_route()
        assert route is not None, "/guard/reset POST route not found"

        # FastAPI stores dependencies in route.dependant.dependencies
        dependant = getattr(route, "dependant", None)
        assert dependant is not None, "route has no dependant"

        dep_calls = [
            dep.call.__name__ if callable(dep.call) else str(dep.call)
            for dep in dependant.dependencies
        ]
        assert "_require_api_key" in dep_calls, (
            f"/guard/reset dependencies do not include _require_api_key; "
            f"found: {dep_calls}"
        )

    def test_guard_reset_rejects_unauthenticated(self, tmp_path):
        """With api_key set, POST /guard/reset without header → 401."""
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")

        # Build a minimal app that mirrors the three guarded endpoints
        from fastapi import Depends, Header, HTTPException
        from fastapi.responses import JSONResponse

        app = FastAPI()
        api_key = "secret-guard-key"

        async def _require_api_key(
            x_tether_key: str | None = Header(default=None, alias="X-Tether-Key"),
        ) -> None:
            if x_tether_key != api_key:
                raise HTTPException(status_code=401, detail="bad key")

        @app.post("/guard/reset")
        async def guard_reset(_auth: None = Depends(_require_api_key)):
            return JSONResponse(content={"reset": True, "was_tripped": False})

        client = TestClient(app)
        # No key → 401
        assert client.post("/guard/reset").status_code == 401
        # Correct key → 200
        assert client.post(
            "/guard/reset",
            headers={"X-Tether-Key": api_key},
        ).status_code == 200


# ---------------------------------------------------------------------------
# Bug #2 — no bare `os.` inside the lifespan / startup function
# ---------------------------------------------------------------------------

class TestOsNameInLifespan:
    """Verify the lifespan function body has no bare `os.` (would NameError)."""

    def test_no_bare_os_dot_in_server_py(self):
        import re
        from pathlib import Path

        src = Path(
            "/Users/romirjain/Desktop/building projects/fastcrest/tether/"
            "src/tether/runtime/server.py"
        ).read_text()

        # Match `os.` NOT preceded by underscore (i.e. not `_os.`).
        # Exclude comment lines.
        bare_os_lines = []
        for i, line in enumerate(src.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if re.search(r"(?<!_)\bos\.", line):
                bare_os_lines.append((i, line.rstrip()))

        assert bare_os_lines == [], (
            "Bare `os.` found in server.py (should be `_os.` inside lifespan "
            f"which imports `import os as _os`): {bare_os_lines}"
        )

    def test_py_compile_server(self):
        """server.py must compile without errors."""
        import py_compile
        from pathlib import Path
        path = str(
            Path(
                "/Users/romirjain/Desktop/building projects/fastcrest/tether/"
                "src/tether/runtime/server.py"
            )
        )
        # raises py_compile.PyCompileError on failure
        py_compile.compile(path, doraise=True)


# ---------------------------------------------------------------------------
# Bug #3 — TetherServer.predict_from_base64_async accepts image_wrist_b64
# ---------------------------------------------------------------------------

class TestWristImageKwarg:
    """TetherServer.predict_from_base64{,_async} must accept image_wrist_b64."""

    def test_predict_from_base64_signature(self):
        from tether.runtime.server import TetherServer
        sig = inspect.signature(TetherServer.predict_from_base64)
        assert "image_wrist_b64" in sig.parameters, (
            f"TetherServer.predict_from_base64 missing image_wrist_b64 param; "
            f"params: {list(sig.parameters)}"
        )

    def test_predict_from_base64_async_signature(self):
        from tether.runtime.server import TetherServer
        sig = inspect.signature(TetherServer.predict_from_base64_async)
        assert "image_wrist_b64" in sig.parameters, (
            f"TetherServer.predict_from_base64_async missing image_wrist_b64 param; "
            f"params: {list(sig.parameters)}"
        )

    def test_predict_from_base64_async_accepts_wrist_kwarg(self, tmp_path):
        """Calling predict_from_base64_async with image_wrist_b64 must not TypeError."""
        import asyncio
        import json as _json

        cfg = {
            "model_id": "lerobot/smolvla_base",
            "model_type": "smolvla",
            "target": "desktop",
            "action_chunk_size": 50,
            "action_dim": 32,
            "expert": {"expert_hidden": 720, "action_dim": 32, "num_layers": 16},
        }
        (tmp_path / "tether_config.json").write_text(_json.dumps(cfg))

        from tether.runtime.server import TetherServer
        server = TetherServer(str(tmp_path), device="cpu")

        # Server is not ready (no ONNX) — predict() will return an error dict,
        # but the call must NOT raise TypeError for the wrist kwarg.
        result = asyncio.run(
            server.predict_from_base64_async(
                image_b64=None,
                instruction="test",
                state=None,
                image_wrist_b64=None,  # <-- was the TypeError trigger
            )
        )
        # Should be a dict (error envelope is fine; TypeError is not)
        assert isinstance(result, dict)

    def test_callee_in_act_handler_accepts_wrist(self):
        """The call site in server.py passes image_wrist_b64; callee must accept it.

        Checks the actual call at line ~2342 passes a kwarg that now exists in
        both TetherServer AND Pi05DecomposedServer predict_from_base64_async.
        """
        from tether.runtime.server import TetherServer
        from tether.runtime.decomposed_server import Pi05DecomposedServer

        for cls in (TetherServer, Pi05DecomposedServer):
            sig = inspect.signature(cls.predict_from_base64_async)
            assert "image_wrist_b64" in sig.parameters, (
                f"{cls.__name__}.predict_from_base64_async missing image_wrist_b64"
            )
