"""Repository-level checks for the retired legacy data-worker upload API.

Runtime behavior is covered by ``infra/data-worker/test/retired-upload.test.js``.
These checks prevent documentation or route wiring from quietly reintroducing
the former client-asserted anonymization upload path.
"""

from __future__ import annotations

from pathlib import Path


WORKER = Path(__file__).parents[1] / "infra" / "data-worker" / "worker.js"


def test_legacy_upload_routes_are_wired_to_gone_response() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert 'path === "/v1/episodes/upload"' in source
    assert 'path === "/v1/episodes/upload-url"' in source
    assert source.count("return retiredUploadResponse();") == 4
    assert 'error: "upload_endpoint_retired"' in source
    assert 'service: "contribution-worker"' in source
    assert 'sign_endpoint: "/v1/uploads/sign"' in source


def test_retirement_router_does_not_call_legacy_upload_handlers() -> None:
    source = WORKER.read_text(encoding="utf-8")
    router = source[: source.index("function retiredUploadResponse")]
    assert "handleUpload(" not in router
    assert "handlePresignedUrl(" not in router
