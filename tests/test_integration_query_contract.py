"""RTSM query contract at calabi-inc/rtsm 85b94e603b25e32fe9c3484070807081da07dca0.

Source: rtsm/api/server.py (_obj_detail, list_objects, semantic_search,
spatial_search) and tests/test_api_contract.py. These transport tests do not
establish policy consumption, provider performance or freshness admission.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import requests

from tether.integrations import connector


@pytest.fixture
def transport(monkeypatch):
    calls = []
    body: Any = None
    status = 200

    def get(url, **kwargs):
        calls.append((url, kwargs))
        response = requests.Response()
        response.status_code = status
        response.url = url
        response.json = lambda: deepcopy(body)
        return response

    def configure(payload, code=200):
        nonlocal body, status
        body, status = payload, code
        return calls

    monkeypatch.setattr(connector.requests, "get", get)
    return configure


def test_objects_preserves_vectors_pagination_and_age_fields(transport):
    body = {
        "total": 3,
        "offset": 1,
        "limit": 1,
        "count": 1,
        "objects": [
            {
                "id": "obj_7",
                "xyz_world": [0.5, 0.8, 1.5],
                "label_primary": "mug",
                "confirmed": True,
                "stability": 0.8,
                "emb_mean": [0.0] * 768,
                "last_seen_wall_utc": 1751000000.0,
            }
        ],
    }
    calls = transport(body)
    assert (
        connector.query_objects(
            "http://rtsm.invalid/",
            include_vectors=True,
            offset=1,
            limit=1,
        )
        == body
    )
    assert calls == [
        (
            "http://rtsm.invalid/objects",
            {
                "params": {"include_vectors": True, "offset": 1, "limit": 1},
                "timeout": 5,
            },
        )
    ]


def test_semantic_query_preserves_atomic_pose_and_results(transport):
    body = {
        "query": "mug",
        "robot_pose": {"frame_epoch": 2, "age_s": 0.1},
        "results": [{"id": "obj_7", "score": 0.42, "confirmed": True}],
    }
    calls = transport(body)
    assert connector.semantic_search("http://rtsm.invalid/", "mug", top_k=3) == body
    assert calls[0][1]["params"] == {"query": "mug", "top_k": 3}


def test_spatial_radius_reaches_upstream_in_meters(transport):
    body = {
        "center": [1, 2, 3],
        "radius_m": 0.25,
        "total": 0,
        "offset": 0,
        "limit": 50,
        "count": 0,
        "results": [],
    }
    calls = transport(body)
    # Keep the public Python argument `radius`, but transmit the server's name.
    assert connector.spatial_search("http://rtsm.invalid/", 1, 2, 3, radius=0.25) == body
    assert calls == [
        (
            "http://rtsm.invalid/search/spatial",
            {
                "params": {"x": 1, "y": 2, "z": 3, "radius_m": 0.25},
                "timeout": 5,
            },
        )
    ]


@pytest.mark.parametrize(
    "invoke,key",
    [
        (lambda: connector.query_objects("http://rtsm.invalid"), "objects"),
        (lambda: connector.semantic_search("http://rtsm.invalid", "mug"), "results"),
        (lambda: connector.spatial_search("http://rtsm.invalid", 1, 2, 3), "results"),
    ],
)
@pytest.mark.parametrize("malformed", [None, [], {"error": "unavailable"}, "scalar", 4])
def test_invalid_response_is_not_accepted_as_a_snapshot(transport, invoke, key, malformed):
    transport(malformed)
    with pytest.raises(ValueError, match=key):
        invoke()


@pytest.mark.parametrize("collection", [None, {}, "objects", [None], ["obj_7"]])
def test_malformed_object_collection_refuses(transport, collection):
    transport({"objects": collection})
    with pytest.raises(ValueError, match="objects"):
        connector.query_objects("http://rtsm.invalid")


def test_upstream_http_failure_is_preserved(transport):
    transport({"detail": "proximity index unavailable"}, code=503)
    with pytest.raises(requests.HTTPError):
        connector.spatial_search("http://rtsm.invalid", 1, 2, 3)
