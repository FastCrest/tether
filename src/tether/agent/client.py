"""Cloud client for the Tether Agent v0.1 API."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping

from tether.agent.models import (
    AgentCommand,
    CommandAck,
    EnrollRequest,
    EnrollResponse,
    FailureEventPayload,
    FleetHeartbeatPayload,
    HeartbeatPayload,
    JsonDict,
)

DEFAULT_TIMEOUT_SECONDS = 10.0


class AgentClientError(RuntimeError):
    """Raised for failed Cloud API calls."""


class AgentClient:
    def __init__(
        self,
        cloud_url: str,
        *,
        device_token: str | None = None,
        fleet_device_token: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        session: Any | None = None,
    ) -> None:
        self.cloud_url = cloud_url.rstrip("/")
        self.device_token = device_token
        self.fleet_device_token = fleet_device_token
        self.timeout_seconds = timeout_seconds
        self._session = session if session is not None else self._make_httpx_session(timeout_seconds)

    def enroll(self, request: EnrollRequest) -> EnrollResponse:
        data = self._request("POST", "/agent/enroll", json_body=request.to_dict(), auth=False)
        return EnrollResponse.from_dict(data)

    def heartbeat(self, device_id: str, payload: HeartbeatPayload) -> JsonDict:
        return self._request(
            "POST",
            f"/agent/devices/{urllib.parse.quote(device_id, safe='')}/heartbeat",
            json_body=payload.to_dict(),
            auth=True,
        )

    def poll_commands(self, device_id: str, *, after: str | None = None) -> list[AgentCommand]:
        params = {"after": after} if after else None
        data = self._request(
            "GET",
            f"/agent/devices/{urllib.parse.quote(device_id, safe='')}/commands",
            params=params,
            auth=True,
        )
        commands = data if isinstance(data, list) else data.get("commands", [])
        return [AgentCommand.from_dict(command) for command in commands]

    def ack_command(self, device_id: str, command_id: str, ack: CommandAck) -> JsonDict:
        return self._request(
            "POST",
            (
                f"/agent/devices/{urllib.parse.quote(device_id, safe='')}/commands/"
                f"{urllib.parse.quote(command_id, safe='')}/ack"
            ),
            json_body=ack.to_dict(),
            auth=True,
        )

    def create_failure(
        self,
        device_id: str,
        payload: FailureEventPayload | Mapping[str, Any],
        *,
        device_token: str | None = None,
    ) -> JsonDict:
        failure_token = device_token or self.fleet_device_token
        if not failure_token:
            raise AgentClientError("fleet device token is required for failure uploads")
        if hasattr(payload, "to_dict"):
            body = payload.to_dict()
        else:
            body = dict(payload)
        return self._request(
            "POST",
            f"/fleet/devices/{urllib.parse.quote(device_id, safe='')}/failures",
            json_body=body,
            auth=True,
            auth_token=failure_token,
        )

    def fleet_heartbeat(
        self,
        device_id: str,
        payload: FleetHeartbeatPayload | Mapping[str, Any],
        *,
        device_token: str | None = None,
    ) -> JsonDict:
        heartbeat_token = device_token or self.fleet_device_token
        if not heartbeat_token:
            raise AgentClientError("fleet device token is required for fleet heartbeats")
        if hasattr(payload, "to_dict"):
            body = payload.to_dict()
        else:
            body = dict(payload)
        return self._request(
            "POST",
            f"/fleet/devices/{urllib.parse.quote(device_id, safe='')}/heartbeat",
            json_body=body,
            auth=True,
            auth_token=heartbeat_token,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        auth: bool,
        auth_token: str | None = None,
    ) -> Any:
        url = self._url(path, params)
        headers = {"Content-Type": "application/json"}
        if auth:
            token = auth_token or self.device_token
            if not token:
                raise AgentClientError("device token is required for authenticated agent calls")
            headers["Authorization"] = f"Bearer {token}"

        if self._session is not None:
            response = self._session.request(
                method,
                url,
                headers=headers,
                json=dict(json_body) if json_body is not None else None,
                timeout=self.timeout_seconds,
            )
            return self._decode_response(response)
        return self._urllib_request(method, url, headers, json_body)

    def _url(self, path: str, params: Mapping[str, Any] | None = None) -> str:
        url = f"{self.cloud_url}{path}"
        if params:
            query = urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
            if query:
                url = f"{url}?{query}"
        return url

    def _decode_response(self, response: Any) -> Any:
        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            text = getattr(response, "text", "")
            raise AgentClientError(f"agent API request failed with HTTP {status_code}: {text}")
        if hasattr(response, "json"):
            return response.json()
        text = getattr(response, "text", "")
        return json.loads(text) if text else {}

    def _urllib_request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any] | None,
    ) -> Any:
        body = None
        if json_body is not None:
            body = json.dumps(dict(json_body)).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                text = response.read().decode("utf-8")
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AgentClientError(f"agent API request failed with HTTP {exc.code}: {detail}") from exc

    @staticmethod
    def _make_httpx_session(timeout_seconds: float) -> Any | None:
        try:
            import httpx
        except Exception:
            return None
        return httpx.Client(timeout=timeout_seconds)


def make_default_client(
    cloud_url: str,
    *,
    device_token: str | None = None,
    fleet_device_token: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> AgentClient:
    return AgentClient(
        cloud_url,
        device_token=device_token,
        fleet_device_token=fleet_device_token,
        timeout_seconds=timeout_seconds,
    )
