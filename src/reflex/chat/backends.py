"""HTTP backend that talks to the FastCrest proxy (Cloudflare Worker).

NOTE: `reflex chat` is an ONLINE convenience surface — it routes to a hosted
model via this proxy and REQUIRES network. Reflex's offline / air-gapped
guarantee is the *serving* path (`reflex serve` / `/act` inference runs fully
on-device); it does NOT extend to chat. Network failures raise OfflineError
that states this distinction. Set FASTCREST_PROXY_URL to self-host the proxy
if you need chat inside a closed network.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import httpx

DEFAULT_PROXY_URL = "https://chat.fastcrest.com"

_OFFLINE_MSG = (
    "`reflex chat` needs network: it routes to a hosted model at {url}. "
    "Reflex's offline/air-gapped guarantee covers `reflex serve` / `/act` "
    "inference (fully on-device), NOT the chat helper. Self-host the proxy and "
    "set FASTCREST_PROXY_URL to use chat inside a closed network."
)


@dataclass
class ChatBackend:
    """Stateless wrapper around POST /chat on the FastCrest proxy."""

    proxy_url: str = field(default_factory=lambda: os.environ.get("FASTCREST_PROXY_URL", DEFAULT_PROXY_URL))
    client_id: str = field(default_factory=lambda: os.environ.get("FASTCREST_CLIENT_ID", "reflex-cli"))
    timeout_s: float = 60.0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": messages, "client_id": self.client_id}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                r = client.post(f"{self.proxy_url}/chat", json=body)
        except httpx.HTTPError as e:  # ConnectError / timeout / DNS — i.e. no network
            raise OfflineError(_OFFLINE_MSG.format(url=self.proxy_url)) from e
        if r.status_code == 429:
            raise RateLimitError(r.json().get("message", "rate limit"))
        if r.status_code >= 400:
            raise ProxyError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> Iterator[dict[str, Any]]:
        """Yield parsed OpenAI delta chunks. Caller assembles into a final message."""
        body: dict[str, Any] = {"messages": messages, "client_id": self.client_id, "stream": True}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                with client.stream("POST", f"{self.proxy_url}/chat", json=body) as r:
                    if r.status_code == 429:
                        raise RateLimitError(r.read().decode().strip())
                    if r.status_code >= 400:
                        raise ProxyError(f"HTTP {r.status_code}: {r.read().decode()[:300]}")
                    for line in r.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            return
                        try:
                            yield json.loads(payload)
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPError as e:  # ConnectError / timeout / DNS — i.e. no network
            raise OfflineError(_OFFLINE_MSG.format(url=self.proxy_url)) from e

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{self.proxy_url}/health")
        r.raise_for_status()
        return r.json()


def assemble_stream(
    chunks: Iterator[dict[str, Any]],
    on_token: Callable[[str], None] | None = None,
    on_tool_call_progress: Callable[[int, str, str], None] | None = None,
) -> dict[str, Any]:
    """Consume an OpenAI streaming response, returning the final assembled message dict.

    on_token: called with each content fragment as it arrives.
    on_tool_call_progress: called with (index, name, arguments_so_far) as tool_call deltas arrive.
    """
    content_parts: list[str] = []
    # tool_calls keyed by index, each with id/type/function{name,arguments}
    tool_calls: dict[int, dict[str, Any]] = {}
    role = "assistant"
    finish_reason: str | None = None

    for chunk in chunks:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        if not finish_reason:
            finish_reason = choice.get("finish_reason")
        if "role" in delta:
            role = delta["role"] or role
        if delta.get("content"):
            content_parts.append(delta["content"])
            if on_token is not None:
                on_token(delta["content"])
        for tc_delta in delta.get("tool_calls") or []:
            idx = tc_delta.get("index", 0)
            tc = tool_calls.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if tc_delta.get("id"):
                tc["id"] = tc_delta["id"]
            fn_delta = tc_delta.get("function") or {}
            if fn_delta.get("name"):
                tc["function"]["name"] += fn_delta["name"]
            if fn_delta.get("arguments"):
                tc["function"]["arguments"] += fn_delta["arguments"]
                if on_tool_call_progress is not None:
                    on_tool_call_progress(idx, tc["function"]["name"], tc["function"]["arguments"])

    msg: dict[str, Any] = {"role": role, "content": "".join(content_parts) if content_parts else None}
    if tool_calls:
        msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return msg


class ProxyError(RuntimeError):
    pass


class RateLimitError(RuntimeError):
    pass


class OfflineError(RuntimeError):
    """Raised when `reflex chat` can't reach the hosted proxy (no network).

    Distinct from ProxyError (server reachable but errored): OfflineError means
    the offline/air-gap boundary was hit. Its message states that Reflex's
    offline guarantee is the serving path (`reflex serve` / `/act`), not chat.
    """
    pass
