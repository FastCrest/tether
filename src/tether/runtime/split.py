"""Tether Split — cloud-edge VLA orchestration.

Routes VLA inference between cloud GPU (big model, high latency) and
edge device (small model, low latency) based on network conditions
and task difficulty.

Usage:
    from tether.runtime.split import SplitOrchestrator
    orch = SplitOrchestrator(cloud_url="http://cloud:8000", edge_server=local_server)
    actions = orch.infer(image, instruction, state)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class InferenceTarget(Enum):
    CLOUD = "cloud"
    EDGE = "edge"
    FALLBACK = "fallback"


@dataclass
class SplitConfig:
    """Configuration for cloud-edge split inference."""

    cloud_url: str = ""
    edge_latency_budget_ms: float = 100.0
    cloud_latency_budget_ms: float = 500.0
    network_timeout_ms: float = 200.0
    fallback_mode: str = "last_action"  # "last_action", "zero", "edge_small"
    prefer: str = "edge"  # "edge", "cloud", "auto"
    health_check_interval_s: float = 5.0


@dataclass
class SplitResult:
    """Result from split inference."""

    actions: np.ndarray
    target_used: InferenceTarget
    latency_ms: float
    cloud_available: bool
    fallback_triggered: bool
    reason: str


class SplitOrchestrator:
    """Route VLA inference between cloud and edge based on conditions."""

    def __init__(self, config: SplitConfig | None = None):
        self.config = config or SplitConfig()
        self._cloud_available = False
        self._last_cloud_check = 0.0
        self._last_actions: np.ndarray | None = None
        self._cloud_latency_history: list[float] = []

    def check_cloud_health(self) -> bool:
        """Check if cloud endpoint is reachable."""
        if not self.config.cloud_url:
            self._cloud_available = False
            return False

        try:
            import httpx

            start = time.perf_counter()
            resp = httpx.get(
                f"{self.config.cloud_url}/health",
                timeout=self.config.network_timeout_ms / 1000,
            )
            latency = (time.perf_counter() - start) * 1000
            self._cloud_available = resp.status_code == 200
            self._cloud_latency_history.append(latency)
            if len(self._cloud_latency_history) > 20:
                self._cloud_latency_history = self._cloud_latency_history[-20:]
            self._last_cloud_check = time.time()
            return self._cloud_available
        except Exception:
            self._cloud_available = False
            return False

    def _should_check_cloud(self) -> bool:
        return time.time() - self._last_cloud_check > self.config.health_check_interval_s

    def _select_target(self) -> InferenceTarget:
        """Decide where to run inference."""
        if self.config.prefer == "edge":
            return InferenceTarget.EDGE

        if self.config.prefer == "cloud":
            if self._cloud_available:
                return InferenceTarget.CLOUD
            return InferenceTarget.FALLBACK

        # Auto mode: use cloud if available and latency is acceptable
        if self._cloud_available and self._cloud_latency_history:
            avg_latency = sum(self._cloud_latency_history) / len(self._cloud_latency_history)
            if avg_latency < self.config.cloud_latency_budget_ms:
                return InferenceTarget.CLOUD

        return InferenceTarget.EDGE

    def _get_fallback_actions(self, action_dim: int, chunk_size: int) -> np.ndarray:
        """Get fallback actions when neither cloud nor edge available."""
        if self.config.fallback_mode == "last_action" and self._last_actions is not None:
            return self._last_actions
        return np.zeros((chunk_size, action_dim), dtype=np.float32)

    def infer_cloud(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> SplitResult | None:
        """Run inference on cloud endpoint."""
        if not self.config.cloud_url:
            return None

        try:
            import httpx

            start = time.perf_counter()
            resp = httpx.post(
                f"{self.config.cloud_url}/act",
                json={"image": image_b64, "instruction": instruction, "state": state},
                timeout=self.config.cloud_latency_budget_ms / 1000,
            )
            latency = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                actions = np.array(data["actions"], dtype=np.float32)
                self._last_actions = actions
                return SplitResult(
                    actions=actions,
                    target_used=InferenceTarget.CLOUD,
                    latency_ms=latency,
                    cloud_available=True,
                    fallback_triggered=False,
                    reason=f"cloud inference in {latency:.1f}ms",
                )
        except Exception as e:
            logger.warning("Cloud inference failed: %s", e)
            self._cloud_available = False
        return None

    def infer_edge(
        self,
        edge_server: Any,
        image: np.ndarray | None = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> SplitResult:
        """Run inference on local edge server."""
        start = time.perf_counter()
        result = edge_server.predict(image=image, instruction=instruction, state=state)
        latency = (time.perf_counter() - start) * 1000

        actions = np.array(result.get("actions", []), dtype=np.float32)
        self._last_actions = actions

        return SplitResult(
            actions=actions,
            target_used=InferenceTarget.EDGE,
            latency_ms=latency,
            cloud_available=self._cloud_available,
            fallback_triggered=False,
            reason=f"edge inference in {latency:.1f}ms",
        )

    def infer(
        self,
        edge_server: Any,
        image: np.ndarray | None = None,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
        action_dim: int = 32,
        chunk_size: int = 50,
    ) -> SplitResult:
        """Run inference with automatic cloud/edge routing."""
        # Periodic health check
        if self._should_check_cloud():
            self.check_cloud_health()

        target = self._select_target()

        if target == InferenceTarget.CLOUD:
            result = self.infer_cloud(image_b64, instruction, state)
            if result is not None:
                return result
            # Cloud failed, fall through to edge
            logger.info("Cloud failed, falling back to edge")

        if target in (InferenceTarget.EDGE, InferenceTarget.CLOUD):
            return self.infer_edge(edge_server, image, instruction, state)

        # Fallback
        actions = self._get_fallback_actions(action_dim, chunk_size)
        return SplitResult(
            actions=actions,
            target_used=InferenceTarget.FALLBACK,
            latency_ms=0.0,
            cloud_available=False,
            fallback_triggered=True,
            reason=f"fallback: {self.config.fallback_mode}",
        )
