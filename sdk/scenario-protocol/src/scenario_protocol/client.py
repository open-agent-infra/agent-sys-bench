"""ScenarioClient — HTTP dispatch to agent fast-service endpoints.

The loadgen uses this client to send requests to the agent service. No agent
code is imported. The client serializes the pydantic Request model to JSON,
POSTs to f"{endpoint}/{rpc_name}", and deserializes the response into the
scenario-defined Result model.
"""

from __future__ import annotations

import time
import logging
from typing import Type, TypeVar

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_TIMEOUT = 3600  # SWE-bench tasks can take 10+ minutes


class ScenarioClient:
    """HTTP client for dispatching scenario requests to an agent service.

    Args:
        endpoint: Base URL of the agent service (e.g. "http://mini-sweagent-service:1801")
        timeout: Per-request timeout in seconds.
    """

    def __init__(self, endpoint: str, timeout: int = DEFAULT_TIMEOUT):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def dispatch(
        self,
        rpc_name: str,
        request: BaseModel,
        response_type: Type[T] | None = None,
        param_name: str = "req",
    ) -> T | None:
        """Send a request to the agent service and return the deserialized result.

        Args:
            rpc_name: Fast-service RPC path (e.g. "agsbench_mini_swe_agent.request.solve_instance")
            request: Pydantic model instance (the request body)
            response_type: Pydantic model class to deserialize the response into.
                If None, returns None.
            param_name: The fast-service function's parameter name. fast-service
                wraps the body as {param_name: <request>} on the wire (matching
                the function signature). Default "req" — agents should name
                their request parameter "req" to match this convention.

        Returns:
            Deserialized response model, or None if response_type is None.

        Raises:
            RuntimeError: If the HTTP request fails or the response cannot be parsed.
        """
        url = f"{self.endpoint}/{rpc_name}"
        body = {param_name: request.model_dump(mode="json")}

        start = time.perf_counter()
        try:
            resp = self._session.post(url, json=body, timeout=self.timeout)
            elapsed = time.perf_counter() - start
        except requests.RequestException as e:
            elapsed = time.perf_counter() - start
            logger.error(f"dispatch {rpc_name} failed after {elapsed:.2f}s: {e}")
            raise RuntimeError(f"HTTP dispatch to {url} failed: {e}") from e

        if resp.status_code != 200:
            elapsed = time.perf_counter() - start
            raise RuntimeError(
                f"dispatch {rpc_name} returned HTTP {resp.status_code} "
                f"after {elapsed:.2f}s: {resp.text[:500]}"
            )

        if response_type is None:
            return None

        try:
            data = resp.json()
            return response_type.model_validate(data)
        except Exception as e:
            logger.error(f"dispatch {rpc_name} response parse failed: {e}")
            raise RuntimeError(f"Failed to parse response from {url}: {e}") from e

    def health(self) -> bool:
        """Check if the agent service is reachable (GET /docs)."""
        try:
            resp = self._session.get(f"{self.endpoint}/docs", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def close(self):
        self._session.close()
