"""
Connection management module for service communication.

This module provides connection pooling and request management
for efficient service-to-service communication.
"""

from typing import Dict, Optional, List, Any, Tuple
import uuid
import asyncio
import aiohttp
from datetime import datetime, timedelta
from dataclasses import dataclass
from .service.base import BaseServiceRequest, BaseServiceResponse


@dataclass
class RequestBatch:
    """A batch of requests to be processed together."""

    requests: List[BaseServiceRequest]
    futures: List[asyncio.Future]
    created_at: datetime


class ConnectionManager:
    """Manages HTTP connections and request batching for service communication.

    This class provides:
    - Connection pooling for HTTP requests
    - Request batching for high-throughput request
    - Request deduplication using request IDs
    """

    def __init__(self, max_connections: int = 100, batch_window_ms: int = 10):
        """Initialize the connection manager.

        Args:
            max_connections: Maximum number of concurrent connections in the pool
            batch_window_ms: Maximum time in milliseconds to wait before sending a batch
        """
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_connections = max_connections
        self.batch_window = batch_window_ms / 1000.0  # Convert to seconds
        self.pending_batches: Dict[Tuple[str, str], RequestBatch] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.request_cache: Dict[str, asyncio.Future] = {}
        self.cache_ttl = timedelta(minutes=5)
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Initialize the connection pool."""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections, ttl_dns_cache=300
            )
            self.session = aiohttp.ClientSession(
                connector=connector, timeout=aiohttp.ClientTimeout(total=30)
            )

    async def close(self) -> None:
        """Close all connections and clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None

        # Cancel any pending tasks
        for task in self.active_tasks.values():
            if not task.done():
                task.cancel()

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())

    async def _process_batch(
        self, service_url: str, batch: RequestBatch, request_id: str
    ) -> None:
        """Process a batch of requests.

        Args:
            service_url: The target service URL
            batch: The batch of requests to process
            request_id: The batch request ID
        """
        try:
            if not self.session or self.session.closed:
                await self.start()

            async with self.session.post(
                f"{service_url}/batch",
                json={"requests": [r.dict() for r in batch.requests]},
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Batch request failed: {error}")

                results = await response.json()

                # Complete all futures with their respective responses
                for future, result in zip(batch.futures, results):
                    if not future.done():
                        future.set_result(result)

        except Exception as e:
            # If there's an error, fail all requests in the batch
            for future in batch.futures:
                if not future.done():
                    future.set_exception(e)
        finally:
            # Clean up the batch
            async with self._lock:
                self.active_tasks.pop(request_id, None)

    async def send_request(
        self,
        request: BaseServiceRequest,
        service_url: str,
        use_batching: bool = True,
        deduplicate: bool = True,
    ) -> BaseServiceResponse:
        """Send a request to a service with optional batching and deduplication.

        Args:
            request: The service request to send
            service_url: The base URL of the target service
            use_batching: Whether to use request batching
            deduplicate: Whether to deduplicate requests

        Returns:
            The service response
        """
        if not self.session or self.session.closed:
            await self.start()

        # Generate a unique request ID for deduplication
        request_id = self._generate_request_id()

        # Check for duplicate requests if deduplication is enabled
        cache_key = f"{service_url}:{request.service_name}:{request.operation_name}:{str(request.raw_data)}"

        if deduplicate:
            async with self._lock:
                if cache_key in self.request_cache:
                    cached_future = self.request_cache[cache_key]
                    if not cached_future.done():
                        return await cached_future

                    # If the cached future is done but still in cache, check if it's too old
                    if (
                        datetime.now()
                        - cached_future.result().get("timestamp", datetime.min)
                    ) < self.cache_ttl:
                        return cached_future.result()

                # Create a new future for this request
                future = asyncio.Future()
                self.request_cache[cache_key] = future

                # Schedule cache cleanup
                asyncio.create_task(self._cleanup_cache(cache_key))
        else:
            future = asyncio.Future()

        # If batching is disabled, send the request immediately
        if not use_batching:
            return await self._send_single_request(service_url, request, future)

        # Otherwise, add to batch
        batch_key = (service_url, request.service_name)

        async with self._lock:
            if batch_key not in self.pending_batches:
                # Create a new batch
                batch = RequestBatch(
                    requests=[request], futures=[future], created_at=datetime.now()
                )
                self.pending_batches[batch_key] = batch

                # Schedule batch processing after the batch window
                task = asyncio.create_task(
                    self._process_batch_after_delay(service_url, batch_key, request_id)
                )
                self.active_tasks[request_id] = task
            else:
                # Add to existing batch
                batch = self.pending_batches[batch_key]
                batch.requests.append(request)
                batch.futures.append(future)

        return await future

    async def _process_batch_after_delay(
        self, service_url: str, batch_key: Tuple[str, str], request_id: str
    ) -> None:
        """Process a batch after the batch window has elapsed."""
        try:
            await asyncio.sleep(self.batch_window)

            async with self._lock:
                if batch_key in self.pending_batches:
                    batch = self.pending_batches.pop(batch_key)
                    await self._process_batch(service_url, batch, request_id)
        except Exception as e:
            # Handle any errors during batch processing
            async with self._lock:
                if batch_key in self.pending_batches:
                    batch = self.pending_batches.pop(batch_key)
                    for future in batch.futures:
                        if not future.done():
                            future.set_exception(e)

    async def _send_single_request(
        self, service_url: str, request: BaseServiceRequest, future: asyncio.Future
    ) -> BaseServiceResponse:
        """Send a single request without batching."""
        try:
            if not self.session or self.session.closed:
                await self.start()

            async with self.session.post(
                f"{service_url}/{request.service_name}/{request.operation_name}",
                json=request.raw_data,
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Request failed: {error}")

                result = await response.json()
                response_obj = BaseServiceResponse(
                    request_id=request.request_id, **result
                )

                if not future.done():
                    future.set_result(response_obj)

                return response_obj

        except Exception as e:
            if not future.done():
                future.set_exception(e)
            raise

    async def _cleanup_cache(self, cache_key: str) -> None:
        """Clean up old cache entries."""
        await asyncio.sleep(self.cache_ttl.total_seconds())
        async with self._lock:
            if cache_key in self.request_cache:
                del self.request_cache[cache_key]


# Global connection manager instance
connection_manager = ConnectionManager()
