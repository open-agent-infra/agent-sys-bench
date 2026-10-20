"""
Service manager module for centralized service management.

This module provides a central registry for all services in the application,
allowing for dependency injection, service discovery with health monitoring,
and performance optimizations like connection pooling and request batching.
"""

import asyncio
import time
import logging
import uuid
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any, Union, Tuple

from .service.base import (
    BaseService,
    BaseServiceInstance,
    BaseServiceRequest,
    BaseServiceResponse,
)
from .connection_manager import connection_manager

# Configure logging
logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Represents the health status of a service."""

    HEALTHY = auto()
    UNHEALTHY = auto()
    DEGRADED = auto()
    UNKNOWN = auto()


class HealthCheckResult:
    """Represents the result of a health check."""

    def __init__(
        self, status: ServiceStatus, message: str = "", details: Dict[str, Any] = None
    ):
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()

    def is_healthy(self) -> bool:
        return self.status == ServiceStatus.HEALTHY

    def __str__(self) -> str:
        return f"{self.status.name}: {self.message}"


class BaseServiceManager:
    """Manages service instances and provides service discovery with health monitoring.

    This class acts as a central registry for all services in the application,
    allowing components to retrieve service instances by name without
    direct dependencies on concrete implementations. It provides:
    - Connection pooling for efficient service communication
    - Request batching for high-throughput request
    - Request deduplication using request IDs
    - Health monitoring and automatic failure handling

    Attributes:
        services (Dict[str, BaseService]): Dictionary mapping service names to service instances
        health_checks (Dict[str, Callable[[BaseService], HealthCheckResult]]): Health check functions
        health_status (Dict[str, HealthCheckResult]): Current health status of services
        unhealthy_services (Set[str]): Set of currently unhealthy services
        health_check_interval (int): Interval in seconds between health checks
        _health_check_task (Optional[asyncio.Task]): Background health check task
        _connection_manager: Manages HTTP connections and request batching
    """

    def __init__(
        self,
        services: Optional[Dict[str, BaseService]] = None,
        health_check_interval: int = 30,
        max_connections: int = 100,
        batch_window_ms: int = 10,
    ):
        """Initialize the service manager with health monitoring and performance optimizations.

        Args:
            services: Optional dictionary of pre-configured services
            health_check_interval: Interval in seconds between health checks
            max_connections: Maximum number of concurrent connections in the pool
            batch_window_ms: Maximum time in milliseconds to wait before sending a batch
        """
        self.services: Dict[str, BaseService] = services if services is not None else {}
        self.health_checks: Dict[str, Callable[[BaseService], HealthCheckResult]] = {}
        self.health_status: Dict[str, HealthCheckResult] = {}
        self.unhealthy_services: Set[str] = set()
        self.health_check_interval = health_check_interval
        self._health_check_task: Optional[asyncio.Task] = None
        self._connection_initialized = False
        self._max_connections = max_connections
        self._batch_window_ms = batch_window_ms
        self._init_lock = asyncio.Lock()

    async def _ensure_connection_manager_initialized(self) -> None:
        """Ensure the connection manager is initialized."""
        if not self._connection_initialized:
            async with self._init_lock:
                if not self._connection_initialized:  # Double-checked locking
                    connection_manager.max_connections = self._max_connections
                    connection_manager.batch_window = self._batch_window_ms / 1000.0
                    await connection_manager.start()
                    self._connection_initialized = True

    async def _initialize_connection_manager(self) -> None:
        """Initialize the connection manager asynchronously."""
        try:
            await self._ensure_connection_manager_initialized()
            logger.info("Connection manager initialized successfully")
        except Exception as e:
            logger.exception(f"Failed to initialize connection manager: {e}")
            raise

    async def call_service(
        self,
        service_name: str,
        operation_name: str,
        data: Dict[str, Any],
        model: Optional[str] = None,
        use_batching: bool = True,
        deduplicate: bool = True,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a service operation with performance optimizations.

        This method provides several performance optimizations:
        - Connection pooling for HTTP requests
        - Request batching for high-throughput request
        - Request deduplication using request IDs

        Args:
            service_name: Name of the service to call
            operation_name: Name of the operation to perform
            data: Operation-specific data
            model: Optional model to use for the operation
            use_batching: Whether to enable request batching
            deduplicate: Whether to enable request deduplication
            request_id: Optional request ID for deduplication (auto-generated if not provided)
            metadata: Additional metadata for the request

        Returns:
            The operation result

        Raises:
            ValueError: If the service is not found or other validation fails
        """
        # Ensure connection manager is initialized
        await self._ensure_connection_manager_initialized()

        if service_name not in self.services:
            raise ValueError(f"Service not found: {service_name}")

        service = self.services[service_name]

        # Create the request with metadata and request ID
        request = BaseServiceRequest(
            service_name=service_name,
            operation_name=operation_name,
            raw_data=data,
            request_id=request_id or str(uuid.uuid4()),
            metadata=metadata or {},
        )

        try:
            # Use the connection manager for HTTP services, fall back to direct call for local services
            if hasattr(service, "is_http") and service.is_http:
                # For HTTP services, use the connection manager
                response = await connection_manager.send_request(
                    request=request,
                    service_url=service.base_url,
                    use_batching=use_batching,
                    deduplicate=deduplicate,
                )
                return response.raw_data
            else:
                # For local services, use direct method call
                response = await service.serve_async(request, model=model)
                return response.raw_data

        except Exception as e:
            logger.exception(
                f"Error calling service {service_name}.{operation_name}: {str(e)}",
                exc_info=True,
            )
            raise

    async def batch_call_services(
        self,
        service_requests: List[BaseServiceRequest],
        max_batch_size: int = 10,
        deduplicate: bool = True,
    ) -> Dict[str, Union[Dict[str, Any], Exception]]:
        """Batch multiple service calls together for better performance.

        This method provides significant performance improvements by:
        - Batching multiple requests into a single HTTP call when possible
        - Using connection pooling
        - Parallelizing independent service calls
        - Supporting request deduplication

        Args:
            service_requests: List of service requests to batch
            max_batch_size: Maximum number of requests to include in a single batch
            deduplicate: Whether to enable request deduplication

        Returns:
            Dictionary mapping request IDs to their results or exceptions
        """
        # Ensure connection manager is initialized
        await self._ensure_connection_manager_initialized()

        # Group requests by service and operation for batching
        batched_requests: Dict[Tuple[str, str], List[BaseServiceRequest]] = {}
        results: Dict[str, Union[Dict[str, Any], Exception]] = {}
        tasks = []

        for req in service_requests:
            if req.service_name not in self.services:
                results[req.request_id] = ValueError(
                    f"Service not found: {req.service_name}"
                )
                continue

            service = self.services[req.service_name]
            if hasattr(service, "is_http") and service.is_http:
                # For HTTP services, group by (service_name, operation_name) for batching
                key = (req.service_name, req.operation_name)
                batched_requests.setdefault(key, []).append(req)
            else:
                # For local services, execute immediately
                tasks.append(
                    asyncio.create_task(self._process_single_request(req, results))
                )

        # Process batched HTTP requests
        for (service_name, operation_name), requests in batched_requests.items():
            # Split into batches of max_batch_size
            for i in range(0, len(requests), max_batch_size):
                batch = requests[i : i + max_batch_size]
                tasks.append(
                    asyncio.create_task(
                        self._process_batch_requests(
                            service_name, operation_name, batch, results, deduplicate
                        )
                    )
                )

        # Wait for all tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def _process_single_request(
        self,
        request: BaseServiceRequest,
        results: Dict[str, Union[Dict[str, Any], Exception]],
    ) -> None:
        """Process a single service request and store the result.

        Args:
            request: The service request to process
            results: Dictionary to store the results
        """
        try:
            service = self.services[request.service_name]
            response = await service.serve_async(request)
            results[request.request_id] = response.raw_data
        except Exception as e:
            logger.exception(
                f"Error processing request {request.request_id}: {str(e)}",
                exc_info=True,
            )
            results[request.request_id] = e

    async def _process_batch_requests(
        self,
        service_name: str,
        operation_name: str,
        requests: List[BaseServiceRequest],
        results: Dict[str, Union[Dict[str, Any], Exception]],
        deduplicate: bool = True,
    ) -> None:
        """Process a batch of requests for the same service and operation.

        Args:
            service_name: Name of the service
            operation_name: Name of the operation
            requests: List of requests to process
            results: Dictionary to store the results
            deduplicate: Whether to enable request deduplication
        """
        service = self.services[service_name]

        try:
            # For HTTP services, use the connection manager
            if hasattr(service, "is_http") and service.is_http:
                batch_results = await connection_manager.batch_send_requests(
                    requests=requests,
                    service_url=service.base_url,
                    deduplicate=deduplicate,
                )
                results.update(batch_results)
            else:
                # For local services, process each request individually
                for req in requests:
                    await self._process_single_request(req, results)
        except Exception as e:
            logger.exception(
                f"Error processing batch {service_name}.{operation_name}: {str(e)}",
                exc_info=True,
            )
            # Store the error for all requests in the batch
            for req in requests:
                results[req.request_id] = e

    def add_service(
        self,
        service_name: str,
        service: BaseService,
        health_check: Optional[Callable[[BaseService], HealthCheckResult]] = None,
    ) -> None:
        """Add a service to the manager with an optional health check.

        Args:
            service_name: Name of the service to add
            service: The service instance to add
            health_check: Optional health check function for this service
        """
        self.services[service_name] = service
        if health_check:
            self.health_checks[service_name] = health_check
        # Initial health status
        self.health_status[service_name] = HealthCheckResult(
            ServiceStatus.UNKNOWN, "Initializing"
        )

        # Start health checks if not already running
        if self._health_check_task is None or self._health_check_task.done():
            self._start_health_checks()

    def remove_service(self, service_name: str) -> None:
        """Remove a service from the manager.

        Args:
            service_name: Name of the service to remove
        """
        if service_name in self.services:
            del self.services[service_name]
        if service_name in self.health_checks:
            del self.health_checks[service_name]
        if service_name in self.health_status:
            del self.health_status[service_name]
        self.unhealthy_services.discard(service_name)

        # Stop health checks if no services left
        if not self.services and self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None

    def get_service(self, service_name: str, check_health: bool = True) -> BaseService:
        """Get a service by name, optionally checking its health status.

        Args:
            service_name: Name of the service to retrieve
            check_health: If True, will raise an exception if service is unhealthy

        Returns:
            The requested service instance

        Raises:
            ValueError: If service is not found or is unhealthy and check_health is True
        """
        service = self.services.get(service_name)
        if service is None:
            raise ValueError(f"Service {service_name} not found")

        if check_health and service_name in self.unhealthy_services:
            last_status = self.health_status.get(
                service_name,
                HealthCheckResult(ServiceStatus.UNKNOWN, "No health status available"),
            )
            raise ValueError(f"Service {service_name} is unhealthy: {last_status}")

        return service

    def get_service_list(self, only_healthy: bool = False) -> List[BaseService]:
        """Get a list of all services, optionally filtering by health status.

        Args:
            only_healthy: If True, returns only healthy services

        Returns:
            List of service instances
        """
        if not only_healthy:
            return list(self.services.values())

        # Get all services that are not in the unhealthy_services set
        # and have a health status that is not UNHEALTHY
        healthy_services = []
        for name, svc in self.services.items():
            if name not in self.unhealthy_services:
                # Double check the health status if available
                health_status = self.health_status.get(name)
                if (
                    health_status is None
                    or health_status.status != ServiceStatus.UNHEALTHY
                ):
                    healthy_services.append(svc)

        return healthy_services

    def get_health_status(self, service_name: str) -> HealthCheckResult:
        """Get the current health status of a service.

        Args:
            service_name: Name of the service to check

        Returns:
            HealthCheckResult with the current status
        """
        return self.health_status.get(
            service_name, HealthCheckResult(ServiceStatus.UNKNOWN, "Service not found")
        )

    def is_healthy(self, service_name: str) -> bool:
        """Check if a service is currently healthy.

        Args:
            service_name: Name of the service to check

        Returns:
            bool: True if the service is healthy, False otherwise
        """
        return service_name not in self.unhealthy_services

    async def _check_service_health(
        self, service_name: str, service: BaseService
    ) -> None:
        """Check the health of a single service and update its status."""
        try:
            if service_name in self.health_checks:
                # If we're in an event loop, run the health check directly
                try:
                    loop = asyncio.get_running_loop()
                    if asyncio.iscoroutinefunction(self.health_checks[service_name]):
                        result = await self.health_checks[service_name](service)
                    else:
                        # Run sync health check in a thread
                        result = await loop.run_in_executor(
                            None, self.health_checks[service_name], service
                        )
                except RuntimeError:
                    # No event loop, run synchronously
                    if asyncio.iscoroutinefunction(self.health_checks[service_name]):
                        result = await self.health_checks[service_name](service)
                    else:
                        result = self.health_checks[service_name](service)
            else:
                # Default health check: check if service has any instances
                instances = service.get_instance_list()
                if instances:
                    result = HealthCheckResult(
                        ServiceStatus.HEALTHY, f"Service has {len(instances)} instances"
                    )
                else:
                    result = HealthCheckResult(
                        ServiceStatus.UNHEALTHY, "No instances available"
                    )

            self.health_status[service_name] = result

            # Update unhealthy services set
            if result.is_healthy():
                self.unhealthy_services.discard(service_name)
            else:
                self.unhealthy_services.add(service_name)

        except Exception as e:
            error_result = HealthCheckResult(
                ServiceStatus.UNHEALTHY,
                f"Health check failed: {str(e)}",
                {"error": str(e), "type": type(e).__name__},
            )
            self.health_status[service_name] = error_result
            self.unhealthy_services.add(service_name)

    async def _health_check_loop(self) -> None:
        """Background task that periodically checks service health."""
        while True:
            try:
                # Check all services with health checks
                tasks = [
                    self._check_service_health(name, svc)
                    for name, svc in self.services.items()
                    if name in self.health_checks
                    or not self.health_checks  # Check all if no specific checks
                ]

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                # Log error but keep the health check loop running
                print(f"Error in health check loop: {e}")

            await asyncio.sleep(self.health_check_interval)

    def _start_health_checks(self) -> None:
        """Start the background health check task."""
        if self._health_check_task is None or self._health_check_task.done():
            # Create a new event loop if one isn't running
            try:
                loop = asyncio.get_running_loop()
                self._health_check_task = loop.create_task(self._health_check_loop())
            except RuntimeError:
                # No event loop running, create a new one in a separate thread
                self._start_health_checks_in_thread()

    def _start_health_checks_in_thread(self) -> None:
        """Start health checks in a separate thread with its own event loop."""
        import threading

        def run_health_checks():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._health_check_task = loop.create_task(self._health_check_loop())
            loop.run_forever()

        self._health_check_thread = threading.Thread(
            target=run_health_checks, daemon=True, name="HealthCheckThread"
        )
        self._health_check_thread.start()

    async def stop(self) -> None:
        """Stop the service manager and clean up resources asynchronously."""
        if hasattr(self, "_health_check_task") and self._health_check_task:
            if not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except (asyncio.CancelledError, RuntimeError):
                    pass
            self._health_check_task = None

    def stop_sync(self) -> None:
        """Stop the service manager and clean up resources synchronously."""
        if hasattr(self, "_health_check_task") and self._health_check_task:
            if not self._health_check_task.done():
                self._health_check_task.cancel()
            self._health_check_task = None

        # If we have a dedicated health check thread, stop it
        if (
            hasattr(self, "_health_check_thread")
            and self._health_check_thread.is_alive()
        ):
            # Try to get the event loop and stop it
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
            self._health_check_thread.join(timeout=1.0)
