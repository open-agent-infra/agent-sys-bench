"""
Base service module defining service interfaces.

This module provides the base classes and interfaces for all services
in the application, establishing a consistent service architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Literal, Optional, FrozenSet
from pydantic import BaseModel, Field, field_validator
import random
import uuid
import re
from datetime import datetime
import time


# Supported scheduling policy types
SchedulingPolicyType = Literal["round_robin", "random", "least_busy", "first_available"]

# Supported stateful routing assignment strategies
StatefulAssignmentStrategyType = Literal[
    "consistent_hashing", "load_balanced", "explicit"
]


def validate_tag_format(tag: str) -> bool:
    """Validate that a tag follows the allowed format.

    Tags must be non-empty strings containing only alphanumeric characters,
    underscores, hyphens, forward slashes, and dots (to support model names).

    Args:
        tag: The tag string to validate

    Returns:
        True if the tag is valid, False otherwise
    """
    if not tag or not isinstance(tag, str):
        return False
    return bool(re.match(r"^[a-zA-Z0-9_/.-]+$", tag))


class BaseServiceInstanceConfig(BaseModel):
    """Configuration for a service instance.

    Attributes:
        location (str): Location of the service instance (local:auto or ip:port)
        api_key (str): API key for the service instance (optional)
        tags (FrozenSet[str]): Set of tags for categorizing this instance
    """

    location: str = "local:auto"  # or "ip:port"
    api_key: Optional[str] = None
    tags: FrozenSet[str] = Field(default_factory=frozenset)
    caching_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuration for caching, e.g., {'enabled': True, 'default_ttl': 3600}",
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate that all tags follow the allowed format."""
        if v is None:
            return frozenset()

        # Convert to frozenset if it's not already
        if not isinstance(v, frozenset):
            v = frozenset(v) if v else frozenset()

        # Validate all tags
        invalid_tags = [tag for tag in v if not validate_tag_format(tag)]
        if invalid_tags:
            raise ValueError(
                f"Invalid tag format(s): {invalid_tags}. Tags must be "
                "non-empty strings containing only alphanumeric characters, "
                "underscores, hyphens, forward slashes, and dots."
            )
        return v


class BaseServiceRequest(BaseModel):
    """Request to a service.

    Attributes:
        service_name (str): Name of the target service
        operation_name (str): Name of the operation to perform
        raw_data (Dict): Operation-specific request data
        hash_key (str): Optional key for stateful routing (e.g., collection_name, user_id)
        request_id (str): Unique identifier for the request (auto-generated if not provided)
        timestamp (datetime): When the request was created
        metadata (Dict): Additional metadata for the request
    """

    service_name: str
    operation_name: str
    raw_data: Dict = Field(default_factory=dict)
    hash_key: Optional[str] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def model_dump(self, *, mode: str = "python", **kwargs):
        if mode == "json":
            return super().model_dump(mode="json", **kwargs)
        data = super().model_dump(**kwargs)
        if isinstance(self.timestamp, datetime):
            data["timestamp"] = self.timestamp.isoformat()
        return data

    def get_statistics(self) -> dict:
        """Get statistics for the service request."""
        return {
            "service_name": self.service_name,
            "operation_name": self.operation_name,
            "raw_data_keys": (
                list(self.raw_data.keys()) if isinstance(self.raw_data, dict) else []
            ),
            "hash_key": self.hash_key,
            "request_id": self.request_id,
            "timestamp": (
                self.timestamp.isoformat()
                if isinstance(self.timestamp, datetime)
                else self.timestamp
            ),
            "metadata_keys": (
                list(self.metadata.keys()) if isinstance(self.metadata, dict) else []
            ),
        }


class BaseServiceResponse(BaseModel):
    """Response from a service.

    Attributes:
        service_name (str): Name of the responding service
        instance_name (str): Name of the specific service instance
        operation_name (str): Name of the performed operation
        raw_data (Dict): Operation-specific response data
        request_id (str): ID of the original request
        timestamp (datetime): When the response was created
        metadata (Dict): Additional metadata for the response
    """

    service_name: str
    instance_name: str
    operation_name: str
    raw_data: Dict = Field(default_factory=dict)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = Field(
        default=True, description="Indicates if the operation was successful"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the operation failed"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    elapsed_time: Optional[float] = Field(
        default=None, description="Elapsed time for the operation in seconds"
    )
    scheduling_time: Optional[float] = Field(
        default=None, description="Time taken to schedule the request in seconds"
    )
    from_cache: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cache data metadata if the response was served from cache",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def model_dump(self, *, mode: str = "python", **kwargs):
        if mode == "json":
            return super().model_dump(mode="json", **kwargs)
        data = super().model_dump(**kwargs)
        if isinstance(self.timestamp, datetime):
            data["timestamp"] = self.timestamp.isoformat()
        return data

    def get_statistics(self) -> dict:
        """Get statistics for the service response."""
        return {
            "service_name": self.service_name,
            "instance_name": self.instance_name,
            "operation_name": self.operation_name,
            "raw_data_keys": (
                list(self.raw_data.keys()) if isinstance(self.raw_data, dict) else []
            ),
            "request_id": self.request_id,
            "success": self.success,
            "error": self.error,
            "timestamp": (
                self.timestamp.isoformat()
                if isinstance(self.timestamp, datetime)
                else self.timestamp
            ),
            "elapsed_time": self.elapsed_time,
            "scheduling_time": self.scheduling_time,
            "from_cache": self.from_cache,
            "metadata_keys": (
                list(self.metadata.keys()) if isinstance(self.metadata, dict) else []
            ),
        }


class BaseServiceInstance(ABC):
    def __init__(
        self,
        service_name: str,
        instance_name: str,
        group_name: str,
        config: BaseServiceInstanceConfig,
    ):
        """Initialize a service instance.

        Args:
            service_name: Name of the service this instance belongs to
            instance_name: Unique name for this instance
            group_name: Name of the group this instance belongs to
            config: Configuration for this instance (includes tags)

        Raises:
            ValueError: If any tag has an invalid format (validated by config)
        """
        self.service_name: str = service_name
        self.instance_name: str = instance_name
        self.group_name: str = group_name
        self.config: BaseServiceInstanceConfig = config
        self.active_requests: int = 0
        self.total_requests: int = 0

        # Get tags from config (validation already done by Pydantic)
        self.tags: FrozenSet[str] = config.tags

    @abstractmethod
    def launch(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def _serve_impl(self, request: BaseServiceRequest) -> BaseServiceResponse:
        """Implementation of serve logic in subclasses."""
        pass

    @abstractmethod
    async def _serve_async_impl(
        self, request: BaseServiceRequest
    ) -> BaseServiceResponse:
        """Implementation of async serve logic in subclasses."""
        pass

    def serve(self, request: BaseServiceRequest) -> BaseServiceResponse:
        """Serve a request with request tracking.

        This method handles the tracking of active and total requests,
        delegating the actual implementation to _serve_impl.

        Args:
            request: The service request to serve

        Returns:
            The service response
        """
        self.active_requests += 1
        self.total_requests += 1

        try:
            response = self._serve_impl(request)
            response.instance_name = self.instance_name
            return response
        except Exception as e:
            # Handle exceptions and ensure active_requests is decremented
            response = BaseServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=request.operation_name,
                raw_data={},
                success=False,
                error=str(e),
                timestamp=datetime.utcnow(),
                metadata=request.metadata,
            )
            return response
        finally:
            self.active_requests -= 1

    async def serve_async(self, request: BaseServiceRequest) -> BaseServiceResponse:
        """Serve a request asynchronously with request tracking.

        This method handles the tracking of active and total requests,
        delegating the actual implementation to _serve_async_impl.

        Args:
            request: The service request to serve

        Returns:
            The service response
        """
        self.active_requests += 1
        self.total_requests += 1

        try:
            response = await self._serve_async_impl(request)
            response.instance_name = self.instance_name
            return response
        except Exception as e:
            # Handle exceptions and ensure active_requests is decremented
            response = BaseServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=request.operation_name,
                raw_data={},
                success=False,
                error=str(e),
                timestamp=datetime.utcnow(),
                metadata=request.metadata,
            )
            return response
        finally:
            self.active_requests -= 1

    def call(self, request: BaseServiceRequest) -> BaseServiceResponse:
        """Synchronous call interface for serving requests.

        This method allows the instance to be called like a function,
        delegating to the serve method.

        Args:
            request: The service request to serve

        Returns:
            The service response
        """
        return self.serve(request)

    async def call_async(self, request: BaseServiceRequest) -> BaseServiceResponse:
        """Asynchronous call interface for serving requests.

        This method allows the instance to be called like a function,
        delegating to the serve_async method.

        Args:
            request: The service request to serve

        Returns:
            The service response
        """
        return await self.serve_async(request)

    def __call__(self, request: BaseServiceRequest) -> BaseServiceResponse:
        return self.serve(request)

    async def __call_async__(self, request: BaseServiceRequest) -> BaseServiceResponse:
        return await self.serve_async(request)

    def check_health(self) -> Dict[str, Any]:
        """Perform a health check on the service instance.

        Returns:
            Dictionary with health status and instance information
        """
        return {
            "healthy": True,
            "status": "healthy",
            "service_name": self.service_name,
            "instance_name": self.instance_name,
            "group_name": self.group_name,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "tags": list(self.tags),
        }

    def is_healthy(self) -> bool:
        """Check if the service instance is healthy.

        Returns:
            True if the instance is healthy, False otherwise
        """
        # For now, we assume all instances are healthy unless overridden
        health_status = self.check_health()
        return (
            health_status.get("healthy", False)
            and health_status.get("status") == "healthy"
        )


class BaseService:
    def __init__(
        self,
        service_name: str,
        scheduling_policy: SchedulingPolicyType = "round_robin",
        stateful_routing: bool = False,
        stateful_assignment_strategy: StatefulAssignmentStrategyType = "consistent_hashing",
    ):
        self.service_name: str = service_name
        self.instances: Dict[str, BaseServiceInstance] = {}
        self.instance_names: List[str] = []
        self.instance_groups: Dict[str, str] = {}
        self.scheduling_policy: SchedulingPolicyType = scheduling_policy

        # Stateful routing configuration
        self.stateful_routing: bool = stateful_routing
        self.stateful_assignment_strategy: StatefulAssignmentStrategyType = (
            stateful_assignment_strategy
        )

        # Track instances by tags for efficient tag-based scheduling
        self.tag_to_instances: Dict[str, List[str]] = {}
        # Track round-robin indices for different instance groups
        self.round_robin_indices: Dict[str, int] = {"default": 0}

        # Stateful routing state
        self.hash_key_to_instance: Dict[str, str] = {}
        self.instance_hash_key_count: Dict[str, int] = {}

        # Map of scheduling policy names to their implementation functions
        self.scheduling_policies: Dict[str, Callable] = {
            "round_robin": self._round_robin_policy,
            "random": self._random_policy,
            "least_busy": self._least_busy_policy,
            "first_available": self._first_available_policy,
        }

    def set_scheduling_policy(self, policy: SchedulingPolicyType) -> None:
        """Set the scheduling policy for this service.

        Args:
            policy: The scheduling policy to use
        """
        if policy not in self.scheduling_policies:
            raise ValueError(f"Unsupported scheduling policy: {policy}")
        self.scheduling_policy = policy

    def add_instance(self, instance_name: str, instance: BaseServiceInstance) -> None:
        if instance_name in self.instances:
            raise Exception("Instance already exists")
        self.instances[instance_name] = instance
        self.instance_names.append(instance_name)
        self.instance_groups[instance_name] = instance.group_name

        # Add to tag-based lookup
        for tag in instance.tags:
            if tag not in self.tag_to_instances:
                self.tag_to_instances[tag] = []
            self.tag_to_instances[tag].append(instance_name)

        # Initialize stateful routing tracking for this instance
        if self.stateful_routing:
            self.instance_hash_key_count[instance_name] = 0

    def remove_instance(self, instance_name: str) -> None:
        instance = self.instances.get(instance_name)
        if not instance:
            return

        # Handle stateful routing cleanup
        if self.stateful_routing:
            # Find hash keys assigned to this instance
            affected_hash_keys = [
                hash_key
                for hash_key, assigned_instance in self.hash_key_to_instance.items()
                if assigned_instance == instance_name
            ]

            if affected_hash_keys:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Removing instance {instance_name} which has {len(affected_hash_keys)} "
                    f"assigned hash keys: {affected_hash_keys}. Stateful data may be lost!"
                )

                # Remove hash key mappings (could implement migration here)
                for hash_key in affected_hash_keys:
                    del self.hash_key_to_instance[hash_key]

            # Clean up tracking
            if instance_name in self.instance_hash_key_count:
                del self.instance_hash_key_count[instance_name]

        # Remove from tag-based lookup
        for tag in instance.tags:
            if tag in self.tag_to_instances:
                if instance_name in self.tag_to_instances[tag]:
                    self.tag_to_instances[tag].remove(instance_name)
                # Clean up empty tag lists
                if not self.tag_to_instances[tag]:
                    del self.tag_to_instances[tag]

        # Remove from main collections
        del self.instances[instance_name]
        self.instance_names.remove(instance_name)
        del self.instance_groups[instance_name]

    def get_instance(self, instance_name: str) -> BaseServiceInstance:
        instance = self.instances.get(instance_name)
        if instance is None:
            raise Exception(f"Instance {instance_name} not found")
        return instance

    def get_instance_list(self) -> List[BaseServiceInstance]:
        return list(self.instances.values())

    def get_instance_count(self) -> int:
        return len(self.instances)

    def _filter_instances_by_tags(
        self, instance_names: List[str], required_tags: FrozenSet[str]
    ) -> List[str]:
        """Filter instances that have all required tags.

        Args:
            instance_names: List of instance names to filter
            required_tags: Set of tags that instances must have (subset match)

        Returns:
            List of instance names that have all required tags
        """
        if not required_tags:
            return instance_names

        filtered_instances = []
        for instance_name in instance_names:
            instance = self.instances[instance_name]
            # Check if instance has all required tags (subset match)
            if required_tags.issubset(instance.tags):
                filtered_instances.append(instance_name)

        return filtered_instances

    # Stateful routing methods
    def extract_hash_key(self, request: BaseServiceRequest) -> Optional[str]:
        """Extract hash key from agsbench_request for stateful routing.

        This method should be overridden by subclasses to define their
        specific hash key extraction logic (e.g., collection_name for VDB,
        user_id for user services, etc.).

        Args:
            request: The service request

        Returns:
            Hash key string or None if no stateful routing needed
        """
        return request.hash_key

    def _assign_hash_key_to_instance(
        self, hash_key: str, tags: Optional[FrozenSet[str]] = None
    ) -> str:
        """Assign a hash key to an instance using the configured strategy.

        Args:
            hash_key: Hash key to assign
            tags: Optional tags to filter candidate instances

        Returns:
            Name of the assigned instance
        """
        # Get candidate instances (filtered by tags if provided)
        candidate_instances = self.instance_names.copy()
        if tags:
            candidate_instances = self._filter_instances_by_tags(
                candidate_instances, tags
            )

        if not candidate_instances:
            raise Exception("No suitable instances available for hash key assignment")

        if self.stateful_assignment_strategy == "consistent_hashing":
            return self._assign_by_consistent_hashing(hash_key, candidate_instances)
        elif self.stateful_assignment_strategy == "load_balanced":
            return self._assign_by_load_balancing(hash_key, candidate_instances)
        elif self.stateful_assignment_strategy == "explicit":
            # For explicit strategy, use round-robin as fallback
            return self._round_robin_policy(candidate_instances)
        else:
            raise NotImplementedError(
                f"Stateful assignment strategy {self.stateful_assignment_strategy} not implemented"
            )

    def _assign_by_consistent_hashing(
        self, hash_key: str, candidate_instances: List[str]
    ) -> str:
        """Assign hash key using consistent hashing for deterministic assignment."""
        import hashlib

        # Create a hash of the hash key
        key_hash = hashlib.md5(hash_key.encode()).hexdigest()

        # Convert to integer and use modulo to select instance
        hash_int = int(key_hash, 16)
        instance_index = hash_int % len(candidate_instances)

        return candidate_instances[instance_index]

    def _assign_by_load_balancing(
        self, hash_key: str, candidate_instances: List[str]
    ) -> str:
        """Assign hash key to the instance with the fewest hash keys."""
        # Find instance with minimum hash key count
        min_count = float("inf")
        selected_instance = candidate_instances[0]

        for instance_name in candidate_instances:
            count = self.instance_hash_key_count.get(instance_name, 0)
            if count < min_count:
                min_count = count
                selected_instance = instance_name

        return selected_instance

    # Scheduling policy implementations
    def _round_robin_policy(self, instance_names: List[str]) -> str:
        """Round-robin scheduling policy."""
        if not instance_names:
            raise Exception("No instance available")

        # Use a hash of the instance names as a key for this specific group
        group_key = str(hash(tuple(sorted(instance_names))))

        # Initialize index for this group if needed
        if group_key not in self.round_robin_indices:
            self.round_robin_indices[group_key] = 0

        # Get current index and instance
        current_index = self.round_robin_indices[group_key]
        instance_name = instance_names[current_index]

        # Update index for next time
        self.round_robin_indices[group_key] = (current_index + 1) % len(instance_names)

        return instance_name

    def _random_policy(self, instance_names: List[str]) -> str:
        """Random scheduling policy."""
        if not instance_names:
            raise Exception("No instance available")
        return random.choice(instance_names)

    def _least_busy_policy(self, instance_names: List[str]) -> str:
        """Least busy scheduling policy - select instance with fewest active requests."""
        if not instance_names:
            raise Exception("No instance available")

        return min(
            instance_names, key=lambda name: self.instances[name].active_requests
        )

    def _first_available_policy(self, instance_names: List[str]) -> str:
        """First available scheduling policy - select first instance with no active requests."""
        if not instance_names:
            raise Exception("No instance available")

        # Try to find an instance with no active requests
        for name in instance_names:
            if self.instances[name].active_requests == 0:
                return name

        # Fall back to least busy if all are busy
        return self._least_busy_policy(instance_names)

    def schedule(
        self,
        request: BaseServiceRequest,
        instance_name: str = None,
        tags: Optional[FrozenSet[str]] = None,
    ) -> BaseServiceInstance:
        """Schedule a request to an appropriate instance.

        Args:
            request: The service request to schedule
            instance_name: Optional instance name override. If provided, uses this instance if available
            tags: Optional set of tags. If provided, only instances with all these tags will be considered

        Returns:
            The selected service instance
        """
        # If instance_name is explicitly provided, use it
        if instance_name is not None:
            if instance_name in self.instances:
                return self.instances[instance_name]
            else:
                raise Exception(f"Instance {instance_name} not found")

        # Check for stateful routing
        if self.stateful_routing:
            hash_key = self.extract_hash_key(request)
            if hash_key:
                # Check if hash key is already assigned to an instance
                if hash_key in self.hash_key_to_instance:
                    assigned_instance = self.hash_key_to_instance[hash_key]

                    # Verify the assigned instance still exists and matches tag requirements
                    if assigned_instance in self.instances:
                        instance = self.instances[assigned_instance]

                        # Check tag compatibility if tags are specified
                        if tags and not tags.issubset(instance.tags):
                            import logging

                            logger = logging.getLogger(__name__)
                            logger.warning(
                                f"Hash key {hash_key} is assigned to instance {assigned_instance} "
                                f"but it doesn't match required tags {tags}. Using assigned instance anyway "
                                f"for consistency."
                            )

                        return instance
                    else:
                        # Assigned instance no longer exists, remove the mapping
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Hash key {hash_key} was assigned to non-existent instance "
                            f"{assigned_instance}. Reassigning..."
                        )
                        del self.hash_key_to_instance[hash_key]

                # Assign hash key to a new instance
                try:
                    assigned_instance = self._assign_hash_key_to_instance(
                        hash_key, tags
                    )
                    self.hash_key_to_instance[hash_key] = assigned_instance

                    # Update hash key count for load balancing
                    self.instance_hash_key_count[assigned_instance] = (
                        self.instance_hash_key_count.get(assigned_instance, 0) + 1
                    )

                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"Assigned hash key {hash_key} to instance {assigned_instance} "
                        f"using {self.stateful_assignment_strategy} strategy"
                    )

                    return self.instances[assigned_instance]

                except Exception as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.exception(f"Failed to assign hash key {hash_key}: {str(e)}")
                    # Fall through to default scheduling

        # Default scheduling (no stateful routing or no hash key)
        # Start with all instances
        candidate_instances = self.instance_names.copy()

        # Apply tag filtering if tags are specified
        if tags:
            candidate_instances = self._filter_instances_by_tags(
                candidate_instances, tags
            )

        if not candidate_instances:
            raise Exception("No suitable instance available")

        # Use the configured scheduling policy
        policy_func = self.scheduling_policies.get(self.scheduling_policy)
        if not policy_func:
            raise NotImplementedError(
                f"Scheduling policy {self.scheduling_policy} not implemented"
            )

        selected_name = policy_func(candidate_instances)
        return self.instances[selected_name]

    def serve(
        self,
        request: BaseServiceRequest,
        instance_name: str = None,
        tags: Optional[FrozenSet[str]] = None,
    ) -> BaseServiceResponse:
        """Serve a request with optional routing parameters.

        Args:
            request: The service request to serve
            instance_name: Optional instance name override
            tags: Optional set of tags for instance filtering

        Returns:
            The service response
        """
        st = time.time()
        instance = self.schedule(request, instance_name, tags)
        scheduling_time = time.time() - st
        resp = instance.serve(request)
        elapsed_time = time.time() - st
        resp.elapsed_time = elapsed_time
        resp.scheduling_time = scheduling_time
        return resp

    async def serve_async(
        self,
        request: BaseServiceRequest,
        instance_name: str = None,
        tags: Optional[FrozenSet[str]] = None,
    ) -> BaseServiceResponse:
        """Serve a request asynchronously with optional routing parameters.

        Args:
            request: The service request to serve
            tags: Optional set of tags for instance filtering

        Returns:
            The service response
        """
        st = time.time()
        instance = self.schedule(request, instance_name, tags)
        scheduling_time = time.time() - st
        resp = await instance.serve_async(request)
        elapsed_time = time.time() - st
        resp.elapsed_time = elapsed_time
        resp.scheduling_time = scheduling_time
        return resp

    def call(
        self,
        request: BaseServiceRequest,
        instance_name: str = None,
        tags: Optional[FrozenSet[str]] = None,
    ) -> BaseServiceResponse:
        """Synchronous call interface for serving requests.

        This method allows the service to be called like a function,
        delegating to the serve method.

        Args:
            request: The service request to serve
            instance_name: Optional instance name override
            tags: Optional set of tags for instance filtering

        Returns:
            The service response
        """
        return self.serve(request=request, instance_name=instance_name, tags=tags)

    async def call_async(
        self,
        request: BaseServiceRequest,
        instance_name: str = None,
        tags: Optional[FrozenSet[str]] = None,
    ) -> BaseServiceResponse:
        """Asynchronous call interface for serving requests.

        This method allows the service to be called like a function,
        delegating to the serve_async method.

        Args:
            request: The service request to serve
            instance_name: Optional instance name override
            tags: Optional set of tags for instance filtering

        Returns:
            The service response
        """
        return await self.serve_async(
            request=request, instance_name=instance_name, tags=tags
        )

    # Stateful routing management methods
    def get_hash_key_mapping(self) -> Dict[str, str]:
        """Get the current hash key-to-instance mapping.

        Returns:
            Dictionary mapping hash keys to instance names
        """
        return self.hash_key_to_instance.copy()

    def set_hash_key_mapping(self, hash_key: str, instance_name: str) -> None:
        """Explicitly set a hash key-to-instance mapping.

        Args:
            hash_key: Hash key to map
            instance_name: Name of the instance to assign the hash key to

        Raises:
            ValueError: If the instance doesn't exist
        """
        if instance_name not in self.instances:
            raise ValueError(f"Instance {instance_name} does not exist")

        old_instance = self.hash_key_to_instance.get(hash_key)
        if old_instance and old_instance != instance_name:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Reassigning hash key {hash_key} from {old_instance} to {instance_name}. "
                f"Stateful data may be inconsistent!"
            )
            # Update hash key counts
            if old_instance in self.instance_hash_key_count:
                self.instance_hash_key_count[old_instance] = max(
                    0, self.instance_hash_key_count[old_instance] - 1
                )

        self.hash_key_to_instance[hash_key] = instance_name
        self.instance_hash_key_count[instance_name] = (
            self.instance_hash_key_count.get(instance_name, 0) + 1
        )

        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"Explicitly assigned hash key {hash_key} to instance {instance_name}"
        )

    def clear_hash_key_mapping(self, hash_key: str) -> None:
        """Clear the mapping for a specific hash key.

        Args:
            hash_key: Hash key to clear mapping for
        """
        if hash_key in self.hash_key_to_instance:
            instance_name = self.hash_key_to_instance[hash_key]
            del self.hash_key_to_instance[hash_key]

            # Update hash key count
            if instance_name in self.instance_hash_key_count:
                self.instance_hash_key_count[instance_name] = max(
                    0, self.instance_hash_key_count[instance_name] - 1
                )

            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Cleared mapping for hash key {hash_key}")

    def get_instance_hash_key_count(self) -> Dict[str, int]:
        """Get the hash key count per instance.

        Returns:
            Dictionary mapping instance names to their hash key counts
        """
        return self.instance_hash_key_count.copy()
