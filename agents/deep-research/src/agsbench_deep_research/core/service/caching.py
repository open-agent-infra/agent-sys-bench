"""
Caching service module for result caching operations.

This module provides service implementations for interacting with
various cache backends, including in-memory and file system caches.
"""

import os
import time
import threading
import pickle
from abc import abstractmethod
from typing import Any, Dict, Optional
from pydantic import Field
import logging
import redis
from .base import (
    BaseServiceInstanceConfig,
    BaseServiceInstance,
    BaseService,
    BaseServiceRequest,
    BaseServiceResponse,
)

caching_logger = logging.getLogger(__name__)


# --- Config ---
class CachingServiceInstanceConfig(BaseServiceInstanceConfig):
    backend: str = Field(
        ..., description="The cache backend type (memory, file, redis, etc.)"
    )
    location: Optional[str] = Field(
        None,
        description=(
            "Location for file or remote backends. For Redis, use redis://host:port/db."
        ),
    )
    default_ttl: Optional[float] = Field(
        None, description="Default TTL in seconds (None=Infinity)"
    )
    # Redis-specific options
    redis_host: Optional[str] = Field(
        None, description="Redis host (overrides location if set)"
    )
    redis_port: Optional[int] = Field(None, description="Redis port")
    redis_db: Optional[int] = Field(None, description="Redis DB number")
    redis_password: Optional[str] = Field(None, description="Redis password")


# --- Request/Response ---
class CachingServiceRequest(BaseServiceRequest):
    service_name: str = Field(
        default="caching", description="Service name for the caching service."
    )
    operation_name: str = Field(..., description="Operation: get, set, delete, clear.")
    key: str = Field(..., description="Cache key.")
    value: Optional[Any] = Field(
        None, description="Value to cache (for set operations)."
    )
    ttl: Optional[float] = Field(None, description="TTL in seconds (None=Infinity)")
    backend: Optional[str] = Field(
        None,
        description="Optional backend override for routing (e.g., 'memory', 'file', 'redis')",
    )
    hash_key: Optional[str] = Field(
        None,
        description=(
            "Optional hash key for consistent routing. If set, overrides key for routing."
        ),
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "service_name": self.service_name,
                "operation_name": self.operation_name,
                "key": self.key,
                "value_type": (
                    type(self.value).__name__ if self.value is not None else None
                ),
                "ttl": self.ttl,
                "backend": self.backend,
                "hash_key": self.hash_key,
            }
        )
        return stats


class CachingServiceResponse(BaseServiceResponse):
    service_name: str = Field(
        default="caching", description="Service name for the caching service."
    )
    operation_name: str = Field(..., description="Operation: get, set, delete, clear.")
    value: Optional[Any] = None
    found: Optional[bool] = None
    backend: Optional[str] = Field(
        None,
        description="Backend used for the operation (e.g., 'memory', 'file', 'redis')",
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "service_name": self.service_name,
                "operation_name": self.operation_name,
                "value_type": (
                    type(self.value).__name__ if self.value is not None else None
                ),
                "found": self.found,
                "backend": self.backend,
            }
        )
        return stats


# --- Instance Base ---
class CachingServiceInstance(BaseServiceInstance):
    def launch(self):
        pass

    def shutdown(self):
        pass

    @abstractmethod
    def get(self, key: str) -> Any:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        pass

    @abstractmethod
    def delete(self, key: str):
        pass

    @abstractmethod
    def clear(self):
        pass

    def _serve_impl(self, request: CachingServiceRequest) -> CachingServiceResponse:
        op = request.operation_name
        if op == "get":
            value = self.get(request.key)
            found = value is not None
            return CachingServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=op,
                value=value,
                found=found,
                backend=getattr(self.config, "backend", None),
                success=True,
            )
        elif op == "set":
            self.set(request.key, request.value, request.ttl)
            return CachingServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=op,
                backend=getattr(self.config, "backend", None),
                success=True,
            )
        elif op == "delete":
            self.delete(request.key)
            return CachingServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=op,
                backend=getattr(self.config, "backend", None),
                success=True,
            )
        elif op == "clear":
            self.clear()
            return CachingServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=op,
                backend=getattr(self.config, "backend", None),
                success=True,
            )
        else:
            return CachingServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=op,
                backend=getattr(self.config, "backend", None),
                success=False,
                error=f"Unknown operation: {op}",
            )

    async def _serve_async_impl(
        self, request: CachingServiceRequest
    ) -> CachingServiceResponse:
        return self._serve_impl(request)


# --- In-Memory Backend ---
class InMemoryCacheServiceInstance(CachingServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: CachingServiceInstanceConfig
    ):
        super().__init__("caching", instance_name, group_name, config)
        self._cache: Dict[str, Any] = {}
        self._expirations: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.default_ttl = config.default_ttl

    def get(self, key: str) -> Any:
        with self._lock:
            exp = self._expirations.get(key)
            if exp is not None and exp < time.time():
                self._cache.pop(key, None)
                self._expirations.pop(key, None)
                return None
            return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        with self._lock:
            self._cache[key] = value
            if ttl is None:
                ttl = self.default_ttl
            if ttl is not None:
                self._expirations[key] = time.time() + ttl
            elif key in self._expirations:
                self._expirations.pop(key)

    def delete(self, key: str):
        with self._lock:
            self._cache.pop(key, None)
            self._expirations.pop(key, None)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._expirations.clear()


# --- File System Backend ---
class FileSystemCacheServiceInstance(CachingServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: CachingServiceInstanceConfig
    ):
        super().__init__("caching", instance_name, group_name, config)
        self.location = config.location or "/tmp/legoagent_cache"
        self.default_ttl = config.default_ttl
        os.makedirs(self.location, exist_ok=True)

    def _file_path(self, key: str) -> str:
        # key may contain special characters, so sanitize it
        key = key.replace("/", "_")
        return os.path.join(self.location, f"{key}.pkl")

    def get(self, key: str) -> Any:
        path = self._file_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            exp = data.get("exp")
            if exp is not None and exp < time.time():
                os.remove(path)
                return None
            return data["value"]
        except Exception as e:
            caching_logger.exception(f"File cache get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        path = self._file_path(key)
        if ttl is None:
            ttl = self.default_ttl
        exp = time.time() + ttl if ttl is not None else None
        try:
            with open(path, "wb") as f:
                pickle.dump({"value": value, "exp": exp}, f)
        except Exception as e:
            caching_logger.exception(f"File cache set error: {e}")

    def delete(self, key: str):
        path = self._file_path(key)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            caching_logger.exception(f"File cache delete error: {e}")

    def clear(self):
        try:
            for fname in os.listdir(self.location):
                if fname.endswith(".pkl"):
                    os.remove(os.path.join(self.location, fname))
                    caching_logger.info(f"Cleared cache file: {fname}")
        except Exception as e:
            caching_logger.exception(f"File cache clear error: {e}")


# --- Redis Backend ---
class RedisCacheServiceInstance(CachingServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: CachingServiceInstanceConfig
    ):
        super().__init__("caching", instance_name, group_name, config)
        # Prefer explicit host/port/db if provided, else use location URI
        if config.redis_host:
            self._client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port or 6379,
                db=config.redis_db or 0,
                password=config.redis_password,
                decode_responses=False,
            )
        else:
            # Use location as redis://host:port/db
            self._client = redis.from_url(
                config.location or "redis://localhost:6379/0", decode_responses=False
            )
        self.default_ttl = config.default_ttl

    def get(self, key: str) -> Any:
        try:
            data = self._client.get(key)
            if data is None:
                return None
            # Unpickle value
            return pickle.loads(data)
        except Exception as e:
            caching_logger.exception(f"Redis cache get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        try:
            data = pickle.dumps(value)
            expire = ttl if ttl is not None else self.default_ttl
            if expire is not None:
                self._client.setex(key, int(expire), data)
            else:
                self._client.set(key, data)
        except Exception as e:
            caching_logger.exception(f"Redis cache set error: {e}")

    def delete(self, key: str):
        try:
            self._client.delete(key)
        except Exception as e:
            caching_logger.exception(f"Redis cache delete error: {e}")

    def clear(self):
        try:
            self._client.flushdb()
        except Exception as e:
            caching_logger.exception(f"Redis cache clear error: {e}")


# --- Main Service ---
class CachingService(BaseService):
    def __init__(
        self,
        scheduling_policy: str = "round_robin",
        stateful_assignment_strategy: str = "load_balanced",
    ):
        super().__init__(
            "caching",
            scheduling_policy=scheduling_policy,
            stateful_routing=True,
            stateful_assignment_strategy=stateful_assignment_strategy,
        )

    def extract_hash_key(self, request: CachingServiceRequest) -> Optional[str]:
        """Extracts the hash key from the request.
        If hash_key is explicitly set, use it; otherwise, use the key.
        This allows for consistent routing based on the key or hash_key.
        If neither is set, it will return None.
        """
        # First check if hash_key is explicitly set
        if request.hash_key:
            return request.hash_key
        elif request.key:
            return request.key
        else:
            caching_logger.warning("No hash_key or key provided in request.")
            return None

    def serve(
        self,
        request: CachingServiceRequest,
        backend: str = None,
        instance_name: str = None,
        tags=None,
    ) -> CachingServiceResponse:
        backend = backend or request.backend
        if backend is not None:
            tags = tags or set()
            if backend not in tags:
                tags.add(backend)
        return super().serve(request, instance_name, tags)

    async def serve_async(
        self,
        request: CachingServiceRequest,
        backend: str = None,
        instance_name: str = None,
        tags=None,
    ) -> CachingServiceResponse:
        backend = backend or request.backend
        if backend is not None:
            tags = tags or set()
            if backend not in tags:
                tags.add(backend)
        return await super().serve_async(request, instance_name, tags)


# --- Instance Factory ---
def get_cache_instance_class(backend: str) -> type[CachingServiceInstance]:
    """Returns the appropriate cache instance class based on the backend type."""
    backend = backend.lower()
    if backend == "memory":
        return InMemoryCacheServiceInstance
    elif backend == "file":
        return FileSystemCacheServiceInstance
    elif backend == "redis":
        return RedisCacheServiceInstance
    else:
        raise ValueError(f"Unknown cache backend: {backend}")
