"""
Dependency injection module for service resolution.

This module provides a registry for dependency injection, allowing
components to request dependencies by type rather than creating
them directly, promoting loose coupling and testability.
"""

from typing import Dict, Any, Type, TypeVar, cast, Callable

T = TypeVar("T")


class DependencyRegistry:
    """Dependency injection registry for lego-agent.

    This class provides a central registry for service instances and
    factory functions, allowing components to resolve dependencies
    without direct instantiation.

    Attributes:
        _services (Dict[str, Any]): Dictionary of registered service instances
        _factories (Dict[str, Callable[[], Any]]): Dictionary of factory functions
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}

    def register(self, service_type: Type[T], instance: T) -> None:
        """Register a service instance."""
        self._services[service_type.__name__] = instance

    def register_factory(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function for a service."""
        self._factories[service_type.__name__] = factory

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service instance."""
        if service_type.__name__ in self._services:
            return cast(T, self._services[service_type.__name__])

        if service_type.__name__ in self._factories:
            instance = self._factories[service_type.__name__]()
            self._services[service_type.__name__] = instance
            return cast(T, instance)

        raise ValueError(f"No registration for {service_type.__name__}")


# Global registry instance
dependency_registry = DependencyRegistry()
