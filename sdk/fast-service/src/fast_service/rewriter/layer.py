from abc import ABCMeta, abstractmethod
from typing import Callable, Optional
import inspect

from ..description import FunctionDescriptor


class ExecutionLayer(metaclass=ABCMeta):
    next_layer: Optional["ExecutionLayer"]

    def __init__(self):
        self.next_layer = None

    def set_next_layer(self, next_layer: "ExecutionLayer" = None):
        self.next_layer = next_layer

    def setup_client(self, f: Callable, raw_descriptor: FunctionDescriptor) -> Callable:
        if self.next_layer is not None:
            f = self.next_layer.setup_client(f, raw_descriptor)
        return self._setup_client(f, raw_descriptor)

    def setup_service(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        if self.next_layer is not None:
            f = self.next_layer.setup_service(f, raw_descriptor)
        return self._setup_service(f, raw_descriptor)

    def setup_async_client(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        if self.next_layer is not None:
            f = self.next_layer.setup_async_client(f, raw_descriptor)
        return self._setup_async_client(f, raw_descriptor)

    def setup_async_service(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        if self.next_layer is not None:
            f = self.next_layer.setup_async_service(f, raw_descriptor)
        return self._setup_async_service(f, raw_descriptor)

    def setup_local_function(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        if raw_descriptor.is_async:
            return self.setup_async_service(f, raw_descriptor)
        else:
            return self.setup_service(f, raw_descriptor)

    @abstractmethod
    def _setup_client(self, f: Callable, raw_descriptor: FunctionDescriptor):
        pass

    @abstractmethod
    def _setup_service(self, f: Callable, raw_descriptor: FunctionDescriptor):
        pass

    def _setup_async_client(self, f: Callable, raw_descriptor: FunctionDescriptor):
        return self._setup_client(f, raw_descriptor)

    def _setup_async_service(self, f: Callable, raw_descriptor: FunctionDescriptor):
        return self._setup_service(f, raw_descriptor)
