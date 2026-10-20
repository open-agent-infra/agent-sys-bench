from typing import Callable

from ..description import FunctionDescriptor, ContextParamUsage
from ..file import FileParamUsage
from ..intermediate import IntermediateManager

from .layer import ExecutionLayer


class IntermediateLayer(ExecutionLayer):

    intermediate_manager: IntermediateManager

    def __init__(
        self,
        intermediate_manager: IntermediateManager,
    ):
        super().__init__()

        self.intermediate_manager = intermediate_manager

    def _setup_client(self, f: Callable, raw_descriptor: FunctionDescriptor):
        return f

    def _setup_service(self, f: Callable, raw_descriptor: FunctionDescriptor):
        if raw_descriptor.monitor_only:
            return f

        context_usage = ContextParamUsage(raw_descriptor)
        if context_usage.use_context():
            file_usage = FileParamUsage(raw_descriptor)
            return self.intermediate_manager.record_intermediate(
                f, raw_descriptor, context_usage, file_usage
            )

        return f

    def _setup_async_client(self, f: Callable, raw_descriptor: FunctionDescriptor):
        return self._setup_client(f, raw_descriptor)

    def _setup_async_service(self, f: Callable, raw_descriptor: FunctionDescriptor):
        return self._setup_service(f, raw_descriptor)
