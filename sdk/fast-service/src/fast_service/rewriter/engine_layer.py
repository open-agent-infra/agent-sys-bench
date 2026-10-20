from typing import Callable
from functools import wraps

from ..description import (
    RequestContext,
    FunctionName,
    FunctionDescriptor,
)
from ..file import FileParamUsage
from ..engine import IFastServiceEngine

from .layer import ExecutionLayer


class EngineLayer(ExecutionLayer):

    engine: IFastServiceEngine
    service_dict: dict

    def __init__(
        self,
        engine: IFastServiceEngine,
        service_dict: dict,
    ):
        super().__init__()

        self.engine = engine
        self.service_dict = service_dict

    def _setup_client(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        file_usage = FileParamUsage(raw_descriptor)
        service = self.service_dict[
            (raw_descriptor.name.module_name, raw_descriptor.name.function_name)
        ]

        if file_usage.use_file():
            return self.engine.setup_client_with_files(
                f,
                service.ip,
                service.port,
                raw_descriptor.name.module_name,
                raw_descriptor.name.function_name,
            )
        else:
            return self.engine.setup_client(f, service.ip, service.port)

    def _setup_async_client(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        file_usage = FileParamUsage(raw_descriptor)
        service = self.service_dict[
            (raw_descriptor.name.module_name, raw_descriptor.name.function_name)
        ]

        if file_usage.use_file():
            return self.engine.setup_async_client_with_files(
                f,
                service.ip,
                service.port,
                raw_descriptor.name.module_name,
                raw_descriptor.name.function_name,
            )
        else:
            return self.engine.setup_async_client(f, service.ip, service.port)

    def _setup_service(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        file_usage = FileParamUsage(raw_descriptor)

        if file_usage.use_file():
            self.engine.setup_service_with_files(
                f,
                raw_descriptor.name.module_name,
                raw_descriptor.name.function_name,
            )
        else:
            self.engine.setup_service(f)

        return f

    def _setup_async_service(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        file_usage = FileParamUsage(raw_descriptor)

        if file_usage.use_file():
            self.engine.setup_async_service_with_files(
                f,
                raw_descriptor.name.module_name,
                raw_descriptor.name.function_name,
            )
        else:
            self.engine.setup_async_service(f)

        return f
