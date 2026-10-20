from typing import Callable

from ..description import (
    RequestContext,
    FunctionName,
    FunctionDescriptor,
    ContextParamUsage,
)
from ..monitor import FastServiceMonitorManager

from .layer import ExecutionLayer


def get_request_id(context_name: str):
    def wrapper(**kwargs):
        if isinstance(kwargs[context_name], RequestContext):
            return str(kwargs[context_name].request_id)
        else:
            raise TypeError("Request context is not defined in given function.")

    return wrapper


def get_parent_service(context_name: str):
    def wrapper(**kwargs):
        if isinstance(kwargs[context_name], RequestContext):
            return kwargs[context_name].get_parent_service().__str__()
        else:
            raise TypeError("Request context is not defined in given function.")

    return wrapper


def get_call_id(context_name: str):
    def wrapper(**kwargs):
        if isinstance(kwargs[context_name], RequestContext):
            return str(kwargs[context_name].get_call_id())
        else:
            raise TypeError("Request context is not defined in given function.")

    return wrapper


def get_parent_call_id(context_name: str):
    def wrapper(**kwargs):
        if isinstance(kwargs[context_name], RequestContext):
            return str(kwargs[context_name].get_parent_call_id())
        else:
            raise TypeError("Request context is not defined in given function.")

    return wrapper


class MonitorLayer(ExecutionLayer):
    """Layer that monitors function execution times and delays."""

    monitor_manager: FastServiceMonitorManager

    def __init__(
        self,
        monitor_manager: FastServiceMonitorManager,
    ):
        super().__init__()

        self.monitor_manager = monitor_manager

    def _setup_client(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        context_usage = ContextParamUsage(raw_descriptor)
        if context_usage.use_context():
            return self.monitor_manager.monitor_remote_delay(
                f,
                get_request_id(context_usage.context_name()),
                get_parent_service(context_usage.context_name()),
                raw_descriptor.name.__str__(),
                get_call_id(context_usage.context_name()),
                get_parent_call_id(context_usage.context_name()),
            )
        return f

    def _setup_service(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        context_usage = ContextParamUsage(raw_descriptor)
        if context_usage.use_context():
            return self.monitor_manager.monitor_e2e_delay(
                f,
                get_request_id(context_usage.context_name()),
                get_parent_service(context_usage.context_name()),
                raw_descriptor.name.__str__(),
                get_call_id(context_usage.context_name()),
                get_parent_call_id(context_usage.context_name()),
            )
        return f

    def _setup_async_client(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        context_usage = ContextParamUsage(raw_descriptor)
        if context_usage.use_context():
            return self.monitor_manager.monitor_remote_delay(
                f,
                get_request_id(context_usage.context_name()),
                get_parent_service(context_usage.context_name()),
                raw_descriptor.name.__str__(),
                get_call_id(context_usage.context_name()),
                get_parent_call_id(context_usage.context_name()),
            )
        return f

    def _setup_async_service(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        context_usage = ContextParamUsage(raw_descriptor)
        if context_usage.use_context():
            return self.monitor_manager.monitor_e2e_delay(
                f,
                get_request_id(context_usage.context_name()),
                get_parent_service(context_usage.context_name()),
                raw_descriptor.name.__str__(),
                get_call_id(context_usage.context_name()),
                get_parent_call_id(context_usage.context_name()),
            )
        return f
