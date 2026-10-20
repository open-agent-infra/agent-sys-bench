from typing import Callable
from functools import wraps
import inspect
import asyncio

from ..description import (
    RequestContext,
    FunctionName,
    FunctionDescriptor,
    ContextParamUsage,
)

from .layer import ExecutionLayer


def _init_context(
    f: Callable,
    function_descriptor: FunctionDescriptor,
    context_usage: ContextParamUsage,
):
    """Initialize context for synchronous functions."""
    is_async = inspect.iscoroutinefunction(f)

    if is_async:

        @wraps(f)
        async def async_wrapper(*args, **kwargs):
            context_name = context_usage.context_name()

            request_id = None
            if context_name in kwargs:
                parent_context = kwargs[context_name]
                if parent_context:
                    if not isinstance(parent_context, RequestContext):
                        raise TypeError(
                            f"Context {context_name} must be of type RequestContext"
                        )
                    request_id = parent_context.request_id
            else:
                parent_context = None

            if request_id is None:
                request_id = RequestContext.generate_request_id()

            context = RequestContext(
                request_id=request_id,
                function_name=function_descriptor.name,
                call_id=RequestContext.generate_call_id(),
            )
            context.set_parent_context(parent_context)

            kwargs[context_name] = context
            res = await f(*args, **kwargs)

            return res

        return async_wrapper
    else:

        @wraps(f)
        def wrapper(*args, **kwargs):
            context_name = context_usage.context_name()

            request_id = None
            if context_name in kwargs:
                parent_context = kwargs[context_name]
                if parent_context:
                    if not isinstance(parent_context, RequestContext):
                        raise TypeError(
                            f"Context {context_name} must be of type RequestContext"
                        )
                    request_id = parent_context.request_id
            else:
                parent_context = None

            if request_id is None:
                request_id = RequestContext.generate_request_id()

            context = RequestContext(
                request_id=request_id,
                function_name=function_descriptor.name,
                call_id=RequestContext.generate_call_id(),
            )
            context.set_parent_context(parent_context)

            kwargs[context_name] = context
            res = f(*args, **kwargs)

            return res

        return wrapper


class ContextLayer(ExecutionLayer):
    """Layer that initializes and manages request context for functions."""

    def __init__(self):
        super().__init__()

    def _setup_client(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        context_usage = ContextParamUsage(raw_descriptor)
        if context_usage.use_context():
            return _init_context(f, raw_descriptor, context_usage)
        return f

    def _setup_service(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        context_usage = ContextParamUsage(raw_descriptor)
        if context_usage.use_context():
            return _init_context(f, raw_descriptor, context_usage)
        return f

    def _setup_async_client(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        context_usage = ContextParamUsage(raw_descriptor)
        if context_usage.use_context():
            return _init_context(f, raw_descriptor, context_usage)
        return f

    def _setup_async_service(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        context_usage = ContextParamUsage(raw_descriptor)
        if context_usage.use_context():
            return _init_context(f, raw_descriptor, context_usage)
        return f
