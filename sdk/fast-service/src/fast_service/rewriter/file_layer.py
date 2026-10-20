from inspect import iscoroutinefunction
from typing import Callable
from functools import wraps

from ..description import (
    FunctionDescriptor,
)

from ..file import FileParamUsage, FastServiceFile

from .layer import ExecutionLayer


def _rewrite_service_with_files(f, file_usage: FileParamUsage):

    if iscoroutinefunction(f):

        @wraps(f)
        async def async_wrapper(*args, **kwargs) -> f.__annotations__.get("return"):
            kwargs = kwargs.copy()
            if file_usage.single_file():
                kwargs[file_usage.file_param_name()] = kwargs[
                    file_usage.file_param_name()
                ][0]
            else:
                kwargs[file_usage.file_param_name()] = kwargs[
                    file_usage.file_param_name()
                ]
            res = await f(**kwargs)
            return res

        async_wrapper.__annotations__[file_usage.file_param_name()] = list[
            FastServiceFile
        ]

        return async_wrapper
    else:

        @wraps(f)
        def wrapper(*args, **kwargs) -> f.__annotations__.get("return"):
            kwargs = kwargs.copy()
            if file_usage.single_file():
                kwargs[file_usage.file_param_name()] = kwargs[
                    file_usage.file_param_name()
                ][0]
            else:
                kwargs[file_usage.file_param_name()] = kwargs[
                    file_usage.file_param_name()
                ]
            return f(**kwargs)

        wrapper.__annotations__[file_usage.file_param_name()] = list[FastServiceFile]

        return wrapper


def _rewrite_client_with_files(f, raw_function, file_usage: FileParamUsage):

    if iscoroutinefunction(f):

        @wraps(raw_function)
        async def async_wrapper(*args, **kwargs):

            params = {}
            file_param = None

            args_len = len(args)
            args_index = 0
            for k, v in raw_function.__annotations__.items():
                if k == file_usage.file_param_name():
                    if args_index < args_len:
                        file_param = args[args_index]
                        args_index += 1
                    else:
                        file_param = kwargs[file_usage.file_param_name()]
                else:
                    if args_index < args_len:
                        params[k] = args[args_index]
                        args_index += 1
                    elif k in kwargs:
                        params[k] = kwargs[k]

            if file_usage.single_file():
                file_param = [file_param]

            res = await f(params=params, files=file_param)
            return res

        return async_wrapper
    else:

        @wraps(raw_function)
        def wrapper(*args, **kwargs):

            params = {}
            file_param = None

            args_len = len(args)
            args_index = 0
            for k, v in raw_function.__annotations__.items():
                if k == file_usage.file_param_name():
                    if args_index < args_len:
                        file_param = args[args_index]
                        args_index += 1
                    else:
                        file_param = kwargs[file_usage.file_param_name()]
                else:
                    if args_index < args_len:
                        params[k] = args[args_index]
                        args_index += 1
                    elif k in kwargs:
                        params[k] = kwargs[k]

            if file_usage.single_file():
                file_param = [file_param]

            return f(params=params, files=file_param)

        return wrapper


class FileLayer(ExecutionLayer):

    def __init__(self):
        super().__init__()

    def _setup_client(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        file_usage = FileParamUsage(raw_descriptor)
        if file_usage.use_file():
            return _rewrite_client_with_files(f, raw_descriptor.f, file_usage)
        else:
            return f

    def _setup_service(
        self, f: Callable, raw_descriptor: FunctionDescriptor
    ) -> Callable:
        file_usage = FileParamUsage(raw_descriptor)
        if file_usage.use_file():
            return _rewrite_service_with_files(f, file_usage)
        else:
            return f
