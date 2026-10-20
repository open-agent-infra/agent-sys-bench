import os

from typing import Callable, get_origin
from functools import wraps
from pydantic import BaseModel

from ..description import FunctionDescriptor, ContextParamUsage, RequestContext
from ..file import FileParamUsage, FastServiceFile

from .storage_manager import StorageManager
from .record import IntermediateRequest, IntermediateResponse
from .request_file_cache import RequestFileCache
import inspect
import logging

logger = logging.getLogger(__name__)


class IntermediateManager:

    dir_path: str

    file_cache: RequestFileCache
    storage_manager: StorageManager

    def start(self, dir_path: str):
        self.dir_path = dir_path
        self.file_cache = RequestFileCache(os.path.join(self.dir_path, "file_cache"))
        self.storage_manager = StorageManager(os.path.join(self.dir_path, "requests"))

        self.storage_manager.start()

    def stop(self):
        self.storage_manager.stop()

    def record_intermediate(
        self,
        f: Callable,
        descriptor: FunctionDescriptor,
        context_usage: ContextParamUsage,
        file_usage: FileParamUsage,
    ):
        if file_usage.use_file():
            return self._record_intermediate_with_file(
                f, descriptor, context_usage, file_usage
            )
        else:
            return self._record_intermediate_without_file(f, descriptor, context_usage)

    def _record_intermediate_without_file(
        self,
        f: Callable,
        descriptor: FunctionDescriptor,
        context_usage: ContextParamUsage,
    ):

        def _record_request(request_id, *args, **kwargs):
            try:
                reqeust_body = descriptor.extract_args_as_dict([], *args, **kwargs)

                record = IntermediateRequest(
                    descriptor.name.__str__(), request_id, reqeust_body
                )
                self.storage_manager.store_req_record(record)
            except Exception as e:
                logger.exception(
                    f"Failed to serialize request of function {descriptor.name}: {e}"
                )
                raise ValueError(
                    f"Failed to serialize request of function {descriptor.name}: {e}"
                )

        if inspect.iscoroutinefunction(f):

            @wraps(f)
            async def async_wrapper(*args, **kwargs):

                context_arg: RequestContext = context_usage.get_context_arg(
                    *args, **kwargs
                )
                request_id = context_arg.request_id

                _record_request(request_id, *args, **kwargs)
                rtn = await f(*args, **kwargs)
                self._record_response(request_id, rtn, f, descriptor)

                return rtn

            return async_wrapper
        else:

            @wraps(f)
            def wrapper(*args, **kwargs):

                context_arg: RequestContext = context_usage.get_context_arg(
                    *args, **kwargs
                )
                request_id = context_arg.request_id

                _record_request(request_id, *args, **kwargs)
                rtn = f(*args, **kwargs)
                self._record_response(request_id, rtn, f, descriptor)

                return rtn

            return wrapper

    def _record_intermediate_with_file(
        self,
        f: Callable,
        descriptor: FunctionDescriptor,
        context_usage: ContextParamUsage,
        file_usage: FileParamUsage,
    ):

        def _record_request(request_id, *args, **kwargs):
            reqeust_body = descriptor.extract_args_as_dict(
                [file_usage.file_param_name()], *args, **kwargs
            )

            file_arg = file_usage.get_file_arg(*args, **kwargs)

            if file_usage.single_file():
                file_arg: FastServiceFile = file_arg
                file_list = [file_arg]
                reqeust_body[file_usage.file_param_name()] = file_arg.get_filename()
            else:
                file_list: list[FastServiceFile] = file_arg
                reqeust_body[file_usage.file_param_name()] = [
                    _.get_filename() for _ in file_list
                ]

            cache_dir, reopened_file_list = self.file_cache.cache_request_files(
                request_id, file_list
            )
            record = IntermediateRequest(
                descriptor.name.__str__(), request_id, reqeust_body, cache_dir
            )
            self.storage_manager.store_req_record(record)

            if file_usage.single_file():
                args, kwargs = file_usage.rewrite_file_arg(
                    reopened_file_list[0], *args, **kwargs
                )
            else:
                args, kwargs = file_usage.rewrite_file_arg(
                    reopened_file_list, *args, **kwargs
                )
            return args, kwargs

        if inspect.iscoroutinefunction(f):

            @wraps(f)
            async def async_wrapper(*args, **kwargs):

                context_arg: RequestContext = context_usage.get_context_arg(
                    *args, **kwargs
                )
                request_id = context_arg.request_id

                args, kwargs = _record_request(request_id, *args, **kwargs)
                rtn = await f(*args, **kwargs)
                self._record_response(request_id, rtn, f, descriptor)

                return rtn

            return async_wrapper
        else:

            @wraps(f)
            def wrapper(*args, **kwargs):
                context_arg: RequestContext = context_usage.get_context_arg(
                    *args, **kwargs
                )
                request_id = context_arg.request_id

                args, kwargs = _record_request(request_id, *args, **kwargs)
                rtn = f(*args, **kwargs)
                self._record_response(request_id, rtn, f, descriptor)

                return rtn

            return wrapper

    def _record_response(
        self,
        request_id,
        rtn,
        f: Callable,
        descriptor: FunctionDescriptor,
    ):
        return_type = getattr(f, "__annotations__").get("return")
        try:
            if get_origin(return_type) is list:
                resp_content = str(
                    [_.model_dump() if isinstance(_, BaseModel) else _ for _ in rtn]
                )
            elif return_type is None:
                resp_content = str(None)
            # if is dict
            elif get_origin(return_type) is dict:
                resp_content = str(
                    {
                        k: v.model_dump() if isinstance(v, BaseModel) else v
                        for k, v in rtn.items()
                    }
                )
            else:
                # only support Pydantic.BaseModel for now
                assert issubclass(
                    return_type, BaseModel
                ), f"Return type {return_type} of function {descriptor.name} must be a Pydantic BaseModel"
                rtn_model: BaseModel = rtn
                resp_content = rtn_model.model_dump()
        except Exception as e:
            logger.exception(
                f"Failed to serialize response of function {descriptor.name}: {e}"
            )
            raise ValueError(
                f"Failed to serialize response of function {descriptor.name}: {e}"
            )

        resp_record = IntermediateResponse(
            descriptor.name.__str__(), request_id, resp_content
        )
        self.storage_manager.store_resp_record(resp_record)
