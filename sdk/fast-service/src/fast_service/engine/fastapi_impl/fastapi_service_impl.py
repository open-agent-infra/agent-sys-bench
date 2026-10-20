import json
import os
import uvicorn
import requests
import logging
import asyncio
import inspect
import aiohttp
from aiohttp import ClientTimeout

from fastapi import FastAPI, APIRouter, Form, UploadFile
from pydantic import BaseModel
from functools import wraps
from collections.abc import Iterable
from typing import Annotated, get_origin, get_args

from .file import FastApiFile
from ..engine_interface import IFastServiceEngine
from ...file import FastServiceFile

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


CLIENT_SIDE_TIMEOUT = int(
    os.environ.get("CLIENT_SIDE_TIMEOUT", 60 * 60 * 24 * 30)
)  # Default to 1 month


def _generate_service_name(module_name: str, func_name: str) -> str:
    # Use a simpler path format that's more compatible with FastAPI routing
    return f"{module_name}.{func_name}"


def _generate_request_url(ip: str, port: int, module_name: str, func_name: str) -> str:
    return f"http://{ip}:{port}/{_generate_service_name(module_name, func_name)}"


def _execute_service(router: APIRouter, ip: str, port: int, *args, **kwargs):
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app, host=ip, port=port)


def _transform_param(param_value):
    if isinstance(param_value, BaseModel):
        return param_value.model_dump()
    elif isinstance(param_value, str):
        return param_value
    elif isinstance(param_value, Iterable):
        items = []
        cls = param_value.__class__
        for item in param_value:
            items.append(_transform_param(item))
        return cls.__call__(items)
    else:
        return param_value


def _append_to_request_body(request_body: dict, param_name, param_value):
    request_body[param_name] = _transform_param(param_value)


def _extract_param(request_field, field_type):
    origin_type = get_origin(field_type)
    if origin_type is None:
        # raw type or raw class
        if field_type is BaseModel or issubclass(field_type, BaseModel):
            return field_type.parse_obj(request_field)
        else:
            return request_field
    else:
        # wrapped by some container
        item_type = get_args(field_type)[0]
        return origin_type.__call__(
            [_extract_param(_, item_type) for _ in request_field]
        )


class FastServiceOnFastAPI(IFastServiceEngine):
    router: APIRouter

    def __init__(self):
        self.router = APIRouter()

    def setup_service(self, f):
        service_name = _generate_service_name(
            getattr(f, "__module__"), getattr(f, "__name__")
        )
        res = self.router.post("/" + service_name)(f)

        logger.info(f"Successfully setup service {service_name}")
        return res

    def setup_async_service(self, f):
        service_name = _generate_service_name(
            getattr(f, "__module__"), getattr(f, "__name__")
        )
        res = self.router.post("/" + service_name)(f)

        logger.info(f"Successfully setup async service {service_name}")
        return res

    def setup_client(self, f, ip: str, port: int):
        # the decoration function inherit the metadata of the decorated function
        # thus python interpreter will help us validate the input params
        @wraps(f)
        def request_remote(*args, **kwargs):
            # make sure request of any type be transformed to dict
            input_annotations = []
            request_body = {}
            args_index = 0
            args_len = len(args)
            rtn_cls = None
            for k, v in getattr(f, "__annotations__").items():
                # we assume that the order of __annotations__ is fixed, after Python 3.7,
                # which is consists of each defined param description and one return value description in order

                if k == "return":
                    rtn_cls = v
                    break

                input_annotations.append(k)
                if args_index < args_len:
                    _append_to_request_body(request_body, k, args[args_index])
                    args_index += 1
                elif k in kwargs:
                    _append_to_request_body(request_body, k, kwargs[k])

            # if there's only one param, unwrap the outside dict
            if len(input_annotations) == 1:
                for k, v in request_body.items():
                    request_body = v

            x = requests.post(
                _generate_request_url(
                    ip, port, getattr(f, "__module__"), getattr(f, "__name__")
                ),
                json=request_body,
                timeout=CLIENT_SIDE_TIMEOUT,
            )

            # if rtn_cls is list of something, we should return a list of that thing
            response_data = x.json()
            if get_origin(rtn_cls) is list:
                if isinstance(response_data, list):
                    return list(
                        [
                            (
                                rtn_cls.__args__[0](**item)
                                if isinstance(item, dict)
                                else item
                            )
                            for item in response_data
                        ]
                    )
                else:
                    logger.error(f"Expected list response but got: {response_data}")
                    return list()
            elif rtn_cls is None:
                return None
            else:
                if isinstance(response_data, dict):
                    return rtn_cls(**response_data)
                else:
                    logger.error(f"Expected dict response but got: {response_data}")
                    return rtn_cls()

        return request_remote

    def setup_async_client(self, f, ip: str, port: int):
        # the decoration function inherit the metadata of the decorated function
        # thus python interpreter will help us validate the input params
        @wraps(f)
        async def request_remote_async(*args, **kwargs):
            # make sure request of any type be transformed to dict
            input_annotations = []
            request_body = {}
            args_index = 0
            args_len = len(args)
            rtn_cls = None
            for k, v in getattr(f, "__annotations__").items():
                if k == "return":
                    rtn_cls = v
                    break

                input_annotations.append(k)
                if args_index < args_len:
                    _append_to_request_body(request_body, k, args[args_index])
                    args_index += 1
                elif k in kwargs:
                    _append_to_request_body(request_body, k, kwargs[k])

            # if there's only one param, unwrap the outside dict
            if len(input_annotations) == 1:
                for k, v in request_body.items():
                    request_body = v

            # Use aiohttp for truly asynchronous HTTP requests
            url = _generate_request_url(
                ip, port, getattr(f, "__module__"), getattr(f, "__name__")
            )

            async with aiohttp.ClientSession(
                timeout=ClientTimeout(total=CLIENT_SIDE_TIMEOUT)
            ) as session:
                async with session.post(
                    url,
                    json=request_body,
                    timeout=ClientTimeout(total=CLIENT_SIDE_TIMEOUT),
                ) as response:
                    response_json = await response.json()

                    # if rtn_cls is list of something, we should return a list of that thing
                    if get_origin(rtn_cls) is list:
                        return rtn_cls.__call__(
                            [rtn_cls.__args__[0](**_) for _ in response_json]
                        )
                    elif rtn_cls is None:
                        return None
                    else:
                        return rtn_cls(**(response_json))

        return request_remote_async

    def setup_service_with_files(self, f, module_name: str, function_name: str):
        return_type = getattr(f, "__annotations__").get("return")

        def wrapper(
            params: Annotated[str, Form()], files: list[UploadFile]
        ) -> return_type:
            files = [FastApiFile(_) for _ in files]
            kwargs = {}
            params = json.loads(params)
            for k, v in f.__annotations__.items():
                if k == "return":
                    continue
                elif v == list[FastServiceFile]:
                    kwargs[k] = files
                else:
                    kwargs[k] = _extract_param(params[k], v)

            return f(**kwargs)

        return self.router.post(
            "/" + _generate_service_name(module_name, function_name)
        )(wrapper)

    def setup_async_service_with_files(self, f, module_name: str, function_name: str):
        return_type = getattr(f, "__annotations__").get("return")

        async def async_wrapper(
            params: Annotated[str, Form()], files: list[UploadFile]
        ) -> return_type:
            files = [FastApiFile(_) for _ in files]
            kwargs = {}
            params = json.loads(params)
            for k, v in f.__annotations__.items():
                if k == "return":
                    continue
                elif v == list[FastServiceFile]:
                    kwargs[k] = files
                else:
                    kwargs[k] = _extract_param(params[k], v)

            return await f(**kwargs)

        return self.router.post(
            "/" + _generate_service_name(module_name, function_name)
        )(async_wrapper)

    def setup_client_with_files(
        self, f, ip: str, port: int, module_name: str, function_name: str
    ):
        return_type = getattr(f, "__annotations__").get("return")

        def request_remote(params: dict, files: list[FastServiceFile]) -> return_type:
            request_body = {}
            for k, v in params.items():
                _append_to_request_body(request_body, k, v)
            request_body = {"params": json.dumps(request_body)}

            request_files = []
            for file in files:
                request_files.append(("files", (file.get_filename(), file.get_file())))

            x = requests.post(
                _generate_request_url(ip, port, module_name, function_name),
                data=request_body,
                files=request_files,
                timeout=CLIENT_SIDE_TIMEOUT,
            )
            # if return_type is list of something, we should return a list of that thing
            if get_origin(return_type) is list:
                return return_type.__call__(
                    [return_type.__args__[0](**_) for _ in x.json()]
                )
            elif return_type is None:
                return None
            else:
                return return_type(**(x.json()))

        return request_remote

    def setup_async_client_with_files(
        self, f, ip: str, port: int, module_name: str, function_name: str
    ):
        return_type = getattr(f, "__annotations__").get("return")

        async def async_request_remote(
            params: dict, files: list[FastServiceFile]
        ) -> return_type:
            request_body = {}
            for k, v in params.items():
                _append_to_request_body(request_body, k, v)
            request_body_json = json.dumps(request_body)

            # Prepare form data for aiohttp
            form_data = aiohttp.FormData()
            form_data.add_field("params", request_body_json)

            # Add files to form data
            for file in files:
                form_data.add_field(
                    "files", file.get_file(), filename=file.get_filename()
                )

            # Use aiohttp for truly asynchronous HTTP requests with files
            url = _generate_request_url(ip, port, module_name, function_name)

            async with aiohttp.ClientSession(
                timeout=ClientTimeout(total=CLIENT_SIDE_TIMEOUT)
            ) as session:
                async with session.post(
                    url,
                    data=form_data,
                    timeout=ClientTimeout(total=CLIENT_SIDE_TIMEOUT),
                ) as response:
                    response_json = await response.json()

                    # if return_type is list of something, we should return a list of that thing
                    if get_origin(return_type) is list:
                        return return_type.__call__(
                            [return_type.__args__[0](**_) for _ in response_json]
                        )
                    elif return_type is None:
                        return None
                    else:
                        return return_type(**(response_json))

        return async_request_remote

    def execute_service(self, ip: str, port: int, *args, **kwargs):
        _execute_service(self.router, ip, port, *args, **kwargs)

    def stop_service(self):
        pass
