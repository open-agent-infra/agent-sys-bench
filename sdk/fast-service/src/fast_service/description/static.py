from typing import Optional, Callable, Iterable, get_origin, get_args
from typing import Annotated
from pydantic import BaseModel
from abc import ABC
import inspect
import asyncio


class FunctionName:

    module_name: str
    function_name: str

    def __init__(self, module_name: str, function_name: str):
        self.module_name = module_name
        self.function_name = function_name

    def __str__(self):
        return f"{self.module_name}.{self.function_name}"

    def __eq__(self, other):
        if not isinstance(other, FunctionName):
            return False
        return (
            self.module_name == other.module_name
            and self.function_name == other.function_name
        )

    @classmethod
    def of(cls, f: Callable):
        return FunctionName(getattr(f, "__module__"), getattr(f, "__name__"))

    @classmethod
    def str_of(cls, f: Callable):
        return cls.of(f).__str__()


class ParamDescriptor:

    name: str
    cls: type
    index: int
    is_list: bool

    def __init__(self, name: str, cls: type, index: int, is_list: bool):
        self.name = name
        self.cls = cls
        self.index = index
        self.is_list = is_list

    @classmethod
    def parse(
        cls, param_name: str, param_cls: type, param_idx: int
    ) -> "ParamDescriptor":
        is_list = False
        if get_origin(param_cls) is list:
            is_list = True
            param_cls = param_cls.__args__[0]
        if param_name == "return":
            param_idx = -1
        return ParamDescriptor(param_name, param_cls, param_idx, is_list)


class FunctionDescriptor:

    f: Callable
    name: FunctionName
    arg_dict: dict[str, ParamDescriptor]
    cls_dict: dict[type : list[ParamDescriptor]]
    rtn_param: Optional[ParamDescriptor | None]
    is_async: bool
    monitor_only: bool

    def __init__(
        self,
        f: Callable,
        arg_dict: dict[str, ParamDescriptor],
        cls_dict: dict[type:ParamDescriptor],
        rtn_param: ParamDescriptor,
        is_async: bool = False,
        monitor_only: bool = False,
    ):
        self.f = f
        self.name = FunctionName.of(self.f)
        self.arg_dict = arg_dict
        self.cls_dict = cls_dict
        self.rtn_param = rtn_param
        self.is_async = is_async
        self.monitor_only = monitor_only

    def extract_args_as_dict(self, exclude: list[str], *args, **kwargs) -> dict:
        request_body = {}
        args_index = 0
        args_len = len(args)
        for k, v in self.arg_dict.items():
            if k in exclude:
                continue
            if args_index < args_len:
                self._append_to_request_body(request_body, k, args[args_index])
                args_index += 1
            elif k in kwargs:
                self._append_to_request_body(request_body, k, kwargs[k])
        return request_body

    def _transform_param(self, param_value):
        if isinstance(param_value, BaseModel):
            return param_value.model_dump()
        elif isinstance(param_value, dict):
            items = {}
            for k, v in param_value.items():
                items[k] = self._transform_param(v)
            return items
        elif isinstance(param_value, (int, float, bool, str)):
            return param_value
        elif isinstance(param_value, Iterable):
            items = []
            cls = param_value.__class__
            for item in param_value:
                items.append(self._transform_param(item))
            return cls.__call__(items)
        else:
            return param_value

    def _append_to_request_body(self, request_body: dict, param_name, param_value):
        request_body[param_name] = self._transform_param(param_value)

    def extract_args_from_json(self, arg_json) -> dict:
        kwargs = {}
        for k, v in self.arg_dict.items():
            cls = list[v.cls] if v.is_list else v.cls
            kwargs[k] = self._extract_param(arg_json[k], cls)
        return kwargs

    def _extract_param(self, request_field, field_type):
        origin_type = get_origin(field_type)
        if origin_type is None:
            # raw type or raw class
            if field_type is BaseModel or issubclass(field_type, BaseModel):
                return field_type.parse_obj(request_field)
            else:
                return request_field
        elif origin_type is Annotated:
            # Annotated type, we need to extract the actual type
            item_type = get_args(field_type)[0]
            return self._extract_param(request_field, item_type)
        else:
            # wrapped by some container
            item_type = get_args(field_type)[0]
            return origin_type.__call__(
                [self._extract_param(_, item_type) for _ in request_field]
            )

    @classmethod
    def of(cls, f: Callable, monitor_only: bool = False) -> "FunctionDescriptor":
        arg_dict: dict[str, ParamDescriptor] = {}
        cls_dict: dict[type:ParamDescriptor] = {}
        rtn_param = None
        index = 0
        for k, v in getattr(f, "__annotations__").items():
            if k == "return":
                rtn_param = ParamDescriptor.parse(k, v, -1)
                continue
            descriptor = ParamDescriptor.parse(k, v, index)

            arg_dict[k] = descriptor

            if descriptor.cls not in cls_dict:
                cls_dict[descriptor.cls] = []
            cls_dict[descriptor.cls].append(descriptor)
            index += 1

        # Check if the function is async
        is_async = inspect.iscoroutinefunction(f)

        return FunctionDescriptor(
            f, arg_dict, cls_dict, rtn_param, is_async, monitor_only
        )
