import uuid

from pydantic import BaseModel
from typing import Optional

from .static import FunctionDescriptor, ParamDescriptor, FunctionName


class RequestContext(BaseModel):

    call_id: str

    request_id: str
    _function_name: FunctionName

    _parent_context: Optional["RequestContext"]

    def __init__(
        self,
        request_id: str = None,
        function_name: FunctionName = None,
        call_id: str = None,
    ):
        super().__init__(call_id=call_id, request_id=request_id)

        self.call_id = call_id

        self.request_id = request_id
        self._function_name = function_name

        self._parent_context = None

    def set_parent_context(self, parent_context: Optional["RequestContext"]):
        self._parent_context = parent_context

    def get_parent_service(self):
        return self._parent_context._function_name if self._parent_context else None

    def get_call_id(self) -> str:
        return self.call_id

    def get_parent_call_id(self):
        return self._parent_context.call_id if self._parent_context else None

    @classmethod
    def generate_request_id(cls):
        return str(uuid.uuid4())

    @classmethod
    def generate_call_id(cls):
        return str(uuid.uuid4())


class ContextParamUsage:

    f: FunctionDescriptor

    descriptor: Optional[ParamDescriptor | None]

    def __init__(self, f: FunctionDescriptor):
        self.f = f

        descriptor_list = f.cls_dict.get(RequestContext)

        if descriptor_list:
            if len(descriptor_list) > 1:
                raise TypeError(
                    "RequestContext param is repeated defined in given function."
                )

            if descriptor_list[0].is_list:
                raise TypeError("RequestContext param shall not be used in list.")

            self.descriptor = descriptor_list[0]
        else:
            self.descriptor = None

    def use_context(self) -> bool:
        return self.descriptor is not None

    def context_name(self) -> str:
        return self.descriptor.name

    def param_index(self) -> int:
        return self.descriptor.index

    def get_context_arg(self, *args, **kwargs) -> RequestContext | None:
        if len(args) > self.descriptor.index:
            return args[self.descriptor.index]
        elif self.descriptor.name in kwargs:
            return kwargs[self.descriptor.name]
        else:
            return None
