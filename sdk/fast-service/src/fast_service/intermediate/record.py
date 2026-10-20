import json
from typing import Any
from pydantic import BaseModel
from pathlib import PosixPath
from pathlib import Path


class IntermediateRecord:

    function_name: str  # function name

    request_id: str
    content: Any  # must be json-serializable

    def __init__(self, function_name: str, request_id: str, content: Any):
        self.function_name = function_name
        self.request_id = request_id
        self.content = content

    def dump_content(self):
        if isinstance(self.content, BaseModel):
            return self.content.model_dump_json()
        else:

            def convert_path(obj):
                if isinstance(obj, (PosixPath, Path)):
                    return str(obj)
                return obj

            return json.dumps(self.content, default=convert_path)

    def dump_record(self):
        res = {"request_id": self.request_id, "content": self.content}
        return json.dumps(res)

    def dump_to_tuple(self):
        return self.function_name, self.request_id, self.dump_content()

    def has_file(self) -> bool:
        return False

    def get_cache_dir(self) -> str:
        pass


class IntermediateRequest(IntermediateRecord):

    cache_dir: str

    def __init__(
        self, function_name: str, request_id: str, content: Any, cache_dir: str = None
    ):
        super().__init__(function_name, request_id, content)
        self.cache_dir = cache_dir

    def dump_to_tuple(self):
        return self.function_name, self.request_id, self.dump_content(), self.cache_dir

    def has_file(self) -> bool:
        return self.cache_dir is not None

    def get_cache_dir(self) -> str:
        return self.cache_dir


class IntermediateResponse(IntermediateRecord):

    function_name: str  # function name

    request_id: str
    content: Any  # must be json-serializable

    def __init__(self, function_name: str, request_id: str, content: Any):
        super().__init__(function_name, request_id, content)
