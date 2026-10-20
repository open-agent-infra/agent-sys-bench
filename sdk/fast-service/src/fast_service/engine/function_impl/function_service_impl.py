from typing import Callable, Any, BinaryIO
from pydantic import BaseModel
import asyncio

from ..engine_interface import IFastServiceEngine


class FastServiceOnFunction(IFastServiceEngine):

    def setup_service(self, f):
        return f

    def setup_service_with_files(self, f, module_name: str, function_name: str):
        return f

    def setup_client(self, f, ip: str, port: int):
        return f

    def setup_client_with_files(
        self, f, ip: str, port: int, module_name: str, function_name: str
    ):
        pass

    def setup_async_service(self, f):
        return f

    def setup_async_client(self, f, ip: str, port: int):
        return f

    def setup_async_service_with_files(self, f, module_name: str, function_name: str):
        return f

    def setup_async_client_with_files(
        self, f, ip: str, port: int, module_name: str, function_name: str
    ):
        pass

    def execute_service(self, ip: str, port: int, *args, **kwargs):
        pass

    def stop_service(self):
        pass
