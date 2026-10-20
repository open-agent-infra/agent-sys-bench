from abc import ABCMeta, abstractmethod


class IFastServiceEngine(metaclass=ABCMeta):

    @abstractmethod
    def setup_service(self, f):
        pass

    @abstractmethod
    def setup_client(self, f, ip: str, port: int):
        pass

    @abstractmethod
    def setup_service_with_files(self, f, module_name: str, function_name: str):
        pass

    @abstractmethod
    def setup_client_with_files(
        self, f, ip: str, port: int, module_name: str, function_name: str
    ):
        pass

    @abstractmethod
    def setup_async_service(self, f):
        pass

    @abstractmethod
    def setup_async_client(self, f, ip: str, port: int):
        pass

    @abstractmethod
    def setup_async_service_with_files(self, f, module_name: str, function_name: str):
        pass

    @abstractmethod
    def setup_async_client_with_files(
        self, f, ip: str, port: int, module_name: str, function_name: str
    ):
        pass

    @abstractmethod
    def execute_service(self, ip: str, port: int, *args, **kwargs):
        pass

    @abstractmethod
    def stop_service(self):
        pass
