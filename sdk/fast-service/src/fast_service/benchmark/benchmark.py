import inspect
from typing import Callable, Any, Optional, Union, Awaitable


from .bm_config import FastServiceBenchmarkConfig
from .request_loader import RequestLoader, SimpleRequestLoader
from .request_scheduler import RequestScheduler, SchedulerFactory
from .response_storage import ResponseStorage


class FastServiceBenchmark:
    config: FastServiceBenchmarkConfig

    line_to_request: Callable[[str], Any]
    invoke_service: Union[Callable[[Any], Any], Callable[[Any], Awaitable[Any]]]
    is_async_service: bool

    request_loader: Optional[RequestLoader]
    request_scheduler: Optional[RequestScheduler]
    response_storage: Optional[ResponseStorage]

    def __init__(
        self,
        config: FastServiceBenchmarkConfig,
        line_to_request: Callable[[str], Any] = None,
        invoke_service: Union[
            Callable[[Any], Any], Callable[[Any], Awaitable[Any]]
        ] = None,
        request_loader: Optional[RequestLoader] = None,
    ):
        self.config = config
        self.line_to_request = line_to_request
        self.invoke_service = invoke_service
        self.is_async_service = (
            invoke_service is not None and inspect.iscoroutinefunction(invoke_service)
        )
        self.request_loader = request_loader

    def execute(self):
        if self.request_loader is None:
            self.request_loader = SimpleRequestLoader(
                self.line_to_request,
                self.config.file_path,
                self.config.request_num,
                self.config.buffer_size,
                sampling_params=self.config.sampling_params,
            )

        if self.config.output_file is None:
            self.response_storage = None
        else:
            self.response_storage = ResponseStorage(output_file=self.config.output_file)

        self.request_scheduler = SchedulerFactory.create_scheduler(
            self.request_loader,
            self.response_storage,
            self.invoke_service,
            **self.config.__dict__,
        )

        try:
            self.request_scheduler.execute()
        finally:
            self.request_loader.close()
