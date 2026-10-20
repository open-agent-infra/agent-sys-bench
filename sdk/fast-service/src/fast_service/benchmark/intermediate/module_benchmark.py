import os
import json
import asyncio


from ...description import FunctionDescriptor, ContextParamUsage
from ...file import FastServiceFile, FileParamUsage
from ...fast_service import FastServiceManager

from ..benchmark import FastServiceBenchmark

from .module_bm_config import FastServiceModuleBenchmarkConfig


def get_function_invoker(
    module_name: str,
    function_name: str,
    fast_service_manager: FastServiceManager,
    async_server: bool,
):
    func, descriptor = fast_service_manager.get_service(module_name, function_name)

    def invoke_service(kwargs):
        if async_server:
            return asyncio.run(func(**kwargs))
        else:
            return func(**kwargs)

    return invoke_service, descriptor


def get_request_loader(dir_path: str, descriptor: FunctionDescriptor):

    def load_request(line: str):
        record = json.loads(line)
        request_id = record["request_id"]
        args = json.loads(record["content"])
        kwargs = descriptor.extract_args_from_json(args)

        context_usage = ContextParamUsage(descriptor)
        if context_usage.use_context():
            kwargs.pop(context_usage.context_name())

        file_usage = FileParamUsage(descriptor)
        if file_usage.use_file() and file_usage.file_param_name() in kwargs:
            file_path_list = kwargs[file_usage.file_param_name()]
            request_dir = os.path.join(dir_path, request_id)
            file_list = [
                FastServiceFile.load(open(os.path.join(request_dir, _), "rb"))
                for _ in file_path_list
            ]
            kwargs[file_usage.file_param_name()] = file_list

        return kwargs

    return load_request


class FastServiceModuleBenchmark:

    benchmark: FastServiceBenchmark

    config: FastServiceModuleBenchmarkConfig

    fast_service_manager: FastServiceManager

    def __init__(
        self,
        config: FastServiceModuleBenchmarkConfig,
        fast_service_manager: FastServiceManager,
    ):

        self.config = config
        self.config.base_config.file_path = os.path.join(
            self.config.dir_path, "requests.txt"
        )
        if config.request_version is not None:
            if config.request_version != "latest":
                self.config.base_config.file_path = os.path.join(
                    self.config.dir_path, f"requests.txt.{config.request_version}"
                )

        self.fast_service_manager = fast_service_manager

        self.invoke_service, descriptor = get_function_invoker(
            config.module_name,
            config.function_name,
            fast_service_manager,
            config.async_server,
        )

        self.line_to_request = get_request_loader(self.config.dir_path, descriptor)

        self.benchmark = FastServiceBenchmark(
            self.config.base_config, self.line_to_request, self.invoke_service
        )

    def execute(self):
        self.benchmark.execute()
