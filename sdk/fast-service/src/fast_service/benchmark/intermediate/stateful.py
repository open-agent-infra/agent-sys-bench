import os
import json
import asyncio
import time
import uuid
import logging

from typing import Callable, Any
from pydantic import BaseModel

from ...description import FunctionDescriptor, ContextParamUsage, RequestContext
from ...file import FastServiceFile, FileParamUsage
from ...fast_service import FastServiceManager

from ..request_loader import RequestLoader, SimpleRequestLoader
from ..bm_config import FastServiceBenchmarkConfig
from ..benchmark import FastServiceBenchmark


logger = logging.getLogger(__name__)


class GroupedRequestLoader(RequestLoader):

    module_name: str

    # parse json to module request
    # (module_name, req_data) -> req
    request_parser: Callable[[str, str], Any]
    # monitor file storing the execution sequence of module interfaces
    sequence_file_path: str
    # directory that store all req data of module interfaces
    function_file_path_dict: dict[str, str]

    must_have_funcs: list[str] | None = None

    request_num: int
    buffer_size: int
    sampling_params: dict

    request_buffer: list | None = None
    buffer_valid_size: int = 0
    buffer_index: int = 0

    # request_id -> list(interval, function)
    request_sequence: list[tuple[str, list[tuple[float, str]]]]
    sub_loader_dict: dict[str, SimpleRequestLoader]

    request_index: int
    # request_id -> list(interval, function, request)
    next_request: tuple[str, list[tuple[float, str, Any]]] | None

    def __init__(
        self,
        module_name: str,
        request_parser: Callable[[str, str], Any],
        sequence_file_path: str,
        function_file_path_dict: dict[str, str],
        must_have_funcs: list[str] | None = None,
        request_num: int = -1,
        buffer_size: int = -1,
        sampling_params: dict = None,
        match_policy: str = "first",
    ):
        self.module_name = module_name
        self.match_policy = match_policy

        self.request_parser = request_parser
        self.sequence_file_path = sequence_file_path
        self.function_file_path_dict = function_file_path_dict
        self.must_have_funcs = must_have_funcs

        self.request_num = request_num
        self.buffer_size = buffer_size
        self.request_buffer = None
        self.buffer_valid_size = 0
        self.buffer_index = 0
        self.sampling_params = sampling_params

        self._init_request_sequence(self.sequence_file_path)
        self._init_single_file_loaders(self.function_file_path_dict)

        self.sub_loader_requests = None
        if self.match_policy in ["same_request_id", "exact"]:
            self._load_all_sub_loaders()

        self.request_index = 0
        self.next_request = None
        self._get_next_batch()

    def _init_request_sequence(self, sequence_file_path: str):
        self.request_sequence = []
        request_sequences = {}
        request_start_time = {}
        with open(sequence_file_path, "r") as sequence_file:
            lines = sequence_file.readlines()
            for i, line in enumerate(lines):
                entries = line.split(",")
                op_finish_time = float(entries[0])
                req_id = entries[1]
                op_name = entries[3]
                op_execution_time = float(entries[4])
                op_start_time = op_finish_time - op_execution_time
                if req_id not in request_sequences:
                    request_sequences[req_id] = []
                    request_start_time[req_id] = op_start_time
                request_start_time[req_id] = min(
                    request_start_time[req_id], op_start_time
                )
                request_sequences[req_id].append(
                    {
                        "op_name": op_name,
                        "start_time": op_start_time,
                        "end_time": op_finish_time,
                        "execution_time": op_execution_time,
                    }
                )
        request_sequences = dict(
            sorted(request_sequences.items(), key=lambda x: request_start_time[x[0]])
        )
        for req_id, ops in request_sequences.items():
            ops.sort(key=lambda x: x["start_time"])
            group_ops = [
                (op["start_time"], op["op_name"])
                for op in ops
                if self.module_name in op["op_name"]
            ]
            # make start_time is relative to the first operation
            if group_ops:
                first_start_time = group_ops[0][0]
                group_ops = [
                    (start_time - first_start_time, op_name)
                    for start_time, op_name in group_ops
                ]
            self.request_sequence.append(
                (
                    req_id,
                    group_ops,
                )
            )

    def _init_single_file_loaders(self, function_file_path_dict: dict[str, str]):
        self.sub_loader_dict = {}
        for function_name, req_file in function_file_path_dict.items():

            def get_single_func_parser(func_name: str):
                def single_func_parser(line):
                    return self.request_parser(func_name, line)

                return single_func_parser

            self.sub_loader_dict[function_name] = SimpleRequestLoader(
                line_to_request=get_single_func_parser(function_name),
                file_path=req_file,
                request_num=-1,
                buffer_size=self.buffer_size,
            )

    def _load_all_sub_loaders(self):
        assert self.match_policy in ["same_request_id", "exact"]
        self.sub_loader_requests = {}
        for op_name, sub_loader in self.sub_loader_dict.items():
            if op_name not in self.sub_loader_requests:
                self.sub_loader_requests[op_name] = {}
            while sub_loader.has_next():
                req = sub_loader.next()
                request_id = req["request_id"]
                content = req["content"]
                if request_id not in self.sub_loader_requests[op_name]:
                    self.sub_loader_requests[op_name][request_id] = []
                self.sub_loader_requests[op_name][request_id].append(content)

        # reset all sub_loaders
        for subloader in self.sub_loader_dict.values():
            subloader.reset()

    def _get_next_request(self) -> tuple[str, list[tuple[float, str, Any]]]:
        if self.request_index >= len(self.request_sequence):
            raise Exception("No next request")
        req_meta = self.request_sequence[self.request_index]
        request_id = req_meta[0]
        op_sequence: list[tuple[float, str]] = req_meta[1]
        sub_request_list: list[tuple[float, str, Any]] = []
        for op_interval, op_name in op_sequence:
            if not self.sub_loader_dict[op_name].has_next():
                raise Exception(f"No next request of {op_name}")
            if self.match_policy == "first":
                req_data = self.sub_loader_dict[op_name].next()
                sub_request_list.append((op_interval, op_name, req_data))
            elif self.match_policy in ["same_request_id", "exact"]:
                try:
                    if op_name not in self.sub_loader_requests:
                        raise Exception(
                            f"No requests for {op_name} in sub_loader_requests"
                        )
                    if request_id not in self.sub_loader_requests[op_name]:
                        raise Exception(
                            f"Request id {request_id} not found in {op_name} sub_loader_requests"
                        )
                    if not self.sub_loader_requests[op_name][request_id]:
                        raise Exception(
                            f"No requests for {request_id} in {op_name} sub_loader_requests"
                        )

                    sub_request_list.append(
                        (
                            op_interval,
                            op_name,
                            self.sub_loader_requests[op_name][request_id].pop(0),
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error while processing request {request_id} for operation {op_name}: {e}"
                    )
                    req_data = self.sub_loader_dict[op_name].next()
                    sub_request_list.append((op_interval, op_name, req_data["content"]))
            else:
                raise Exception(f"Unknown match policy: {self.match_policy}")
        self.next_request = (request_id, sub_request_list)
        logger.debug(f"request data for {request_id} : {self.next_request}")
        self.request_index += 1
        return (request_id, sub_request_list)

    def _get_next_batch(self):
        if self.request_buffer is None:
            self.request_buffer = []
        batch_size = len(self.request_sequence) - self.request_index
        if self.buffer_size > 0:
            batch_size = min(batch_size, self.buffer_size - len(self.request_buffer))
        if self.request_num > 0:
            batch_size = min(batch_size, self.request_num)
        batch_size = max(batch_size, 0)
        for _ in range(batch_size):
            req = self._get_next_request()
            if self.sampling_params:
                if (
                    "method" in self.sampling_params
                    and self.sampling_params["method"] == "uniform"
                ):
                    step = self.sampling_params["step"]
                    offset = self.sampling_params["offset"]
                    if (self.request_index - 1) % step != offset:
                        continue
                else:
                    raise Exception(f"Unknown sampling method: {self.sampling_params}")
            op_names = [op[1] for op in req[1]]
            if self.must_have_funcs:
                skip = False
                for func in self.must_have_funcs:
                    contain_func = any(func in op_name for op_name in op_names)
                    if not contain_func:
                        skip = True
                        break
                if skip:
                    logger.info(
                        f"Skip request {req[0]} because it does not have all must have functions"
                    )
                    continue
            logger.info(
                f"got request for request {req[0]} with {len(req[1])} operations"
            )
            self.request_buffer.append(req)
            self.buffer_valid_size += 1

    def has_next(self) -> bool:
        # not read file yet
        if self.request_buffer is None:
            self._get_next_batch()

        # end of file
        if self.buffer_valid_size == 0:
            return False

        if self.buffer_size == -1:
            return self.buffer_index < self.buffer_valid_size
        else:
            if self.buffer_index < self.buffer_valid_size:
                return True

            # buffer run out, get next batch
            self._get_next_batch()
            if self.buffer_valid_size == 0:
                return False
            else:
                self.buffer_index = 0
                return True

    def next(self):
        if not self.has_next():
            raise Exception("No next request")
        result = self.request_buffer[self.buffer_index]
        self.buffer_index += 1
        return result

    def close(self):
        for sub_loader in self.sub_loader_dict.values():
            sub_loader.close()


class StatefulModuleBenchmarkConfig:

    module_name: str
    sequence_file_path: str
    dir_path: str

    init_function: str
    context_id_param: str
    cleanup_function: str

    base_config: FastServiceBenchmarkConfig
    request_version: str
    async_server: bool
    speedup_factor: float

    match_policy: str

    def __init__(
        self,
        module_name: str,
        sequence_file_path: str,
        dir_path: str,
        base_config: FastServiceBenchmarkConfig,
        init_function: str | None = None,
        context_id_param: str | None = None,
        cleanup_function: str | None = None,
        request_version: str = None,
        async_server: bool = False,
        speedup_factor: float = 1.0,
        match_policy: str = "first",
    ):
        self.module_name = module_name
        self.sequence_file_path = sequence_file_path
        self.dir_path = dir_path
        self.init_function = init_function
        self.context_id_param = context_id_param
        self.cleanup_function = cleanup_function
        self.base_config = base_config
        self.request_version = request_version
        self.async_server = async_server
        self.speedup_factor = speedup_factor
        self.match_policy = match_policy

        if (init_function is None) != (context_id_param is None):
            raise Exception(
                "init_function and context_id_param must be specified simultaneously"
            )


def get_function_invoker(
    module_name: str,
    function_set,
    init_function: str,
    context_id_param: str,
    fast_service_manager: FastServiceManager,
    async_server: bool,
    speedup_factor: float = 1.0,
):
    func_dict = {}
    descriptor_dict = {}
    context_usage_dict = {}

    for function_name in function_set:
        function, descriptor = fast_service_manager.get_service(
            module_name, function_name.split(".")[-1]
        )
        func_dict[function_name] = function
        descriptor_dict[function_name] = descriptor
        context_usage = ContextParamUsage(descriptor)
        context_usage_dict[function_name] = context_usage

    def invoke_service(func_name: str, kwargs, context: RequestContext):
        func = func_dict[func_name]
        func_context_usage = context_usage_dict[func_name]
        if func_context_usage.use_context():
            kwargs[func_context_usage.context_name()] = context

        if async_server:
            return asyncio.run(func(**kwargs))
        else:
            return func(**kwargs)

    def sequential_invoke(grouped_request: tuple[str, list[tuple[float, str, Any]]]):

        context_id = None

        request_id = str(uuid.uuid4())
        root_call_id = str(uuid.uuid4())
        start_time = time.time()
        for relative_ts, func_name, req_data in grouped_request[1]:
            if init_function is not None and init_function not in func_name:
                for arg in req_data.values():
                    if isinstance(arg, dict):
                        if context_id_param in arg:
                            arg[context_id_param] = context_id
                    elif isinstance(arg, BaseModel):
                        if arg.__getattribute__(context_id_param) is not None:
                            arg.__setattr__(context_id_param, context_id)

            cur_ts = time.time()
            schedule_ts = start_time + (relative_ts / speedup_factor)
            if schedule_ts > cur_ts:
                time.sleep(schedule_ts - cur_ts)
            res = invoke_service(
                func_name,
                req_data,
                RequestContext(request_id=request_id, call_id=root_call_id),
            )

            if init_function is not None and init_function in func_name:
                if isinstance(res, BaseModel):
                    context_id = res.__getattribute__(context_id_param)
                elif isinstance(res, str):
                    context_id = res
                else:
                    raise Exception(f"Unexpected type {type(res)} for state context id")

        logger.debug(f"Request {request_id} finished")

    return sequential_invoke, descriptor_dict


def get_request_parser(
    dir_path: str,
    descriptor_dict: dict[str, FunctionDescriptor],
    return_request_id: bool = False,
) -> Callable:

    def parse_request(func_name: str, line: str):
        descriptor = descriptor_dict[func_name]

        record = json.loads(line)
        request_id = record["request_id"]
        args = json.loads(record["content"])
        kwargs = descriptor.extract_args_from_json(args)

        context_usage = ContextParamUsage(descriptor)
        if context_usage.use_context():
            kwargs.pop(context_usage.context_name())

        file_usage = FileParamUsage(descriptor)
        if file_usage.use_file() and file_usage.file_param_name() in kwargs:
            file_path_record = kwargs[file_usage.file_param_name()]
            request_dir = os.path.join(dir_path, func_name, request_id)
            if file_usage.single_file():
                file_param = FastServiceFile.load(
                    open(os.path.join(request_dir, file_path_record), "rb")
                )
            else:
                file_param = [
                    FastServiceFile.load(open(os.path.join(request_dir, _), "rb"))
                    for _ in file_path_record
                ]
            kwargs[file_usage.file_param_name()] = file_param
        if return_request_id:
            return {"request_id": request_id, "content": kwargs}
        else:
            return kwargs

    return parse_request


class StatefulModuleBenchmark:

    benchmark: FastServiceBenchmark

    config: StatefulModuleBenchmarkConfig

    fast_service_manager: FastServiceManager

    def __init__(
        self,
        config: StatefulModuleBenchmarkConfig,
        fast_service_manager: FastServiceManager,
    ):

        self.config = config
        self.config.base_config.file_path = self.config.sequence_file_path

        self.fast_service_manager = fast_service_manager

        function_file_dict = self._get_function_file_dict()

        self.invoke_service, descriptor_dict = get_function_invoker(
            config.module_name,
            function_file_dict.keys(),
            config.init_function,
            config.context_id_param,
            fast_service_manager,
            config.async_server,
            speedup_factor=config.speedup_factor,
        )

        must_have_funcs = []
        if self.config.init_function is not None:
            must_have_funcs.append(self.config.init_function)
        if self.config.cleanup_function is not None:
            must_have_funcs.append(self.config.cleanup_function)

        request_loader = GroupedRequestLoader(
            module_name=self.config.module_name,
            request_parser=get_request_parser(
                self.config.dir_path,
                descriptor_dict,
                return_request_id=self.config.match_policy
                in ["same_request_id", "exact"],
            ),
            sequence_file_path=self.config.sequence_file_path,
            function_file_path_dict=function_file_dict,
            must_have_funcs=must_have_funcs or None,
            request_num=self.config.base_config.request_num,
            buffer_size=self.config.base_config.buffer_size,
            sampling_params=self.config.base_config.sampling_params,
            match_policy=self.config.match_policy,
        )

        self.benchmark = FastServiceBenchmark(
            config=self.config.base_config,
            invoke_service=self.invoke_service,
            request_loader=request_loader,
        )

    def _get_function_file_dict(self) -> dict[str, str]:
        function_dict: dict[str, str] = {}
        for root, dirs, files in os.walk(self.config.dir_path):
            if "requests.txt" in files and self.config.module_name in root:
                function_name = os.path.basename(root)
                function_dict[function_name] = os.path.join(root, "requests.txt")
        logger.info(f"load requests from: \n{'\n'.join(function_dict.values())}")

        return function_dict

    def execute(self):
        self.benchmark.execute()
