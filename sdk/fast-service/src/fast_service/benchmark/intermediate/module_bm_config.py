from ..bm_config import FastServiceBenchmarkConfig


class FastServiceModuleBenchmarkConfig:

    module_name: str
    function_name: str
    dir_path: str
    base_config: FastServiceBenchmarkConfig
    request_version: str
    async_server: bool

    def __init__(
        self,
        module_name: str,
        function_name: str,
        dir_path: str,
        base_config: FastServiceBenchmarkConfig,
        request_version: str = None,
        async_server: bool = False,
    ):
        self.module_name = module_name
        self.function_name = function_name
        self.dir_path = dir_path
        self.base_config = base_config
        self.request_version = request_version
        self.async_server = async_server
