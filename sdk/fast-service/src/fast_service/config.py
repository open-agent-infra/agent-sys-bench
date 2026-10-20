import argparse
import yaml

ARG_FULL_NAME_PREFIX = "--"
FAST_SERVICE_CONFIG = "fast_service_config"
DEFAULT_FAST_SERVICE_CONFIG = "./config.yml"


def _transform2arg(name: str) -> str:
    return ARG_FULL_NAME_PREFIX + name


def _read_yaml(path: str) -> dict:
    result = None
    with open(path, "r", encoding="utf-8") as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    if result is None:
        raise IOError(f"Fail to read yaml file {path}")
    if not isinstance(result, dict):
        raise SyntaxError(f"Only support dict style yaml file.")
    return result


FAST_API_IMPL = "fastapi"
LOCAL_FUNCTION_IMPL = "local_function"
GRPC_IMPL = "grpc"
SERVICE_IMPL_LIST = [FAST_API_IMPL, LOCAL_FUNCTION_IMPL, GRPC_IMPL]

DEFAULT_IP = "127.0.0.1"
DEFAULT_PORT = 8000


class SingleService:
    module_name: str
    name: str
    ip: str
    port: int
    service_impl: str = None

    def __init__(
        self, module_name: str, name: str, ip: str, port: int, service_impl: str = None
    ):
        self.module_name = module_name
        self.name = name
        self.ip = ip
        self.port = port
        self.service_impl = service_impl


class MonitorStorageService:
    type: str
    store_dir: str
    ip: str = None
    port: int = None

    def __init__(
        self, type: str = "csv", store_dir: str = None, ip: str = None, port: int = None
    ):
        self.type = type
        self.ip = ip
        self.port = port
        self.store_dir = store_dir


class IntermediateStorage:
    store_dir: str

    def __init__(self, store_dir: str = "./.cache/intermediate"):
        self.store_dir = store_dir


class FastServiceConfig:
    service_impl: str = FAST_API_IMPL
    ip: str = DEFAULT_IP
    port: int = DEFAULT_PORT
    service_dict: dict[(str, str), SingleService] = None
    client_mode: bool = False
    monitor: bool = False
    monitor_storage: MonitorStorageService = None
    store_intermediate: bool = False
    intermediate_storage: IntermediateStorage = None

    def __init__(
        self,
        service_impl: str,
        service_list: list[dict],
        client_mode: bool = False,
        ip: str = None,
        port: int = None,
        monitor: bool = False,
        monitor_storage: dict = None,
        store_intermediate: bool = False,
        intermediate_storage: dict = None,
    ):
        if service_impl not in SERVICE_IMPL_LIST:
            raise ValueError(
                f"service_impl {service_impl} not one of candidate implementation {SERVICE_IMPL_LIST}"
            )
        self.service_impl = service_impl
        self.ip = ip
        self.port = port
        self.service_dict = {}
        for single_service_config in service_list:
            single_service = SingleService(**single_service_config)
            self.service_dict[(single_service.module_name, single_service.name)] = (
                single_service
            )

        self.client_mode = client_mode

        self.monitor = monitor

        if monitor_storage is None:
            self.monitor_storage = MonitorStorageService()
        else:
            self.monitor_storage = MonitorStorageService(**monitor_storage)

        self.store_intermediate = store_intermediate
        if intermediate_storage is None:
            self.intermediate_storage = IntermediateStorage()
        else:
            self.intermediate_storage = IntermediateStorage(**intermediate_storage)

    @classmethod
    def load_from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            _transform2arg(FAST_SERVICE_CONFIG),
            type=str,
            default=DEFAULT_FAST_SERVICE_CONFIG,
            help="The config file for fast-service.",
        )
        args = vars(parser.parse_args())
        config_path = args[FAST_SERVICE_CONFIG]

        config_dict = _read_yaml(config_path)

        return FastServiceConfig(**config_dict)

    @classmethod
    def load_from_file(cls, path: str):
        config_dict = _read_yaml(path)

        return FastServiceConfig(**config_dict)

    @classmethod
    def load_from_dict(cls, config_dict: dict):
        return FastServiceConfig(**config_dict)
