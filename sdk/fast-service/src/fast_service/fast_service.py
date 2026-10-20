import atexit
import signal
import sys
import logging
import inspect

from .config import FastServiceConfig
from .engine import IFastServiceEngine, create_fast_service_engine
from .monitor import FastServiceMonitorManager, MonitorConfig
from .description import (
    FunctionDescriptor,
)
from .rewriter import FastServiceRewriter
from .intermediate import IntermediateManager

from typing import Callable, Optional, Any


formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


GLOBAL_INSTANCE_MAP: dict[str, "FastServiceManager"] = {}


class ServiceDescriptor:

    f: Callable

    raw_function: FunctionDescriptor

    def __init__(
        self,
        f: Callable,
        raw_function: FunctionDescriptor,
    ):
        self.f = f
        self.raw_function = raw_function

    @classmethod
    def init(cls, f: Callable, monitor_only: bool = False) -> "ServiceDescriptor":

        raw_function = FunctionDescriptor.of(f, monitor_only=monitor_only)

        return ServiceDescriptor(f, raw_function)

    @classmethod
    def rewrite(cls, descriptor: "ServiceDescriptor", f: Callable):
        return ServiceDescriptor(f, descriptor.raw_function)


class FastServiceManager:

    config: FastServiceConfig

    registered_service: dict[str, ServiceDescriptor]

    engine: Optional[IFastServiceEngine]
    monitor_manager: Optional[FastServiceMonitorManager]
    intermediate_manager: Optional[IntermediateManager]

    def __init__(self, config: FastServiceConfig = None):
        super().__init__()
        self.config = config

        frame = inspect.currentframe().f_back
        caller_module = inspect.getmodule(frame).__name__
        GLOBAL_INSTANCE_MAP[caller_module] = self

        self.registered_service = {}
        self.engine = None
        self.monitor_manager = None
        self.intermediate_manager = None

    def fast_service(self, f: Callable = None, monitor_only: bool = False):

        def decorate_function(f: Callable) -> Callable:
            service_id = (getattr(f, "__module__"), getattr(f, "__name__"))
            descriptor: ServiceDescriptor = ServiceDescriptor.init(
                f, monitor_only=monitor_only
            )
            self.registered_service[service_id] = descriptor

            def dynamic_invoke(*args, **kwargs):
                return self.registered_service[service_id].f(*args, **kwargs)

            return dynamic_invoke

        if f is None:
            return decorate_function
        else:
            return decorate_function(f)

    def async_fast_service(self, f: Callable = None, monitor_only: bool = False):
        """Register an async function with the service"""

        def decorate_function(f: Callable) -> Callable:
            service_id = (getattr(f, "__module__"), getattr(f, "__name__"))
            self.registered_service[service_id] = ServiceDescriptor.init(
                f, monitor_only=monitor_only
            )

            async def async_dynamic_invoke(*args, **kwargs):
                return await self.registered_service[service_id].f(*args, **kwargs)

            return async_dynamic_invoke

        if f is None:
            return decorate_function
        else:
            return decorate_function(f)

    def _update_config(self, config: FastServiceConfig):
        if config is not None:
            self.config = config
        elif self.config is None:
            self.config = FastServiceConfig.load_from_args()

    def _check_config(self, *args, **kwargs):
        if kwargs["client_mode"] is not None:
            expected_client_mode = kwargs["client_mode"]
            if expected_client_mode and not self.config.client_mode:
                raise Exception("Client mode is disabled while setting up client mode.")
            elif not expected_client_mode and self.config.client_mode:
                raise Exception("Client mode is enabled while setting up service.")

    def _create_component(self):
        self.engine = create_fast_service_engine(self.config.service_impl)
        if self.config.monitor:
            self.monitor_manager = FastServiceMonitorManager()
        if self.config.store_intermediate:
            self.intermediate_manager = IntermediateManager()

    def _rewrite_registered_functions(self):
        rewriter: FastServiceRewriter = FastServiceRewriter(
            self.config, self.engine, self.monitor_manager, self.intermediate_manager
        )
        for service_id, descriptor in self.registered_service.items():
            descriptor: ServiceDescriptor = descriptor
            f = rewriter.rewrite(descriptor.raw_function)
            self.registered_service[service_id] = ServiceDescriptor.rewrite(
                descriptor,
                f,
            )

    def execute(self, config: FastServiceConfig = None, *args, **kwargs):
        self._update_config(config)
        self._check_config(client_mode=False)

        self._create_component()

        self._rewrite_registered_functions()

        if self.config.monitor:
            self.monitor_manager.start(
                MonitorConfig.from_dict(self.config.monitor_storage.__dict__)
            )
        if self.config.store_intermediate:
            self.intermediate_manager.start(self.config.intermediate_storage.store_dir)
        self.engine.execute_service(
            ip=self.config.ip, port=self.config.port, *args, **kwargs
        )

    def setup_client_mode(self, config: FastServiceConfig = None, *args, **kwargs):
        self._update_config(config)
        self._check_config(client_mode=True)

        self._create_component()

        self._rewrite_registered_functions()

        if self.config.monitor:
            self.monitor_manager.start(
                MonitorConfig.from_dict(self.config.monitor_storage.__dict__)
            )
        if self.config.store_intermediate:
            self.intermediate_manager.start(self.config.intermediate_storage.store_dir)

    def get_registered_service(self) -> list[tuple[str, str]]:
        result = []
        for descriptor in self.registered_service.values():
            result.append(
                (getattr(descriptor.f, "__module__"), getattr(descriptor.f, "__name__"))
            )
        return result

    def get_service(
        self, module_name: str, service_name: str
    ) -> (Callable, FunctionDescriptor):
        service_descriptor = self.registered_service[(module_name, service_name)]
        return service_descriptor.f, service_descriptor.raw_function

    @classmethod
    def get_all_registered_service(cls) -> list[tuple[str, str]]:
        result = []
        for k, v in GLOBAL_INSTANCE_MAP.items():
            result.extend(v.get_registered_service())
        return result

    def stop_server(self):
        if self.config is None:
            # not running
            return
        if self.engine is not None:
            self.engine.stop_service()
        if self.config.monitor:
            self.monitor_manager.stop()
        if self.config.store_intermediate:
            self.intermediate_manager.stop()

    def stop_client(self):
        if self.config is None:
            # not running
            return
        if self.config.client_mode:
            if self.config.monitor:
                self.monitor_manager.stop()
            if self.config.store_intermediate:
                self.intermediate_manager.stop()


fast_service_manager = FastServiceManager()


def _clear_client_resource():
    # at any time, clear client resource on process exit
    running_client_manager = []
    for v in GLOBAL_INSTANCE_MAP.values():
        if v.config is not None and v.config.client_mode:
            running_client_manager.append(v)
    for m in running_client_manager:
        m.stop_client()

    logger.info("Clean fast service resource on process exit.")


atexit.register(_clear_client_resource)


def _process_exit_signal(signum, frame):
    # managers in server mode tend to be killed manually
    for v in GLOBAL_INSTANCE_MAP.values():
        v.stop_server()

    sys.exit(0)


signal.signal(signal.SIGTERM, _process_exit_signal)
signal.signal(signal.SIGINT, _process_exit_signal)
