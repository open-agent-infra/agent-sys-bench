from .monitor_cache import MonitorMemoryCache
from .monitor_pusher import MonitorPusher
from .monitor_config import MonitorConfig
from .record import E2EDelayRecord, RemoteDelayRecord

from typing import Callable
from functools import wraps
import inspect
import time


class FastServiceMonitorManager:
    """Manager for monitoring service execution times and delays."""

    config: MonitorConfig

    memory_cache: MonitorMemoryCache
    pusher: MonitorPusher

    def start(self, config: MonitorConfig):
        """Start the monitoring system with the given configuration."""
        self.config = config
        self.memory_cache = MonitorMemoryCache()
        self.pusher = MonitorPusher(self.memory_cache, self.config)
        self.pusher.start()

    def stop(self):
        """Stop the monitoring system."""
        self.pusher.stop()

    def monitor_e2e_delay(
        self,
        f: Callable,
        get_request_id: Callable,
        get_invoker_name: Callable,
        service_name: str,
        get_call_id: Callable,
        get_parent_call_id: Callable,
    ):
        """Monitor end-to-end delay for a function.

        Automatically detects if the function is async and applies the appropriate wrapper.
        """
        if inspect.iscoroutinefunction(f):
            return self._monitor_async_e2e_delay(
                f,
                get_request_id,
                get_invoker_name,
                service_name,
                get_call_id,
                get_parent_call_id,
            )
        else:

            @wraps(f)
            def decorator(*args, **kwargs):
                start = time.time()
                res = f(*args, **kwargs)
                end = time.time()
                delay = end - start
                request_id = get_request_id(**kwargs)
                record = E2EDelayRecord(
                    end_time=end,
                    request_id=request_id,
                    invoker_name=get_invoker_name(**kwargs),
                    service_name=service_name,
                    delay=delay,
                    call_id=get_call_id(**kwargs),
                    parent_call_id=get_parent_call_id(**kwargs),
                )
                self.record_e2e_delay(record)
                return res

            return decorator

    def _monitor_async_e2e_delay(
        self,
        f: Callable,
        get_request_id: Callable,
        get_invoker_name: Callable,
        service_name: str,
        get_call_id: Callable,
        get_parent_call_id: Callable,
    ):
        """Monitor end-to-end delay for an async function."""

        @wraps(f)
        async def async_decorator(*args, **kwargs):
            start = time.time()
            res = await f(*args, **kwargs)
            end = time.time()
            delay = end - start
            request_id = get_request_id(**kwargs)
            record = E2EDelayRecord(
                end_time=end,
                request_id=request_id,
                invoker_name=get_invoker_name(**kwargs),
                service_name=service_name,
                delay=delay,
                call_id=get_call_id(**kwargs),
                parent_call_id=get_parent_call_id(**kwargs),
            )
            self.record_e2e_delay(record)
            return res

        return async_decorator

    def monitor_remote_delay(
        self,
        f: Callable,
        get_request_id: Callable,
        get_invoker_name: Callable,
        remote_service_name: str,
        get_call_id: Callable,
        get_parent_call_id: Callable,
    ):
        """Monitor remote call delay for a function.

        Automatically detects if the function is async and applies the appropriate wrapper.
        """
        if inspect.iscoroutinefunction(f):
            return self._monitor_async_remote_delay(
                f,
                get_request_id,
                get_invoker_name,
                remote_service_name,
                get_call_id,
                get_parent_call_id,
            )
        else:

            @wraps(f)
            def decorator(*args, **kwargs):
                start = time.time()
                res = f(*args, **kwargs)
                end = time.time()
                delay = end - start
                request_id = get_request_id(**kwargs)
                invoker_name = get_invoker_name(**kwargs)
                record = RemoteDelayRecord(
                    end_time=end,
                    request_id=request_id,
                    invoker_name=invoker_name,
                    remote_service_name=remote_service_name,
                    delay=delay,
                    call_id=get_call_id(**kwargs),
                    parent_call_id=get_parent_call_id(**kwargs),
                )
                self.record_remote_delay(record)
                return res

            return decorator

    def _monitor_async_remote_delay(
        self,
        f: Callable,
        get_request_id: Callable,
        get_invoker_name: Callable,
        remote_service_name: str,
        get_call_id: Callable,
        get_parent_call_id: Callable,
    ):
        """Monitor remote call delay for an async function."""

        @wraps(f)
        async def async_decorator(*args, **kwargs):
            start = time.time()
            res = await f(*args, **kwargs)
            end = time.time()
            delay = end - start
            request_id = get_request_id(**kwargs)
            invoker_name = get_invoker_name(**kwargs)
            record = RemoteDelayRecord(
                end_time=end,
                request_id=request_id,
                invoker_name=invoker_name,
                remote_service_name=remote_service_name,
                delay=delay,
                call_id=get_call_id(**kwargs),
                parent_call_id=get_parent_call_id(**kwargs),
            )
            self.record_remote_delay(record)
            return res

        return async_decorator

    def record_e2e_delay(self, record: E2EDelayRecord):
        """Record an end-to-end delay measurement."""
        self.memory_cache.put_e2e_delay(record)

    def record_remote_delay(self, record: RemoteDelayRecord):
        """Record a remote call delay measurement."""
        self.memory_cache.put_remote_delay(record)
