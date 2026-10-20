from abc import ABCMeta, abstractmethod
from ..monitor_config import MonitorConfig
from ..record import E2EDelayRecord, RemoteDelayRecord


class IMonitorPersistence(metaclass=ABCMeta):

    @abstractmethod
    def store_e2e_delay(self, record: E2EDelayRecord):
        pass

    @abstractmethod
    def store_remote_delay(self, record: RemoteDelayRecord):
        pass

    @abstractmethod
    def execute(self, config: MonitorConfig, *args, **kwargs):
        pass

    @abstractmethod
    def close(self, *args, **kwargs):
        pass

    @abstractmethod
    def flush(self):
        pass
