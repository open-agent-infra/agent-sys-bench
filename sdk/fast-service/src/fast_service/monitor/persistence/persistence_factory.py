from .persistence_interface import IMonitorPersistence
from .csv_impl import CSVMonitorPersistence


def create_monitor_persistence(persistence_impl: str) -> IMonitorPersistence:
    return CSVMonitorPersistence()
