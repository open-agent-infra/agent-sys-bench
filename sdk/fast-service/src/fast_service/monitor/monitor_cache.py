from queue import Queue

from .record import E2EDelayRecord, RemoteDelayRecord


class MonitorMemoryCache:

    e2e_delay_queue: Queue[E2EDelayRecord]

    remote_delay_queue: Queue[RemoteDelayRecord]

    def __init__(self):
        self.e2e_delay_queue = Queue()
        self.remote_delay_queue = Queue()

    def put_e2e_delay(self, record: E2EDelayRecord):
        self.e2e_delay_queue.put(record)

    def poll_e2e_delay(self) -> E2EDelayRecord | None:
        if self.e2e_delay_queue.empty():
            return None
        return self.e2e_delay_queue.get()

    def put_remote_delay(self, record: RemoteDelayRecord):
        self.remote_delay_queue.put(record)

    def poll_remote_delay(self) -> RemoteDelayRecord | None:
        if self.remote_delay_queue.empty():
            return None
        return self.remote_delay_queue.get()

    def is_empty(self) -> bool:
        return self.e2e_delay_queue.empty() and self.remote_delay_queue.empty()
