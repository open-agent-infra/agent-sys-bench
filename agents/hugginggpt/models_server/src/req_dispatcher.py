import uuid

from queue import Queue
from typing import Any
from threading import Thread, Event, Semaphore, Lock

from model_worker import MultiModelMsgChannel


class InferenceFuture:

    _raw_request: Any

    _result: Any
    _is_done: Event

    def __init__(self, raw_request: Any):
        self._raw_request = raw_request

        self._is_done = Event()
        self._result = None

    def is_done(self) -> bool:
        return self._is_done.is_set()

    def get_result(self) -> Any:
        self._is_done.wait()
        return self._result

    def set_is_done(self):
        self._is_done.set()

    def get_raw_request(self) -> Any:
        return self._raw_request

    def set_result(self, result: Any):
        self._result = result


class InferenceDispatcher:

    raw_req_queue: Queue

    dispatcher_thread: Thread
    recv_thread: Thread
    on_fly_semaphore: Semaphore
    stop_event: Event

    req_dict_lock: Lock
    dispatched_req_dict: dict[str, InferenceFuture]

    msg_channel: MultiModelMsgChannel

    def __init__(self, msg_channel: MultiModelMsgChannel):
        self.raw_req_queue = Queue()

        self.dispatcher_thread = Thread(target=self._dispatch)
        self.recv_thread = Thread(target=self._receive)
        self.on_fly_semaphore = Semaphore(0)
        self.stop_event = Event()

        self.dispatched_req_dict = {}
        self.req_dict_lock = Lock()

        self.msg_channel = msg_channel

        self.dispatcher_thread.start()
        self.recv_thread.start()

    def submit(self, model_name: str, req: Any) -> InferenceFuture:
        future = InferenceFuture(req)
        self.raw_req_queue.put((model_name, future))
        return future

    def _dispatch(self):
        while not self.stop_event.is_set():
            model_name, future = self.raw_req_queue.get()
            future: InferenceFuture = future
            req_id = str(uuid.uuid4())
            with self.req_dict_lock:
                self.dispatched_req_dict[req_id] = future
            self.msg_channel.send_input(model_name, (req_id, future.get_raw_request()))
            self.on_fly_semaphore.release()

    def _receive(self):
        while not self.stop_event.is_set():
            if not self.on_fly_semaphore.acquire(timeout=5):
                continue
            req_id, result = self.msg_channel.recv_output()
            with self.req_dict_lock:
                future = self.dispatched_req_dict[req_id]
                self.dispatched_req_dict.pop(req_id)
            future.set_result(result)
            future.set_is_done()

    def stop(self):
        self.stop_event.set()
        self.dispatcher_thread.join()
        self.recv_thread.join()
