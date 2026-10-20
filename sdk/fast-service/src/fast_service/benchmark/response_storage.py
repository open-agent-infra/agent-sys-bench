import os
import time
import threading
import multiprocessing
import queue

from typing import Optional, Any


def _serialize_resp(
    resp_cache: queue.Queue,
    io_queue: multiprocessing.Queue,
    stop_event: threading.Event,
):
    while not stop_event.is_set() or not resp_cache.empty():
        if resp_cache.empty():
            time.sleep(1)
            continue

        resp = resp_cache.get()
        io_queue.put(resp)


def _persistent_resp(
    config: dict,
    io_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
):
    output_file: str = config["output_file"]
    abs_path = os.path.abspath(output_file)
    dir_path = os.path.dirname(abs_path)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    with open(output_file, "a") as f:
        count = 0
        while not stop_event.is_set() or not io_queue.empty():
            if io_queue.empty():
                time.sleep(1)
                continue

            resp = io_queue.get()
            f.write(f"{resp}\n")

            count += 1
            if count % 10 == 0:
                f.flush()


class ResponseStorage:

    output_file: str

    resp_cache: queue.Queue
    io_queue: multiprocessing.Queue

    serializer: Optional[threading.Thread | None]
    serializer_stop_event: Optional[threading.Event | None]

    io_executor: Optional[multiprocessing.Process | None]
    io_stop_event: Optional[Any]

    def __init__(self, output_file: str):
        self.output_file = output_file

        self.resp_cache = queue.Queue()
        self.io_queue = multiprocessing.Queue()

        self.serializer = None
        self.serializer_stop_event = None

        self.io_executor = None
        self.io_stop_event = None

    def start(self):
        if self.serializer is not None or self.io_executor is not None:
            return

        self.serializer_stop_event = threading.Event()
        self.serializer = threading.Thread(
            target=_serialize_resp,
            args=(
                self.resp_cache,
                self.io_queue,
                self.serializer_stop_event,
            ),
        )
        self.serializer.start()

        self.io_stop_event = multiprocessing.Event()
        self.io_executor = multiprocessing.Process(
            target=_persistent_resp,
            args=(
                {
                    "output_file": self.output_file,
                },
                self.io_queue,
                self.io_stop_event,
            ),
        )
        self.io_executor.start()

    def stop(self):
        if self.serializer is None or self.io_executor is None:
            return

        self.serializer_stop_event.set()
        self.serializer.join()
        self.serializer_stop_event = None
        self.serializer = None

        self.io_stop_event.set()
        self.io_executor.join()
        self.io_stop_event = None
        self.io_executor = None

    def store_in_str(self, resp: str):
        self.resp_cache.put(resp)
