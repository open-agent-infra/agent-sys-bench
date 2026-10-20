import signal
import time
import threading
import multiprocessing
import queue
import logging

from typing import Optional, Any, Callable

from .record import IntermediateRequest, IntermediateResponse
from .file_manager import FileManager

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_INTERVAL = 1


def _serialize_result(
    req_cache: queue.Queue,
    resp_cache: queue.Queue,
    req_io_queue: multiprocessing.Queue,
    resp_io_queue: multiprocessing.Queue,
    stop_event: threading.Event,
    start_event: threading.Event,
):
    logger.info("Start intermediate result serialization thread.")
    start_event.set()

    main_thread = threading.main_thread()
    while (main_thread.is_alive() and not stop_event.is_set()) or not (
        req_cache.empty() and resp_cache.empty()
    ):
        if req_cache.empty() and resp_cache.empty():
            time.sleep(DEFAULT_INTERVAL)
            continue

        if not req_cache.empty():
            req_record: IntermediateRequest = req_cache.get()
            req_io_queue.put(req_record.dump_to_tuple())

        if not resp_cache.empty():
            resp_record: IntermediateResponse = resp_cache.get()
            resp_io_queue.put(resp_record.dump_to_tuple())

    logger.info(
        f"Stop intermediate result serialization thread. Memory cache is empty: {req_cache.empty()} {resp_cache.empty()}.",
    )


def _persistent_result(
    args: dict,
    req_io_queue: multiprocessing.Queue,
    resp_io_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    start_event: multiprocessing.Event,
):
    logger.info("Start intermediate result I/O process.")
    start_event.set()

    file_manager = FileManager(args["dir_path"])

    def stop_process():
        file_manager.flush()
        file_manager.close()
        logger.info(
            f"Stop intermediate result I/O process. "
            f"I/O queue is empty: {req_io_queue.empty()} {resp_io_queue.empty()}."
        )

    def process_signal(signum, frame):
        stop_process()

    signal.signal(signal.SIGTERM, process_signal)

    try:
        start_time = time.time()
        while not stop_event.is_set() or not (
            req_io_queue.empty() and resp_io_queue.empty()
        ):
            if req_io_queue.empty() and resp_io_queue.empty():
                time.sleep(DEFAULT_INTERVAL)
                continue

            if not req_io_queue.empty():
                req_item = IntermediateRequest(*req_io_queue.get())
                file_manager.write_req(req_item)

            if not resp_io_queue.empty():
                resp_item = IntermediateResponse(*resp_io_queue.get())
                file_manager.write_resp(resp_item)

            if start_time + DEFAULT_INTERVAL < time.time():
                file_manager.flush()
                start_time = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        stop_process()


class StorageManager:

    dir_path: str

    req_cache: queue.Queue
    resp_cache: queue.Queue

    req_io_queue: multiprocessing.Queue
    resp_io_queue: multiprocessing.Queue

    serializer: Optional[threading.Thread | None]
    serializer_stop_event: Optional[threading.Event | None]

    io_executor: Optional[multiprocessing.Process | None]
    io_stop_event: Optional[Any]

    def __init__(self, dir_path: str):
        self.dir_path = dir_path

        self.req_cache = queue.Queue()
        self.resp_cache = queue.Queue()
        self.req_io_queue = multiprocessing.Queue()
        self.resp_io_queue = multiprocessing.Queue()

        self.serializer = None
        self.serializer_stop_event = None

        self.io_executor = None
        self.io_stop_event = None

    def start(self):
        if self.serializer is not None or self.io_executor is not None:
            return

        self.serializer_stop_event = threading.Event()
        serializer_start_event = threading.Event()
        self.serializer = threading.Thread(
            target=_serialize_result,
            args=(
                self.req_cache,
                self.resp_cache,
                self.req_io_queue,
                self.resp_io_queue,
                self.serializer_stop_event,
                serializer_start_event,
            ),
        )
        self.serializer.start()
        serializer_start_event.wait()

        self.io_stop_event = multiprocessing.Event()
        io_start_event = multiprocessing.Event()
        self.io_executor = multiprocessing.Process(
            target=_persistent_result,
            args=(
                {"dir_path": self.dir_path},
                self.req_io_queue,
                self.resp_io_queue,
                self.io_stop_event,
                io_start_event,
            ),
        )
        self.io_executor.start()
        io_start_event.wait()

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

    def store_req_record(self, record: IntermediateRequest):
        self.req_cache.put(record)

    def store_resp_record(self, record: IntermediateResponse):
        self.resp_cache.put(record)
