import signal
import sys

from .monitor_cache import MonitorMemoryCache
from .persistence import IMonitorPersistence, create_monitor_persistence
from .monitor_config import MonitorConfig
from .record import E2EDelayRecord, RemoteDelayRecord

from threading import Thread
from multiprocessing import Process, Queue

import time
import threading
import multiprocessing
import logging

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_TIME_INTERVAL = 1


def _transport_cache(
    memory_cache: MonitorMemoryCache,
    e2e_delay_queue: Queue,
    remote_delay_queue: Queue,
    stop_event: threading.Event,
    start_event: threading.Event,
):
    logger.info("Start monitor data serialization thread.")
    start_event.set()

    main_thread = threading.main_thread()
    while (
        main_thread.is_alive() and not stop_event.is_set()
    ) or not memory_cache.is_empty():
        has_item = False

        e2e_item = memory_cache.poll_e2e_delay()
        if e2e_item is not None:
            e2e_delay_queue.put(e2e_item.to_tuple())
            has_item = True

        remote_item = memory_cache.poll_remote_delay()
        if remote_item is not None:
            remote_delay_queue.put(remote_item.to_tuple())
            has_item = True

        if not has_item:
            time.sleep(DEFAULT_TIME_INTERVAL)

    logger.info(
        f"Stop monitor data serialization thread. Monitor data cache is empty: {memory_cache.is_empty()}.",
    )


def _start_persistence(
    config: dict,
    e2e_delay_queue: Queue,
    remote_delay_queue: Queue,
    stop_event: multiprocessing.Event,
    start_event: multiprocessing.Event,
):
    logger.info("Start monitor data persistence process.")
    config = MonitorConfig(**config)
    persistence: IMonitorPersistence = create_monitor_persistence(config.type)
    persistence.execute(config)
    start_event.set()

    def stop_process():
        persistence.flush()
        persistence.close()
        logger.info(
            f"Stop monitor data persistence process. "
            f"Monitor data queue is empty: {e2e_delay_queue.empty()};{remote_delay_queue.empty()}."
        )

    def process_signal(signum, frame):
        stop_process()

    signal.signal(signal.SIGTERM, process_signal)

    start_time = time.time()
    try:
        while (
            not stop_event.is_set()
            or not e2e_delay_queue.empty()
            or not remote_delay_queue.empty()
        ):
            if start_time + DEFAULT_TIME_INTERVAL < time.time():
                persistence.flush()
                start_time = time.time()

            has_item = False
            if not e2e_delay_queue.empty():
                persistence.store_e2e_delay(E2EDelayRecord(*(e2e_delay_queue.get())))
                has_item = True
            if not remote_delay_queue.empty():
                persistence.store_remote_delay(
                    RemoteDelayRecord(*(remote_delay_queue.get()))
                )
                has_item = True

            if not has_item:
                time.sleep(DEFAULT_TIME_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        stop_process()


class MonitorPusher:
    config: MonitorConfig

    memory_cache: MonitorMemoryCache

    # time, request_id, service_name, delay
    e2e_delay_queue: Queue
    # time, request_id, service_name, remote_service_name, delay
    remote_delay_queue: Queue

    sender: Thread
    sender_stop_event: threading.Event

    consumer: Process
    consumer_stop_event: multiprocessing.Event

    def __init__(self, memory_cache: MonitorMemoryCache, config: MonitorConfig):
        self.memory_cache = memory_cache
        self.config = config

        self.e2e_delay_queue = Queue()
        self.remote_delay_queue = Queue()

    def start(self):
        self.consumer_stop_event = multiprocessing.Event()
        consumer_start_event = multiprocessing.Event()
        self.consumer = Process(
            target=_start_persistence,
            args=(
                self.config.__dict__,
                self.e2e_delay_queue,
                self.remote_delay_queue,
                self.consumer_stop_event,
                consumer_start_event,
            ),
        )
        self.consumer.start()
        consumer_start_event.wait()

        self.sender_stop_event = threading.Event()
        sender_start_event = threading.Event()
        self.sender = Thread(
            target=_transport_cache,
            args=(
                self.memory_cache,
                self.e2e_delay_queue,
                self.remote_delay_queue,
                self.sender_stop_event,
                sender_start_event,
            ),
        )
        self.sender.start()
        sender_start_event.wait()

    def stop(self):
        self.sender_stop_event.set()
        self.sender.join()

        self.consumer_stop_event.set()
        self.consumer.join()
