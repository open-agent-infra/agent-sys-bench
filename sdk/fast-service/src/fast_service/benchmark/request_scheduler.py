import time
import os
import numpy as np
import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

from abc import ABCMeta, abstractmethod
from typing import Callable, Any, Literal, Union, Awaitable
from functools import wraps
import traceback
import logging

from .request_loader import RequestLoader
from .response_storage import ResponseStorage

logger = logging.getLogger(__name__)

CONCURRENCY_TYPE = Literal["async", "thread", "process"]


class RequestScheduler(metaclass=ABCMeta):

    request_loader: RequestLoader
    response_storage: ResponseStorage
    invoke_service: Union[Callable[[Any], Any], Callable[[Any], Awaitable[Any]]]
    is_async_service: bool

    request_num: int = -1
    warm_up: int = 0
    warm_up_time: float = 0

    concurrency: CONCURRENCY_TYPE = "thread"

    def __init__(
        self,
        request_loader: RequestLoader,
        response_storage: ResponseStorage,
        invoke_service: Union[Callable[[Any], Any], Callable[[Any], Awaitable[Any]]],
        request_num: int = -1,
        warm_up: int = 0,
        warm_up_time: float = 0,
        concurrency: CONCURRENCY_TYPE = "thread",
    ):
        self.request_loader = request_loader
        self.response_storage = response_storage
        self.request_num = request_num
        self.warm_up = warm_up
        self.warm_up_time = warm_up_time
        self.concurrency = concurrency
        self.max_nthreads = int(os.environ.get("FSBENCH_NUM_THREADS", 32))
        self.max_nprocesses = int(
            os.environ.get("FSBENCH_NUM_PROCESSES", os.cpu_count())
        )

        # Check if the invoke_service is an async function
        self.is_async_service = inspect.iscoroutinefunction(invoke_service)

        if self.response_storage is not None:
            if self.is_async_service:

                @wraps(invoke_service)
                async def async_resp_storage_wrapper(*args, **kwargs):
                    """Wrapper to store the response in the response storage."""
                    """This wrapper is used to handle async service invocation."""
                    try:
                        resp = await invoke_service(*args, **kwargs)
                    except Exception as e:
                        print(
                            f"Error during async service invocation: {str(e)}, traceback:\n {traceback.format_exc()}",
                            flush=True,
                        )
                        logger.exception("Error during async service invocation")
                        resp = str(e)
                    finally:
                        self.response_storage.store_in_str(str(resp))
                    return resp

                self.invoke_service = async_resp_storage_wrapper
            else:

                @wraps(invoke_service)
                def resp_storage_wrapper(*args, **kwargs):
                    """Wrapper to store the response in the response storage."""
                    try:
                        # Call the synchronous service function
                        resp = invoke_service(*args, **kwargs)
                    except Exception as e:
                        print(f"Error during service invocation: {e}", flush=True)
                        logger.exception("Error during service invocation")
                        resp = str(e)
                    finally:
                        self.response_storage.store_in_str(str(resp))
                    return resp

                self.invoke_service = resp_storage_wrapper
        else:
            # self.invoke_service = invoke_service
            if self.is_async_service:

                @wraps(invoke_service)
                async def async_wrapper(*args, **kwargs):
                    """Wrapper to handle async service invocation."""
                    try:
                        resp = await invoke_service(*args, **kwargs)
                    except Exception as e:
                        print(
                            f"Error during async service invocation: {str(e)}",
                            flush=True,
                        )
                        logging.exception("Error during async service invocation")
                        resp = str(e)
                    return resp

                self.invoke_service = async_wrapper
            else:

                @wraps(invoke_service)
                def sync_wrapper(*args, **kwargs):
                    """Wrapper to handle synchronous service invocation."""
                    try:
                        resp = invoke_service(*args, **kwargs)
                    except Exception as e:
                        print(f"Error during service invocation: {e}", flush=True)
                        logger.exception("Error during service invocation")
                        resp = str(e)
                    return resp

                self.invoke_service = sync_wrapper

    def execute(self):
        if self.response_storage is not None:
            self.response_storage.start()

        if self.warm_up > 0:
            warm_up_start = time.time()
            print(
                f"Warm up for {self.warm_up} requests started at {warm_up_start}",
                flush=True,
            )
            if self.is_async_service:
                asyncio.run(self._warm_up_async())
            else:
                self._warm_up()
            while time.time() - warm_up_start < self.warm_up_time:
                time.sleep(0.1)
            print(
                f"Warm up for {self.warm_up} requests finished at {time.time()}",
                flush=True,
            )
        self._execute()

        if self.response_storage is not None:
            self.response_storage.stop()

    @abstractmethod
    def _execute(self):
        pass

    def _warm_up(self):
        """Synchronous warm-up method."""
        index = 0
        while self.request_loader.has_next() and index < self.warm_up:
            request = self.request_loader.next()
            self.invoke_service(request)
            index += 1

    async def _warm_up_async(self):
        """Asynchronous warm-up method for async functions."""
        index = 0
        while self.request_loader.has_next() and index < self.warm_up:
            request = self.request_loader.next()
            await self.invoke_service(request)
            index += 1

    async def invoke_service_async(self, request):
        """Wrapper for the invoke_service method to run it asynchronously.

        If the service is already async, directly await it.
        If the service is sync, run it in a thread pool.
        """
        if self.is_async_service:
            return await self.invoke_service(request)
        else:
            return await asyncio.to_thread(self.invoke_service, request)


class OneByOneScheduler(RequestScheduler):

    def __init__(
        self,
        request_loader: RequestLoader,
        response_storage: ResponseStorage,
        invoke_service: Union[Callable[[Any], Any], Callable[[Any], Awaitable[Any]]],
        request_num: int = -1,
        warm_up: int = 0,
        warm_up_time: float = 0,
        concurrency: CONCURRENCY_TYPE = "thread",
    ):
        super().__init__(
            request_loader=request_loader,
            response_storage=response_storage,
            invoke_service=invoke_service,
            request_num=request_num,
            warm_up=warm_up,
            warm_up_time=warm_up_time,
            concurrency=concurrency,
        )

    async def _execute_async(self):
        """Execute the scheduler with async service."""
        index = 0
        while self.request_loader.has_next() and index < self.request_num:
            request = self.request_loader.next()
            await self.invoke_service(request)
            index += 1

    def _execute(self):
        """Execute the scheduler with the appropriate method based on service type."""
        if self.is_async_service:
            asyncio.run(self._execute_async())
        else:
            index = 0
            while self.request_loader.has_next() and index < self.request_num:
                request = self.request_loader.next()
                self.invoke_service(request)
                index += 1


class BatchByBatchScheduler(RequestScheduler):

    batch_size: int = 1

    def __init__(
        self,
        request_loader: RequestLoader,
        response_storage: ResponseStorage,
        invoke_service: Union[Callable[[Any], Any], Callable[[Any], Awaitable[Any]]],
        request_num: int = -1,
        warm_up: int = 0,
        warm_up_time: float = 0,
        batch_size: int = 1,
        concurrency: CONCURRENCY_TYPE = "thread",
    ):
        super().__init__(
            request_loader=request_loader,
            response_storage=response_storage,
            invoke_service=invoke_service,
            request_num=request_num,
            warm_up=warm_up,
            warm_up_time=warm_up_time,
            concurrency=concurrency,
        )
        self.batch_size = batch_size

    async def _execute_async(self):
        """Execute the scheduler with async concurrency."""
        index = 0
        while self.request_loader.has_next() and index < self.request_num:
            reqs = []
            for _ in range(self.batch_size):
                if self.request_loader.has_next() and index < self.request_num:
                    request = self.request_loader.next()
                    reqs.append(request)
                    index += 1
                else:
                    break

            # Process the batch asynchronously
            if self.is_async_service:
                # If the service is async, directly create tasks with it
                tasks = [
                    asyncio.create_task(self.invoke_service(request))
                    for request in reqs
                ]
            else:
                # If the service is sync, use the wrapper to run in thread pool
                tasks = [
                    asyncio.create_task(self.invoke_service_async(request))
                    for request in reqs
                ]
            await asyncio.gather(*tasks)

    def _execute_thread(self):
        """Execute the scheduler with thread concurrency."""
        if self.is_async_service:
            print(
                "Warning: Async functions are not supported with thread concurrency. Falling back to synchronous execution."
            )

            def async_wrapper(request):
                asyncio.run(self.invoke_service(request))

            worker_fn = async_wrapper
        else:
            worker_fn = self.invoke_service

        index = 0
        while self.request_loader.has_next() and index < self.request_num:
            reqs = []
            for _ in range(self.batch_size):
                if self.request_loader.has_next() and index < self.request_num:
                    request = self.request_loader.next()
                    reqs.append(request)
                    index += 1
                else:
                    break

            # Process the batch synchronously
            with ThreadPoolExecutor(max_workers=len(reqs)) as executor:
                executor.map(worker_fn, reqs)

    def _execute_process(self):
        """Execute the scheduler with process concurrency."""
        if self.is_async_service:
            # Async functions can't be pickled for multiprocessing
            print(
                "Warning: Async functions are not supported with process concurrency. Falling back to synchronous execution."
            )

            def async_wrapper(request):
                asyncio.run(self.invoke_service(request))

            worker_fn = async_wrapper
        else:
            worker_fn = self.invoke_service

        index = 0
        while self.request_loader.has_next() and index < self.request_num:
            reqs = []
            for _ in range(self.batch_size):
                if self.request_loader.has_next() and index < self.request_num:
                    request = self.request_loader.next()
                    reqs.append(request)
                    index += 1
                else:
                    break

            # Process the batch synchronously
            with ProcessPoolExecutor(max_workers=len(reqs)) as executor:
                executor.map(worker_fn, reqs)

    def _execute(self):
        """Entry point for executing the scheduler."""
        if self.concurrency == "async":
            asyncio.run(self._execute_async())
        elif self.concurrency == "thread":
            self._execute_thread()
        elif self.concurrency == "process":
            self._execute_process()
        else:
            raise ValueError(f"Unsupported concurrency type: {self.concurrency}")


class PoissonScheduler(RequestScheduler):

    lambda_rate: float = 1.0

    def __init__(
        self,
        request_loader: RequestLoader,
        response_storage: ResponseStorage,
        invoke_service: Union[Callable[[Any], Any], Callable[[Any], Awaitable[Any]]],
        request_num: int = -1,
        warm_up: int = 0,
        warm_up_time: float = 0,
        lambda_rate: float = 1.0,
        concurrency: CONCURRENCY_TYPE = "thread",
    ):
        super().__init__(
            request_loader=request_loader,
            response_storage=response_storage,
            invoke_service=invoke_service,
            request_num=request_num,
            warm_up=warm_up,
            warm_up_time=warm_up_time,
            concurrency=concurrency,
        )
        self.lambda_rate = lambda_rate

    def generate_arrival_times(self):
        """Generates arrival times based on Poisson distribution."""
        # Generate inter-arrival times using exponential distribution
        rng = np.random.default_rng(seed=42)
        inter_arrival_times = rng.exponential(1 / self.lambda_rate, self.request_num)
        arrival_times = np.cumsum(
            inter_arrival_times
        )  # Cumulative sum to get actual arrival times
        return arrival_times

    async def _execute_async(self):
        """Execute the scheduler with async concurrency."""
        arrival_times = self.generate_arrival_times()
        start_time = time.time()
        tasks = []
        index = 0

        while self.request_loader.has_next() and index < self.request_num:
            request = self.request_loader.next()
            arrival_time = arrival_times[index]

            time_to_wait = max(0.0, arrival_time - (time.time() - start_time))
            await asyncio.sleep(time_to_wait)  # Wait until the arrival time

            if self.is_async_service:
                # If the service is async, directly create a task with it
                task = asyncio.create_task(self.invoke_service(request))
            else:
                # If the service is sync, use the wrapper to run in thread pool
                task = asyncio.create_task(self.invoke_service_async(request))
            tasks.append(task)
            index += 1

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    def _execute_thread(
        self,
        arrival_times: np.array = None,
        start_time: float = None,
        rank: int = 0,
        size: int = 1,
    ):
        """Execute the scheduler with thread concurrency."""
        if self.is_async_service:
            print(
                "Warning: Async functions are not supported with thread concurrency. Falling back to synchronous execution."
            )

            def async_wrapper(request):
                asyncio.run(self.invoke_service(request))

            worker_fn = async_wrapper
        else:
            worker_fn = self.invoke_service

        if arrival_times is None:
            arrival_times = self.generate_arrival_times()
        if start_time is None:
            start_time = time.time()

        executor = ThreadPoolExecutor(max_workers=self.max_nthreads)
        index = 0
        while self.request_loader.has_next() and index < self.request_num:
            request = self.request_loader.next()
            if index % size == rank:
                arrival_time = arrival_times[index]
                time_to_wait = max(0.0, arrival_time - (time.time() - start_time))
                time.sleep(time_to_wait)

                executor.submit(worker_fn, request)
            index += 1
        executor.shutdown(wait=True)

    def _execute_process(self):
        """Execute the scheduler with process concurrency."""
        if self.is_async_service:
            # Async functions can't be pickled for multiprocessing
            print(
                "Warning: Async functions are not supported with process concurrency. Falling back to synchronous execution."
            )

            def async_wrapper(request):
                asyncio.run(self.invoke_service(request))

            worker_fn = async_wrapper
        else:
            worker_fn = self.invoke_service

        arrival_times = self.generate_arrival_times()
        executor = ProcessPoolExecutor(max_workers=self.max_nprocesses)
        start_time = time.time()
        index = 0

        while self.request_loader.has_next() and index < self.request_num:
            request = self.request_loader.next()
            arrival_time = arrival_times[index]

            time_to_wait = max(0.0, arrival_time - (time.time() - start_time))
            time.sleep(time_to_wait)

            executor.submit(worker_fn, request)
            index += 1
        executor.shutdown(wait=True)

    def _execute_process_thread(self):
        """Execute the scheduler with process-thread concurrency."""
        # combine process and thread execution
        # create multiple process, each process has a threadpool
        # each process send requests in a threadpool
        arrival_times = self.generate_arrival_times()
        start_time = time.time() + 1.0
        processes = []
        for pid in range(self.max_nprocesses):
            process = multiprocessing.Process(
                target=self._execute_thread,
                args=(arrival_times, start_time, pid, self.max_nprocesses),
            )
            processes.append(process)
            process.start()

        # Wait for all processes to complete
        for process in processes:
            process.join()

    def _execute(self):
        """Entry point for executing the scheduler."""
        if self.concurrency == "async":
            asyncio.run(self._execute_async())
        elif self.concurrency == "thread":
            self._execute_thread()
        elif self.concurrency == "process":
            self._execute_process()
        elif self.concurrency == "process-thread":
            self._execute_process_thread()
        else:
            raise ValueError(f"Unsupported concurrency type: {self.concurrency}")


class FixedIntervalScheduler(PoissonScheduler):
    def generate_arrival_times(self):
        """Generates arrival times based on Poisson distribution."""
        # Generate inter-arrival times using exponential distribution
        inter_arrival_times = [1.0 / self.lambda_rate] * self.request_num
        arrival_times = np.cumsum(
            inter_arrival_times
        )  # Cumulative sum to get actual arrival times
        return arrival_times


class SpecifiedScheduler(PoissonScheduler):
    arrival_times: list[float]

    def __init__(
        self,
        request_loader: RequestLoader,
        response_storage: ResponseStorage,
        invoke_service: Union[Callable[[Any], Any], Callable[[Any], Awaitable[Any]]],
        request_num: int = -1,
        warm_up: int = 0,
        warm_up_time: float = 0,
        arrival_times: list[float] = None,
        concurrency: CONCURRENCY_TYPE = "thread",
    ):
        assert request_num > 0
        assert arrival_times is not None
        assert len(arrival_times) >= request_num
        self.arrival_times = arrival_times
        super().__init__(
            request_loader=request_loader,
            response_storage=response_storage,
            invoke_service=invoke_service,
            request_num=request_num,
            warm_up=warm_up,
            warm_up_time=warm_up_time,
            concurrency=concurrency,
        )

    def generate_arrival_times(self):
        return self.arrival_times


class SchedulerFactory:

    @classmethod
    def create_scheduler(
        cls,
        request_loader: RequestLoader,
        response_storage: ResponseStorage,
        invoke_service: Union[Callable[[Any], Any], Callable[[Any], Awaitable[Any]]],
        **kwargs,
    ):
        if kwargs["mode"] == "one-by-one":
            return OneByOneScheduler(
                request_loader=request_loader,
                response_storage=response_storage,
                invoke_service=invoke_service,
                request_num=kwargs["request_num"],
                warm_up=kwargs["warm_up"],
                warm_up_time=kwargs["warm_up_time"],
                concurrency=kwargs.get("concurrency", "thread"),
            )
        elif kwargs["mode"] == "poisson":
            return PoissonScheduler(
                request_loader=request_loader,
                response_storage=response_storage,
                invoke_service=invoke_service,
                request_num=kwargs["request_num"],
                warm_up=kwargs["warm_up"],
                warm_up_time=kwargs["warm_up_time"],
                lambda_rate=kwargs["lambda_rate"],
                concurrency=kwargs.get("concurrency", "thread"),
            )
        elif kwargs["mode"] == "fixed-interval":
            return FixedIntervalScheduler(
                request_loader=request_loader,
                response_storage=response_storage,
                invoke_service=invoke_service,
                request_num=kwargs["request_num"],
                warm_up=kwargs["warm_up"],
                warm_up_time=kwargs["warm_up_time"],
                lambda_rate=kwargs["lambda_rate"],
                concurrency=kwargs.get("concurrency", "thread"),
            )
        elif kwargs["mode"] == "specified":
            return SpecifiedScheduler(
                request_loader=request_loader,
                response_storage=response_storage,
                invoke_service=invoke_service,
                request_num=kwargs["request_num"],
                warm_up=kwargs["warm_up"],
                warm_up_time=kwargs["warm_up_time"],
                arrival_times=kwargs["arrival_times"],
                concurrency=kwargs.get("concurrency", "thread"),
            )
        elif kwargs["mode"] == "batch-by-batch":
            return BatchByBatchScheduler(
                request_loader=request_loader,
                response_storage=response_storage,
                invoke_service=invoke_service,
                request_num=kwargs["request_num"],
                warm_up=kwargs["warm_up"],
                warm_up_time=kwargs["warm_up_time"],
                batch_size=kwargs["batch_size"],
                concurrency=kwargs.get("concurrency", "thread"),
            )
        else:
            raise ValueError(f"Unsupported mode: {kwargs['mode']}")
