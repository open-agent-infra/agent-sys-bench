import time
import json
import asyncio

from example_app import fast_service_manager, Item

from fast_service import (
    FastServiceConfig,
    FastServiceBenchmark,
    FastServiceBenchmarkConfig,
)


def line_to_req(line: str):
    """Convert a line from the data file to a request object."""
    line = line.strip()
    raw_data = json.loads(line)
    items = [Item(**_) for _ in raw_data["items"]]
    return items


def invoke_service(items):
    """Synchronous service invocation."""
    print(f"Sync service called at {time.time()}")
    # Simulate some processing time
    time.sleep(0.5)
    result = {"total_price": sum(item.price for item in items)}
    print(f"Sync service completed at {time.time()}")
    return result


async def invoke_service_async(items):
    """Asynchronous service invocation."""
    print(f"Async service called at {time.time()}")
    # Simulate some processing time without blocking
    await asyncio.sleep(0.5)
    result = {"total_price": sum(item.price for item in items)}
    print(f"Async service completed at {time.time()}")
    return result


if __name__ == "__main__":
    config = FastServiceConfig.load_from_file("./config/example-client.yml")
    fast_service_manager.setup_client_mode(config)

    # Test with synchronous function
    print("\n=== Testing with synchronous function ===")
    for mode in ["one-by-one", "batch-by-batch", "fixed-interval", "poisson"]:
        print(f"\nStart {mode} benchmark with sync function")
        config = FastServiceBenchmarkConfig(
            file_path="./data/bm_data_file.txt",
            mode=mode,
            request_num=5,
            buffer_size=2,
            output_file="./.cache/bm_sync_response.txt",
            warm_up=1,
            batch_size=2,
            lambda_rate=2.0,
            concurrency="thread",
        )
        benchmark = FastServiceBenchmark(
            config=config, line_to_request=line_to_req, invoke_service=invoke_service
        )
        benchmark.execute()
        print(f"Finish {mode} benchmark with sync function")
        time.sleep(1)

    # Test with asynchronous function
    print("\n=== Testing with asynchronous function ===")
    for mode in ["one-by-one", "batch-by-batch", "fixed-interval", "poisson"]:
        print(f"\nStart {mode} benchmark with async function")
        config = FastServiceBenchmarkConfig(
            file_path="./data/bm_data_file.txt",
            mode=mode,
            request_num=5,
            buffer_size=2,
            output_file="./.cache/bm_async_response.txt",
            warm_up=1,
            batch_size=2,
            lambda_rate=2.0,
            concurrency="async",
        )
        benchmark = FastServiceBenchmark(
            config=config,
            line_to_request=line_to_req,
            invoke_service=invoke_service_async,
        )
        benchmark.execute()
        print(f"Finish {mode} benchmark with async function")
        time.sleep(1)

    # Test with asynchronous function but using thread concurrency
    print("\n=== Testing with asynchronous function using thread concurrency ===")
    for mode in ["one-by-one", "batch-by-batch", "fixed-interval", "poisson"]:
        print(f"\nStart {mode} benchmark with async function (thread concurrency)")
        config = FastServiceBenchmarkConfig(
            file_path="./data/bm_data_file.txt",
            mode=mode,
            request_num=5,
            buffer_size=2,
            output_file="./.cache/bm_async_thread_response.txt",
            warm_up=1,
            batch_size=2,
            lambda_rate=2.0,
            concurrency="thread",
        )
        benchmark = FastServiceBenchmark(
            config=config,
            line_to_request=line_to_req,
            invoke_service=invoke_service_async,
        )
        benchmark.execute()
        print(f"Finish {mode} benchmark with async function (thread concurrency)")
        time.sleep(1)
