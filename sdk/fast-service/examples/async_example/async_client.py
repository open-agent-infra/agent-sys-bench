import asyncio
import time
import requests
import aiohttp
from pydantic import BaseModel
from fast_service import FastServiceConfig

from async_app import (
    Item,
    Collection,
    compute_total,
    compute_total_async,
    process_items_parallel,
    fast_service_manager,
)


# Sample data
raw_data = [
    {"name": "apple", "price": 10.0},
    {"name": "pear", "price": 20.0},
    {"name": "peach", "price": 30.0},
]


async def example_http_client():
    """Example of using HTTP client directly with aiohttp"""
    async with aiohttp.ClientSession() as session:
        # Call synchronous function
        start_time = time.time()
        async with session.post(
            "http://127.0.0.1:8003/async_app.compute_total", json=raw_data
        ) as response:
            sync_result = dict(await response.json())
        sync_time = time.time() - start_time
        # Call asynchronous function
        start_time = time.time()
        async with session.post(
            "http://127.0.0.1:8003/async_app.compute_total_async", json=raw_data
        ) as response:
            async_result = dict(await response.json())
        async_time = time.time() - start_time
        # Call parallel function
        start_time = time.time()
        async with session.post(
            "http://127.0.0.1:8003/async_app.process_items_parallel", json=raw_data
        ) as response:
            parallel_result = dict(await response.json())
        parallel_time = time.time() - start_time
    print("HTTP Client Results (using aiohttp):")
    print(f"Sync result: {sync_result} (took {sync_time:.2f}s)")
    print(f"Async result: {async_result} (took {async_time:.2f}s)")
    print(f"Parallel result: {parallel_result} (took {parallel_time:.2f}s)")


async def example_fast_service_async_client():
    """Example of using fast-service async client"""
    config = FastServiceConfig.load_from_file("./config/async-client.yml")
    fast_service_manager.setup_client_mode(config)

    request = [Item(**_) for _ in raw_data]
    # Call synchronous function
    start_time = time.time()
    sync_result = compute_total(items=request).model_dump()
    sync_time = time.time() - start_time
    # Call asynchronous functions
    start_time = time.time()
    async_result = await compute_total_async(items=request)
    async_result = async_result.model_dump()
    async_time = time.time() - start_time
    start_time = time.time()
    parallel_result = await process_items_parallel(items=request)
    parallel_result = parallel_result.model_dump()
    parallel_time = time.time() - start_time
    print("\nFast-Service Async Client Results:")
    print(f"Sync result: {sync_result} (took {sync_time:.2f}s)")
    print(f"Async result: {async_result} (took {async_time:.2f}s)")
    print(f"Parallel result: {parallel_result} (took {parallel_time:.2f}s)")


async def main():
    # Run HTTP client example with aiohttp
    await example_http_client()

    # Run async client example
    await example_fast_service_async_client()


if __name__ == "__main__":
    asyncio.run(main())
