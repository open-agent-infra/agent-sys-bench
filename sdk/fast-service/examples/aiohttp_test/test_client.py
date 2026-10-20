import asyncio
import time
import aiohttp
import requests

# Create test data
test_items = [
    {"name": "apple", "price": 10.0},
    {"name": "pear", "price": 20.0},
    {"name": "peach", "price": 30.0},
    {"name": "banana", "price": 15.0},
    {"name": "orange", "price": 25.0},
]


async def test_direct_http_requests():
    """Test using direct HTTP requests with both requests and aiohttp"""
    print("\nDirect HTTP Request Tests:")

    # Test with synchronous requests
    start_time = time.time()
    response = requests.post(
        "http://127.0.0.1:8001/test_app.compute_total", json=test_items
    )
    sync_result = response.json()
    sync_time = time.time() - start_time
    print(f"Sync requests result: {sync_result} (took {sync_time:.2f}s)")
    # Test with aiohttp for async function
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://127.0.0.1:8001/test_app.compute_total_async", json=test_items
        ) as response:
            aiohttp_result = await response.json()
    aiohttp_time = time.time() - start_time
    print(f"Aiohttp async result: {aiohttp_result} (took {aiohttp_time:.2f}s)")
    # Test parallel processing with aiohttp
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://127.0.0.1:8001/test_app.process_items_parallel", json=test_items
        ) as response:
            parallel_result = await response.json()
    parallel_time = time.time() - start_time
    print(f"Parallel aiohttp result: {parallel_result} (took {parallel_time:.2f}s)")

    return sync_time, aiohttp_time, parallel_time


async def test_concurrent_requests():
    """Test making multiple concurrent requests"""
    print("\nConcurrent Request Tests:")

    # Create 5 concurrent requests to the parallel processing endpoint
    start_time = time.time()
    tasks = []
    for _ in range(5):
        task = asyncio.create_task(
            test_single_request("http://127.0.0.1:8001/test_app.process_items_parallel")
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    print(f"Completed 5 concurrent requests in {total_time:.2f}s")
    print(f"Average time per request: {total_time/5:.2f}s")

    return total_time


async def test_single_request(url):
    """Make a single request to the given URL"""
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=test_items) as response:
            result = await response.json()
            return result


async def main():
    print("Testing aiohttp implementation in fast-service")

    # Run the tests
    http_times = await test_direct_http_requests()
    concurrent_time = await test_concurrent_requests()

    # Print summary
    print("\nPerformance Summary:")
    print(
        f"Direct HTTP requests - Sync: {http_times[0]:.2f}s, Async: {http_times[1]:.2f}s, Parallel: {http_times[2]:.2f}s"
    )
    print(
        f"5 concurrent requests: {concurrent_time:.2f}s (avg: {concurrent_time/5:.2f}s per request)"
    )


if __name__ == "__main__":
    asyncio.run(main())
