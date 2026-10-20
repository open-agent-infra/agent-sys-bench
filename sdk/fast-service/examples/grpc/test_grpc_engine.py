import asyncio
import threading
import time
from example_app import (
    Item,
    fast_service_manager,
    compute_total,
    update_items,
    async_compute_total,
)
from fast_service import FastServiceConfig


def start_server():
    config = FastServiceConfig.load_from_file("./config/example-computing-service.yml")
    fast_service_manager.execute(config)


async def run_client():
    # Wait for server to start
    time.sleep(2)

    # Load client configuration
    config = FastServiceConfig.load_from_file("./config/example-client.yml")

    # Set up client mode
    fast_service_manager.setup_client_mode(config)

    # Create test data
    items = [
        Item(name="item1", price=10.0),
        Item(name="item2", price=20.0),
        Item(name="item3", price=30.0),
    ]

    # Test synchronous functions
    print("Testing synchronous functions:")
    start_time = time.time()
    result = compute_total(items=items)
    end_time = time.time()
    print(f"Compute total result: {result}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    updated_items = update_items(items=items)
    end_time = time.time()
    print(f"Updated items: {updated_items}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # Test asynchronous function
    print("\nTesting asynchronous function:")
    start_time = time.time()
    async_result = await async_compute_total(items=items)
    end_time = time.time()
    print(f"Async compute total result: {async_result}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    print("\nAll tests completed successfully!")
    return True


async def main():
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    # Run client tests
    success = await run_client()

    # If tests were successful, exit
    if success:
        import os

        os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
