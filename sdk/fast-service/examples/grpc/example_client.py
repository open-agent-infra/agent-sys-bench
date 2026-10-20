import asyncio
import time
from example_app import (
    Item,
    fast_service_manager,
    compute_total,
    update_items,
    async_compute_total,
)
from fast_service import FastServiceConfig


async def main():
    # Load client configuration
    config = FastServiceConfig.load_from_file("./config/example-client.yml")

    # Setup client mode
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


if __name__ == "__main__":
    asyncio.run(main())
