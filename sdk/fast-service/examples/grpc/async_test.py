import asyncio
import threading
import time
import sys
from example_app import Item, fast_service_manager, async_compute_total
from fast_service import FastServiceConfig


def start_server():
    config = FastServiceConfig.load_from_file("./config/example-computing-service.yml")
    fast_service_manager.execute(config)


async def run_client():
    # Wait for server to start
    await asyncio.sleep(2)

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

    # Test async function
    print("Testing async function:")

    try:
        start_time = time.time()
        result = await async_compute_total(items=items)
        end_time = time.time()
        print(f"Async compute total result: {result}")
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        print("\nAsync test completed successfully!")
    except Exception as e:
        print(f"Error in async test: {e}")
        print("\nAsync test failed!")

    sys.exit(0)


async def main():
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    # Run client tests
    await run_client()


if __name__ == "__main__":
    asyncio.run(main())
