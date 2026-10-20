import threading
import time
import sys
from example_app import Item, fast_service_manager, compute_total, update_items
from fast_service import FastServiceConfig


def start_server():
    config = FastServiceConfig.load_from_file("./config/example-computing-service.yml")
    fast_service_manager.execute(config)


def run_client():
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

    # Test gRPC client functions
    print("Testing gRPC client functions:")

    # Test compute_total
    start_time = time.time()
    result = compute_total(items=items)
    end_time = time.time()
    print(f"Compute total result: {result}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # Test update_items
    start_time = time.time()
    updated_items = update_items(items=items)
    end_time = time.time()
    print(f"Updated items: {updated_items}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    print("\nAll tests completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    # Run client tests
    run_client()
