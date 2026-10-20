"""
Multi-engine client example.

This script demonstrates how to use both FastAPI and gRPC engines
from a single client.
"""

import os
import sys
import time
import asyncio
from pathlib import Path

from fast_service import FastServiceConfig
from multi_engine_service import fastapi_service_manager, grpc_service_manager
from multi_engine_service import Item
from multi_engine_service import (
    compute_total,
    update_items,
    async_compute_total,
    get_items,
    fastapi_async_cost_benchmark,
    grpc_async_cost_benchmark,
)


async def main():
    """Run the client example."""

    # Load client configuration
    fastapi_config = FastServiceConfig.load_from_file("config/client-fastapi.yml")
    grpc_config = FastServiceConfig.load_from_file("config/client-grpc.yml")

    # Setup client mode
    fastapi_service_manager.setup_client_mode(fastapi_config)
    grpc_service_manager.setup_client_mode(grpc_config)

    # Create test data
    items = [
        Item(name="item1", price=10.0),
        Item(name="item2", price=20.0),
        Item(name="item3", price=30.0),
    ]

    print("\n=== Testing FastAPI Functions ===")

    # Test compute_total via FastAPI
    start_time = time.time()
    result = compute_total(items, name="fastapi-order")
    end_time = time.time()
    print(f"FastAPI compute_total result: {result}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # Test update_items via FastAPI
    start_time = time.time()
    updated_items = update_items(items, price_increase=2.0)
    end_time = time.time()
    print(f"FastAPI update_items result: {updated_items}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    print("\n=== Testing gRPC Functions ===")

    # Test get_items via gRPC
    start_time = time.time()
    items_from_grpc = get_items()
    end_time = time.time()
    print(f"gRPC get_items result: {items_from_grpc}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # Test async_compute_total via gRPC
    start_time = time.time()
    async_result = await async_compute_total(items, name="grpc-order-async")
    end_time = time.time()
    print(f"gRPC async_compute_total result: {async_result}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    print("\n=== Testing Multiple Calls ===")

    # Test multiple calls to measure connection reuse performance
    print("Making 5 consecutive calls to FastAPI compute_total:")
    for i in range(5):
        start_time = time.time()
        result = compute_total(items, name=f"fastapi-order-{i}")
        end_time = time.time()
        print(f"  Call {i+1}: {end_time - start_time:.4f} seconds")

    print("\nMaking 5 consecutive calls to gRPC get_items:")
    for i in range(5):
        start_time = time.time()
        items_from_grpc = get_items()
        end_time = time.time()
        print(f"  Call {i+1}: {end_time - start_time:.4f} seconds")

    print("\n=== Testing Concurrent Calls ===")

    # Test concurrent calls to FastAPI
    print("Making 5 concurrent calls to FastAPI compute_total:")

    async def run_fastapi_concurrent():
        start_time = time.time()
        tasks = []
        for i in range(5):
            tasks.append(
                asyncio.create_task(
                    asyncio.to_thread(
                        compute_total, items, name=f"fastapi-concurrent-{i}"
                    )
                )
            )
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        print(
            f"  Total time for 5 concurrent calls: {end_time - start_time:.4f} seconds"
        )
        return results

    fastapi_results = await run_fastapi_concurrent()
    print(f"  Received {len(fastapi_results)} results")

    # Test concurrent calls to gRPC
    print("\nMaking 5 concurrent calls to gRPC async_compute_total:")

    async def run_grpc_concurrent():
        start_time = time.time()
        tasks = []
        for i in range(5):
            tasks.append(
                asyncio.create_task(
                    async_compute_total(items, name=f"grpc-concurrent-{i}")
                )
            )
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        print(
            f"  Total time for 5 concurrent calls: {end_time - start_time:.4f} seconds"
        )
        return results

    grpc_results = await run_grpc_concurrent()
    print(f"  Received {len(grpc_results)} results")

    print("\n=== Testing Cost Benchmark ===")
    ntimes = 100
    print("Running cost benchmark for FastAPI:")
    start_time = time.time()
    for i in range(ntimes):
        await fastapi_async_cost_benchmark()
    end_time = time.time()
    print(f"FastAPI  Time taken: {end_time - start_time:.4f} seconds")

    print("Running cost benchmark for gRPC:")
    start_time = time.time()
    for i in range(ntimes):
        await grpc_async_cost_benchmark()
    end_time = time.time()
    print(f"gRPC  Time taken: {end_time - start_time:.4f} seconds")

    print("\nAll tests completed successfully!")

    # Explicitly stop the monitoring system to ensure data is flushed to disk
    print("\nStopping monitoring systems...")
    fastapi_service_manager.stop_client()
    grpc_service_manager.stop_client()
    print("Monitoring systems stopped.")


if __name__ == "__main__":
    asyncio.run(main())
