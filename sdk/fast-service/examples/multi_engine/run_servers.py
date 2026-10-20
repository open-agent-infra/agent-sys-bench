"""
Run both FastAPI and gRPC servers.

This script starts both the FastAPI and gRPC servers in separate threads.
"""

import os
import sys
import threading
import time
from pathlib import Path

from fast_service import FastServiceManager
from fast_service import FastServiceConfig


def run_fastapi_server():
    """Run the FastAPI server."""
    print("Starting FastAPI server...")
    from multi_engine_service import fastapi_service_manager

    config = FastServiceConfig.load_from_file("config/fastapi-service.yml")
    fastapi_service_manager.execute(config)


def run_grpc_server():
    """Run the gRPC server."""
    print("Starting gRPC server...")
    from multi_engine_service import grpc_service_manager

    config = FastServiceConfig.load_from_file("config/grpc-service.yml")
    grpc_service_manager.execute(config)


if __name__ == "__main__":
    # Change to the directory containing this script
    os.chdir(Path(__file__).parent)

    # Start the FastAPI server in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi_server)
    fastapi_thread.daemon = True
    fastapi_thread.start()

    # Start the gRPC server in a separate thread
    grpc_thread = threading.Thread(target=run_grpc_server)
    grpc_thread.daemon = True
    grpc_thread.start()

    print("Both servers are running. Press Ctrl+C to stop.")

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down servers...")
        sys.exit(0)
