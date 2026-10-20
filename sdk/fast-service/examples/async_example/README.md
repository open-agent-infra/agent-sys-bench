# Async Example for Fast-Service

This example demonstrates how to use async functions with fast-service.

## Features

- Register and call async functions in the server
- Call async functions asynchronously from the client
- Process items in parallel using asyncio

## Running the Example

1. Start the server:
```bash
cd /path/to/fast-service/example/async_example
python async_service.py
```

2. In another terminal, run the client:
```bash
cd /path/to/fast-service/example/async_example
python async_client.py
```

## Example Code

The example includes:

1. `async_app.py`: Defines both synchronous and asynchronous functions
2. `async_service.py`: Starts the service
3. `async_client.py`: Client code to test both HTTP and fast-service async clients

## Async Functions

The example includes three functions:

1. `compute_total`: A synchronous function that computes the total price of items
2. `compute_total_async`: An asynchronous version of the same function
3. `process_items_parallel`: An asynchronous function that processes items in parallel

## Configuration

The example includes two configuration files:

1. `config/async-server.yml`: Server configuration
2. `config/async-client.yml`: Client configuration