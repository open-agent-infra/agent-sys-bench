# Fast-Service aiohttp Test

This example demonstrates the performance benefits of using aiohttp for asynchronous HTTP requests in fast-service.

## Components

- `test_app.py`: Contains both synchronous and asynchronous functions for testing
- `test_service.py`: Sets up the service with both sync and async functions
- `test_client.py`: Tests different client implementations and measures performance

## Running the Example

1. Start the service:
   ```
   python test_service.py
   ```

2. In another terminal, run the client:
   ```
   python test_client.py
   ```

## What This Example Demonstrates

1. **Performance Comparison**: Compares the performance of synchronous requests vs. aiohttp-based async requests
2. **Concurrency Benefits**: Shows how aiohttp enables efficient concurrent requests
3. **Parallel Processing**: Demonstrates how async functions can process items in parallel

## Expected Results

- Synchronous functions with delays will take the full delay time
- Asynchronous functions with the same delay will also take similar time for a single request
- Parallel processing within an async function will be faster
- Multiple concurrent requests will be efficiently handled with aiohttp