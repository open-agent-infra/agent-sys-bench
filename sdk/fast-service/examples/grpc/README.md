# gRPC Example

This example demonstrates how to use the gRPC engine with fast-service.

## Running the Example

1. Start the service:
   ```
   python example_computing_service.py
   ```

2. In another terminal, run the client:
   ```
   python example_client.py
   ```

## Files

- `example_app.py`: Defines the functions to be exposed as services
- `example_computing_service.py`: Starts the gRPC service
- `example_client.py`: Client that calls the gRPC service
- `config/`: Configuration files for the service and client