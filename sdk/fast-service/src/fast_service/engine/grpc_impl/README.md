# gRPC Engine for Fast-Service

This module provides a gRPC implementation of the Fast-Service engine interface, allowing services to be exposed via gRPC.

## Features

- Synchronous and asynchronous function calls
- JSON serialization for request and response data
- Support for Pydantic models
- Automatic type conversion between Python and gRPC types

## Usage

To use the gRPC engine, set the `service_impl` parameter to `"grpc"` in your configuration:

```yaml
service_impl: grpc
ip: 0.0.0.0
port: 8000
service_list: []
```

## Implementation Details

The gRPC implementation consists of:

1. **Protocol Buffers Definition**: The `fast_service.proto` file defines the service interface.
2. **Service Implementation**: The `FastServiceGRPCServicer` class handles incoming gRPC requests.
3. **Engine Implementation**: The `FastServiceOnGRPC` class implements the `IFastServiceEngine` interface.

### Performance Optimizations

The gRPC engine includes several performance optimizations:

1. **Channel and Stub Caching**: Channels and stubs are cached and reused to avoid the overhead of creating new connections for each request.
2. **Connection Error Handling**: If a connection error occurs, the channel is automatically recreated.
3. **Thread-Safe Implementation**: All channel operations are protected by a lock to ensure thread safety.

### Asynchronous Function Support

The gRPC engine now provides true asynchronous function support using gRPC's asynchronous API:

1. **Unified Servicer**: A single servicer handles both sync and async functions seamlessly
2. **Async Server**: When async functions are registered, the engine automatically uses `grpc.aio.server` for true async processing
3. **Async Client**: Client calls to async functions use `grpc.aio.insecure_channel` for non-blocking communication
4. **Direct Await**: Async functions are executed directly with `await` rather than through a separate event loop
5. **Efficient Resource Management**: The implementation maintains connection pools for both sync and async channels

This implementation provides significantly better performance and resource utilization for async workloads compared to the previous approach that used synchronous gRPC with event loops.

## Example

See the `/example/grpc` directory for a complete example of using the gRPC engine.

## Future Enhancements

- File transfer support
- Streaming responses
- Authentication and authorization
- TLS/SSL support
- Bidirectional streaming for real-time communication