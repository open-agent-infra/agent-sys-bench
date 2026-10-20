# Multi-Engine Example

This example demonstrates how to use multiple engines (FastAPI and gRPC) in a single application.

## Overview

The example consists of:

1. **Service Implementation**: A Python module with functions exposed through both FastAPI and gRPC
2. **Server Configuration**: Separate configuration files for FastAPI and gRPC servers
3. **Client Configuration**: A single configuration file that references both servers
4. **Server Script**: A script that starts both servers in separate threads
5. **Client Script**: A script that demonstrates calling functions from both engines

## Running the Example

1. Start the servers:

```bash
python run_servers.py
```

This will start:
- FastAPI server on port 12000
- gRPC server on port 12001

2. In a separate terminal, run the client:

```bash
python run_client.py
```

## Architecture

This example demonstrates a microservices architecture where:

- Synchronous functions are exposed via FastAPI for HTTP-based access
- Asynchronous and high-performance functions are exposed via gRPC
- A single client can seamlessly access both types of services

## Performance Comparison

The client script includes a performance test that makes multiple consecutive calls to both engines. This demonstrates the performance benefits of connection reuse in the gRPC implementation.

## Use Cases

This multi-engine approach is useful for:

1. **Gradual Migration**: Migrate from REST to gRPC incrementally
2. **Mixed Workloads**: Use REST for simple operations and gRPC for high-performance needs
3. **Public/Private APIs**: Expose REST APIs publicly while using gRPC for internal services