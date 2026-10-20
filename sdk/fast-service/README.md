# Fast-Service

## Description

This is a simple framework to help python developer quickly deploy their codes as network service in demand.

Developer can decide which part of their code to be deployed as a service with a few annotations and no changes to their business code.

Fast-Service supports both synchronous and asynchronous functions, allowing you to register async functions in the server and call them asynchronously from the client. The async client implementation uses aiohttp for efficient non-blocking HTTP requests.

## Environment Preparation

**1. Make sure there's python3.**

**2. Install fast-service to your local environment**

```pip install -e .```

**3. If you need to use monitor, see [monitor doc](docs/Monitor.md) to prepare the components.**

## Examples

There are several examples in the `example` directory. You can run them to see how fast-service works:

1. [FastAPI Example](example/fastapi/README.md)
2. [Async Example](example/async_example/README.md) - Demonstrates how to use async functions with fast-service

## Deploy

To run a service on a host, directly run your service script.

To run a service in a docker container, follow this guide [docker-deploy](docs/docker.md).

## Maintain

To support load balance, follow this guide [load-balancer-guide](docs/Load-Balancer.md).

## How to Contribute

see [contribution-guide](./docs/contribution-guide.md)