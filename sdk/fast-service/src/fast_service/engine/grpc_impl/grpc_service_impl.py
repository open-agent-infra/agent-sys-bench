import asyncio
import concurrent.futures
import functools
import inspect
import json
import os
import signal
import sys
import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_type_hints,
    get_origin,
    get_args,
)

import grpc
import grpc.aio  # Import the async gRPC API
from google.protobuf.json_format import MessageToDict, ParseDict
from pydantic import BaseModel

from fast_service.engine.engine_interface import IFastServiceEngine
from fast_service.engine.grpc_impl import fast_service_pb2, fast_service_pb2_grpc


class PydanticJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Pydantic models."""

    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)


def reconstruct_pydantic_models(data, return_type):
    """Reconstruct Pydantic models from JSON data based on return type annotation."""
    if data is None:
        return None

    # If return type is a Pydantic model
    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        return return_type.model_validate(data)

    # If return type is a List, Tuple, etc.
    origin = get_origin(return_type)
    if origin is not None:
        args = get_args(return_type)
        if (
            origin is list
            and args
            and isinstance(args[0], type)
            and issubclass(args[0], BaseModel)
        ):
            return [args[0].model_validate(item) for item in data]
        elif origin is dict:
            # Handle dict with Pydantic values if needed
            pass

    # Default: return data as is
    return data


def serialize_result(result):
    """Serialize a result to JSON bytes."""
    if isinstance(result, BaseModel):
        # Convert Pydantic model to dict
        result_dict = result.model_dump()
        return json.dumps(result_dict).encode("utf-8")
    elif isinstance(result, list) and result and isinstance(result[0], BaseModel):
        # Convert list of Pydantic models to list of dicts
        result_dict = [item.model_dump() for item in result]
        return json.dumps(result_dict).encode("utf-8")
    else:
        # Use standard JSON serialization
        return json.dumps(result).encode("utf-8")


def parse_request(request, func):
    """Parse a gRPC request and convert arguments to appropriate types."""
    # Get function signature
    sig = inspect.signature(func)
    param_types = get_type_hints(func)

    # Parse JSON arguments
    raw_args = json.loads(request.args) if request.args else []
    raw_kwargs = json.loads(request.kwargs) if request.kwargs else {}

    # Convert arguments to Pydantic models if needed
    args = []
    for i, arg in enumerate(raw_args):
        param_name = list(sig.parameters.keys())[i]
        if param_name in param_types:
            arg_type = param_types[param_name]
            args.append(reconstruct_pydantic_models(arg, arg_type))
        else:
            args.append(arg)

    # Convert keyword arguments to Pydantic models if needed
    kwargs = {}
    for key, value in raw_kwargs.items():
        if key in param_types:
            arg_type = param_types[key]
            kwargs[key] = reconstruct_pydantic_models(value, arg_type)
        else:
            kwargs[key] = value

    return args, kwargs


class FastServiceGRPCServicer(fast_service_pb2_grpc.FastServiceServicer):
    """gRPC servicer implementation for FastService."""

    def __init__(
        self, function_map: Dict[str, Callable], is_async_server: bool = False
    ):
        """Initialize the servicer with a function map and server type."""
        self.function_map = function_map
        self.is_async_server = is_async_server
        self.async_function_map = {
            name: func
            for name, func in function_map.items()
            if inspect.iscoroutinefunction(func)
        }

        # Create a thread pool for running sync functions from async context
        self.max_nthreads = int(os.environ.get("FS_GRPC_NUM_THREADS", 32))
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_nthreads
        )

    # ===== Synchronous server methods =====

    def Call(self, request, context):
        """Handle synchronous function calls."""
        if self.is_async_server:
            # This should never be called in async server mode
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Call() called on async server - this is a bug")
            return fast_service_pb2.CallResponse(error="Internal server error")

        function_name = request.function_name

        # Check if the function exists
        if function_name in self.function_map:
            func = self.function_map[function_name]
            is_async = inspect.iscoroutinefunction(func)
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Function {function_name} not found")
            return fast_service_pb2.CallResponse()

        try:
            # Parse the request
            args, kwargs = parse_request(request, func)

            # Execute the function
            if is_async:
                # For async functions, run in a new event loop
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(func(*args, **kwargs))
                loop.close()
            else:
                # For sync functions, just call directly
                result = func(*args, **kwargs)

            # Serialize the result
            serialized_result = serialize_result(result)

            # Return the response
            return fast_service_pb2.CallResponse(result=serialized_result)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            import traceback

            print(f"Error in Call: {traceback.format_exc()}")
            return fast_service_pb2.CallResponse(error=str(e))

    def CallWithFiles(self, request_iterator, context):
        """Handle function calls with file transfers."""
        # Implementation for file transfers would go here
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("File transfer not implemented yet")
        yield fast_service_pb2.FileChunk()

    def AsyncCallWithFiles(self, request_iterator, context):
        """Handle async function calls with file transfers."""
        # Implementation for async file transfers would go here
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Async file transfer not implemented yet")
        yield fast_service_pb2.FileChunk()

    # ===== Asynchronous server methods =====

    async def Call_async(self, request, context):
        """Handle function calls in asynchronous server context."""
        function_name = request.function_name

        # Check if the function exists
        if function_name in self.function_map:
            func = self.function_map[function_name]
            is_async = inspect.iscoroutinefunction(func)
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Function {function_name} not found")
            return fast_service_pb2.CallResponse()

        try:
            # Parse the request
            args, kwargs = parse_request(request, func)

            # Execute the function
            if is_async:
                # For async functions, await directly
                result = await func(*args, **kwargs)
            else:
                # For sync functions, run in a thread pool to avoid blocking
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self.thread_pool, lambda: func(*args, **kwargs)
                )

            # Serialize the result
            serialized_result = serialize_result(result)

            # Return the response
            return fast_service_pb2.CallResponse(result=serialized_result)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            import traceback

            print(f"Error in async Call: {traceback.format_exc()}")
            return fast_service_pb2.CallResponse(error=str(e))

    async def AsyncCall_async(self, request, context):
        """Handle async function calls in asynchronous server context."""
        function_name = request.function_name

        # Check if the function exists and is async
        if function_name not in self.async_function_map:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Async function {function_name} not found")
            return fast_service_pb2.CallResponse()

        func = self.async_function_map[function_name]

        try:
            # Parse the request
            args, kwargs = parse_request(request, func)

            # Execute the async function directly with await
            result = await func(*args, **kwargs)

            # Serialize the result
            serialized_result = serialize_result(result)

            # Return the response
            return fast_service_pb2.CallResponse(result=serialized_result)
        except Exception as e:
            import traceback

            print(f"Error in async AsyncCall: {traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return fast_service_pb2.CallResponse(error=str(e))

    # For compatibility with grpc.aio.server
    async def Call(self, request, context):
        """Async version of Call for use with grpc.aio.server."""
        if not self.is_async_server:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Async Call() called on sync server - this is a bug")
            return fast_service_pb2.CallResponse(error="Internal server error")
        return await self.Call_async(request, context)

    async def AsyncCall(self, request, context):
        """Async version of AsyncCall for use with grpc.aio.server."""
        if not self.is_async_server:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                "Async AsyncCall() called on sync server - this is a bug"
            )
            return fast_service_pb2.CallResponse(error="Internal server error")
        return await self.AsyncCall_async(request, context)


class FastServiceOnGRPC(IFastServiceEngine):
    """gRPC implementation of the FastServiceEngine interface."""

    def __init__(self):
        self.function_map = {}
        self.server = None
        self.server_thread = None
        self.stop_event = threading.Event()
        self.use_async_server = False
        self.async_server = None
        self.async_server_task = None

        # Channel cache for performance optimization
        self._channel_cache = {}
        self._stub_cache = {}
        self._async_channel_cache = {}
        self._async_stub_cache = {}
        self._channel_lock = threading.RLock()  # Use RLock for thread safety

        self.max_nthreads = int(os.environ.get("FS_GRPC_NUM_THREADS", 32))

    def setup_service(self, f):
        """Set up a synchronous service function."""
        self.function_map[f.__name__] = f
        return f

    def setup_client(self, f, ip: str, port: int):
        """Set up a synchronous client function."""
        return_type = get_type_hints(f).get("return")
        endpoint = f"{ip}:{port}"

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get cached channel and stub
            _, stub = self._get_channel_and_stub(endpoint)

            # Serialize arguments
            serialized_args = json.dumps(args, cls=PydanticJSONEncoder).encode("utf-8")
            serialized_kwargs = json.dumps(kwargs, cls=PydanticJSONEncoder).encode(
                "utf-8"
            )

            request = fast_service_pb2.CallRequest(
                function_name=f.__name__,
                args=serialized_args,
                kwargs=serialized_kwargs,
            )

            try:
                response = stub.Call(request)
                if response.error:
                    raise Exception(response.error)
                data = json.loads(response.result)
                return reconstruct_pydantic_models(data, return_type)
            except grpc.RpcError as e:
                # If we get a connection error, try to recreate the channel
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    with self._channel_lock:
                        if endpoint in self._channel_cache:
                            try:
                                self._channel_cache[endpoint].close()
                            except Exception:
                                pass
                            del self._channel_cache[endpoint]
                            del self._stub_cache[endpoint]
                raise Exception(f"gRPC error: {e.code()}: {e.details()}")

        return wrapper

    def setup_service_with_files(self, f, module_name: str, function_name: str):
        """Set up a service function with file handling."""
        self.function_map[function_name] = f
        return f

    def setup_client_with_files(
        self, f, ip: str, port: int, module_name: str, function_name: str
    ):
        """Set up a client function with file handling."""

        # This is a placeholder for future implementation
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            raise NotImplementedError("File transfer not implemented yet")

        return wrapper

    def setup_async_service(self, f):
        """Set up an asynchronous service function."""
        self.function_map[f.__name__] = f
        # If we have any async functions, we should use the async server
        self.use_async_server = True
        return f

    def setup_async_client(self, f, ip: str, port: int):
        """Set up an asynchronous client function."""
        return_type = get_type_hints(f).get("return")
        endpoint = f"{ip}:{port}"

        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            # Serialize arguments
            serialized_args = json.dumps(args, cls=PydanticJSONEncoder).encode("utf-8")
            serialized_kwargs = json.dumps(kwargs, cls=PydanticJSONEncoder).encode(
                "utf-8"
            )

            # Get or create async channel and stub
            async_channel, async_stub = await self._get_async_channel_and_stub(endpoint)

            # Create request
            request = fast_service_pb2.CallRequest(
                function_name=f.__name__, args=serialized_args, kwargs=serialized_kwargs
            )

            try:
                # Use AsyncCall with true async gRPC
                response = await async_stub.AsyncCall(request)

                if response.error:
                    raise Exception(response.error)
                data = json.loads(response.result)
                return reconstruct_pydantic_models(data, return_type)
            except grpc.RpcError as e:
                # If we get a connection error, try to recreate the channel
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    with self._channel_lock:
                        if endpoint in self._async_channel_cache:
                            try:
                                await self._async_channel_cache[endpoint].close()
                            except Exception:
                                pass
                            del self._async_channel_cache[endpoint]
                            del self._async_stub_cache[endpoint]
                raise Exception(f"gRPC error: {e.code()}: {e.details()}")

        return wrapper

    def setup_async_service_with_files(self, f, module_name: str, function_name: str):
        """Set up an async service function with file handling."""
        self.function_map[function_name] = f
        self.use_async_server = True
        return f

    def setup_async_client_with_files(
        self, f, ip: str, port: int, module_name: str, function_name: str
    ):
        """Set up an async client function with file handling."""

        # This is a placeholder for future implementation
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            raise NotImplementedError("Async file transfer not implemented yet")

        return wrapper

    async def _run_async_server(self, ip: str, port: int):
        """Run the async gRPC server."""
        self.async_server = grpc.aio.server()
        servicer = FastServiceGRPCServicer(self.function_map, is_async_server=True)
        fast_service_pb2_grpc.add_FastServiceServicer_to_server(
            servicer, self.async_server
        )

        self.async_server.add_insecure_port(f"{ip}:{port}")
        await self.async_server.start()

        print(f"Async gRPC server started on {ip}:{port}")

        # Wait for server to be terminated
        await self.async_server.wait_for_termination()

    def execute_service(self, ip: str, port: int):
        """Start the gRPC server."""
        if self.use_async_server:
            # Start async server
            print(f"Starting async gRPC server on {ip}:{port}")

            # Create a new event loop for the async server
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Start the server in the background
            self.async_server_task = loop.create_task(self._run_async_server(ip, port))

            # Run the event loop in a separate thread
            self.server_thread = threading.Thread(
                target=self._run_async_event_loop, args=(loop,)
            )
            self.server_thread.daemon = True
            self.server_thread.start()

            # Wait for server to be terminated
            try:
                while not self.stop_event.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_service()
        else:
            # Start sync server
            self.server = grpc.server(
                concurrent.futures.ThreadPoolExecutor(max_workers=self.max_nthreads)
            )
            servicer = FastServiceGRPCServicer(self.function_map, is_async_server=False)
            fast_service_pb2_grpc.add_FastServiceServicer_to_server(
                servicer, self.server
            )

            self.server.add_insecure_port(f"{ip}:{port}")
            self.server.start()

            print(f"gRPC server started on {ip}:{port}")

            # Start server in a separate thread
            self.stop_event.clear()
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()

            # Wait for server to be terminated
            try:
                while not self.stop_event.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_service()

    def _run_async_event_loop(self, loop):
        """Run the async event loop in a separate thread."""
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            loop.close()

    def _run_server(self):
        """Run the server and handle signals."""
        # Don't set signal handlers in a non-main thread
        try:
            while not self.stop_event.is_set():
                time.sleep(1)
        finally:
            if self.server:
                self.server.stop(0)

    def stop_service(self):
        """Stop the gRPC server."""
        self.stop_event.set()

        if self.server:
            self.server.stop(0)

        if self.async_server:
            # We need to stop the async server in the event loop
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_running():
                loop.create_task(self.async_server.stop(0))
            else:
                # If no event loop is running, create a new one
                async def stop_server():
                    await self.async_server.stop(0)

                loop = asyncio.new_event_loop()
                loop.run_until_complete(stop_server())
                loop.close()

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)

        print("gRPC server stopped")

        # Close all cached channels
        with self._channel_lock:
            # Close sync channels
            for channel in self._channel_cache.values():
                try:
                    channel.close()
                except Exception:
                    pass
            self._channel_cache.clear()
            self._stub_cache.clear()

            # Close async channels
            for channel in self._async_channel_cache.values():
                try:
                    # We need to close async channels in the event loop
                    loop = asyncio.get_event_loop_policy().get_event_loop()
                    if loop.is_running():
                        loop.create_task(channel.close())
                    else:
                        # If no event loop is running, create a new one
                        async def close_channel(ch):
                            await ch.close()

                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(close_channel(channel))
                        loop.close()
                except Exception:
                    pass
            self._async_channel_cache.clear()
            self._async_stub_cache.clear()

    def __del__(self):
        """Clean up resources."""
        self.stop_service()

    def _get_channel_and_stub(self, endpoint):
        """Get or create a synchronous channel and stub for the given endpoint.

        Args:
            endpoint: A string in the format "ip:port"

        Returns:
            A tuple of (channel, stub)
        """
        with self._channel_lock:
            if endpoint not in self._channel_cache:
                # Create new channel and stub
                channel = grpc.insecure_channel(endpoint)
                stub = fast_service_pb2_grpc.FastServiceStub(channel)
                self._channel_cache[endpoint] = channel
                self._stub_cache[endpoint] = stub

            return self._channel_cache[endpoint], self._stub_cache[endpoint]

    async def _get_async_channel_and_stub(self, endpoint):
        """Get or create an asynchronous channel and stub for the given endpoint.

        Args:
            endpoint: A string in the format "ip:port"

        Returns:
            A tuple of (channel, stub)
        """
        with self._channel_lock:
            if endpoint not in self._async_channel_cache:
                # Create new async channel and stub
                channel = grpc.aio.insecure_channel(endpoint)
                stub = fast_service_pb2_grpc.FastServiceStub(channel)
                self._async_channel_cache[endpoint] = channel
                self._async_stub_cache[endpoint] = stub

            return self._async_channel_cache[endpoint], self._async_stub_cache[endpoint]
