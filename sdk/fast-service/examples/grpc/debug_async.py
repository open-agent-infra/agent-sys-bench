import asyncio
import grpc
import json
import inspect
import time
from typing import List, get_type_hints
from pydantic import BaseModel
from fast_service.engine.grpc_impl import fast_service_pb2, fast_service_pb2_grpc


class Item(BaseModel):
    name: str
    price: float


class Collection(BaseModel):
    name: str
    total_price: float
    item_list: list[Item]


def reconstruct_pydantic_models(data, return_type):
    """Reconstruct Pydantic models from JSON data based on return type annotation."""
    if data is None:
        return None

    # If return type is a Pydantic model
    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        return return_type.model_validate(data)

    # If return type is a List, Tuple, etc.
    origin = getattr(return_type, "__origin__", None)
    if origin is not None:
        args = getattr(return_type, "__args__", None)
        if (
            origin is list
            and args
            and isinstance(args[0], type)
            and issubclass(args[0], BaseModel)
        ):
            return [args[0].model_validate(item) for item in data]

    # Default: return data as is
    return data


class SimpleServer(fast_service_pb2_grpc.FastServiceServicer):
    def __init__(self):
        self.async_function_map = {}

        # Register the async function
        @self.register_async
        async def async_compute_total(items: List[Item]) -> Collection:
            print(f"async_compute_total start: {time.time()}")
            await asyncio.sleep(0.1)

            name = f"{len(items)}-items-async"
            total_price = 0
            for item in items:
                total_price += item.price
            item_list = items

            result = Collection(name=name, total_price=total_price, item_list=item_list)
            print(f"async_compute_total end: {time.time()}")
            return result

    def register_async(self, f):
        self.async_function_map[f.__name__] = f
        return f

    async def AsyncCall(self, request, context):
        """Handle asynchronous function calls."""
        function_name = request.function_name
        if function_name not in self.async_function_map:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Async function {function_name} not found")
            return fast_service_pb2.CallResponse()

        func = self.async_function_map[function_name]

        try:
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

            # Execute the function
            result = await func(*args, **kwargs)

            # Serialize the result
            if isinstance(result, BaseModel):
                # Convert Pydantic model to dict
                result_dict = result.model_dump()
                serialized_result = json.dumps(result_dict).encode("utf-8")
            elif (
                isinstance(result, list) and result and isinstance(result[0], BaseModel)
            ):
                # Convert list of Pydantic models to list of dicts
                result_dict = [item.model_dump() for item in result]
                serialized_result = json.dumps(result_dict).encode("utf-8")
            else:
                # Use standard JSON serialization
                serialized_result = json.dumps(result).encode("utf-8")

            # Return the response
            return fast_service_pb2.CallResponse(result=serialized_result)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            import traceback

            print(f"Error in AsyncCall: {traceback.format_exc()}")
            return fast_service_pb2.CallResponse(error=str(e))


async def serve():
    server = grpc.aio.server()
    fast_service_pb2_grpc.add_FastServiceServicer_to_server(SimpleServer(), server)
    server.add_insecure_port("[::]:12000")
    await server.start()
    print("Server started on port 12000")
    await server.wait_for_termination()


async def client():
    # Wait for server to start
    await asyncio.sleep(1)

    async with grpc.aio.insecure_channel("localhost:12000") as channel:
        stub = fast_service_pb2_grpc.FastServiceStub(channel)

        # Create test data
        items = [
            Item(name="item1", price=10.0),
            Item(name="item2", price=20.0),
            Item(name="item3", price=30.0),
        ]

        # Create a request
        request = fast_service_pb2.CallRequest(
            function_name="async_compute_total",
            args=json.dumps([]).encode("utf-8"),
            kwargs=json.dumps({"items": [item.model_dump() for item in items]}).encode(
                "utf-8"
            ),
        )

        try:
            response = await stub.AsyncCall(request)
            print(f"Response received: {response}")
            if response.error:
                print(f"Error: {response.error}")
            else:
                result = json.loads(response.result)
                print(f"Parsed result: {result}")
        except Exception as e:
            print(f"Error: {e}")


async def main():
    # Start server in background
    server_task = asyncio.create_task(serve())

    # Run client
    await client()

    # Cancel server task
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
