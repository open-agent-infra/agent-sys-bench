"""
Multi-engine service example.

This module demonstrates how to create a service that can be accessed
through multiple engines (FastAPI and gRPC).
"""

import time
import asyncio
from typing import List, Annotated
from pydantic import BaseModel
from fastapi import Body
from fast_service import FastServiceManager, FastServiceConfig, RequestContext

# Create separate service managers for each engine
fastapi_service_manager = FastServiceManager()
grpc_service_manager = FastServiceManager()


class Item(BaseModel):
    """Item model with name and price."""

    name: str
    price: float


class Order(BaseModel):
    """Order model with name, total price, and list of items."""

    name: str
    total_price: float
    item_list: List[Item]


# Sample data
ITEMS = [
    Item(name="item1", price=10.0),
    Item(name="item2", price=20.0),
    Item(name="item3", price=30.0),
]


# Functions exposed via FastAPI
@fastapi_service_manager.fast_service
def compute_total(
    items: List[Item],
    name: Annotated[str, Body()] = "order",
    context: RequestContext = None,
) -> Order:
    """Compute the total price of a list of items.

    This function is exposed via FastAPI.
    """
    print(f"compute_total start: {time.time()}")
    total = sum(item.price for item in items)
    print(f"compute_total end: {time.time()}")
    return Order(name=name, total_price=total, item_list=items)


@fastapi_service_manager.fast_service
def update_items(
    items: List[Item],
    price_increase: Annotated[float, Body()] = 1.0,
    context: RequestContext = None,
) -> List[Item]:
    """Update the price of all items.

    This function is exposed via FastAPI.
    """
    print(f"update_items start: {time.time()}")
    updated_items = [
        Item(name=item.name, price=item.price + price_increase) for item in items
    ]
    print(f"update_items end: {time.time()}")
    return updated_items


# Functions exposed via gRPC
@grpc_service_manager.fast_service
async def async_compute_total(
    items: List[Item],
    name: Annotated[str, Body()] = "order-async",
    context: RequestContext = None,
) -> Order:
    """Compute the total price of a list of items asynchronously.

    This function is exposed via gRPC.
    """
    print(f"async_compute_total start: {time.time()}")
    # Simulate async processing
    await asyncio.sleep(0.1)
    total = sum(item.price for item in items)
    print(f"async_compute_total end: {time.time()}")
    return Order(name=name, total_price=total, item_list=items)


@grpc_service_manager.fast_service
def get_items(context: RequestContext = None) -> List[Item]:
    """Get the list of available items.

    This function is exposed via gRPC.
    """
    print(f"get_items start: {time.time()}")
    print(f"get_items end: {time.time()}")
    return ITEMS


@fastapi_service_manager.fast_service
async def fastapi_async_cost_benchmark(context: RequestContext = None) -> None:
    """Async function for cost benchmarking.

    This function is exposed via FastAPI.
    """
    pass


@grpc_service_manager.fast_service
async def grpc_async_cost_benchmark(context: RequestContext = None) -> None:
    """Async function for cost benchmarking.

    This function is exposed via gRPC.
    """
    pass
