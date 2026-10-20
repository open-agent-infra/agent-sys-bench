from pydantic import BaseModel
from fast_service import FastServiceManager
import asyncio
import time

fast_service_manager = FastServiceManager()


class Item(BaseModel):
    name: str
    price: float


class Collection(BaseModel):
    name: str
    total_price: float
    item_list: list[Item]


@fast_service_manager.fast_service
def compute_total(items: list[Item]) -> Collection:
    """Synchronous function to compute total price of items"""
    print(f"compute_total start: {time.time()}")
    name = f"{len(items)}-items"
    total_price = 0
    for item in items:
        total_price += item.price
    item_list = items
    # Simulate some processing time
    time.sleep(1)

    result = Collection(name=name, total_price=total_price, item_list=item_list)
    print(f"compute_total end: {time.time()}")
    return result


@fast_service_manager.async_fast_service
async def compute_total_async(items: list[Item]) -> Collection:
    """Asynchronous function to compute total price of items"""
    print(f"compute_total_async start: {time.time()}")
    name = f"{len(items)}-items-async"
    total_price = 0
    for item in items:
        total_price += item.price
    item_list = items
    # Simulate some processing time without blocking
    await asyncio.sleep(1)

    result = Collection(name=name, total_price=total_price, item_list=item_list)
    print(f"compute_total_async end: {time.time()}")
    return result


@fast_service_manager.async_fast_service
async def process_items_parallel(items: list[Item]) -> Collection:
    """Process items in parallel using asyncio"""
    print(f"process_items_parallel start: {time.time()}")

    # Simulate parallel processing of items
    async def process_item(item: Item):
        print(f"process_item start: {time.time()}")
        await asyncio.sleep(0.2)  # Simulate processing time
        print(f"process_item end: {time.time()}")
        return item.price

    # Process all items in parallel
    prices = await asyncio.gather(*[process_item(item) for item in items])

    total_price = sum(prices)
    name = f"{len(items)}-items-parallel"

    result = Collection(name=name, total_price=total_price, item_list=items)
    print(f"process_items_parallel end: {time.time()}")
    return result
