import asyncio
import time
from typing import List, Dict, Any
from pydantic import BaseModel
from fast_service import FastServiceManager

fast_service_manager = FastServiceManager()


class Item(BaseModel):
    name: str
    price: float


class ItemList(BaseModel):
    name: str
    total_price: float
    item_list: List[Item]


@fast_service_manager.async_fast_service
async def compute_total_async(items: List[Item]) -> ItemList:
    """
    Async function that computes the total price of items with a simulated delay
    """
    print(f"Processing {len(items)} items asynchronously...")

    # Simulate processing delay
    await asyncio.sleep(1)

    total = sum(item.price for item in items)
    return ItemList(
        name=f"{len(items)}-items-async", total_price=total, item_list=items
    )


@fast_service_manager.async_fast_service
async def process_items_parallel(items: List[Item]) -> ItemList:
    """
    Async function that processes items in parallel
    """
    print(f"Processing {len(items)} items in parallel...")

    # Simulate parallel processing of each item
    async def process_item(item: Item):
        await asyncio.sleep(0.2)  # Each item takes 0.2s to process
        return item

    # Process all items concurrently
    processed_items = await asyncio.gather(*(process_item(item) for item in items))

    total = sum(item.price for item in processed_items)
    return ItemList(
        name=f"{len(items)}-items-parallel",
        total_price=total,
        item_list=processed_items,
    )


@fast_service_manager.fast_service
def compute_total(items: List[Item]) -> ItemList:
    """
    Synchronous function that computes the total price of items with a simulated delay
    """
    print(f"Processing {len(items)} items synchronously...")

    # Simulate processing delay
    time.sleep(1)

    total = sum(item.price for item in items)
    return ItemList(name=f"{len(items)}-items", total_price=total, item_list=items)
