import time
from pydantic import BaseModel
from fast_service import FastServiceManager

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
    print(f"compute_total start: {time.time()}")
    name = f"{len(items)}-items"
    total_price = 0
    for item in items:
        total_price += item.price
    item_list = items

    result = Collection(name=name, total_price=total_price, item_list=item_list)
    print(f"compute_total end: {time.time()}")
    return result


@fast_service_manager.fast_service
def redirect_compute(items: list[Item]) -> Collection:
    print(f"redirect_compute start: {time.time()}")
    result = compute_total(items=items)
    print(f"redirect_compute end: {time.time()}")
    return result


@fast_service_manager.fast_service
def update_items(items: list[Item]) -> list[Item]:
    print(f"update_items start: {time.time()}")
    items = [Item(name=item.name, price=item.price + 1) for item in items]
    print(f"update_items end: {time.time()}")
    return items


# Async example
@fast_service_manager.fast_service
async def async_compute_total(items: list[Item]) -> Collection:
    # Simulate async processing
    print(f"async_compute_total start: {time.time()}")
    import asyncio

    await asyncio.sleep(0.1)

    name = f"{len(items)}-items-async"
    total_price = 0
    for item in items:
        total_price += item.price
    item_list = items

    result = Collection(name=name, total_price=total_price, item_list=item_list)
    print(f"async_compute_total end: {time.time()}")
    return result
