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
    name = f"{len(items)}-items"
    total_price = 0
    for item in items:
        total_price += item.price
    item_list = items

    result = Collection(name=name, total_price=total_price, item_list=item_list)

    return result


@fast_service_manager.fast_service
def redirect_compute(items: list[Item]) -> Collection:
    return compute_total(items=items)


@fast_service_manager.fast_service
def update_items(items: list[Item]) -> list[Item]:
    items = [Item(name=item.name, price=item.price + 1) for item in items]
    return items
