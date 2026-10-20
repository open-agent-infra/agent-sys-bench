from pydantic import BaseModel
from fast_service import FastServiceManager, RequestContext

fast_service_manager = FastServiceManager()


class Item(BaseModel):
    name: str
    price: float


class Collection(BaseModel):
    name: str
    total_price: float
    item_list: list[Item]


@fast_service_manager.fast_service
def compute_total(items: list[Item], context: RequestContext = None) -> Collection:
    name = f"{len(items)}-items"
    total_price = 0
    for item in items:
        total_price += item.price
    item_list = items

    result = Collection(name=name, total_price=total_price, item_list=item_list)

    return result


@fast_service_manager.fast_service
def redirect_compute(items: list[Item], context: RequestContext = None) -> Collection:
    return compute_total(items=items, context=context)
