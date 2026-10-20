import json

from pydantic import BaseModel
from fast_service import FastServiceManager, RequestContext, FastServiceFile

fast_service_manager = FastServiceManager()


class Item(BaseModel):
    name: str
    price: float


class Collection(BaseModel):
    name: str
    total_price: float
    item_list: list[Item]


def execute_compute(items: list[Item]) -> Collection:
    name = f"{len(items)}-items"
    total_price = 0
    for item in items:
        total_price += item.price
    item_list = items

    return Collection(name=name, total_price=total_price, item_list=item_list)


@fast_service_manager.fast_service
def compute_total(
    task_name: str,
    items: list[Item],
    files: list[FastServiceFile],
    context: RequestContext = None,
) -> Collection:
    result = execute_compute(items)

    for file in files:
        items_in_file = [
            Item.parse_obj(_) for _ in json.load(file.get_file()).get("items")
        ]
        file_result = execute_compute(items_in_file)
        print(file.get_filename(), ":", file_result.model_dump())

    return result


@fast_service_manager.fast_service
def redirect_compute(
    task_name: str,
    items: list[Item],
    files: list[FastServiceFile],
    context: RequestContext = None,
) -> Collection:
    return compute_total(task_name, items, files, context)
