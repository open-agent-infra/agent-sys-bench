import json
import time

from typing import Optional
from pydantic import BaseModel
from fast_service import FastServiceManager, RequestContext, FastServiceFile

fast_service_manager = FastServiceManager()


class Item(BaseModel):
    name: str
    price: float


class Collection(BaseModel):
    history_id: str
    name: Optional[str] = None
    total_price: Optional[float] = None
    item_list: Optional[list[Item]] = None


class ComputeTask(BaseModel):
    history_id: Optional[str] = None
    task_name: str
    items: list[Item]


def execute_compute(history_id: str, items: list[Item]) -> Collection:
    name = f"{len(items)}-items"
    total_price = 0
    for item in items:
        total_price += item.price
    item_list = items

    return Collection(
        history_id=history_id, name=name, total_price=total_price, item_list=item_list
    )


HISTORY_DICT: dict[str, int] = {}


@fast_service_manager.fast_service
def create_context(task: ComputeTask, context: RequestContext) -> Collection:
    history_id = f"{task.task_name}-{time.time()}"
    HISTORY_DICT[history_id] = 0
    print(f"CREATE STATE CONTEXT: {history_id}")
    return Collection(history_id=history_id)


@fast_service_manager.fast_service
def monitored_compute(
    task: ComputeTask,
    context: RequestContext = None,
) -> Collection:
    history_id = task.history_id
    if history_id not in HISTORY_DICT or HISTORY_DICT[history_id] != 0:
        print(f"Inconsistent state of {history_id}")
    else:
        HISTORY_DICT[history_id] += 1
    result = execute_compute(task.history_id, task.items)
    print(f"MONITORED-RESULT: {result}")
    return result


@fast_service_manager.fast_service
def single_file_compute(
    task: ComputeTask,
    file: FastServiceFile,
    context: RequestContext = None,
) -> Collection:
    history_id = task.history_id
    if history_id not in HISTORY_DICT or HISTORY_DICT[history_id] != 1:
        print(f"Inconsistent state of {history_id}")
    else:
        HISTORY_DICT[history_id] += 1

    result = execute_compute(task.history_id, task.items)

    items_in_file = [Item.parse_obj(_) for _ in json.load(file.get_file()).get("items")]
    file_result = execute_compute(task.history_id, items_in_file)
    print(f"SINGLE-FILE-COMPUTE: {file.get_filename()} {file_result.model_dump()}")
    return result


@fast_service_manager.fast_service
def multi_file_compute(
    task: ComputeTask,
    files: list[FastServiceFile],
    context: RequestContext = None,
) -> Collection:
    history_id = task.history_id
    if history_id not in HISTORY_DICT or HISTORY_DICT[history_id] != 2:
        print(f"Inconsistent state of {history_id}")
    else:
        HISTORY_DICT[history_id] += 1

    result = execute_compute(task.history_id, task.items)

    for file in files:
        items_in_file = [
            Item.parse_obj(_) for _ in json.load(file.get_file()).get("items")
        ]
        file_result = execute_compute(task.history_id, items_in_file)
        print(f"MULTI-FILE-COMPUTE: {file.get_filename()} {file_result.model_dump()}")

    return result


@fast_service_manager.fast_service(monitor_only=True)
def redirect_compute(
    task: ComputeTask,
    files: list[FastServiceFile],
    context: RequestContext = None,
) -> Collection:
    history_id = create_context(task, context=context).history_id
    task.history_id = history_id
    time.sleep(1)
    monitored_compute(task, context=context)
    time.sleep(2)
    single_file_compute(task, files[0], context=context)
    time.sleep(3)
    return multi_file_compute(task, [files[1]], context=context)
