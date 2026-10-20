from example_app import (
    fast_service_manager,
    redirect_compute,
    Item,
    compute_total,
    update_items,
)

import requests
from fast_service import FastServiceConfig

raw_data = [
    {"name": "apple", "price": "10"},
    {"name": "pear", "price": "20"},
    {"name": "peach", "price": "30"},
]


def example_http_client():
    x = requests.post(
        "http://127.0.0.1:8000/example_app.redirect_compute", json=raw_data
    )
    remote_result = dict(x.json())
    print(remote_result)

    local_result = redirect_compute([Item(**_) for _ in raw_data]).model_dump()
    print(local_result)

    print(remote_result == local_result)


def example_fast_service_client():
    config = FastServiceConfig.load_from_file("./config/example-client.yml")
    fast_service_manager.setup_client_mode(config)

    request = [Item(**_) for _ in raw_data]
    result_from_internal_service = redirect_compute(items=request).model_dump()
    print(result_from_internal_service)

    result_from_computing_service = compute_total(items=request).model_dump()
    print(result_from_computing_service)

    print(result_from_internal_service == result_from_computing_service)

    updated_items = [item.model_dump() for item in update_items(items=request)]
    print(updated_items)


if __name__ == "__main__":
    example_http_client()
    example_fast_service_client()
