import time

from example_app import (
    fast_service_manager,
    redirect_compute,
    simple_compute,
    monitored_compute,
    multi_file_compute,
    single_file_compute,
    Item,
)

from fast_service import FastServiceConfig, FastServiceFile

raw_data = {
    "items": [
        {"name": "apple", "price": "10"},
        {"name": "pear", "price": "20"},
        {"name": "peach", "price": "30"},
    ],
}


if __name__ == "__main__":
    config = FastServiceConfig.load_from_file("./config/example-client.yml")
    fast_service_manager.setup_client_mode(config)

    k = 0
    while k < 4:
        local_result = simple_compute(
            [Item(**_) for _ in raw_data["items"]],
        ).model_dump()
        print(local_result)

        time.sleep(1)
        k += 1

    k = 0
    while k < 4:
        local_result = monitored_compute(
            [Item(**_) for _ in raw_data["items"]],
        ).model_dump()
        print(local_result)

        time.sleep(1)
        k += 1

    k = 0
    while k < 4:
        file = open("./file/req_0.json", "rb")
        local_result = single_file_compute(
            "compute",
            [Item(**_) for _ in raw_data["items"]],
            FastServiceFile.load(file),
        ).model_dump()
        print(local_result)
        file.close()

        time.sleep(1)
        k += 1

    k = 0
    while k < 4:
        file_0 = open("./file/req_0.json", "rb")
        file_1 = open("./file/req_1.json", "rb")
        local_result = redirect_compute(
            "compute",
            [Item(**_) for _ in raw_data["items"]],
            [FastServiceFile.load(file_0), FastServiceFile.load(file_1)],
        ).model_dump()
        print(local_result)
        file_0.close()
        file_1.close()

        time.sleep(1)
        k += 1
