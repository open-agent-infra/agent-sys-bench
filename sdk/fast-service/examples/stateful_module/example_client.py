import time

from example_app import fast_service_manager, redirect_compute, Item, ComputeTask

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
        file_0 = open("./file/req_0.json", "rb")
        file_1 = open("./file/req_1.json", "rb")
        local_result = redirect_compute(
            ComputeTask(
                task_name=f"compute-{k}",
                items=[Item(**_) for _ in raw_data["items"]],
            ),
            [FastServiceFile.load(file_0), FastServiceFile.load(file_1)],
        ).model_dump()
        print(local_result)
        file_0.close()
        file_1.close()

        time.sleep(1)
        k += 1
