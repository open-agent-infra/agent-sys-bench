import time

from example_app import fast_service_manager, redirect_compute, Item

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
    while k < 10:
        file = open("./file/req.json", "rb")
        local_result = redirect_compute(
            "compute",
            [Item(**_) for _ in raw_data["items"]],
            [FastServiceFile.load(file)],
        ).model_dump()
        print(local_result)
        file.close()

        time.sleep(1)
        k += 1

    # example_fast_service_client()
