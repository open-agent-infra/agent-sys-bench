import time
import json

from example_app import fast_service_manager, redirect_compute, Item

from fast_service import (
    FastServiceConfig,
    FastServiceBenchmark,
    FastServiceBenchmarkConfig,
)


def line_to_req(line: str):
    line = line.strip()
    raw_data = json.loads(line)
    items = [Item(**_) for _ in raw_data["items"]]
    return items


def invoke_service(items):
    local_result = redirect_compute(items=items).model_dump()
    print(local_result)
    return local_result


if __name__ == "__main__":
    config = FastServiceConfig.load_from_file("./config/example-client.yml")
    fast_service_manager.setup_client_mode(config)

    for mode in [
        "one-by-one",
        "poisson",
        "fixed-interval",
        "specified",
        "batch-by-batch",
    ]:
        print(f"start {mode} benchmark")
        config = FastServiceBenchmarkConfig(
            file_path="./data/bm_data_file.txt",
            mode=mode,
            request_num=6,
            buffer_size=2,
            output_file="./.cache/bm_response.txt",
            warm_up=2,
            arrival_times=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        benchmark = FastServiceBenchmark(
            config=config, line_to_request=line_to_req, invoke_service=invoke_service
        )
        benchmark.execute()
        print(f"finish {mode} benchmark")
        time.sleep(1)
