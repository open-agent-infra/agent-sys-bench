from example_app import fast_service_manager

from fast_service import (
    FastServiceConfig,
    FastServiceBenchmarkConfig,
    FastServiceModuleBenchmarkConfig,
    FastServiceModuleBenchmark,
)

if __name__ == "__main__":
    config = FastServiceConfig.load_from_file("./config/example-module-bm.yml")
    fast_service_manager.setup_client_mode(config)

    print("start module benchmark")
    base_config = FastServiceBenchmarkConfig(
        file_path="./data/bm_data_file.txt",
        mode="poisson",
        request_num=6,
        lambda_rate=2,
        output_file="./.cache/bm_response.txt",
    )
    module_config = FastServiceModuleBenchmarkConfig(
        module_name="example_app",
        function_name="multi_file_compute",
        dir_path="./.cache/computing/intermediate/requests/example_app.multi_file_compute",
        base_config=base_config,
    )
    module_benchmark = FastServiceModuleBenchmark(
        config=module_config, fast_service_manager=fast_service_manager
    )
    module_benchmark.execute()
    print("finish module benchmark")
