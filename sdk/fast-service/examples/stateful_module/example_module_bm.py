from example_app import fast_service_manager

from fast_service import (
    FastServiceConfig,
    FastServiceBenchmarkConfig,
    StatefulModuleBenchmarkConfig,
    StatefulModuleBenchmark,
)

if __name__ == "__main__":
    config = FastServiceConfig.load_from_file("./config/example-module-bm.yml")
    fast_service_manager.setup_client_mode(config)

    print("start stateful module benchmark")
    base_config = FastServiceBenchmarkConfig(
        file_path="./data/bm_data_file.txt",
        mode="poisson",
        request_num=4,
        lambda_rate=2,
        output_file="./.cache/bm_response.txt",
    )
    module_config = StatefulModuleBenchmarkConfig(
        module_name="example_app",
        sequence_file_path="./.cache/internal/remote_delay.csv",
        dir_path="./.cache/computing/intermediate/requests/",
        init_function="create_context",
        context_id_param="history_id",
        base_config=base_config,
    )
    module_benchmark = StatefulModuleBenchmark(
        config=module_config, fast_service_manager=fast_service_manager
    )
    module_benchmark.execute()
    print("finish stateful module benchmark")
