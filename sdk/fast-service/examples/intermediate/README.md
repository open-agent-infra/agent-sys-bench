# Intermediate Result Storage and Module Benchmark Example

This example's logic is similar to [file example](../file/README.md).

Additionally, we add intermediate result related logic.

Here we introduce key points to save the intermediate results of your service and use them to benchmark related module in the service.

## 1. Configure intermediate result storage

```yml
...

store_intermediate: True
intermediate_storage:
  store_dir: ./.cache/computing/intermediate

...
```

## 2. Run your service and client to get intermediate result

The intermediate results are stored in your configured dir.

The inner structure of this dir looks like:

```yml
- service_name
  - requests.txt 
  - request_dir_0 # if you define file usage in your service
    - request_file_0
    - request_file_1
    ...
  - request_dir_1
  - request_dir_2
  ...
```

The file, requests.txt, stores all the requests.

Each line represents an intermediate result in json format, including request_id and all input args and names of uploaded files.

## 3. Implement module benchmark and run it

```python
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
```

