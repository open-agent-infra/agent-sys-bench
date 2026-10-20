# Benchmark and Monitor Example

This example's logic is similar to [fastapi example](../fastapi/README.md).

Here we introduce key points to inject monitor logic to your services and how to implement a benchmark.

## 1. Steps to Implement Monitor

### 1.1. Define Service Interface with RequestContext

```python
@fast_service_manager.fast_service
def compute_total(items: list[Item], context: RequestContext = None) -> Collection:
    ...
```

### 1.2. Turn on the Monitor in Your Config

```yaml
monitor: True
monitor_storage:
  type: csv
  store_dir: ./monitor/internal
```


## 2. Implement a Benchmark (WorkloadGenerator)

Currently, the benchmark framework suppose all the test data are organized in one txt file with each line as one request.

The encoding of the data file shall be ***utf-8***

To use the benchmark framework, you need to define two function, which will be used by the benchmark, in advance:
1. transform logic from txt line to request
2. service invocation logic with request

Example of all the steps are included in  [example_benchmark.py](example_benchmark.py)

### 2.1. Define the line-request transform logic 

In [example file](data/bm_data_file.txt), each line is a string of one json object.

```text
{"items": [{"name": "apple", "price": "10"}, {"name": "pear", "price": "20"}, {"name": "peach", "price": "30"}] }
{"items": [{"name": "apple", "price": "20"}, {"name": "pear", "price": "20"}, {"name": "peach", "price": "30"}] }
...
```

We define a function to transform the string into our target request.

```python
def line_to_req(line: str):
    line = line.strip()
    raw_data = json.loads(line)
    items = [Item(**_) for _ in raw_data["items"]]
    return items
```

### 2.2. Define the service invocation logic

We use the client mode of fast service to invoke deployed service. 

The invocation logic should be encapsulated as one function.

```python
def invoke_service(items):
    local_result = redirect_compute(items=items).model_dump()
    print(local_result)

if __name__ == "__main__":
    config = FastServiceConfig.load_from_file("./config/example-client.yml")
    fast_service_manager.setup_client_mode(config)
```

### 2.3. Configure your benchmark

Create a ```FastServiceBenchmarkConfig``` with desired configuration.

Currently, the benchmark framework support one-by-one mode and poisson mode.

In each mode, you need to configure the file path and request num.

```python
obo_config = FastServiceBenchmarkConfig(
        file_path="./data/bm_data_file.txt",
        mode="one-by-one",
        request_num=5,
    )
```

### 2.4. Start your benchmark

Create a FastServiceBenchmark and execute it.

```python
obo_benchmark = FastServiceBenchmark(
        config=obo_config, line_to_request=line_to_req, invoke_service=invoke_service
    )
obo_benchmark.execute()
```

## 3. Collect the Monitor data

### 3.1. Execute and Monitor

Run the [example_computing_service.py](example_computing_service.py), [example_internal_service.py](example_internal_service.py) and [example_benchmark.py](example_benchmark.py)

You will see monitor data in dir ```./cache``` with three sub-dir ```computing```, ```internal``` and ```client```.

The data files in each dir will be ```e2e_delay.csv``` and ```remote_delay.csv```.

### 3.2. Description of Built-in Monitor 

There are two metrics of built-in Monitor:
1. ***e2e delay***: The delay of each request executed in current service. The time consumption includes the time used on waiting the remote services' responses.
2. ***remote delay***: The delay of current service taken to wait for the remote services' response.

With the two above metric, we can deduce more valuable metrics for analysis, including service self time consumption and inter-service communication delay.

