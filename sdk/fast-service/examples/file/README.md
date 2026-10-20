# File Upload Example

This example's logic is similar to [fastapi example](../fastapi/README.md).

Additionally, we add file upload logic.

Here we introduce key points to upload files to your services.

## 1. Define Service Interface with FastServiceFiles

```python
@fast_service_manager.fast_service
def compute_total(
    task_name: str,
    items: list[Item],
    files: list[FastServiceFile],
    context: RequestContext = None,
) -> Collection:
    ...
```

## 2. Implement your logic with the uploaded files

```python
@fast_service_manager.fast_service
def compute_total(
    task_name: str,
    items: list[Item],
    files: list[FastServiceFile],
    context: RequestContext = None,
) -> Collection:
    result = execute_compute(items)

    for file in files:
        items_in_file = [
            Item.parse_obj(_) for _ in json.load(file.get_file()).get("items")
        ]
        file_result = execute_compute(items_in_file)
        print(file.get_filename(), ":", file_result.model_dump())

    return result
```

## 3. Open files in client and pass them to the service interface

```python
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
```

