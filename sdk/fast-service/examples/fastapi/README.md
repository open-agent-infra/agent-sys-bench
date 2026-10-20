# Example

The service logic is implemented in [example_app.py](example_app.py), which the reader should start with to learn the code.

The example service and client program is set up via: 
1. [example_computing_service.py](example_computing_service.py): the service executing computing logic indeed.
2. [example_internal_service.py](example_internal_service.py): the service redirecting requests to example_computing_service.
3. [example_client.py](example_client.py): client to test each service.

## 1. Define Classes of Request and Response.

It is recommended to wrap all the params of one function, which is aimed to be deployed as a service, into an object.

The class of this object shall extend ```pydantic.BaseModel```, which helps param validation and auto serialization/deserialization.

Complex class is supported only if all related classes have extended ```pydantic.BaseModel```.

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

class Collection(BaseModel):
    name: str
    total_price: float
    item_list: list[Item]
```

## 2. Define Functions to be deployed as Service

Here's a simple example. 

Function ```compute_total``` is aimed to compute a sum value of the input and return result in a defined format.

Function ```redirect_compute``` just invokes ```compute_total```.

```python
def compute_total(items: list[Item]) -> Collection:
    name = f'{len(items)}-items'
    total_price = 0
    for item in items:
        total_price += item.price
    item_list = items
    result = Collection(name=name, total_price=total_price, item_list=item_list)
    return result

def redirect_compute(items: list[Item]) -> Collection:
    return compute_total(items=items)

```

## 3. Annotate Functions to be deployed as Service

1. Declare a ```fast_service_manager``` of class ```FastServiceManager``` imported from ```fast_service```
2. Annotate target functions with ```@fast_service_manager.fast_service```

```python
from fast_service import FastServiceManager

fast_service_manager = FastServiceManager()

@fast_service_manager.fast_service
def compute_total(items: list[Item]) -> Collection:
    ...

@fast_service_manager.fast_service
def redirect_compute(items: list[Item]) -> Collection:
    ...
```

During python code and module loading stage ```fast_service.fast_service_manager``` helps:
1. record all functions that may be deployed as service 
2. rewrite the functions, which is one of the preparation for execution

## 4. Prepare the YAML file to Configure the Deployment

One example is [example-computing-service.yml](config/example-computing-service.yml).

Here are some description of key configs.

```YAML
service_impl: fastapi # The implementation of service, currently only support fastapi
ip: 127.0.0.1 # IP address of current instance
port: 8001 # Port listened by current instance
service_list: # Services provide in current instance
  - module_name: example_app # module name of the targe function, could be retrieved from getattr(f, "__module__") 
    name: compute_total # function name
    ip: 127.0.0.1 # IP address of the service. 
    port: 8001 # Port of the service.
  - module_name: example_app
    name: redirect_compute
    ip: 127.0.0.1
    port: 8001
```


If **ip:port** of one service is **different from** current instance, related logic will be automatically rewritten as **remote service invoke**. 

**Otherwise**, this function will be deployed **as a service** in current instance.

For those **not configured in yml file**, they will **remain normal local function** and will be executed without any special process. 

To easily identify the service name, please refer to [service_name_checker](../../docs/tools.md#service-name-check-tool).

## 5. Execute fast_service_manager with config in main process Explicitly

```python
from example_app import fast_service_manager

from fast_service import FastServiceConfig

if __name__ == "__main__":
    config = FastServiceConfig.load_from_file("./config/example-internal-service.yml")
    fast_service_manager.execute(config)

if __name__ == '__main__':
    fast_service_manager.execute()
```



During execution, ```fast_service.fast_service_manager``` will
1. startup recorded functions as service
2. automatically take over the communication logic, local function invoking or remote service invoking, during service serving stage

Due to the implementation, ```fast_service_manager``` execution may block the rest logic of main process, which means the codes following ```fast_service_manager.execute()``` won't be executed.


## 6. Implement a Client

Since the example is based on Fastapi, [example_client.py](example_client.py) provides two manners to implement client:
1. http based client
2. fast service setup client

```python
def example_http_client():
    x = requests.post(
        "http://127.0.0.1:8000/example_app.redirect_compute", json=raw_data
    )
    remote_result = dict(x.json())
    print(remote_result)

def example_fast_service_client():
    config = FastServiceConfig.load_from_file("./config/example-client.yml")
    fast_service_manager.setup_client_mode(config)

    request = [Item(**_) for _ in raw_data]
    result_from_internal_service = redirect_compute(items=request).model_dump()
    print(result_from_internal_service)
```

To directly invoke the function as invoking the remote service, which is an automatic client manner provided by fast service, a yml should be prepared to specify the client mode and remote services.

```yaml
client_mode: True
service_impl: fastapi
service_list:
  - module_name: example_app
    name: compute_total
    ip: 127.0.0.1
    port: 8001
  - module_name: example_app
    name: redirect_compute
    ip: 127.0.0.1
    port: 8000
```


## 7. Run the Example Services and Client

In this example, there are two service instances, example_computing_service and example_internal_service. They share the same code, [example_app](example_app.py), but different config of deployment.

Computing service will be provided on ```127.0.0.1:8001```.

Internal service will be provided on ```127.0.0.1:8000```, via invoking computing service on ```127.0.0.1:8001```.

Deploy each service in individual command line.

```shell
python example_computing_service.py --fast_service_config ./config/example-server.yml
```

```shell
python example_internal_service.py --fast_service_config ./config/example-client.yml
```

Run the example_client to see the test result.

```shell
{'name': '3-items', 'total_price': 60.0, 'item_list': [{'name': 'apple', 'price': 10.0}, {'name': 'pear', 'price': 20.0}, {'name': 'peach', 'price': 30.0}]}
{'name': '3-items', 'total_price': 60.0, 'item_list': [{'name': 'apple', 'price': 10.0}, {'name': 'pear', 'price': 20.0}, {'name': 'peach', 'price': 30.0}]}
True
{'name': '3-items', 'total_price': 60.0, 'item_list': [{'name': 'apple', 'price': 10.0}, {'name': 'pear', 'price': 20.0}, {'name': 'peach', 'price': 30.0}]}
{'name': '3-items', 'total_price': 60.0, 'item_list': [{'name': 'apple', 'price': 10.0}, {'name': 'pear', 'price': 20.0}, {'name': 'peach', 'price': 30.0}]}
True
```

When running program via shell command, the config could be specified via program params. 

```shell
python example_internal_service.py --fast_service_config ./config/example-internal-service.yml
```

Config specified by program params won't shadow the config already defined in the code. If there's no config defined and used explicitly in code, the config will be loaded via program params from file.
