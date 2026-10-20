# Useful Tools


## Service Name check Tool

The script [service_name_checker.py](../tools/service_name_checker.py) helps developers to easily identify what service are annotated in the code.

To use the tool, run the script in the following way:

```shell
python service_name_checker.py -f <path of your file>
```

For example:

```shell
# instruction
python .\tools\service_name_checker.py -f .\example\fastapi\example_computing_service.py

# result
        module_name     function_name
1       example_app     compute_total
2       example_app     redirect_compute
```