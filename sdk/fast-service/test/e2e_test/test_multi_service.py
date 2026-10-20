import os
import signal
import requests
import time

from pydantic import BaseModel
from fast_service import FastServiceManager, FastServiceConfig

IP = "localhost"
INTERNAL_PORT = 38341
FINAL_PORT = 38343

# Since this test is run by pytest, the module name will be the py file name rather than "__main__".
MODULE_NAME = "test_multi_service"

internal_config = FastServiceConfig(
    "fastapi",
    IP,
    INTERNAL_PORT,
    [
        {
            "module_name": MODULE_NAME,
            "name": "internal_step",
            "ip": IP,
            "port": INTERNAL_PORT,
        },
        {
            "module_name": MODULE_NAME,
            "name": "final_step",
            "ip": IP,
            "port": FINAL_PORT,
        },
    ],
)

final_config = FastServiceConfig(
    "fastapi",
    IP,
    FINAL_PORT,
    [
        {
            "module_name": MODULE_NAME,
            "name": "final_step",
            "ip": IP,
            "port": FINAL_PORT,
        },
    ],
)

internal_manager = FastServiceManager(internal_config)
final_manager = FastServiceManager(final_config)


def wait_for_connection(port: int):
    response = None
    while response is None:
        try:
            response = requests.get(f"http://{IP}:{port}/docs")
        except requests.exceptions.ConnectionError:
            pass
    return response


def wait_for_server(port: int):
    response = wait_for_connection(port)
    while response.status_code != 200:
        time.sleep(0.1)
        response = wait_for_connection(port)


class InvokePath(BaseModel):
    interfaces: list[str]


@internal_manager.fast_service
def final_step(path: InvokePath) -> InvokePath:
    path.interfaces.append(f"{IP}.{INTERNAL_PORT}.step2")
    return path


@internal_manager.fast_service
def internal_step(path: InvokePath) -> InvokePath:
    path.interfaces.append(f"{IP}.{INTERNAL_PORT}.step1")
    return final_step(path=path)


# internal_manager.async_execute()
wait_for_server(INTERNAL_PORT)


# Since the original object final_step is not actually used by internal_step, it's ok to redefine it.
@final_manager.fast_service
def final_step(path: InvokePath) -> InvokePath:
    path.interfaces.append(f"{IP}.{FINAL_PORT}.step2")
    return path


# final_manager.async_execute()
wait_for_server(FINAL_PORT)


def setup_module():
    print("Setup service")


def teardown_module():
    print("Teardown service")
    pid = os.getpid()
    os.kill(pid, signal.SIGINT)


def test_multi_service():
    r1 = (
        requests.post(
            f"http://{IP}:{INTERNAL_PORT}/{MODULE_NAME}.internal_step",
            json={"interfaces": ["client"]},
        )
        .json()
        .get("interfaces")
    )
    assert len(r1) == 3
    assert r1[2] == f"{IP}.{FINAL_PORT}.step2"

    r2 = (
        requests.post(
            f"http://{IP}:{FINAL_PORT}/{MODULE_NAME}.final_step",
            json={"interfaces": ["client"]},
        )
        .json()
        .get("interfaces")
    )
    assert len(r2) == 2
    assert r2[1] == f"{IP}.{FINAL_PORT}.step2"
