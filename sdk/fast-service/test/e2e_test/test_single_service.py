import os
import signal
import requests
import time

from pydantic import BaseModel
from fast_service import FastServiceManager, FastServiceConfig

IP = "localhost"
PORT = 38341

# Since this test is run by pytest, the module name will be the py file name rather than "__main__".
MODULE_NAME = "test_single_service"

config = FastServiceConfig(
    "fastapi",
    IP,
    PORT,
    [
        {
            "module_name": MODULE_NAME,
            "name": "remote_invoke",
            "ip": IP,
            "port": PORT,
        },
        {
            "module_name": MODULE_NAME,
            "name": "local_invoke",
            "ip": IP,
            "port": PORT,
        },
    ],
)

manager = FastServiceManager(config)


class InvokePath(BaseModel):
    interfaces: list[str]


@manager.fast_service
def remote_invoke(path: InvokePath) -> InvokePath:
    path.interfaces.append(f"{IP}.{PORT}.remote_invoke")
    return path


@manager.fast_service
def local_invoke(path: InvokePath) -> InvokePath:
    path.interfaces.append(f"{IP}.{PORT}.local_invoke")
    return remote_invoke(path=path)


def wait_for_connection():
    response = None
    while response is None:
        try:
            response = requests.get(f"http://{IP}:{PORT}/docs")
        except requests.exceptions.ConnectionError:
            pass
    return response


def setup_module():
    print("Setup service")
    # manager.async_execute()
    response = wait_for_connection()
    while response.status_code != 200:
        time.sleep(0.1)
        response = wait_for_connection()


def teardown_module():
    print("Teardown service")
    pid = os.getpid()
    os.kill(pid, signal.SIGINT)


def test_single_service():
    r1 = requests.post(
        f"http://{IP}:{PORT}/{MODULE_NAME}.remote_invoke",
        json={"interfaces": ["client"]},
    )
    assert len(r1.json().get("interfaces")) == 2
    r2 = requests.post(
        f"http://{IP}:{PORT}/{MODULE_NAME}.local_invoke",
        json={"interfaces": ["client"]},
    )
    assert len(r2.json().get("interfaces")) == 3
