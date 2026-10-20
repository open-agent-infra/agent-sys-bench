from .engine_interface import IFastServiceEngine
from .function_impl import FastServiceOnFunction
from .fastapi_impl import FastServiceOnFastAPI
from .grpc_impl import FastServiceOnGRPC

FAST_API_IMPL = "fastapi"
LOCAL_FUNCTION_IMPL = "local_function"
GRPC_IMPL = "grpc"
SERVICE_IMPL_DICT = {
    FAST_API_IMPL: FastServiceOnFastAPI,
    LOCAL_FUNCTION_IMPL: FastServiceOnFunction,
    GRPC_IMPL: FastServiceOnGRPC,
}


def create_fast_service_engine(service_impl: str) -> IFastServiceEngine:
    if service_impl not in SERVICE_IMPL_DICT:
        raise KeyError(f"{service_impl} not implemented")
    return SERVICE_IMPL_DICT[service_impl]()
