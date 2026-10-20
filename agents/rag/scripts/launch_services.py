import os
import argparse
from fast_service import FastServiceConfig
from fast_service.config import _read_yaml
from agsbench_rag import agsbench_rag_fsm, init_settings
from agsbench_rag.settings import get_settings
import multiprocessing


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="../config/rag-server.yml")
    parser.add_argument("-s", "--settings", type=str, default="../config/settings.yml")
    return parser.parse_args()


def launch_services(config_path: str, pid: int):
    config_dict = _read_yaml(config_path)
    if pid is not None:
        get_settings().log_file = f"{get_settings().log_file}_{pid}"
        config_dict["port"] += pid
        config_dict["monitor_storage"][
            "store_dir"
        ] = f"{config_dict['monitor_storage']['store_dir']}_{pid}"
        for service in config_dict["service_list"]:
            service["port"] += pid
    config = FastServiceConfig.load_from_dict(config_dict)
    print(f"config: {config.service_dict}")
    agsbench_rag_fsm.execute(config=config)


if __name__ == "__main__":
    args = get_args()
    settings_file = args.settings
    init_settings(settings_file)
    print(f"settings: {get_settings()}")

    uvicorn_nworkers = get_settings().uvicorn_nworkers
    if uvicorn_nworkers is None or uvicorn_nworkers <= 1:
        config = FastServiceConfig.load_from_file(args.config)
        print(f"config: {config.service_dict}")
        agsbench_rag_fsm.execute(config=config)
    else:
        processes = []
        for i in range(uvicorn_nworkers):
            p = multiprocessing.Process(
                target=launch_services, args=(args.config, i + 1)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
