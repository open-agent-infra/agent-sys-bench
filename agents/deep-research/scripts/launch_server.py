import os
import logging

root_logger = logging.getLogger()
root_logger.setLevel(os.environ.get("ROOT_LOG_LEVEL", "INFO").upper())
if os.environ.get("ROOT_LOG_FILE", None):
    file_handler = logging.FileHandler(os.environ["ROOT_LOG_FILE"], mode="w+")
    file_handler.setLevel(os.environ.get("ROOT_LOG_LEVEL", "INFO").upper())
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

legoagent_logger = logging.getLogger("agsbench_deep_research")
legoagent_logger.setLevel(os.environ.get("LEGOAGENT_LOG_LEVEL", "INFO").upper())
if os.environ.get("LEGOAGENT_LOG_FILE", None):
    file_handler = logging.FileHandler(os.environ["LEGOAGENT_LOG_FILE"], mode="w+")
    file_handler.setLevel(os.environ.get("LEGOAGENT_LOG_LEVEL", "INFO").upper())
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(formatter)
    legoagent_logger.addHandler(file_handler)


import argparse
from fast_service import FastServiceConfig
from fast_service.config import _read_yaml
from agsbench_deep_research.agents.deep_research.langchain_drs import agsbench_deep_research_fsm
from agsbench_deep_research.agents.deep_research.langchain_drs import DeepResearchWorkflowRequest
from agsbench_deep_research.agents.deep_research.langchain_drs import DeepResearchWorkflowResponse
from agsbench_deep_research.agents.deep_research.langchain_drs import deep_research_workflow
from agsbench_deep_research.agents.deep_research.langchain_drs import deep_research_workflow_async
from agsbench_deep_research.agents.deep_research.langchain_drs import get_drs_config_value
import multiprocessing


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="./config/deployment/server.yaml"
    )
    args = parser.parse_args()
    return args


def launch_services(config_path: str, pid: int):
    config_dict = _read_yaml(config_path)
    if pid is not None:
        config_dict["port"] += pid
        config_dict["monitor_storage"][
            "store_dir"
        ] = f"{config_dict['monitor_storage']['store_dir']}_{pid}"
        for service in config_dict["service_list"]:
            service["port"] += pid
    config = FastServiceConfig.load_from_dict(config_dict)
    print(f"config: {config.service_dict}")
    agsbench_deep_research_fsm.execute(config=config)


if __name__ == "__main__":
    args = get_args()

    uvicorn_nworkers = int(os.getenv("UVICORN_NWORKERS", "1"))
    if uvicorn_nworkers <= 1:
        config = FastServiceConfig.load_from_file(args.config)
        print(f"config: {config.service_dict}")
        agsbench_deep_research_fsm.execute(config=config)
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
