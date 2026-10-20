import os
import sys

abs_path = os.path.abspath(f"{os.environ.get("HUGGINGGPT_HOME")}/src/agsbench_hugginggpt")

module_dir = os.path.dirname(abs_path)
if module_dir not in sys.path:
    sys.path.append(module_dir)

from fast_service import FastServiceConfig
from agsbench_hugginggpt import hugginggpt_fsm

if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__),
        f"{os.environ.get("HUGGINGGPT_HOME")}/config/hugginggpt_server.yml",
    )
    config = FastServiceConfig.load_from_file(config_path)
    print(f"config: {config.service_dict}")
    hugginggpt_fsm.execute(config=config)
