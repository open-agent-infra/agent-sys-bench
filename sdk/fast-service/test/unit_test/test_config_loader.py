import yaml
import os
from fast_service import FastServiceConfig


class TestConfigLoader:
    file_path = "./config.yml"

    content: dict = {
        "service_impl": "fastapi",
        "ip": "127.0.0.1",
        "port": 8080,
        "service_list": [
            {
                "module_name": "__main__",
                "name": "compute_total",
                "ip": "127.0.0.1",
                "port": 8080,
            },
            {
                "module_name": "__main__",
                "name": "redirect_compute",
                "ip": "127.0.0.1",
                "port": 8080,
            },
        ],
    }

    def setup_method(self):
        print(f"Create tmp {self.file_path}")
        with open(self.file_path, "w", encoding="utf-8") as f:
            yaml.dump(data=self.content, stream=f, allow_unicode=True)

    def teardown_method(self):
        os.remove(self.file_path)
        print(f"Successfully remove tmp {self.file_path}")

    def compare_result(self, config):
        assert config.service_impl == self.content["service_impl"]
        assert config.ip == self.content["ip"]
        assert config.port == self.content["port"]
        assert (
            config.service_dict[
                (
                    self.content["service_list"][0]["module_name"],
                    self.content["service_list"][0]["name"],
                )
            ]
            is not None
        )

    def test_load_config_from_dict(self):
        config = FastServiceConfig.load_from_dict(self.content)
        self.compare_result(config)

    def test_load_config_from_file(self):
        config = FastServiceConfig.load_from_file(self.file_path)
        self.compare_result(config)

    def test_load_monitor_config(self):
        new_content = self.content.copy()
        new_content["monitor"] = True
        config = FastServiceConfig.load_from_dict(new_content)
        assert config.monitor is True

        new_content["monitor_storage"] = {
            "type": "prometheus",
            "ip": "127.0.0.1",
            "port": 8080,
        }

        config = FastServiceConfig.load_from_dict(new_content)
        assert config.monitor_storage.type == "prometheus"

        config = FastServiceConfig.load_from_dict(self.content)
        assert not config.monitor
        assert config.monitor_storage.type == "csv"
        assert config.monitor_storage.ip is None

    def test_load_intermediate_config(self):
        new_content = self.content.copy()
        new_content["store_intermediate"] = True

        config = FastServiceConfig.load_from_dict(new_content)
        assert config.store_intermediate is True

        new_content["intermediate_storage"] = {"store_dir": "./intermediate"}

        config = FastServiceConfig.load_from_dict(new_content)
        assert config.intermediate_storage.store_dir == "./intermediate"
