from example_app import fast_service_manager

from fast_service import FastServiceConfig

if __name__ == "__main__":
    config = FastServiceConfig.load_from_file("./config/example-computing-service.yml")
    fast_service_manager.execute(config)
