from test_app import fast_service_manager
from fast_service import FastServiceConfig

if __name__ == "__main__":
    # Load config
    config = FastServiceConfig.load_from_file("config.yml")

    # Start the service
    print(f"Starting service on {config.ip}:{config.port}")
    fast_service_manager.execute(config)
