from azure_openai import fast_service_manager


if __name__ == "__main__":
    print(f"start to launch server")
    fast_service_manager.execute()
    print(f"server stoped")
