class MonitorConfig:
    type: str
    store_dir: str
    ip: str
    port: int

    def __init__(
        self,
        type: str = "csv",
        store_dir: str = "./monitor",
        ip: str = None,
        port: int = None,
    ):
        self.type = type
        self.ip = ip
        self.port = port
        self.store_dir = store_dir

    @classmethod
    def from_dict(cls, data: dict):
        return MonitorConfig(**data)
