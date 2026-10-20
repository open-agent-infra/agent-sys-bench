MODE_LIST = ["one-by-one", "poisson", "fixed-interval", "specified", "batch-by-batch"]


class FastServiceBenchmarkConfig:
    request_num: int = 0
    file_path: str = None
    buffer_size: int = -1
    mode: str = "one-by-one"
    lambda_rate: float = 1.0
    output_file: str = None
    warm_up: int = 0
    warm_up_time: float = 0.0
    batch_size: int = 1
    arrival_times: list[float] = None
    concurrency: str = "thread"
    sampling_params: dict = None

    def __init__(
        self,
        request_num: int = 0,
        file_path: str = None,
        buffer_size: int = -1,
        mode: str = "one-by-one",
        lambda_rate: float = 1.0,
        output_file: str = None,
        warm_up: int = 0,
        warm_up_time: float = 0.0,
        batch_size: int = 1,
        arrival_times: list[float] = None,
        concurrency: str = "thread",
        sampling_params: dict = None,
    ):
        if request_num < 0:
            raise ValueError("Request number must be positive")
        self.request_num: int = request_num

        self.file_path: str = file_path
        self.buffer_size: int = buffer_size

        self.mode: str = mode
        self.lambda_rate: float = lambda_rate

        self.output_file = output_file

        self.warm_up = warm_up
        self.warm_up_time = warm_up_time
        self.batch_size = batch_size
        self.arrival_times = arrival_times
        self.concurrency = concurrency
        self.sampling_params = sampling_params
