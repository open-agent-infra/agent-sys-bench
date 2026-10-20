class E2EDelayRecord:

    end_time: float
    request_id: str
    invoker_name: str
    service_name: str
    delay: float
    call_id: str
    parent_call_id: str

    def __init__(
        self,
        end_time: float,
        request_id: str,
        invoker_name: str,
        service_name: str,
        delay: float,
        call_id: str,
        parent_call_id: str,
    ):
        self.end_time = end_time
        self.request_id = request_id
        self.invoker_name = invoker_name
        self.service_name = service_name
        self.delay = delay
        self.call_id = call_id
        self.parent_call_id = parent_call_id

    def to_tuple(self):
        return (
            self.end_time,
            self.request_id,
            self.invoker_name,
            self.service_name,
            self.delay,
            self.call_id,
            self.parent_call_id,
        )


class RemoteDelayRecord:

    end_time: float
    request_id: str
    invoker_name: str
    remote_service_name: str
    delay: float
    call_id: str
    parent_call_id: str

    def __init__(
        self,
        end_time: float,
        request_id: str,
        invoker_name: str,
        remote_service_name: str,
        delay: float,
        call_id: str,
        parent_call_id: str,
    ):
        self.end_time = end_time
        self.request_id = request_id
        self.invoker_name = invoker_name
        self.remote_service_name = remote_service_name
        self.delay = delay
        self.call_id = call_id
        self.parent_call_id = parent_call_id

    def to_tuple(self):
        return (
            self.end_time,
            self.request_id,
            self.invoker_name,
            self.remote_service_name,
            self.delay,
            self.call_id,
            self.parent_call_id,
        )
