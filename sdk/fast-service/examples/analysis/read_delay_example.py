from fast_service.analysis import (
    DelayRecord,
    DelayFile,
    RequestDelay,
    RequestDelayManager,
)


EXAMPLE_DELAY_FILE_LIST = [
    "delay_file/client/remote_delay.csv",
    "delay_file/env_manager/e2e_delay.csv",
    "delay_file/execution_coordinator/remote_delay.csv",
    "delay_file/execution_coordinator/e2e_delay.csv",
    "delay_file/llm/e2e_delay.csv",
]


if __name__ == "__main__":
    delay_file_list = [DelayFile("webarena", _) for _ in EXAMPLE_DELAY_FILE_LIST]
    request_delay_manager = RequestDelayManager("webarena", delay_file_list)
    request_delay_manager.prepare_in_request()
    request_delay_manager.print_flatten_operations()

    count = 0
    for request_delay in request_delay_manager.request_dict.values():
        if count == 2:
            break

        print(request_delay.request_id)

        record_list = request_delay.stage_list.sort(
            key=lambda x: (x.end_time - x.delay)
        )
        for record in request_delay.stage_list:
            print(
                f'on remote {record.invoker_name == "None"},',
                record.function_name,
                record.delay,
            )

        count += 1
