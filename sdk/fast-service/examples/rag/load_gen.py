import time
import numpy as np
from typing import List, Dict
import concurrent.futures

from fast_service import FastServiceConfig, FastServiceFile
from naive_rag import rag_e2e, RAGE2EOutput
from naive_rag import fast_service_manager


def load_requests(file_path: str = None) -> List[Dict]:
    requests = []
    question = "Please summarize the paper for me."
    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            session_id, api_name, pdf_path = line.strip().split(", ")
            request = {
                "req_id": i,
                "session_id": session_id,
                "api_name": api_name,
                "api_input": {
                    "collection_name": f"session_{session_id}",
                    "files": [FastServiceFile.load(open(pdf_path, "rb"))],
                    "question": question,
                    "context_limit": 3,
                },
            }
            requests.append(request)
    return requests


def generate_arrival_times(num_requests: int, lambda_rate: float):
    """Generates arrival times based on Poisson distribution."""
    # Generate inter-arrival times using exponential distribution
    rng = np.random.default_rng(seed=42)
    inter_arrival_times = rng.exponential(1 / lambda_rate, num_requests)
    arrival_times = np.cumsum(
        inter_arrival_times
    )  # Cumulative sum to get actual arrival times
    return arrival_times


def process_request(request) -> Dict:
    """Simulate processing a request."""
    print(f"Processing request: {request}")
    # Add your actual request processing logic here
    ret: RAGE2EOutput = rag_e2e(
        collection_name=request["api_input"]["collection_name"],
        files=request["api_input"]["files"],
        question=request["api_input"]["question"],
        context_limit=request["api_input"]["context_limit"],
    ).model_dump()
    print(f"finished {request['req_id']} ==> ret={ret}")
    return ret


def workload_generator(user_requests, arrival_times):
    """Generates and processes requests based on Poisson arrival times."""

    responses = []
    # Use a ThreadPoolExecutor to handle concurrent requests
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_request = {}
        start_time = time.time()

        # Schedule requests for processing based on arrival times
        for request, arrival_time in zip(user_requests, arrival_times):
            time_to_wait = max(0, arrival_time - (time.time() - start_time))
            time.sleep(time_to_wait)  # Wait until the arrival time
            future = executor.submit(process_request, request)
            future_to_request[future] = request

        # Collect responses as they complete
        for future in concurrent.futures.as_completed(future_to_request):
            request = future_to_request[future]
            try:
                response = future.result()
                responses.append(response)
            except Exception as exc:
                print(f"Request {request} generated an exception: {exc}")

    return responses


def main():
    requests_file = ".cache/requests.txt"
    all_requests = load_requests(requests_file)

    lambda_rate = 1.0
    arrival_times = generate_arrival_times(
        num_requests=len(all_requests), lambda_rate=lambda_rate
    )
    print(f"arrival_times: {arrival_times.tolist()}")

    config = FastServiceConfig.load_from_file("./config/naive-rag-client.yml")
    fast_service_manager.setup_client_mode(config)

    workload_generator(all_requests, arrival_times)


if __name__ == "__main__":
    main()
