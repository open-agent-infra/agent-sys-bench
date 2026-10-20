import os

import yaml
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException


from multiprocessing import Event, Queue

from image_manager import ImageManager
from model_request import InferenceRequest
from model_worker import Worker, MultiModelMsgChannel
from req_dispatcher import InferenceDispatcher, InferenceFuture

worker_dict: dict[str, Worker] = {}
inference_dispatcher: InferenceDispatcher
image_manager: ImageManager

app = FastAPI()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default=7070)
    parser.add_argument(
        "-f", "--models_config", default="../../config/models_server_config.yaml"
    )
    parser.add_argument("-hf", "--hf_home", default="/mnt/data/huggingface")
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-t", "--waiting_interval", default=1)
    parser.add_argument("--save_dir", default="../../public/images")
    # tempory fix for image batching
    parser.add_argument("--image_height", default=1024)
    parser.add_argument("--image_width", default=1024)

    args = parser.parse_args()
    return args


@app.get("/status/{model_name:path}")
def get_model_status(model_name: str):
    global worker_dict
    if model_name in worker_dict:
        return {
            "model_name": model_name,
            "loaded": True,
        }
    else:
        return {"model_name": model_name, "error": "Model not found."}


@app.post("/models/{model_name:path}")
def submit(model_name: str, request: InferenceRequest):
    """Endpoint to submit a request to the specified model"""

    global worker_dict, inference_dispatcher, image_manager
    if model_name not in worker_dict:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")

    if request.img_url is not None and not image_manager.is_valid_path(request.img_url):
        return {"error": "Invalid path."}

    future: InferenceFuture = inference_dispatcher.submit(model_name, request)
    result = future.get_result()
    return result


def main():
    global worker_dict, inference_dispatcher, image_manager
    args = get_args()
    with open(args.models_config, "r") as file:
        models_config = yaml.safe_load(file)

    arg_dict: dict = args.__dict__
    arg_dict["log_path"] = os.path.join(
        models_config["log_dir"], models_config["log_name"]
    )

    image_manager = ImageManager(args.save_dir)

    msg_channel = MultiModelMsgChannel()
    for model_config in models_config["models"]:
        model_name = model_config["model_name"]
        input_queue = Queue()
        msg_channel.register_channel(model_name, input_queue)
        model_worker = Worker(model_config, input_queue, msg_channel.output_queue)
        worker_dict[model_name] = model_worker

    inference_dispatcher = InferenceDispatcher(msg_channel)

    print("Loading models...", flush=True)
    finish_load_event_list: list[Event] = []
    for worker in worker_dict.values():
        finish_load_event: Event = Event()
        finish_load_event_list.append(finish_load_event)
        worker.start(finish_load_event, arg_dict)

    for finish_load_event in finish_load_event_list:
        finish_load_event.wait()
    print("Models loaded.", flush=True)

    print("Starting server...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
