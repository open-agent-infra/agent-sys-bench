import os
import argparse
import time
import traceback

import yaml
import torch

from typing import Any, Optional
from multiprocessing import Process, Queue, Event
from queue import Empty


from transformers import pipeline
from transformers import AutoModelForImageSegmentation
from datasets import load_dataset
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import DiffusionPipeline
from PIL import ImageDraw
from torchvision import transforms

from image_manager import ImageManager
from model_request import InferenceRequest

from functools import lru_cache
import logging


@lru_cache
def get_logger(path: str):
    model_server_logger = logging.getLogger(name="hugginggpt_model_server")
    model_server_logger.setLevel(logging.DEBUG)
    cache_dir = os.path.dirname(path)
    os.makedirs(cache_dir, exist_ok=True)
    fh = logging.FileHandler(filename=path)
    fh.setLevel(logging.DEBUG)
    model_server_logger.addHandler(fh)
    return model_server_logger


class MultiModelMsgChannel:

    input_queue_dict: dict[str, Queue]

    output_queue: Queue

    def __init__(self):
        self.input_queue_dict: dict[str, Queue] = {}
        self.output_queue = Queue()

    def register_channel(self, model_name: str, input_queue: Queue):
        self.input_queue_dict[model_name] = input_queue

    def has_output(self):
        return not self.output_queue.empty()

    def recv_output(self):
        return self.output_queue.get()

    def send_output(self, req: tuple[str, Any]):
        self.output_queue.put(req)

    def send_input(self, model_name: str, req: tuple[str, Any]):
        self.input_queue_dict[model_name].put(req)


def load_model(
    model_config: dict[str, Any],
    hf_home: str,
    batch_size: int,
) -> dict:
    print(f"Loading model {model_config['model_name']}...", flush=True)
    if model_config["model_name"] == "microsoft/speecht5_tts":
        result = {
            "model": pipeline(
                "text-to-speech",
                model="microsoft/speecht5_tts",
                device=model_config["device"],
                batch_size=batch_size,
            ),
            "embeddings_dataset": load_dataset(
                "Matthijs/cmu-arctic-xvectors", split="validation"
            ),
            "device": model_config["device"],
        }
    elif model_config["model_name"] == "stabilityai/stable-diffusion-xl-refiner-1.0":
        result = {
            "model": StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ),
            "device": model_config["device"],
        }
        device = torch.device(int(model_config["device"]))
        result["model"] = result["model"].to(device)
    elif model_config["model_name"] == "briaai/RMBG-2.0":
        result = {
            "model": AutoModelForImageSegmentation.from_pretrained(
                model_config["model_name"],
                trust_remote_code=True,
            ),
            "device": model_config["device"],
        }
        device = torch.device(int(model_config["device"]))
        result["model"] = result["model"].to(device)
    elif model_config["model_name"] == "stabilityai/stable-diffusion-xl-base-1.0":
        result = {
            "model": DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ),
            "device": model_config["device"],
        }
        device = torch.device(int(model_config["device"]))
        result["model"] = result["model"].to(device)
    else:
        result = {
            "model": pipeline(
                task=model_config["task"],
                model=f"{model_config['model_name']}",
                device=model_config["device"],
                batch_size=batch_size,
            ),
            "device": model_config["device"],
        }
    print(
        f"Loaded model {model_config['model_name']}, task: {model_config['task']}, device: {model_config['device']}",
        flush=True,
    )
    return result


def execute(
    model_name: str,
    requests_list: list[InferenceRequest],
    model_detail: dict[str, Any],
    image_server: ImageManager,
    image_height: int,
    image_width: int,
):
    """Endpoint to run inference on the specified model"""

    model = model_detail["model"]
    device = model_detail["device"]

    # NLP tasks
    if model_name == "Qwen/Qwen2.5-1.5B-Instruct":
        messages = [_.messages for _ in requests_list]
        results = model(messages, max_new_tokens=1024)
        return results

    # object detection
    elif model_name == "facebook/detr-resnet-50":
        img_urls = [_.img_url for _ in requests_list]
        images = []
        for img_url in img_urls:
            images.append(image_server.load_image(img_url))
        inference_results = model(images, threshold=0.9)
        results = []
        for i, inference_result in enumerate(inference_results):
            draw = ImageDraw.Draw(images[i])
            for result in inference_result:
                box = result["box"]
                label = result["label"]
                score = result["score"]
                xmin, ymin, xmax, ymax = box.values()
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
                text = f"{label}: {score:.2f}"
                text_bbox = draw.textbbox((0, 0), text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_position = (
                    xmin,
                    max(0, ymin - text_height - 5),
                )
                draw.rectangle(
                    [
                        text_position,
                        (xmin + text_width, text_position[1] + text_height),
                    ],
                    fill="red",
                )
                draw.text(text_position, text, fill="white")
            output_path = image_server.save_image(images[i])
            results.append({"generated_image": output_path})
        return results

    # image classification
    elif model_name == "Falconsai/nsfw_image_detection":
        img_urls = [_.img_url for _ in requests_list]
        images = []
        for img_url in img_urls:
            images.append(image_server.load_image(img_url))
        results = model(images)
        return results

    # image to image
    elif model_name == "stabilityai/stable-diffusion-xl-refiner-1.0":
        urls = [_.img_url for _ in requests_list]
        texts = [_.text for _ in requests_list]
        texts = [text if text is not None else "" for text in texts]
        img_height = image_height
        img_width = image_width

        images = []
        for url in urls:
            images.append(image_server.load_image(url))

        images = model(texts, image=images, height=img_height, width=img_width).images
        results = []
        for i, image in enumerate(images):
            output_path = image_server.save_image(image)
            results.append({"generated_image": output_path})
        return results

    # image segmentation
    elif model_name == "briaai/RMBG-2.0":
        img_height = image_height
        img_width = image_width
        image_size = (img_height, img_width)

        transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        urls = [_.img_url for _ in requests_list]
        raw_images = []
        images = []
        for url in urls:
            raw_images.append(image_server.load_image(url))
        for i, raw_image in enumerate(raw_images):
            images.append(transform_image(raw_image))
        images = torch.stack(images).to(torch.device(int(device)))
        with torch.no_grad():
            preds = model(images)[-1].sigmoid().cpu()
        results = []
        for i, pred in enumerate(preds):
            image = raw_images[i]
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image.size)
            image.putalpha(mask)
            output_path = image_server.save_image(image)
            results.append({"generated_image": output_path})
        return results

    # image to text
    elif model_name == "nlpconnect/vit-gpt2-image-captioning":
        img_urls = [_.img_url for _ in requests_list]
        images = []
        for img_url in img_urls:
            images.append(image_server.load_image(img_url))
        results = model(images)
        return results
    # text to image
    elif model_name == "stabilityai/stable-diffusion-xl-base-1.0":
        texts = [_.text for _ in requests_list]
        img_height = image_height
        img_width = image_width
        images = model(prompt=texts, height=img_height, width=img_width).images

        results = []
        for i, image in enumerate(images):
            output_path = image_server.save_image(image)
            results.append({"generate_image": output_path})
        return results
    # visual question answering
    elif model_name == "dandelin/vilt-b32-finetuned-vqa":
        urls = [_.img_url for _ in requests_list]
        texts = [_.text for _ in requests_list]
        images = []
        for url in urls:
            images.append(image_server.load_image(url))
        results = model(question=texts, image=images)
        return results

    return {
        "model_name": model_name,
        "device": device,
        "inputs": "xxx",
        "result": "yyy",
    }


def model_execute(
    model_config: dict[str, Any],
    input_queue: Queue,
    output_queue: Queue,
    finish_load_event: Event,
    stop_event: Event,
    process_args: dict[str, Any],
):
    args = argparse.Namespace(**process_args)
    model_name = model_config["model_name"]
    print(f"Start process of {model_name}", flush=True)
    try:
        model_detail = load_model(model_config, args.hf_home, args.batch_size)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        traceback.print_exc()
        return
    finish_load_event.set()

    if "batch_size" in model_config:
        batch_size = model_config["batch_size"]
    else:
        batch_size = args.batch_size

    while not stop_event.is_set():
        num = 0
        request_batch: list[tuple[str, Any]] = []
        start_time = time.time()
        end_time = start_time + args.waiting_interval
        while num < batch_size:
            if time.time() > end_time:
                break
            try:
                next_req = input_queue.get(timeout=(end_time - time.time()))
                request_batch.append(next_req)
                num += 1
            except Empty:
                continue

        if len(request_batch) == 0:
            continue

        try:
            start_time = time.time()
            batch_results = execute(
                model_name,
                [_[1] for _ in request_batch],
                model_detail,
                ImageManager(args.save_dir),
                args.image_height,
                args.image_width,
            )
            end_time = time.time()
            get_logger(args.log_path).info(
                f"{end_time}, {model_name}, {len(request_batch)}, {end_time - start_time}"
            )
            for i in range(len(request_batch)):
                output_queue.put((request_batch[i][0], batch_results[i]))
        except Exception as e:
            print(f"{e}", flush=True)
            traceback.print_exc()
            for i in range(len(request_batch)):
                output_queue.put((request_batch[i][0], {"error": f"{e}"}))


class Worker:

    model_config: dict[str, Any]
    input_queue: Queue
    output_queue: Queue

    model_process: Optional[Process]
    stop_event: Event

    def __init__(
        self,
        model_config: dict[str, Any],
        input_queue: Queue,
        output_queue: Queue,
    ):
        self.model_config = model_config
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.stop_event = Event()
        self.model_process = None

    def start(self, finish_load_event: Event, process_args: dict[str, Any]):
        self.model_process = Process(
            target=model_execute,
            args=(
                self.model_config,
                self.input_queue,
                self.output_queue,
                finish_load_event,
                self.stop_event,
                process_args,
            ),
        )
        self.model_process.start()

    def stop(self):
        self.stop_event.set()
