import re
import json
import requests
import threading
from queue import Queue
from fast_service import RequestContext
from .service_manager import hugginggpt_fsm, ModelSelectionInput, ModelSelectionOutput
from .utils import (
    get_config,
    get_prestep_config,
    get_modelsmap,
    get_inference_mode,
    get_models_server_url,
    get_logger,
    collect_result,
    replace_slot,
    find_and_format_json,
    call_llm,
)


def get_model_status(model_id, local_url, queue=None):
    url = local_url
    endpoint_type = "local"

    r = requests.get(url)
    if r.status_code == 200 and "loaded" in r.json():
        if queue:
            queue.put((model_id, True, endpoint_type))
        get_logger().info(f"     Model {model_id} is available on {endpoint_type}.")
        return True
    else:
        get_logger().info(f"     Model {model_id} is not available on {endpoint_type}.")
        if queue:
            queue.put((model_id, False, None))
        return False


def get_avaliable_models(candidates, topk=5):
    all_available_models = {"local": []}
    threads = []
    result_queue = Queue()

    for candidate in candidates:
        model_id = candidate["id"]

        localStatusUrl = f"{get_models_server_url()}/status/{model_id}"
        thread = threading.Thread(
            target=get_model_status, args=(model_id, localStatusUrl, result_queue)
        )
        threads.append(thread)
        thread.start()

    result_count = len(threads)
    while result_count:
        model_id, status, endpoint_type = result_queue.get()
        if status and model_id not in all_available_models:
            all_available_models[endpoint_type].append(model_id)
        if len(all_available_models["local"]) >= topk:
            break
        result_count -= 1

    for thread in threads:
        thread.join()

    return all_available_models


def choose_model(input, task, metas, call_id: str = None):
    choose_model_demos_or_presteps = open(
        get_prestep_config("choose_model"), "r"
    ).read()
    choose_model_prompt = get_config()["prompt"]["choose_model"]
    choose_model_tprompt = get_config()["tprompt"]["choose_model"]
    prompt = replace_slot(
        choose_model_prompt,
        {
            "input": input,
            "task": task,
            "metas": metas,
        },
    )
    messages = json.loads(choose_model_demos_or_presteps)
    messages[0]["content"] = input
    messages[1]["content"] = str(task)

    messages.insert(0, {"role": "system", "content": choose_model_tprompt})
    messages.append({"role": "user", "content": prompt})

    response, usage = call_llm(messages, "model_selection", call_id=call_id)
    return response, usage


@hugginggpt_fsm.fast_service
def model_selection(
    input: ModelSelectionInput, context: RequestContext = None
) -> ModelSelectionOutput:
    request_index = context.request_id
    get_logger().info(f"[{request_index}] ModelSelection start")

    command = input.command
    results = input.results
    id = command["id"]
    args = command["args"]
    task = command["task"]
    deps = command["dep"]
    hosted_on = ""
    best_model_id = None
    reason = None

    if deps[0] != -1:
        dep_tasks = [results[dep] for dep in deps]
    else:
        dep_tasks = []

    get_logger().info(f"[{request_index}]   Run task: {id} - {task}")

    if deps[0] != -1:
        if "image" in args and "<GENERATED>-" in args["image"]:
            resource_id = int(re.search(r"<GENERATED>-(\d+)", args["image"]).group(1))
            if "generated_image" in results[resource_id]["inference result"]:
                args["image"] = results[resource_id]["inference result"][
                    "generated_image"
                ]
        if "audio" in args and "<GENERATED>-" in args["audio"]:
            resource_id = int(re.search(r"<GENERATED>-(\d+)", args["audio"]).group(1))
            if "generated audio" in results[resource_id]["inference result"]:
                args["audio"] = results[resource_id]["inference result"][
                    "generated audio"
                ]
        if "text" in args and "<GENERATED>-" in args["text"]:
            resource_id = int(re.search(r"<GENERATED>-(\d+)", args["text"]).group(1))
            if "generated text" in results[resource_id]["inference result"]:
                args["text"] = results[resource_id]["inference result"][
                    "generated text"
                ]

    text = image = audio = None
    for dep_task in dep_tasks:
        if "generated text" in dep_task["inference result"]:
            text = dep_task["inference result"]["generated text"]
            get_logger().info(
                f"[{request_index}]     Detect the generated text of dependency task (from results): {text}"
            )
        elif "text" in dep_task["task"]["args"]:
            text = dep_task["task"]["args"]["text"]
            get_logger().info("Detect the text of dependency task (from args): " + text)
        if "generated_image" in dep_task["inference result"]:
            image = dep_task["inference result"]["generated_image"]
            get_logger().info(
                f"[{request_index}]     Detect the generated_image of dependency task (from results): {image}"
            )
        elif "image" in dep_task["task"]["args"]:
            image = dep_task["task"]["args"]["image"]
            get_logger().info(
                f"[{request_index}]     Detect the image of dependency task (from args): {image}"
            )
        if "generated audio" in dep_task["inference result"]:
            audio = dep_task["inference result"]["generated audio"]
            get_logger().info(
                f"[{request_index}]     Detect the generated audio of dependency task (from results): {audio}"
            )
        elif "audio" in dep_task["task"]["args"]:
            audio = dep_task["task"]["args"]["audio"]
            get_logger().info(
                f"[{request_index}]     Detect the audio of dependency task (from args): {audio}"
            )

    if "image" in args and "<GENERATED>" in args["image"]:
        if image:
            args["image"] = image
    if "audio" in args and "<GENERATED>" in args["audio"]:
        if audio:
            args["audio"] = audio
    if "text" in args and "<GENERATED>" in args["text"]:
        if text:
            args["text"] = text

    for resource in ["image", "audio"]:
        if (
            resource in args
            and not args[resource].startswith("public/")
            and len(args[resource]) > 0
            and not args[resource].startswith("http")
        ):
            args[resource] = f"{args[resource]}"

    if "-text-to-image" in command["task"] and "text" not in args:
        get_logger().info(
            f"[{request_index}]     control-text-to-image task, but text is empty, so we use control-generation instead."
        )
        control = task.split("-")[0]

        if control == "seg":
            task = "image-segmentation"
            command["task"] = task
        elif control == "depth":
            task = "depth-estimation"
            command["task"] = task
        else:
            task = f"{control}-control"

    command["args"] = args
    get_logger().info(f"[{request_index}]     parsed task: {command}")

    if task.endswith("-text-to-image") or task.endswith("-control"):
        if task.endswith("-text-to-image"):
            control = task.split("-")[0]
            best_model_id = f"lllyasviel/sd-controlnet-{control}"
        else:
            best_model_id = task
        hosted_on = "local"
        reason = "ControlNet is the best model for this task."
        choose = {"id": best_model_id, "reason": reason}
        get_logger().info(f"[{request_index}]     chosen model: {choose}")
    elif task in [
        "token-classification",
        "text2text-generation",
        "summarization",
        "translation",
        "question-answering",
        "conversational",
        "text-generation",
        "sentence-similarity",
        "document-question-answering",
    ]:
        best_model_id = "LLM"
        reason = "LLM performs well on some NLP tasks as well."
        choose = {"id": best_model_id, "reason": reason}
    else:
        if task not in get_modelsmap():
            get_logger().info(
                f"[{request_index}]     no available models on {task} task."
            )
            inference_result = {
                "error": f"{command['task']} not found in available tasks."
            }
            results[id] = collect_result(command, "", inference_result)
            return ModelSelectionOutput(
                id="-1",
                reason="No available models on this task.",
                hosted_on="None",
                command={},
                results={},
            )

        candidates = get_modelsmap()[task][:10]
        all_avaliable_models = get_avaliable_models(
            candidates, get_config()["num_candidate_models"]
        )
        all_avaliable_model_ids = all_avaliable_models["local"]
        get_logger().info(
            f"[{request_index}]     avaliable models on {command['task']}: {all_avaliable_models}"
        )

        if len(all_avaliable_model_ids) == 0:
            get_logger().info(f"no available models on {command['task']}")
            inference_result = {
                "error": f"no available models on {command['task']} task."
            }
            results[id] = collect_result(command, "", inference_result)
            return ModelSelectionOutput(
                id="-1",
                reason="No available models on this task.",
                hosted_on="None",
                command={},
                results={},
            )

        if len(all_avaliable_model_ids) == 1:
            best_model_id = all_avaliable_model_ids[0]
            reason = "Only one model available."
            choose = {"id": best_model_id, "reason": reason}
            get_logger().info(f"[{request_index}]     chosen model: {choose}")
        else:
            cand_models_info = [
                {
                    "id": model["id"],
                    "inference endpoint": "local",
                    "likes": model.get("likes"),
                    "description": model.get("description", "")[
                        : get_config()["max_description_length"]
                    ],
                    # "language": model.get("meta").get("language") if model.get("meta") else None,
                    "tags": (
                        model.get("meta").get("tags") if model.get("meta") else None
                    ),
                }
                for model in candidates
                if model["id"] in all_avaliable_model_ids
            ]

            choose_str, usage = choose_model(
                input.input, command, cand_models_info, context.call_id
            )
            choose_str = find_and_format_json(choose_str)
            try:
                choose = json.loads(choose_str)
            except Exception as e:
                return ModelSelectionOutput(
                    id="-1",
                    reason="Error in choose model.",
                    hosted_on="None",
                    command={},
                    results={},
                )
            best_model_id, reason = choose["id"], choose["reason"]
            get_logger().info(
                f"[{request_index}]     best_model_id: {best_model_id}, reason: {reason}"
            )
    hosted_on = get_inference_mode()

    get_logger().info(
        f"[{request_index}] ModelSelection result: best_model_id: {best_model_id}, reason: {reason}"
    )
    get_logger().info(f"[{request_index}] ModelSelection end")
    return ModelSelectionOutput(
        id=best_model_id,
        reason=reason,
        hosted_on=hosted_on,
        command=command,
        results=results,
        usage=locals().get("usage"),
    )
