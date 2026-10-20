import requests
import os
from fast_service import RequestContext
from .service_manager import hugginggpt_fsm, ModelInferenceInput, ModelInferenceOutput
from .utils import call_llm, get_logger, get_models_server_url


def local_model_inference(model_id, input, command, call_id: str = None):
    data = command["args"]
    task = command["task"]
    task_url = f"{get_models_server_url()}/models/{model_id}"
    if not os.path.exists(f"{os.environ.get("HUGGINGGPT_HOME")}/public/images"):
        os.makedirs(f"{os.environ.get("HUGGINGGPT_HOME")}/public/images")

    if model_id == "LLM":
        messages = [
            {
                "role": "user",
                "content": f"[ {input} ] contains a task in JSON format {command}. Now you are a {command['task']} system, the arguments are {command['args']}. Just help me do {command['task']} and give me the result. The result must be in text form without any urls.",
            }
        ]
        messages.insert(
            0,
            {
                "role": "system",
                "content": "I am a  LLM model, I can help you do some NLP tasks.",
            },
        )
        response, usage = call_llm(messages, "model_inference", call_id=call_id)
        results = {"response": response, "usage": usage}
        return results

    if task == "image-segmentation":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()

    if task == "image-to-image":
        img_url = data["image"]
        text = None
        if "text" in data:
            text = data["text"]
        response = requests.post(task_url, json={"img_url": img_url, "text": text})
        results = response.json()

    if task == "text-to-image":
        text = data["text"]
        response = requests.post(task_url, json={"text": text})
        results = response.json()

    if task == "object-detection":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()

    if task in [
        "image-classification",
        "image-to-text",
        "visual-question-answering",
    ]:
        img_url = data["image"]
        text = None
        if "text" in data:
            text = data["text"]
        response = requests.post(task_url, json={"img_url": img_url, "text": text})
        results = {}
        results["results"] = response.json()
    return results


@hugginggpt_fsm.fast_service
def model_inference(
    model_inference_input: ModelInferenceInput, context: RequestContext = None
) -> ModelInferenceOutput:
    request_index = context.request_id
    get_logger().info(f"[{request_index}] ModelInference start")

    best_model_id = model_inference_input.best_model_id
    hosted_on = model_inference_input.hosted_on
    input = model_inference_input.input
    command = model_inference_input.command
    inference_result = {}
    import traceback

    try:
        get_logger().info(
            f"[{request_index}]      ModelInference input: {best_model_id}, {input}, {command}"
        )
        inference_result = local_model_inference(
            best_model_id, input, command, context.call_id
        )
        if "error" in inference_result:
            return ModelInferenceOutput(result={"error": str(inference_result)})
    except Exception as e:
        tb = traceback.format_exc()
        get_logger().error(
            f"[{request_index}] Error in model inference: {str(e)}\n{tb}"
        )
        return ModelInferenceOutput(result={"error": str(e)})

    get_logger().info(f"[{request_index}] ModelInference result: {inference_result}")
    get_logger().info(f"[{request_index}] ModelInference end")
    return ModelInferenceOutput(result=inference_result)
