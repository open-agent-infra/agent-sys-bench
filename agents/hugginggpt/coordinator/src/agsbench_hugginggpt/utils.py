import os
import io
import re
import yaml
import json
from functools import lru_cache
import logging

from agent_commons.llm import LLMClient, LLMResponse, LLMClientFactory


GLOBAL_REQUEST_INDEX = 0


@lru_cache
def get_settings():
    env_file = os.path.join(
        os.path.dirname(__file__),
        f"{os.environ.get("HUGGINGGPT_HOME")}/config/settings.yml",
    )
    with open(env_file, "r", encoding="utf-8") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data


@lru_cache
def get_logger():
    hugginggpt_logger = logging.getLogger(name="hugginggpt")
    hugginggpt_logger.setLevel(logging.DEBUG)
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    fh = logging.FileHandler(filename=".cache/debug.log")
    fh.setLevel(logging.DEBUG)
    hugginggpt_logger.addHandler(fh)
    return hugginggpt_logger


@lru_cache
def get_config():
    config_path = f"{os.environ.get("HUGGINGGPT_HOME")}/config/hugginggpt_config.yml"
    _CONFIG = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    return _CONFIG


def get_prestep_config(step: str):
    return os.path.join(
        os.environ.get("HUGGINGGPT_HOME"), get_config()["demos_or_presteps"][step]
    )


def get_all_tasks():
    tasks = {
        0: "token-classification",
        1: "text2text-generation",
        2: "summarization",
        3: "translation",
        4: "question-answering",
        5: "conversational",
        6: "text-generation",
        7: "sentence-similarity",
        16: "document-question-answering",
        # 8: "tabular-classification",
        9: "object-detection",
        10: "image-classification",
        11: "image-to-image",
        # 12: "image-to-text",
        # 13: "text-to-image",
        # 14: "text-to-video",
        # 15: "visual-question-answering",
        17: "image-segmentation",
        12: "image-to-text",
        13: "text-to-image",
        # 14: "text-to-video",
        15: "visual-question-answering",
        # 18: "depth-estimation",
        # 19: "text-to-speech",
        # 20: "automatic-speech-recognition",
        # 21: "audio-to-audio",
        # 22: "audio-classification",
        # 23: "canny-control",
        # 24: "hed-control",
        # 25: "mlsd-control",
        # 26: "normal-control",
        # 27: "openpose-control",
        # 28: "canny-text-to-image",
        # 29: "depth-text-to-image",
        # 30: "hed-text-to-image",
        # 31: "mlsd-text-to-image",
        # 32: "normal-text-to-image",
        # 33: "openpose-text-to-image",
        # 34: "seg-text-to-image",
    }
    return tasks


def get_tasks_to_plan(tasks_to_plan: list[int] = None):
    tasks = get_all_tasks()
    if tasks_to_plan:
        tasks = {i: tasks[i] for i in tasks_to_plan}
    return tasks


@lru_cache
def get_huggingface_token():
    if not os.environ.get("HUGGINGFACE_ACCESS_TOKEN"):
        raise ValueError(
            "Huggingface token must be provided either as an argument or in the environment variable HUGGINGFACE_ACCESS_TOKEN"
        )
    return os.environ.get("HUGGINGFACE_ACCESS_TOKEN")


@lru_cache
def get_modelsmap():
    models_path = os.path.join(
        os.path.dirname(__file__),
        f"{os.environ.get("HUGGINGGPT_HOME")}/config/p0_models.jsonl",
    )
    with open(models_path, "r") as file:
        models = [json.loads(line) for line in file.readlines()]
    _ModelsMap = {}
    for model in models:
        if "task" not in model:
            model["task"] = model["pipeline_tag"]
        tag = model["task"]
        if tag not in _ModelsMap:
            _ModelsMap[tag] = []
        _ModelsMap[tag].append(model)
    return _ModelsMap


@lru_cache
def get_llm_client(task_type: str) -> LLMClient:
    if task_type == "task_planning":
        api_key = get_settings()["openai_api_key_task_planning"]
        base_url = get_settings()["openai_base_url_task_planning"]
    elif task_type == "model_selection":
        api_key = get_settings()["openai_api_key_model_selection"]
        base_url = get_settings()["openai_base_url_model_selection"]
    elif task_type == "model_inference":
        api_key = get_settings()["openai_api_key_model_inference"]
        base_url = get_settings()["openai_base_url_model_inference"]
    elif task_type == "response_generation":
        api_key = get_settings()["openai_api_key_response_generation"]
        base_url = get_settings()["openai_base_url_response_generation"]
    else:
        raise ValueError("Invalid task type provided")
    return LLMClientFactory.create_client(api_key=api_key, base_url=base_url)


@lru_cache
def get_inference_mode():
    return get_settings()["inference_mode"]


@lru_cache
def get_models_server_url():
    return get_settings()["models_server_url"]


def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace(
            "{{" + key + "}}", value.replace('"', "'").replace("\n", "")
        )
    return text


def format_double_quotes(input_str):
    input_str = input_str.replace('"reason"', "'reason'").replace('"id"', "'id'")
    input_str = input_str.replace('"', "")
    input_str = input_str.replace("'reason':", "'reason':\"")
    input_str = input_str.replace("'id':", "'id':\"")
    input_str = re.sub(r"(})", r'"\1', input_str, 1)
    index = input_str.find("'reason'")
    if index != -1:
        comma_index = input_str.rfind(",", 0, index)
        if comma_index != -1:
            input_str = input_str[:comma_index] + '"' + input_str[comma_index:]
    input_str = input_str.replace("'reason'", '"reason"').replace("'id'", '"id"')
    return input_str


def find_and_format_json(s):
    s = s.replace("'", "'")
    s = format_double_quotes(s)
    start = s.find("{")
    end = s.rfind("}")
    res = s[start : end + 1]
    res = res.replace("\n", "")
    return res


def jsonify_str(task_str: str):
    if "[" in task_str and "]" in task_str:
        task_str = "[" + task_str[task_str.index("[") + 1 : task_str.rindex("]")] + "]"
    elif "[" in task_str:
        task_str = "[" + task_str[task_str.index("[") + 1 :] + "]"
    if task_str.endswith(",]"):
        task_str = task_str[:-2] + "]"
    return task_str


def call_llm(messages: list[dict], task_type: str, call_id: str = None):
    if task_type == "task_planning":
        model = get_settings()["llm_model_task_planning"]
    elif task_type == "model_selection":
        model = get_settings()["llm_model_model_selection"]
    elif task_type == "model_inference":
        model = get_settings()["llm_model_model_inference"]
    elif task_type == "response_generation":
        model = get_settings()["llm_model_response_generation"]
    else:
        raise ValueError("Invalid task type provided")

    chat_completions: LLMResponse = get_llm_client(task_type).call_llm(
        model=model,
        messages=messages,
        call_id=call_id,
    )
    response = chat_completions.content
    usage = {
        "prompt_tokens": chat_completions.prompt_tokens,
        "completion_tokens": chat_completions.completion_tokens,
    }
    return response, usage


def collect_result(command, choose, inference_result):
    result = {"task": command}
    result["inference result"] = inference_result
    result["choose model result"] = choose
    get_logger().info(f"inference result: {inference_result}")
    return result


def get_global_request_index():
    global GLOBAL_REQUEST_INDEX
    return GLOBAL_REQUEST_INDEX


def set_global_request_index(index):
    global GLOBAL_REQUEST_INDEX
    GLOBAL_REQUEST_INDEX = index
