import json
from fast_service import RequestContext
from .service_manager import (
    hugginggpt_fsm,
    ResponseGenerationInput,
    ResponseGenerationOutput,
)
from .utils import get_config, get_prestep_config, replace_slot, call_llm, get_logger


@hugginggpt_fsm.fast_service
def response_generation(
    input: ResponseGenerationInput,
    context: RequestContext = None,
) -> ResponseGenerationOutput:
    request_index = context.request_id
    get_logger().info(f"[{request_index}] ResponseGeneration start")

    response_results_demos_or_presteps = open(
        get_prestep_config("response_results"), "r"
    ).read()
    response_results_prompt = get_config()["prompt"]["response_results"]
    response_results_tprompt = get_config()["tprompt"]["response_results"]

    results = input.results
    input = input.input
    results = [v for k, v in sorted(results.items(), key=lambda item: item[0])]
    prompt = replace_slot(
        response_results_prompt,
        {
            "input": input,
        },
    )
    messages = json.loads(response_results_demos_or_presteps)
    messages[0]["content"] = input
    messages[1]["content"] = str(results)
    messages.insert(0, {"role": "system", "content": response_results_tprompt})
    messages.append({"role": "user", "content": prompt})

    response, usage = call_llm(messages, "response_generation", context.call_id)

    get_logger().info(f"[{request_index}] ResponseGeneration result: {response}")
    get_logger().info(f"[{request_index}] ResponseGeneration end")
    return ResponseGenerationOutput(
        result=response,
        usage=usage,
    )
