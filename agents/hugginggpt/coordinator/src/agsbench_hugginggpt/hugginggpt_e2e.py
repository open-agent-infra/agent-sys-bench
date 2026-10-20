import json
import copy
import re
from typing import Optional
from fast_service import RequestContext
from .service_manager import (
    hugginggpt_fsm,
    TaskPlanningInput,
    TaskPlanningOutput,
    ModelSelectionInput,
    ModelSelectionOutput,
    ModelInferenceInput,
    ModelInferenceOutput,
    TaskExecutionInput,
    TaskExecutionOutput,
    ResponseGenerationInput,
    ResponseGenerationOutput,
    HuggingGPTE2EInput,
    HuggingGPTE2EOutput,
)
from .task_planning import task_planning
from .model_selection import model_selection
from .model_inference import model_inference
from .response_generation import response_generation
from .utils import (
    get_logger,
    jsonify_str,
    collect_result,
    get_global_request_index,
    set_global_request_index,
)

total_requests = 0


def unfold(tasks):
    try:
        for task in tasks:
            for key, value in task["args"].items():
                if isinstance(value, str) and "<GENERATED>" in value:
                    generated_items = value.split(",")
                    if len(generated_items) > 1:
                        for item in generated_items:
                            new_task = copy.deepcopy(task)
                            dep_task_id = int(item.split("-")[1])
                            new_task["dep"] = [dep_task_id]
                            new_task["args"][key] = item
                            tasks.append(new_task)
                        tasks.remove(task)
    except Exception as e:
        get_logger().error(f"Error: {str(e)}")
        return None
    return tasks


# "<GENERATED>-1" (as in the planning prompt) or "1 <GENERATED>"; the
# reference may be embedded in free text ("Describe: <GENERATED>-0").
GENERATED_REF = re.compile(r"<GENERATED>\s*-\s*(\d+)|(\d+)\s*<GENERATED>")


def fix_dep(tasks):
    for task in tasks:
        task["dep"] = []
        for k, v in task["args"].items():
            if not isinstance(v, str) or "<GENERATED>" not in v:
                continue
            get_logger().info(f"    v: {v}")
            refs = GENERATED_REF.findall(v)
            if not refs:
                get_logger().error(f"Error: unparsable <GENERATED> reference: {v}")
                return None
            for after, before in refs:
                dep_task_id = int(after or before)
                if dep_task_id not in task["dep"]:
                    task["dep"].append(dep_task_id)
        if len(task["dep"]) == 0:
            task["dep"] = [-1]
    return tasks


def task_execution(
    task_execution_input: TaskExecutionInput,
    context: RequestContext = None,
) -> TaskExecutionOutput:
    request_index = context.request_id
    get_logger().info(f"[{request_index}] TaskExecution start")

    task_str = task_execution_input.task_str
    tasks = json.loads(task_str)

    get_logger().info(f"[{request_index}]     tasks: {tasks}")
    tasks = unfold(tasks)
    if tasks is None:
        return TaskExecutionOutput(results={"model_selection": "error"})
    tasks = fix_dep(tasks)
    if tasks is None:
        return TaskExecutionOutput(results={"model_selection": "error"})
    get_logger().info(f"[{request_index}]     after fix_dep, tasks: {tasks}")

    tasks = tasks[:]
    result_dict = dict()
    for task in tasks:
        for dep_id in task["dep"]:
            if dep_id >= task["id"]:
                task["dep"] = [-1]
                break
        dep = task["dep"]
        if dep[0] == -1 or len(list(set(dep).intersection(result_dict.keys()))) == len(
            dep
        ):
            selection_output: ModelSelectionOutput = model_selection(
                ModelSelectionInput(
                    input=task_execution_input.task_str,
                    command=task,
                    results=result_dict,
                ),
                context=context,
            )
            result_dict = selection_output.results

            if selection_output.id == "-1":
                return TaskExecutionOutput(results={"model_selection": "error"})

            inference_output: ModelInferenceOutput = model_inference(
                ModelInferenceInput(
                    best_model_id=selection_output.id,
                    hosted_on=selection_output.hosted_on,
                    input=task_execution_input.input,
                    command=selection_output.command,
                ),
                context=context,
            )
            if "error" in inference_output.result:
                return TaskExecutionOutput(results={"model_inference": "error"})

            result_dict[task["id"]] = collect_result(task, "", inference_output.result)

    results = result_dict.copy()

    get_logger().info(f"results: {results}")

    return TaskExecutionOutput(
        results=results,
    )


@hugginggpt_fsm.fast_service
def hugginggpt_e2e(
    hugginggpt_e2e_input: HuggingGPTE2EInput,
    context: RequestContext = None,
) -> HuggingGPTE2EOutput:

    request_index = context.request_id

    collection_name = hugginggpt_e2e_input.collection_name
    question = hugginggpt_e2e_input.question
    context_limit = hugginggpt_e2e_input.context_limit

    messages = [{"role": "user", "content": question}]

    get_logger().info(f"[{request_index}] ********************************************")
    get_logger().info(f"[{request_index}] messages: {messages}")

    execution_plan: TaskPlanningOutput = task_planning(
        TaskPlanningInput(
            history=[],
            question=question,
            collection_name=collection_name,
            context_limit=context_limit,
            tasks_to_plan=hugginggpt_e2e_input.tasks_to_plan,
        ),
        context=context,
    )
    if execution_plan.task_str is None:
        return HuggingGPTE2EOutput(
            answer="[ERROR] I'm sorry, there's a problem with the task planning. Please try again."
        )

    execution_output: TaskExecutionOutput = task_execution(
        TaskExecutionInput(
            input=question, task_str=execution_plan.task_str, messages=messages
        ),
        context=context,
    )
    if "model_selection" in execution_output.results:
        return HuggingGPTE2EOutput(
            answer="[ERROR] I'm sorry, there's a problem with the model selection. Please try again."
        )
    if "model_inference" in execution_output.results:
        return HuggingGPTE2EOutput(
            answer="[ERROR] I'm sorry, there's a problem with the model inference. Please try again."
        )

    e2e_response: ResponseGenerationOutput = response_generation(
        ResponseGenerationInput(input=question, results=execution_output.results),
        context=context,
    )

    return HuggingGPTE2EOutput(answer=e2e_response.result)
