import json
from fast_service import RequestContext
from .service_manager import (
    hugginggpt_fsm,
    TaskPlanningInput,
    TaskPlanningOutput,
)
from .utils import (
    get_config,
    get_prestep_config,
    replace_slot,
    call_llm,
    get_tasks_to_plan,
    get_all_tasks,
)
from .utils import get_logger


def check_format(task_str: str):
    try:
        tasks = json.loads(task_str)
    except Exception as e:
        return False

    for task in tasks:
        if "args" not in task:
            return False
    return True


def check_tasks_not_in_plan(task_str: str, tasks_to_plan: list[str]):
    tasks = json.loads(task_str)

    for task in tasks:
        if task["task"] not in tasks_to_plan:
            return True
    return False


@hugginggpt_fsm.fast_service
def task_planning(
    input: TaskPlanningInput,
    context: RequestContext = None,
) -> TaskPlanningOutput:
    request_index = context.request_id

    get_logger().info(f"[{request_index}] TaskPlanning start")

    parse_task_presteps = open(get_prestep_config("parse_task"), "r").read()
    parse_task_prompt = get_config()["prompt"]["parse_task"]
    parse_task_tprompt = get_config()["tprompt"]["parse_task"]
    tasks_to_plan = get_tasks_to_plan(input.tasks_to_plan)
    tasks_to_plan_str = ", ".join(f'"{task}"' for task in tasks_to_plan.values())
    parse_task_tprompt = parse_task_tprompt.replace("{{}}", tasks_to_plan_str)

    messages = json.loads(parse_task_presteps)
    messages.insert(0, {"role": "system", "content": parse_task_tprompt})

    prompt = replace_slot(
        parse_task_prompt, {"input": input.question, "context": input.history}
    )
    messages.append({"role": "user", "content": prompt})

    task_str, usage = call_llm(messages, "task_planning", context.call_id)

    get_logger().info(f"[{request_index}]     task_str: {task_str}")

    if not check_format(task_str):
        get_logger().error(
            f"[{request_index}] Error: The task_str is not in the correct format."
        )
        get_logger().error(f"[{request_index}] task_str: {task_str}")
        return TaskPlanningOutput(task_str=None, query_context=input.question)

    if check_tasks_not_in_plan(task_str, tasks_to_plan.values()):
        get_logger().error(
            f"[{request_index}] Error: The task_str contains tasks not in the plan."
        )
        get_logger().error(f"[{request_index}] task_str: {task_str}")
        return TaskPlanningOutput(task_str=None, query_context=input.question)

    get_logger().info(f"[{request_index}] TaskPlanning done")
    return TaskPlanningOutput(
        task_str=task_str,
        query_context=input.question,
        usage=usage,
    )
