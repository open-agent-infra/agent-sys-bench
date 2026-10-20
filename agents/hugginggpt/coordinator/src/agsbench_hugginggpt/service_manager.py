from typing import Optional, Any
from pydantic import BaseModel
from fast_service import FastServiceManager

hugginggpt_fsm = FastServiceManager()


class TaskPlanningInput(BaseModel):
    history: list[dict]
    question: str
    collection_name: Optional[str] = None
    context_limit: int = 3
    tasks_to_plan: Optional[list[int]] = None


class TaskPlanningOutput(BaseModel):
    task_str: Optional[str] = None
    query_context: str
    usage: Optional[dict] = None


class TaskExecutionInput(BaseModel):
    input: str
    task_str: str
    messages: list[dict]


class TaskExecutionOutput(BaseModel):
    results: dict


class ModelSelectionInput(BaseModel):
    input: str
    command: dict
    results: dict


class ModelSelectionOutput(BaseModel):
    id: str
    reason: str
    hosted_on: str
    command: dict
    results: dict
    usage: Optional[dict] = None


class ModelInferenceInput(BaseModel):
    best_model_id: str
    hosted_on: str
    input: str
    command: dict


class ModelInferenceOutput(BaseModel):
    result: dict


class ResponseGenerationInput(BaseModel):
    input: str
    results: dict


class ResponseGenerationOutput(BaseModel):
    result: str
    usage: Optional[dict] = None


class HuggingGPTE2EInput(BaseModel):
    collection_name: str
    question: str
    context_limit: int = 3
    tasks_to_plan: Optional[list[int]] = None


class HuggingGPTE2EOutput(BaseModel):
    answer: str
