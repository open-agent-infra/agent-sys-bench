import time
from typing import List
from pydantic import BaseModel
from openai.types.chat import ChatCompletion
from azure_openai import OpenAIInput, openai_generate
from fast_service import fast_service_manager


class ChatAgentInput(BaseModel):
    model: str
    prompts: List[str]


class ChatAgentOutput(BaseModel):
    response: str


chat_agent_prompt = "You are a helpful assistant."


@fast_service_manager.fast_service
def chat_agent_generate(item: ChatAgentInput) -> ChatAgentOutput:
    print(f"chat_agent_generate start with {item} ... ")
    st = time.time()
    llm_input = OpenAIInput(**item.model_dump(), system_prompt=chat_agent_prompt)
    print(f"llm_input: {llm_input.model_dump_json()}")
    completion: ChatCompletion = openai_generate(item=llm_input)
    response = completion.choices[0].message.content
    et = time.time()
    print(f"timecost: {et - st}")
    return ChatAgentOutput(response=response)
