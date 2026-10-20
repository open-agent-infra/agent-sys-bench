import time
from typing import List
from pydantic import BaseModel
from openai import OpenAI
from openai.types.chat import ChatCompletion
from fast_service import fast_service_manager


llm_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)


class OpenAIInput(BaseModel):
    model: str
    system_prompt: str
    prompts: List[str]


@fast_service_manager.fast_service
def openai_generate(item: OpenAIInput) -> ChatCompletion:
    model = item.model
    system_prompt = item.system_prompt
    prompts = item.prompts
    print(f"get inputs: {model}, {prompts}")
    st = time.time()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "".join(prompts)},
    ]
    if model.startswith("google/gemma-2-2b-it"):
        if messages[0]["role"] == "system":
            messages[1]["content"] = (
                messages[0]["content"] + " " + messages[1]["content"]
            )
            messages = messages[1:]
    completion = llm_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    et = time.time()

    print(completion.choices[0].message)
    print(f"timecost: {et-st} seconds")
    return completion


if __name__ == "__main__":
    print(f"start to launch server")
    fast_service_manager.execute()
    print(f"server stoped")
