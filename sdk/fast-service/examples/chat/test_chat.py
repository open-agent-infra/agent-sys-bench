import requests

if __name__ == "__main__":
    model = "google/gemma-2-2b-it"
    x = requests.post(
        "http://127.0.0.1:7070/azure_openai.openai_generate",
        json={
            "model": model,
            "system_prompt": "You are a helpful assistant.",
            "prompts": ["Hello! What's your name?"],
        },
    )
    print(f"x => {x.json()}")

    y = requests.post(
        "http://127.0.0.1:7080/chat_agent.chat_agent_generate",
        json={
            "model": model,
            "prompts": ["Hello!"],
        },
    )
    print(f"y => {y.json()}")
