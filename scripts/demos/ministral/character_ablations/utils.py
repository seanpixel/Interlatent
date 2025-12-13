import os
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-2506"

def generate(api_key: str = api_key, model: str = model, prompt: str ="hello") -> str:
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "mistral-small-2506"

    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model = model,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    return chat_response.choices[0].message.content
