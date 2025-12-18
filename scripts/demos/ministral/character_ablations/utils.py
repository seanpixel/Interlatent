import os


def generate(*, prompt: str, api_key: str | None = None, model: str = "mistral-small-2506") -> str:
    """
    Minimal wrapper around the Mistral chat API.

    Designed to be import-safe when MISTRAL_API_KEY is unset so callers can
    implement fallbacks (e.g., templated rewrites).
    """
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")

    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    chat_response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return chat_response.choices[0].message.content
