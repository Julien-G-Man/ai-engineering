import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())


def _normalize_base_url(url: str) -> str:
    base_url = url.strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    return base_url


def _build_client_and_default_model() -> tuple[OpenAI, str]:
    nvidia_key = os.getenv("NVIDIA_OPENAI_API_KEY", "").strip()
    nvidia_model = os.getenv("NVIDIA_OPENAI_MODEL", "").strip()
    nvidia_base_raw = os.getenv("NVIDIA_OPENAI_BASE_URL", "").strip() or os.getenv("NVIDIA_OPENAI_API_URL", "").strip()

    if any([nvidia_key, nvidia_model, nvidia_base_raw]):
        if not all([nvidia_key, nvidia_model, nvidia_base_raw]):
            raise RuntimeError(
                "NVIDIA config is incomplete. Set NVIDIA_OPENAI_API_KEY, "
                "NVIDIA_OPENAI_MODEL, and NVIDIA_OPENAI_BASE_URL "
                "(or NVIDIA_OPENAI_API_URL)."
            )
        client = OpenAI(
            api_key=nvidia_key,
            base_url=_normalize_base_url(nvidia_base_raw),
        )
        return client, nvidia_model

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise RuntimeError(
            "No API credentials found. Configure OPENAI_API_KEY, or "
            "NVIDIA_OPENAI_API_KEY + NVIDIA_OPENAI_MODEL + "
            "NVIDIA_OPENAI_BASE_URL/NVIDIA_OPENAI_API_URL."
        )

    return OpenAI(api_key=openai_key), "gpt-4o-mini"


client, DEFAULT_MODEL = _build_client_and_default_model()


def get_completions(prompt_or_messages, model=None, temperature=0.2):
    model_name = model or DEFAULT_MODEL

    # Backward compatible input handling:
    # - string prompt -> single user message
    # - list[dict] chat history -> passed through directly
    if isinstance(prompt_or_messages, list):
        messages = prompt_or_messages
    else:
        messages = [{"role": "user", "content": str(prompt_or_messages)}]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content
    
