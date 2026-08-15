import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _normalize_base_url(url: str) -> str:
    base_url = url.strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    return base_url


def get_llm(temperature: float = 0.0, model: str = "gpt-4o-mini") -> ChatOpenAI:
    nvidia_key = os.getenv("NVIDIA_OPENAI_API_KEY", "").strip()
    nvidia_model = os.getenv("NVIDIA_OPENAI_MODEL", "").strip()
    nvidia_base_raw = (
        os.getenv("NVIDIA_OPENAI_BASE_URL", "").strip()
        or os.getenv("NVIDIA_OPENAI_API_URL", "").strip()
    )

    if any([nvidia_key, nvidia_model, nvidia_base_raw]):
        if not all([nvidia_key, nvidia_model, nvidia_base_raw]):
            raise RuntimeError(
                "NVIDIA OpenAI-compatible config is incomplete. "
                "Set NVIDIA_OPENAI_API_KEY, NVIDIA_OPENAI_MODEL, and "
                "NVIDIA_OPENAI_BASE_URL (or NVIDIA_OPENAI_API_URL)."
            )

        return ChatOpenAI(
            temperature=temperature,
            model=nvidia_model,
            api_key=nvidia_key,
            base_url=_normalize_base_url(nvidia_base_raw),
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "No model credentials found. Configure either OPENAI_API_KEY, "
            "or NVIDIA_OPENAI_API_KEY + NVIDIA_OPENAI_MODEL + "
            "NVIDIA_OPENAI_BASE_URL/NVIDIA_OPENAI_API_URL."
        )

    return ChatOpenAI(
        temperature=temperature,
        model=model,
        api_key=api_key,
    )
