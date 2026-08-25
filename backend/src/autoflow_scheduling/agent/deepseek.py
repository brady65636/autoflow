from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def create_deepseek_model(
    *,
    api_key: str | None = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Create a DeepSeek OpenAI-compatible chat model in non-thinking mode."""
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is required")
    return ChatOpenAI(
        model=model,
        api_key=key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}},
    )
