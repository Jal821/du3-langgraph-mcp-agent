"""Where the language model comes from.

One place, so that swapping the endpoint is a change to .env rather than a
change to the agent. Any OpenAI-compatible endpoint works: a LiteLLM proxy, an
Ollama or LM Studio server, or OpenAI itself.

temperature=0 because this agent quotes prices. Two identical questions should
select the same modules, and a creative tool-selection loop is a bug here, not
a feature.
"""

import os
import sys

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "auto/best-fast"


def describe() -> str:
    """One line naming what will actually answer, printed before every run.

    Worth the four lines of code: with a proxy in the path, "which model am I
    talking to" is otherwise invisible, and a wrong answer from an unexpected
    model looks like a bug in the agent.
    """
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("OPENAI_BASE_URL")
    where = f" at {base_url}" if base_url else " at api.openai.com"
    return f"MODEL: {model}{where}"


def get_model(**kwargs) -> ChatOpenAI:
    """The chat model the agent runs on.

    Reads OPENAI_API_KEY, and OPENAI_BASE_URL if the endpoint is not OpenAI
    itself. A missing key exits with the fix rather than with a traceback from
    somewhere inside the SDK.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and fill it in.\n"
            "Any OpenAI-compatible endpoint works; set OPENAI_BASE_URL for a local one."
        )

    settings = {
        "model": os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        "api_key": api_key,
        "temperature": 0,
        # A tool-calling loop over five tools is not fast; the default timeout is
        # tight enough that a slow first call looks like a hang.
        "timeout": 120,
        "max_retries": 2,
    }

    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        settings["base_url"] = base_url

    settings.update(kwargs)
    return ChatOpenAI(**settings)
