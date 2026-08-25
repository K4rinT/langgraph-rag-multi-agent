from __future__ import annotations
import os
import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

logger = logging.getLogger(__name__)

REQUIRED_VARS = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY")


def _optional_temperature() -> dict:
    """
    Return temperature kwargs, or {}
    Leave it unset (gpt-5 reject any temperature other than the default)
    """
    raw = os.getenv("LLM_TEMPERATURE")
    if not raw:
        return {}
    try:
        return {"temperature": float(raw)}
    except ValueError:
        raise RuntimeError(f"LLM_TEMPERATURE must be a number, got {raw!r}") from None


def build_llm(**kwargs) -> BaseChatModel:
    """ 
    Create an Azure OpenAI chat client
    """
    missing = [name for name in REQUIRED_VARS if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    options = {**_optional_temperature(), **kwargs}

    if endpoint.endswith("/responses"):
        base_url = endpoint.removesuffix("/responses")
        logger.info("Responses API gateway: %s (model %s)", base_url, deployment)

        return ChatOpenAI(
            base_url = base_url,
            api_key = api_key,
            model = deployment,
            use_responses_api = True,
            default_headers = {"api-key": api_key},
            **options
        )
    logger.info("Azure OpenAI deployment: %s", deployment)

    return AzureChatOpenAI(
        azure_endpoint = endpoint,
        api_key = api_key,
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_deployment = deployment,
        **options
    )