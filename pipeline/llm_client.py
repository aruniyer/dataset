"""Unified LLM client using LiteLLM for routing and cost tracking."""

from dataclasses import dataclass, field

import litellm

DEFAULT_MODEL = "gpt-5.2"


class LLMError(Exception):
    """Unified error for LLM API failures."""
    pass


@dataclass
class LLMUsage:
    """Token usage and cost for an LLM call (or accumulated across calls)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class LLMResponse:
    """Response from an LLM call, including text, usage, and model info."""
    text: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str = ""


def _is_openai_model(model: str) -> bool:
    return any(model.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-"))


def chat(messages: list[dict], model: str = DEFAULT_MODEL, max_tokens: int = 4096) -> LLMResponse:
    """Send messages to an LLM and return the response with usage/cost info.

    Routes to the appropriate provider via LiteLLM.
    Requires OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment.
    """
    # LiteLLM uses max_tokens for Anthropic, max_completion_tokens for OpenAI
    kwargs: dict = {"model": model, "messages": messages}
    if _is_openai_model(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    try:
        response = litellm.completion(**kwargs)
    except litellm.exceptions.APIError as e:
        raise LLMError(str(e)) from e
    except Exception as e:
        raise LLMError(str(e)) from e

    text = response.choices[0].message.content or ""

    # Extract usage
    usage = LLMUsage()
    if response.usage:
        usage.prompt_tokens = response.usage.prompt_tokens or 0
        usage.completion_tokens = response.usage.completion_tokens or 0
        usage.total_tokens = response.usage.total_tokens or 0

    # Extract cost from LiteLLM's hidden params
    try:
        cost = response._hidden_params.get("response_cost", 0.0)
        if cost is not None:
            usage.cost_usd = float(cost)
    except Exception:
        # If cost extraction fails, try completion_cost as fallback
        try:
            usage.cost_usd = float(litellm.completion_cost(completion_response=response))
        except Exception:
            pass

    return LLMResponse(text=text, usage=usage, model=model)
