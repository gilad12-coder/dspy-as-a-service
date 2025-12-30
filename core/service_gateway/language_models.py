from typing import Dict
import dspy
from ..exceptions import ServiceError
from ..models import ModelConfig


def build_language_model(config: ModelConfig, *, configure_global: bool = False) -> dspy.LM:
    """Construct a DSPy language model and optionally configure the global runtime.

    Args:
        config: Declarative language-model configuration.
        configure_global: When True, call ``dspy.configure`` with the created LM.

    Returns:
        dspy.LM: Configured language model ready for use by optimizers.

    Raises:
        ServiceError: If DSPy refuses the settings (e.g., unsupported provider).
    """

    model_name = config.name.strip("/")
    lm_kwargs: Dict[str, object] = {
        "model": model_name,
        "temperature": config.temperature,
    }
    if config.base_url:
        lm_kwargs["base_url"] = config.base_url
    if config.max_tokens is not None:
        lm_kwargs["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        lm_kwargs["top_p"] = config.top_p
    lm_kwargs.update(config.extra)
    try:
        language_model = dspy.LM(**lm_kwargs)
    except ValueError as exc:
        raise ServiceError(f"Failed to build language model '{config.name}': {exc}") from exc

    if configure_global:
        dspy.configure(lm=language_model)
    return language_model
