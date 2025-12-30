import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

import dspy

from ..constants import (
    COMPILE_TRAINSET_KEY,
    COMPILE_VALSET_KEY,
    OPTIMIZER_METRIC_KEY,
    OPTIMIZER_NAME_GEPA,
    OPTIMIZER_NAME_MIPROV2,
    OPTIMIZER_PROMPT_MODEL_KEY,
    OPTIMIZER_REFLECTION_LM_KEY,
    OPTIMIZER_TASK_MODEL_KEY
)
from ..exceptions import ServiceError
from ..models import ModelConfig
from .data import DatasetSplits
from .language_models import build_language_model

logger = logging.getLogger(__name__)


def compile_program(
    *,
    optimizer: Any,
    program: Any,
    splits: DatasetSplits,
    metric: Optional[Any],
    compile_kwargs: Dict[str, Any],
) -> Any:
    """Run the optimizer compile step with derived datasets.

    Args:
        optimizer: DSPy optimizer instance.
        program: DSPy module to tune.
        splits: Dataset partitions generated from user data.
        metric: Optional metric callable.
        compile_kwargs: Additional arguments forwarded to ``optimizer.compile``.

    Returns:
        Any: Compiled DSPy program returned by the optimizer.

    Raises:
        ServiceError: If training data is empty or the optimizer rejects kwargs.
    """

    if not splits.train:
        raise ServiceError(
            "Training split is empty; increase the train fraction or provide more data."
        )

    kwargs = dict(compile_kwargs or {})
    if COMPILE_TRAINSET_KEY not in kwargs:
        kwargs[COMPILE_TRAINSET_KEY] = splits.train

    # Only pass valset if the optimizer's compile method accepts it
    if splits.val and _compile_accepts_valset(optimizer):
        kwargs.setdefault(COMPILE_VALSET_KEY, splits.val)

    try:
        return optimizer.compile(program, **kwargs)
    except TypeError as exc:
        raise ServiceError(
            "Optimizer.compile rejected the provided arguments; update compile_kwargs: "
            f"{exc}"
        ) from exc


def _compile_accepts_valset(optimizer: Any) -> bool:
    """Check if the optimizer's compile method accepts a valset parameter.

    Args:
        optimizer: DSPy optimizer instance.

    Returns:
        bool: True if compile() accepts valset, False otherwise.
    """
    compile_method = getattr(optimizer, "compile", None)
    if compile_method is None:
        return False
    try:
        sig = inspect.signature(compile_method)
        return COMPILE_VALSET_KEY in sig.parameters
    except (ValueError, TypeError):
        return False


def evaluate_on_test(program: Any, test_examples: List[Any], metric) -> Optional[float]:
    """Evaluate a compiled program on the test split.

    Args:
        program: Compiled DSPy module.
        test_examples: Held-out dataset for final evaluation.
        metric: Metric callable used by DSPy evaluators.

    Returns:
        Optional[float]: Numeric score if evaluation succeeds, else ``None``.

    Raises:
        ServiceError: If the evaluator returns a non-numeric result.
    """

    if not test_examples:
        return None

    evaluator = dspy.Evaluate(
        devset=test_examples,
        metric=metric,
        display_progress=True,
    )
    result = evaluator(program)
    if isinstance(result, (int, float)):
        return float(result)
    score = getattr(result, "score", None)
    if isinstance(score, (int, float)):
        return float(score)
    raise ServiceError(
        "Evaluator returned a non-numeric result; ensure the metric's score is a float."
    )


def optimizer_requires_metric(factory: Callable[..., Any]) -> bool:
    """Return True if the optimizer factory signature has a ``metric`` parameter.

    Args:
        factory: Optimizer factory callable or class constructor.

    Returns:
        bool: True when ``metric`` is present in any callable path.
    """

    try:
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        return False
    if "metric" in sig.parameters:
        return True

    if _callable_accepts_metric(factory):
        return True
    for target in _extract_factory_targets(factory):
        if _callable_accepts_metric(target):
            return True
    return False


def validate_optimizer_signature(factory: Callable[..., Any], name: str) -> None:
    """Ensure we can introspect the optimizer factory for logging.

    Args:
        factory: Optimizer factory callable.
        name: Optimizer name used in log messages.

    Returns:
        None
    """

    try:
        inspect.signature(factory)
    except (ValueError, TypeError):
        logger.warning("Unable to introspect optimizer '%s' signature.", name)


def validate_optimizer_kwargs(
    factory: Callable[..., Any], kwargs: Dict[str, Any], name: str
) -> None:
    """Ensure user-supplied optimizer kwargs match the factory signature.

    Args:
        factory: Optimizer factory callable.
        kwargs: Keyword arguments provided by the user.
        name: Optimizer name for contextual error messages.

    Returns:
        None

    Raises:
        ServiceError: If kwargs cannot be bound to the factory signature.
    """

    if not kwargs:
        return
    try:
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        return
    try:
        sig.bind_partial(**kwargs)
    except TypeError as exc:
        raise ServiceError(
            f"optimizer_kwargs contain unsupported entries for '{name}': {exc}"
        ) from exc


def instantiate_optimizer(
    factory: Callable[..., Any],
    optimizer_name: str,
    optimizer_kwargs: Dict[str, Any],
    metric: Callable[..., Any],
    default_model: ModelConfig,
    reflection_model: Optional[ModelConfig],
    prompt_model: Optional[ModelConfig],
    task_model: Optional[ModelConfig],
) -> Any:
    """Instantiate an optimizer, injecting language models and metrics as needed.

    Args:
        factory: Optimizer factory callable resolved from registry/alias.
        optimizer_name: Human-readable optimizer identifier.
        optimizer_kwargs: User-supplied keyword arguments.
        metric: Metric callable to inject when required.
        default_model: ModelConfig used as a fallback LM.
        reflection_model: Optional LM config dedicated to reflection.
        prompt_model: Optional LM config dedicated to prompt generation.
        task_model: Optional LM config dedicated to task evaluation.

    Returns:
        Any: Instantiated optimizer ready for compilation.

    Raises:
        ServiceError: If required models or kwargs are missing.
    """

    optimizer_key = optimizer_name.lower()
    reflection_required_optimizers = {OPTIMIZER_NAME_GEPA}
    dual_lm_optimizers = {OPTIMIZER_NAME_MIPROV2}
    requires_metric = optimizer_requires_metric(factory)
    if not requires_metric and optimizer_key in {OPTIMIZER_NAME_GEPA, OPTIMIZER_NAME_MIPROV2}:
        requires_metric = True

    kwargs = dict(optimizer_kwargs or {})
    if requires_metric and OPTIMIZER_METRIC_KEY not in kwargs:
        kwargs[OPTIMIZER_METRIC_KEY] = metric
    needs_reflection = optimizer_key in reflection_required_optimizers
    if OPTIMIZER_REFLECTION_LM_KEY not in kwargs:
        if reflection_model:
            kwargs[OPTIMIZER_REFLECTION_LM_KEY] = build_language_model(reflection_model)
        elif needs_reflection:
            raise ServiceError(
                f"Optimizer '{optimizer_name}' requires reflection_model_config "
                "or a preconfigured 'reflection_lm' in optimizer_kwargs."
            )
    if optimizer_key in dual_lm_optimizers:
        prompt_cfg = prompt_model or default_model
        task_cfg = task_model or default_model
        if OPTIMIZER_PROMPT_MODEL_KEY not in kwargs:
            if not prompt_cfg:
                raise ServiceError(
                    f"Optimizer '{optimizer_name}' requires prompt_model_config "
                    "or an explicit 'prompt_model' in optimizer_kwargs."
                )
            kwargs[OPTIMIZER_PROMPT_MODEL_KEY] = build_language_model(prompt_cfg)
        prompt_lm = kwargs[OPTIMIZER_PROMPT_MODEL_KEY]
        if OPTIMIZER_TASK_MODEL_KEY not in kwargs:
            if not task_cfg:
                raise ServiceError(
                    f"Optimizer '{optimizer_name}' requires task_model_config "
                    "or an explicit 'task_model' in optimizer_kwargs."
                )
            same_config = False
            if prompt_cfg and task_cfg:
                same_config = prompt_cfg.model_dump() == task_cfg.model_dump()
            if same_config:
                kwargs[OPTIMIZER_TASK_MODEL_KEY] = prompt_lm
            else:
                kwargs[OPTIMIZER_TASK_MODEL_KEY] = build_language_model(task_cfg)
    logger.debug("Creating optimizer %s with kwargs keys=%s", optimizer_name, list(kwargs.keys()))
    return factory(**kwargs)


def _callable_accepts_metric(target: Any) -> bool:
    """Return True when the callable exposes a ``metric`` parameter.

    Args:
        target: Callable object to inspect.

    Returns:
        bool: True when ``metric`` is present in the signature.
    """

    if target is None:
        return False
    try:
        sig = inspect.signature(target)
    except (ValueError, TypeError):
        return False
    return "metric" in sig.parameters


def _extract_factory_targets(factory: Callable[..., Any]) -> list[Any]:
    """Collect potential callable targets from wrappers/closures.

    Args:
        factory: Optimizer factory possibly wrapping other callables.

    Returns:
        list[Any]: Additional callables to inspect for ``metric`` support.
    """

    targets: list[Any] = []
    wrapped = getattr(factory, "__wrapped__", None)
    if wrapped is not None:
        targets.append(wrapped)
    closure_cells = getattr(factory, "__closure__", None)
    if closure_cells:
        for cell in closure_cells:
            targets.append(cell.cell_contents)
    targets.append(getattr(factory, "__call__", None))
    return targets
