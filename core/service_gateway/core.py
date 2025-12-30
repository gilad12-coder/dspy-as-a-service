import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..constants import (
    DETAIL_BASELINE,
    DETAIL_OPTIMIZED,
    DETAIL_TEST,
    DETAIL_TRAIN,
    DETAIL_VAL,
    META_COMPILE_KWARGS,
    META_MODEL_IDENTIFIER,
    META_MODULE_KWARGS,
    META_OPTIMIZER,
    META_OPTIMIZER_KWARGS,
    PROGRESS_BASELINE,
    PROGRESS_OPTIMIZED,
    PROGRESS_SPLITS_READY,
)
from ..exceptions import ServiceError
from ..models import ColumnMapping, RunRequest, RunResponse, SplitCounts
from ..progress import capture_tqdm
from ..registry import ServiceRegistry, UnknownRegistrationError
from ..resolvers import ResolverError, resolve_module_factory, resolve_optimizer_factory
from .artifacts import persist_program
from .data import (
    extract_signature_fields,
    load_metric_from_code,
    load_signature_from_code,
    rows_to_examples,
    split_examples,
)
from .language_models import build_language_model
from .optimizers import (
    compile_program,
    evaluate_on_test,
    instantiate_optimizer,
    validate_optimizer_kwargs,
    validate_optimizer_signature,
)

logger = logging.getLogger(__name__)


class DspyService:
    """High-level coordinator between FastAPI payloads and DSPy runtimes."""

    def __init__(
        self,
        registry: ServiceRegistry,
        default_seed: Optional[int] = None,
        artifacts_root: str | Path = "artifacts",
    ):
        """Create a new DspyService instance.

        Args:
            registry: ServiceRegistry containing registered modules and optimizers.
            default_seed: Default random seed for reproducibility.
            artifacts_root: Directory path for storing program artifacts.
        """
        self.registry = registry
        self.default_seed = default_seed
        self.artifacts_root = Path(artifacts_root)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root = self.artifacts_root.resolve()

    def run(
        self,
        payload: RunRequest,
        *,
        artifact_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> RunResponse:
        """Execute an optimization workflow for a single request.

        Args:
            payload: RunRequest containing module, optimizer, dataset, and config.
            artifact_id: Optional identifier for the saved artifact.
            progress_callback: Optional callback invoked with progress events.

        Returns:
            RunResponse: Results including metrics and the optimized program artifact.

        Raises:
            ServiceError: If validation fails or optimization errors occur.
        """
        run_start = time.perf_counter()
        logger.info(
            "Starting DSPy run: module=%s optimizer=%s dataset_rows=%d",
            payload.module_name,
            payload.optimizer_name,
            len(payload.dataset),
        )

        signature_cls = load_signature_from_code(payload.signature_code)
        signature_inputs, signature_outputs = extract_signature_fields(signature_cls)
        logger.debug(
            "Loaded signature %s with inputs=%s outputs=%s",
            signature_cls.__name__,
            signature_inputs,
            signature_outputs,
        )
        self._require_mapping_matches_signature(
            payload.column_mapping, signature_inputs, signature_outputs
        )
        module_factory, auto_signature = self._get_module_factory(payload.module_name)
        module_kwargs = dict(payload.module_kwargs)
        if auto_signature or "signature" not in module_kwargs:
            module_kwargs["signature"] = signature_cls
        program = module_factory(**module_kwargs)

        metric = load_metric_from_code(payload.metric_code)
        metric_identifier = getattr(metric, "__name__", "inline_metric")

        build_language_model(payload.model_settings, configure_global=True)
        optimizer_factory = self._get_optimizer_factory(payload.optimizer_name)
        optimizer = instantiate_optimizer(
            optimizer_factory,
            payload.optimizer_name,
            payload.optimizer_kwargs,
            metric,
            payload.model_settings,
            payload.reflection_model_settings,
            payload.prompt_model_settings,
            payload.task_model_settings,
        )

        examples = rows_to_examples(payload.dataset, payload.column_mapping)
        logger.info("Converted dataset to %d DSPy examples", len(examples))

        splits = split_examples(
            examples,
            payload.split_fractions,
            shuffle=payload.shuffle,
            seed=payload.seed or self.default_seed,
        )
        logger.info(
            "Split dataset -> train=%d val=%d test=%d",
            len(splits.train),
            len(splits.val),
            len(splits.test),
        )
        if progress_callback:
            progress_callback(
                PROGRESS_SPLITS_READY,
                {
                    DETAIL_TRAIN: len(splits.train),
                    DETAIL_VAL: len(splits.val),
                    DETAIL_TEST: len(splits.test),
                },
            )

        baseline_test_metric = None
        if splits.test:
            baseline_test_metric = evaluate_on_test(program, splits.test, metric)
            logger.info("Baseline test metric: %s", baseline_test_metric)
            if progress_callback and baseline_test_metric is not None:
                progress_callback(
                    PROGRESS_BASELINE,
                    {DETAIL_BASELINE: baseline_test_metric},
                )

        logger.info("Compiling program via optimizer=%s", payload.optimizer_name)
        with capture_tqdm(progress_callback):
            compiled_program = compile_program(
                optimizer=optimizer,
                program=program,
                splits=splits,
                metric=metric,
                compile_kwargs=payload.compile_kwargs,
            )
        logger.info("Optimizer compile completed successfully")

        optimized_test_metric = None
        if splits.test:
            optimized_test_metric = evaluate_on_test(compiled_program, splits.test, metric)
            logger.info("Optimized test metric: %s", optimized_test_metric)
            if progress_callback and optimized_test_metric is not None:
                progress_callback(
                    PROGRESS_OPTIMIZED,
                    {DETAIL_OPTIMIZED: optimized_test_metric},
                )

        program_artifact = persist_program(compiled_program, artifact_id, self.artifacts_root)
        if program_artifact and program_artifact.path:
            logger.info("Saved optimized program to %s", program_artifact.path)

        split_counts = SplitCounts(
            train=len(splits.train), val=len(splits.val), test=len(splits.test)
        )

        details: Dict[str, Any] = {
            DETAIL_TRAIN: split_counts.train,
            DETAIL_VAL: split_counts.val,
            DETAIL_TEST: split_counts.test,
            DETAIL_BASELINE: baseline_test_metric,
            DETAIL_OPTIMIZED: optimized_test_metric,
        }

        optimization_metadata = {
            META_OPTIMIZER: payload.optimizer_name,
            META_OPTIMIZER_KWARGS: payload.optimizer_kwargs,
            META_COMPILE_KWARGS: payload.compile_kwargs,
            META_MODULE_KWARGS: payload.module_kwargs,
            META_MODEL_IDENTIFIER: payload.model_settings.normalized_identifier(),
        }

        runtime_seconds = time.perf_counter() - run_start
        response = RunResponse(
            module_name=payload.module_name,
            optimizer_name=payload.optimizer_name,
            metric_name=metric_identifier,
            split_counts=split_counts,
            baseline_test_metric=baseline_test_metric,
            optimized_test_metric=optimized_test_metric,
            optimization_metadata=optimization_metadata,
            details=details,
            program_artifact_path=program_artifact.path if program_artifact else None,
            program_artifact=program_artifact,
            runtime_seconds=runtime_seconds,
        )

        logger.info(
            "DSPy run finished: module=%s optimizer=%s status=success",
            payload.module_name,
            payload.optimizer_name,
        )
        return response

    def validate_payload(self, payload: RunRequest) -> None:
        """Run additional validations prior to job submission.

        Args:
            payload: RunRequest to validate before execution.

        Returns:
            None.

        Raises:
            ServiceError: If validation fails for signature, metric, module, or optimizer.
        """
        logger.info(
            "Validating payload for module=%s optimizer=%s dataset_rows=%d",
            payload.module_name,
            payload.optimizer_name,
            len(payload.dataset),
        )

        signature_cls = load_signature_from_code(payload.signature_code)
        inputs, outputs = extract_signature_fields(signature_cls)
        self._require_mapping_matches_signature(payload.column_mapping, inputs, outputs)
        load_metric_from_code(payload.metric_code)
        self._get_module_factory(payload.module_name)
        optimizer_factory = self._get_optimizer_factory(payload.optimizer_name)
        validate_optimizer_signature(optimizer_factory, payload.optimizer_name)
        validate_optimizer_kwargs(
            optimizer_factory, payload.optimizer_kwargs, payload.optimizer_name
        )
        logger.info("Payload validation succeeded for module=%s", payload.module_name)

    def _get_module_factory(self, name: str) -> tuple[Callable[..., Any], bool]:
        """Resolve the requested module factory from registry or DSPy.

        Args:
            name: Module name, either registered or a dotted path like 'dspy.ChainOfThought'.

        Returns:
            tuple: (factory_callable, auto_signature_flag) where auto_signature_flag
                indicates whether the signature should be auto-injected.

        Raises:
            ServiceError: If the module cannot be resolved.
        """
        try:
            return self.registry.get_module(name), False
        except UnknownRegistrationError:
            try:
                return resolve_module_factory(name)
            except ResolverError as exc:
                raise ServiceError(str(exc)) from exc

    def _get_optimizer_factory(self, name: str) -> Callable[..., Any]:
        """Resolve the requested optimizer factory from registry or DSPy.

        Args:
            name: Optimizer name, either registered or a dotted path like 'dspy.BootstrapFewShot'.

        Returns:
            Callable: Factory callable that creates the optimizer instance.

        Raises:
            ServiceError: If the optimizer cannot be resolved.
        """
        try:
            return self.registry.get_optimizer(name)
        except UnknownRegistrationError:
            try:
                return resolve_optimizer_factory(name)
            except ResolverError as exc:
                raise ServiceError(str(exc)) from exc

    @staticmethod
    def _require_mapping_matches_signature(
        mapping: ColumnMapping, signature_inputs: List[str], signature_outputs: List[str]
    ) -> None:
        """Ensure the column mapping covers every signature field exactly once.

        Args:
            mapping: ColumnMapping specifying input/output column mappings.
            signature_inputs: List of input field names from the DSPy signature.
            signature_outputs: List of output field names from the DSPy signature.

        Returns:
            None.

        Raises:
            ServiceError: If any signature fields are missing from the mapping.
        """
        missing_inputs = set(signature_inputs) - set(mapping.inputs.keys())
        missing_outputs = set(signature_outputs) - set(mapping.outputs.keys())
        if missing_inputs or missing_outputs:
            raise ServiceError(
                "column_mapping must include every signature field. "
                f"Missing inputs: {sorted(missing_inputs)}; "
                f"missing outputs: {sorted(missing_outputs)}"
            )
