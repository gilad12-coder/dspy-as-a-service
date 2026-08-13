"""Shared protocol-level string constants.

These keys cross subprocess and HTTP boundaries (worker progress events,
payload overview keys, tqdm fields, optimization type discriminators), so
renaming any of them is a wire-protocol change. Keep them frozen unless a
coordinated migration is intended.
"""

RESOLUTION_HINT = "Register it via ServiceRegistry or provide a dotted path beginning with 'dspy.'."

DETAIL_TRAIN = "train_examples"
DETAIL_VAL = "val_examples"
DETAIL_TEST = "test_examples"
DETAIL_BASELINE = "baseline_test_metric"
DETAIL_OPTIMIZED = "optimized_test_metric"

META_OPTIMIZER = "optimizer"
META_OPTIMIZER_KWARGS = "optimizer_kwargs"
META_COMPILE_KWARGS = "compile_kwargs"
META_MODULE_KWARGS = "module_kwargs"
META_MODEL_IDENTIFIER = "model_identifier"

PAYLOAD_OVERVIEW_NAME = "name"
PAYLOAD_OVERVIEW_DESCRIPTION = "description"
PAYLOAD_OVERVIEW_USERNAME = "username"
PAYLOAD_OVERVIEW_MODULE_NAME = "module_name"
PAYLOAD_OVERVIEW_MODULE_KWARGS = "module_kwargs"
# Stored so the load path can reconstruct the optimized program from a
# state-only JSON artifact (no pickle deserialization required).
PAYLOAD_OVERVIEW_SIGNATURE_CODE = "signature_code"
# Workflow runs: the full graph spec (the workflow-mode analogue of
# signature_code) and a scrubbed tool_source ({kind, mcp_url, tool_filter} —
# never the auth header), both needed to rebuild the program shell at serve.
PAYLOAD_OVERVIEW_WORKFLOW = "workflow"
PAYLOAD_OVERVIEW_TOOL_SOURCE = "tool_source"
PAYLOAD_OVERVIEW_OPTIMIZER_NAME = "optimizer_name"
PAYLOAD_OVERVIEW_MODEL_NAME = "model_name"
PAYLOAD_OVERVIEW_MODEL_SETTINGS = "model_settings"
PAYLOAD_OVERVIEW_REFLECTION_MODEL = "reflection_model_name"
PAYLOAD_OVERVIEW_TASK_MODEL = "task_model_name"
PAYLOAD_OVERVIEW_COLUMN_MAPPING = "column_mapping"
PAYLOAD_OVERVIEW_DATASET_ROWS = "dataset_rows"
PAYLOAD_OVERVIEW_DATASET_FILENAME = "dataset_filename"
PAYLOAD_OVERVIEW_SPLIT_FRACTIONS = "split_fractions"
PAYLOAD_OVERVIEW_SHUFFLE = "shuffle"
PAYLOAD_OVERVIEW_SEED = "seed"
PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS = "optimizer_kwargs"
PAYLOAD_OVERVIEW_COMPILE_KWARGS = "compile_kwargs"
PAYLOAD_OVERVIEW_TASK_FINGERPRINT = "task_fingerprint"
# Token source the run bills against: "managed" (Skynet credits) or "byok"
# (the user's own provider key). Threaded from the wizard so billing mode is
# enforced server-side, not advisory.
PAYLOAD_OVERVIEW_TOKEN_SOURCE = "token_source"
# Per-model token sources used for mixed managed/BYOK billing after completion.
PAYLOAD_OVERVIEW_TOKEN_SOURCES_BY_MODEL = "token_sources_by_model"
# Low/high ends of the projected credit bracket the wizard showed at submit.
# Persisted alongside the billing stamp so the estimate can be reconciled
# against the actual charge; advisory only — never gates or bills.
PAYLOAD_OVERVIEW_ESTIMATED_LOW = "estimated_credits_low"
PAYLOAD_OVERVIEW_ESTIMATED_HIGH = "estimated_credits_high"
# The run's effective cost ceiling after the balance clamp, in full-cost
# credits. Persisted in the overview (not just the payload JSON) so the submit
# gate can sum the commitments of a user's still-active runs without loading
# every payload; rows predating this stamp contribute zero to that sum.
PAYLOAD_OVERVIEW_MAX_COST_CREDITS = "max_cost_credits"
PAYLOAD_OVERVIEW_IS_PRIVATE = "is_private"
# Id of the personal-library dataset a run was submitted from, when the submit
# was by-reference. Persisted so the optimization detail surfaces a live link
# back to the dataset and the dataset page can list the runs that used it.
PAYLOAD_OVERVIEW_SOURCE_DATASET_ID = "source_dataset_id"

PROGRESS_SPLITS_READY = "dataset_splits_ready"
PROGRESS_BASELINE = "baseline_evaluated"
PROGRESS_OPTIMIZED = "optimized_evaluated"
PROGRESS_OPTIMIZER = "optimizer_progress"
PROGRESS_CANDIDATE = "candidate"
PROGRESS_REJECTED = "candidate_rejected"
PROGRESS_VALSET = "valset_rows"
PROGRESS_VALSET_OUTPUTS = "valset_outputs"
PROGRESS_MINIBATCH = "minibatch_feedback"

PROGRESS_GRID_PAIR_STARTED = "grid_pair_started"
PROGRESS_GRID_PAIR_COMPLETED = "grid_pair_completed"
PROGRESS_GRID_PAIR_FAILED = "grid_pair_failed"

# Phase-marker events the UI relies on to render pipeline stages and
# per-pair status. They fire once per phase (not per step), so they're
# cheap to keep — but they also happen early, which makes them the
# first casualties of a naive FIFO eviction. The jobstore preserves
# these before touching optimizer_progress and other high-volume rows.
# Candidate events are also preserved: the trajectory tree references
# each candidate's parent by id, so evicting earlier rows produces
# orphan nodes the frontend cannot resolve.
STRUCTURAL_PROGRESS_EVENTS = frozenset(
    {
        PROGRESS_SPLITS_READY,
        PROGRESS_BASELINE,
        PROGRESS_OPTIMIZED,
        PROGRESS_CANDIDATE,
        PROGRESS_REJECTED,
        PROGRESS_VALSET,
        PROGRESS_VALSET_OUTPUTS,
        PROGRESS_GRID_PAIR_STARTED,
        PROGRESS_GRID_PAIR_COMPLETED,
        PROGRESS_GRID_PAIR_FAILED,
    }
)

PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE = "optimization_type"
OPTIMIZATION_TYPE_RUN = "run"
OPTIMIZATION_TYPE_GRID_SEARCH = "grid_search"

# Composition classifier — orthogonal to optimization_type. Distinguishes a run
# over a single atomic DSPy module from one over a workflow (a DAG of module
# nodes). Derived at submit time from ``module_name == WORKFLOW_MODULE_NAME`` and
# hoisted to the indexed ``jobs.composition`` column so it is queryable without
# parsing JSON or string-matching the module name.
PAYLOAD_OVERVIEW_COMPOSITION = "composition"
COMPOSITION_SINGLE = "single"
COMPOSITION_WORKFLOW = "workflow"
# Tagger bulk auto-tag jobs: claimed by the same worker fleet but run in the
# worker thread (no subprocess) — see core.worker.tagging_job.
OPTIMIZATION_TYPE_TAGGING = "tagging_autotag"

# Token source modes. "managed" bills Skynet credits; "byok" runs on the user's
# own provider key and is never billed.
TOKEN_SOURCE_MANAGED = "managed"
TOKEN_SOURCE_BYOK = "byok"

PAYLOAD_OVERVIEW_TOTAL_PAIRS = "total_pairs"
PAYLOAD_OVERVIEW_GENERATION_MODELS = "generation_models"
PAYLOAD_OVERVIEW_REFLECTION_MODELS = "reflection_models"

TQDM_TOTAL_KEY = "tqdm_total"
TQDM_N_KEY = "tqdm_n"
TQDM_ELAPSED_KEY = "tqdm_elapsed"
TQDM_PERCENT_KEY = "tqdm_percent"
TQDM_RATE_KEY = "tqdm_rate"
TQDM_REMAINING_KEY = "tqdm_remaining"
TQDM_DESC_KEY = "tqdm_desc"

# Every optimizer-progress snapshot key shares this prefix; the job store carries
# the family forward across interleaved non-tqdm events so the live progress bar
# does not blink out between rollout ticks.
TQDM_KEY_PREFIX = "tqdm_"

COMPILE_TRAINSET_KEY = "trainset"
COMPILE_VALSET_KEY = "valset"
OPTIMIZER_METRIC_KEY = "metric"
OPTIMIZER_REFLECTION_LM_KEY = "reflection_lm"
OPTIMIZER_LOG_DIR_KEY = "log_dir"

OPTIMIZER_NAME_GEPA = "gepa"
