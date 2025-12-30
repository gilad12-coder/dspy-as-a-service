# user-facing resolution guidance & status copy
RESOLUTION_HINT = "Register it via ServiceRegistry or provide a dotted path beginning with 'dspy.'."
JOB_SUCCESS_MESSAGE = "Optimization completed successfully."

# run-response detail keys
DETAIL_TRAIN = "train_examples"
DETAIL_VAL = "val_examples"
DETAIL_TEST = "test_examples"
DETAIL_BASELINE = "baseline_test_metric"
DETAIL_OPTIMIZED = "optimized_test_metric"

# metadata blocks stored alongside optimizer runs
META_OPTIMIZER = "optimizer"
META_OPTIMIZER_KWARGS = "optimizer_kwargs"
META_COMPILE_KWARGS = "compile_kwargs"
META_MODULE_KWARGS = "module_kwargs"
META_MODEL_IDENTIFIER = "model_identifier"

# payload overview keys cached per job
PAYLOAD_OVERVIEW_MODULE_NAME = "module_name"
PAYLOAD_OVERVIEW_OPTIMIZER_NAME = "optimizer_name"
PAYLOAD_OVERVIEW_DATASET_ROWS = "dataset_rows"
PAYLOAD_OVERVIEW_SPLIT_FRACTIONS = "split_fractions"
PAYLOAD_OVERVIEW_SHUFFLE = "shuffle"
PAYLOAD_OVERVIEW_SEED = "seed"
PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS = "optimizer_kwargs"
PAYLOAD_OVERVIEW_COMPILE_KWARGS = "compile_kwargs"

# progress event identifiers emitted during optimization
PROGRESS_SPLITS_READY = "dataset_splits_ready"
PROGRESS_BASELINE = "baseline_evaluated"
PROGRESS_OPTIMIZED = "optimized_evaluated"
PROGRESS_OPTIMIZER = "optimizer_progress"

# tqdm metric keys captured from progress bars
TQDM_TOTAL_KEY = "tqdm_total"
TQDM_N_KEY = "tqdm_n"
TQDM_ELAPSED_KEY = "tqdm_elapsed"
TQDM_PERCENT_KEY = "tqdm_percent"
TQDM_RATE_KEY = "tqdm_rate"
TQDM_REMAINING_KEY = "tqdm_remaining"
TQDM_DESC_KEY = "tqdm_desc"

# optimizer + compile helper keys
COMPILE_TRAINSET_KEY = "trainset"
COMPILE_VALSET_KEY = "valset"
OPTIMIZER_METRIC_KEY = "metric"
OPTIMIZER_REFLECTION_LM_KEY = "reflection_lm"
OPTIMIZER_PROMPT_MODEL_KEY = "prompt_model"
OPTIMIZER_TASK_MODEL_KEY = "task_model"

# optimizer identifiers that require special handling
OPTIMIZER_NAME_GEPA = "gepa"
OPTIMIZER_NAME_MIPROV2 = "miprov2"
