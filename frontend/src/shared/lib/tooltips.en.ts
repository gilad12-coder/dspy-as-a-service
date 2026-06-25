/**
 * English overlay for TOOLTIPS (src/shared/lib/tooltips.ts).
 *
 * `tip()` reads this in the English locale. The type is the COMPLETE
 * `Record<TooltipKey, string>` (not `Partial`), so every tooltip key must be
 * translated here — adding a key to TOOLTIPS without a matching English string
 * is a TypeScript compile error rather than a silent Hebrew leak in English.
 */

import type { TooltipKey } from "@/shared/lib/tooltips";

export const TOOLTIPS_EN: Record<TooltipKey, string> = {
  "score.baseline": "Baseline score before optimization: how the program did without an improved prompt or selected examples",
  "score.optimized": "Optimized score after optimization: how the program did with the chosen prompt and examples",
  "score.improvement": "The gap between the optimized score and the baseline score. The larger it is, the more the optimization improved the result",
  "score.progression": "How the score changed from attempt to attempt while the optimizer searched for a better prompt",

  "lm.calls_count": "The number of calls to the language model during optimization",
  "lm.avg_response_time": "The average time it took the model to respond to each call",

  "lm_activity.section": "Language model activity by stage — how many calls were made and how long they took, for the generation and reflection models separately",
  "lm_activity.stage.baseline": "Calls made while measuring the baseline score — before the optimizer started working",
  "lm_activity.stage.training": "Calls made during the optimization itself — while the optimizer built prompt candidates",
  "lm_activity.stage.evaluation": "Calls made while measuring the optimized score — after the optimization finished",
  "lm_activity.column.generation": "Calls made to the generation model — the model that produces answers",
  "lm_activity.column.reflection": "Calls made to the reflection model — the model that analyzes errors and suggests improvements",
  "lm_activity.cell.calls": "The number of calls made in this stage",
  "lm_activity.cell.avg_ms": "The average time per call in this stage",
  "lm_activity.total_row": "Total calls and average time across all stages",

  "model.generation": "The model that actually produces the answer during the optimization run",
  "model.reflection": "The model that checks mistakes and suggests how to improve the instructions during optimization",

  "data.split_explanation": "The dataset is split into three parts: train for learning, val for choosing the prompt, and test for the final measurement",
  "data.shuffle_explanation": "Shuffles the order of rows before the split, so the file order does not accidentally affect the results",
  "data.split.train": "Examples the optimizer uses to build prompt candidates",
  "data.split.val": "Examples that rank the candidates during optimization",
  "data.split.test": "Examples reserved for the final measurement, after the prompt has been chosen",
  "data.seed": "A fixed starting number that keeps the same split and the same shuffle on every repeated run",

  "prompt.optimized": "The prompt the optimizer built: improved instructions and examples selected from the dataset",
  "prompt.demonstrations": "Input-output examples (few-shot demonstrations) shown to the model to show it the desired format and answer",

  "module.choice":
    "A DSPy module is a component in the program that calls a language model: it wraps each signature in a prompting technique and defines the structure of the call to the model in order to produce the output defined in the signature. Within this framework the optimizer tunes the module's learnable parameters, such as instructions and examples in the prompt",
  "module.predict": "Predict — the basic module: maps the input to the output in a single call to the model, with no intermediate steps",
  "module.cot":
    "Chain of Thought — adds a reasoning field that leads the model to think step-by-step before the final answer; usually improves accuracy on complex tasks",
  "module.react":
    "ReAct — an agent that combines thinking with calling tools in a loop, until it produces the output in the signature",
  "optimizer.choice": "The method that tries to improve the prompt and find a version with a higher score",

  "react.tool_source": "Where the tool list is loaded from: a live MCP server, or a snapshot of tools from the dataset",
  "react.mcp_url": "The address of the MCP server from which the agent's tools are loaded",
  "react.auth": "Authentication header (Authorization header) for the MCP server. Not stored on the server and not exposed to the chat agent",
  "react.tool_filter": "Limit the tool list to the specified names only, separated by commas",
  "react.optimized_tools": "The tools the agent (ReAct) runs in a loop, with the descriptions and arguments the optimizer refined during optimization",

  "config.section.summary": "The module, the optimizer, and the parameters chosen for this run",
  "config.section.models": "The language models configured — generation for producing answers, reflection for analyzing errors",
  "config.section.data": "Splitting the dataset into train, val and test, and shuffle settings",

  "grid.generation_models": "The models that produce answers. Each pair in the grid search uses a different generation model",
  "grid.reflection_models": "The models that analyze errors and suggest improvements. Each pair uses a different reflection model",
  "grid.score_comparison": "Comparison of the baseline score and the optimized score for each model pair",
  "grid.quality_speed_combined":
    "Quality and speed for each model pair, side by side. The higher the quality and speed, the better the pair.",
  "grid.avg_response_time_per_pair": "Average duration per language model call, by model pair",
  "grid.best_pair_default":
    "Default: the pair with the highest quality score. You can switch to any other pair.",

  "pair.runtime": "The duration of the optimization run for this model pair",

  "serve.section_pair": "API URL and code snippets to integrate the selected pair into your app",
  "serve.section_run": "API URL and code snippets to integrate the improved program into your app",
  "serve.api_url_pair": "The API URL of the selected pair",
  "serve.api_url_run": "The API URL you send POST requests to with the input fields in order to get a prediction from the improved program",
  "serve.api_url_react":
    "The API URL you send a POST request to with the user message; the optimized ReAct agent's response is streamed back over SSE",
  "serve.integration_code": "Ready-to-copy code examples",

  "submit.depth":
    "How wide GEPA's search is: light runs fast with fewer attempts; deeper checks more options and takes more time",
  "submit.reflection_minibatch": "How many examples the model checks in each reflection round to find error patterns",
  "submit.eval_rounds": "How many times to run a full evaluation to check prompt candidates",
  "submit.merge": "When enabled, GEPA can merge and combine ideas from several good candidates into one prompt",

  "model_config.connection_section": "Run the model on your own server: an OpenAI-compatible endpoint (Ollama, vLLM, LM Studio or an enterprise gateway) and an access key. Leave closed to use the built-in providers",
  "model_config.model": "The model that will run the optimization. Choose from the model catalog, or a custom model discovered from the Base URL",
  "model_config.base_url": "The address of your own OpenAI-compatible server — Ollama, vLLM, LM Studio or an enterprise gateway. Leave empty to use the provider's default server",
  "model_config.api_key": "Access key for the model server. Optional — if empty, it is taken from an environment variable. Not stored on the server and removed from the form after submission",
  "model_config.temperature": "How creative the model is — a low value gives consistent answers, a high one more varied",
  "model_config.top_p": "top_p (nucleus sampling): limits the range of words the model considers — a low value narrows it, a high one allows more variety",
  "model_config.max_tokens": "The maximum prediction length — a token is roughly one word",

  "code.signature_metric": "The source code of the Signature and metric defined for this optimization",
  "code.signature": "Defines the input and output fields of the task — what the model receives and what it needs to return",
  "code.metric": "A function that measures the quality of the prediction — returns a numeric score for each example",
  "code.predictions_table": "Results of running the program on the test examples — a score for each example and an overall summary",

  "tagger.upload_file": "Upload a CSV, JSON or Excel file. Each row becomes an item to tag",
  "tagger.text_column": "Choose the column that contains the text to tag. The other columns are kept in the export",
  "tagger.mode": "Choose the tagging type that fits the task: yes/no, selection from a list, or free text",
  "tagger.binary_question":
    "The question shown above the yes/no buttons. It helps to phrase a question that can be answered clearly",
  "tagger.multiclass_categories": "Define the categories available for selection while tagging — at least two",
  "tagger.freetext_instruction": "The instruction shown above the text field. Briefly explain what needs to be written",

  "compare.winner_improvement": "The score improvement percentage of the winning run — the difference between the optimized score and the baseline score",
  "compare.winner_runtime": "The total duration of the winning run, from launch until the optimization finished",
  "compare.winner_models": "The language model pair of the winning run — generation model that produces output, and reflection model that improves the instructions",

  "analytics.score_comparison": "Comparison of the baseline score versus the optimized score for every completed optimization",
  "analytics.runtime_vs_gain": "Analysis of run times and efficiency — how much improvement is gained relative to time",
  "analytics.runtime_minutes": "The duration of the run in minutes for every completed optimization",
  "analytics.improvement_per_minute": "Score improvement percentage per minute of run — a high value means a more efficient optimization",
  "analytics.dataset_size_vs_improvement": "Whether more data leads to better score improvement — each point is one optimization",
  "analytics.submissions_per_day": "The number of optimizations submitted per day",
  "analytics.optimizer_avg_improvement": "Average score improvement percentage each optimizer achieved across all runs",
  "analytics.top_improvements": "The runs that achieved the largest improvement in score, from best to least",
  "analytics.optimizer_comparison_table": "Detailed comparison between the optimizers: average improvement, number of runs, and run time",
  "analytics.model_performance_table": "Performance of the different models: usage frequency and average improvement",
};
