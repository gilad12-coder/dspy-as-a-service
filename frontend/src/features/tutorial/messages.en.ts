// English UI strings for the tutorial slice. Edit directly; keys missing here fall
// back to the Hebrew slice via msg(), so partial translations are safe.

import type { tutorialMessages } from "./messages";

export const tutorialMessagesEn: Partial<Record<keyof typeof tutorialMessages, string>> = {
  "auto.features.tutorial.components.tutorial.menu.1": "Skynet guide",
  "auto.features.tutorial.components.tutorial.menu.2":
    "A short tour of the system's main areas: the dashboard, the submission form, and the results page.",
  "auto.features.tutorial.components.tutorial.menu.3": "Start the tour",
  "auto.features.tutorial.components.tutorial.popover.1": " of ",
  "auto.features.tutorial.components.tutorial.popover.2": "Previous",
  "auto.features.tutorial.lib.steps.literal.1": "Great service, highly recommend!",
  "auto.features.tutorial.lib.steps.literal.2": "The product arrived broken, very disappointing",
  "auto.features.tutorial.lib.steps.literal.3": "Fast shipping, good packaging",
  "auto.features.tutorial.lib.steps.literal.4": "Not worth the price, low quality",
  "auto.features.tutorial.lib.steps.literal.5": "Pleasant shopping experience, I'll be back",
  "auto.features.tutorial.lib.steps.literal.6": "Key metrics",
  "auto.features.tutorial.lib.steps.literal.7":
    "Four cards summarize activity: all runs, active runs, succeeded runs, and failed runs, alongside success and failure rates.",
  "auto.features.tutorial.lib.steps.literal.8": "Sidebar",
  "auto.features.tutorial.lib.steps.literal.9": "Statistics",
  "auto.features.tutorial.lib.steps.literal.10":
    "Select at least two finished runs. An action bar appears at the bottom, and from it you can open a detailed comparison between the runs.",
  "auto.features.tutorial.lib.steps.literal.11": "Config comparison",
  "auto.features.tutorial.lib.steps.literal.12": "Prompt comparison",
  "auto.features.tutorial.lib.steps.literal.13": "Submission form",
  "auto.features.tutorial.lib.steps.literal.14": "Basic details",
  "auto.features.tutorial.lib.steps.literal.15": "Column mapping",
  "auto.features.tutorial.lib.steps.literal.16": "Dataset split",
  "auto.features.tutorial.lib.steps.literal.17": "Search level",
  "auto.features.tutorial.lib.steps.literal.18":
    "Light runs fast with few attempts. Medium balances speed and quality. Deep explores more options and therefore takes longer, but increases the chance of a significant improvement.",
  "auto.features.tutorial.lib.steps.literal.19": "GEPA parameters",
  "auto.features.tutorial.lib.steps.literal.20": "Signature",
  "auto.features.tutorial.lib.steps.literal.21": "Review",
  "auto.features.tutorial.lib.steps.literal.22": "Results page",
  "auto.features.tutorial.lib.steps.literal.23": "Process stages",
  "auto.features.tutorial.lib.steps.literal.24": "Data tab",
  "auto.features.tutorial.lib.steps.literal.25": "Usage",
  "auto.features.tutorial.lib.steps.literal.26": "Logs",
  "auto.features.tutorial.lib.steps.literal.27": "Run config",
  "auto.features.tutorial.lib.steps.literal.28": "Text labeling",
  "auto.features.tutorial.lib.steps.literal.29": "Labeling setup form",
  "auto.features.tutorial.lib.steps.literal.30":
    "Three steps: pick a file and mark one or more input columns (text from several columns is combined in each row), choose a labeling mode, then define a question or categories. After that you move on to the labeling itself, with keyboard navigation and export at the end.",
  "auto.features.tutorial.lib.steps.literal.31": "Labeling modes",
  "auto.features.tutorial.lib.steps.literal.32":
    "Binary classification: a question answered yes or no, for example to detect sentiment. Categories: a choice from a list you define, including selecting multiple categories. Free text: writing an answer for each row, for example to extract information or to translate.",
  "auto.features.tutorial.lib.steps.literal.33": "That's it!",
  "auto.features.tutorial.lib.steps.literal.34": "Guide",
  "auto.features.tutorial.lib.steps.literal.35": "Get to know the system's main capabilities",
  "auto.features.tutorial.lib.steps.literal.36": "Model testing",
  "auto.features.tutorial.lib.steps.literal.37":
    "Before starting a run, you can check that the model is available and measure its response time. This is how you catch connection or configuration problems before launching a long run.",
  "auto.features.tutorial.lib.steps.literal.38": "Semantic search",
  "auto.features.tutorial.lib.steps.literal.39":
    "Free-text search across all runs in the system — yours or other users'. Type a description in English or Hebrew, and Skynet finds runs with similar meaning, not just word matches. You can filter by models, optimizers, status, and date range, and browse a ranked list of results.",
  "auto.features.tutorial.lib.steps.literal.40": "API and integration",
  "auto.features.tutorial.lib.steps.literal.41":
    "Beyond the browser interface, the optimized prompt is immediately available through a REST API. Copy the endpoint URL or use the ready-made code snippets (Python, JavaScript, cURL) to integrate it into your own code.",
  "auto.features.tutorial.lib.steps.literal.42": "AI agent",
  "auto.features.tutorial.lib.steps.literal.43":
    "The floating button in the corner opens the AI agent. The agent knows the system and can help fill in the submission form, explain results, and answer questions.",
  "auto.features.tutorial.lib.steps.literal.44": "Chat with the agent",
  "auto.features.tutorial.lib.steps.literal.45":
    "In the chat window you can talk to the agent and ask it to perform actions. Trust mode determines whether the agent asks for approval before every action or carries out safe actions on its own. The agent sees the state of the submission form and can help fill in fields.",
  "auto.features.tutorial.lib.steps.literal.46": "Candidate tree",
  "auto.features.tutorial.lib.steps.literal.47":
    "GEPA creates new prompt candidates, and each candidate can spawn candidates in the next generation. The tree shows the score of every candidate and the parent-to-child relationship across attempts. The slider above the graph steps through the generations and reveals how the results evolved, and clicking a node opens a drawer with the full prompt, per-example scores, and the feedback that drove the improvement.",
  "auto.features.tutorial.lib.steps.literal.48":
    "GEPA creates new prompt candidates, and each candidate can spawn candidates in the next generation. The tree shows the score of every candidate and the parent-to-child relationship, and rejected proposals also appear as dashed nodes. The slider above the graph steps through the generations, and clicking a node opens a drawer with the full prompt, per-example scores, and the feedback that drove the improvement or rejection.",
  "auto.features.tutorial.lib.steps.literal.49":
    "Free-text search across all runs in the system — yours or other users'. Type a description in English or Hebrew, and Skynet finds runs with similar meaning, not just word matches. When the field is empty it suggests your recent searches and common searches, and you can focus the field from anywhere with the keyboard. You can filter by models, optimizers, status, and date range, and browse a ranked list of results.",
  "auto.features.tutorial.lib.steps.literal.50":
    "In the chat window you can talk to the agent and ask it to perform actions. Trust mode determines whether the agent asks for approval before every action or carries out safe actions on its own. The agent sees the state of the submission form and can help fill in fields. Previous conversations are saved and can be reopened from the history button, and the tools button shows which actions the agent can perform.",
  "auto.features.tutorial.lib.steps.literal.51":
    "Before starting a run, you can pick several models and run a short comparison between them on your dataset. Skynet measures availability, response time, and quality, and ranks the models to help you choose the best fit even before a long run.",
  "auto.features.tutorial.lib.steps.literal.52": "Model activity",
  "auto.features.tutorial.lib.steps.literal.53":
    "The model activity tab breaks down the language models' work in the run: for each stage — baseline measurement, optimization, and final measurement — you see how many calls were made and how long each call took on average, separately for the generation model and the reflection model. This lets you see where most of the calls and time went, and identify a stage or model that is slowing the run down.",
  "auto.features.tutorial.lib.demo.data.literal.1": "Email classification",
  "auto.features.tutorial.lib.demo.data.literal.2": "Starting an optimization: email classification",
  "auto.features.tutorial.lib.demo.data.literal.3": "Starting an optimization: email classification",
  "auto.features.tutorial.lib.demo.data.literal.4": "Starting an optimization: email classification",
  "auto.features.tutorial.lib.demo.data.literal.5": "Starting an optimization: email classification",
  "auto.features.tutorial.lib.demo.data.literal.6": "Starting an optimization: email classification",
  "auto.features.tutorial.lib.demo.data.literal.7": "Email classification",
  "auto.features.tutorial.lib.demo.data.literal.8":
    "Classifying emails into categories: spam, important, promotional",
  "auto.features.tutorial.lib.demo.data.literal.9": "Sentiment analysis",
  "auto.features.tutorial.lib.demo.data.literal.10": "Detecting positive/negative tone in product reviews",
  "auto.features.tutorial.lib.demo.data.literal.11": "Text summarization",
  "auto.features.tutorial.lib.demo.data.literal.12": "Entity extraction",
  "auto.features.tutorial.lib.demo.data.literal.13": "Identifying names and organizations in free text",
  "auto.features.tutorial.lib.demo.data.literal.14": "Hebrew-English translation",
  "auto.features.tutorial.lib.demo.data.literal.15": "Translating short sentences",
  "auto.features.tutorial.lib.demo.data.literal.16": "Email classification",
  "auto.features.tutorial.lib.demo.data.literal.17": "Text summarization",
  "auto.features.tutorial.lib.demo.data.literal.18": "GEPA with a Predict module (no reasoning field)",
  "auto.features.tutorial.lib.demo.data.literal.19":
    "Validating the Signature and metric code…",
  "auto.features.tutorial.lib.demo.data.literal.20": "All checks passed",
  "auto.features.tutorial.lib.demo.data.literal.21": "Validation passed ✓",
  "auto.features.tutorial.lib.demo.data.literal.22": "Splitting 200 examples into train, val, and test…",
  "auto.features.tutorial.lib.demo.data.literal.23": "Dataset split: train=120, val=40, test=40",
  "auto.features.tutorial.lib.demo.data.literal.24": "Measuring the baseline program on the test set…",
  "auto.features.tutorial.lib.demo.data.literal.25": "Baseline program score: 62.0",
  "auto.features.tutorial.lib.demo.data.literal.26": "Found the best program with a score of 0.84",
  "auto.features.tutorial.lib.demo.data.literal.27": "Measuring the optimized program on the test set…",
  "auto.features.tutorial.lib.demo.data.literal.28": "Optimized program score: 84.0",
  "auto.features.tutorial.lib.demo.data.literal.29": "Optimization completed ✓",
  "auto.features.tutorial.lib.demo.data.literal.30": "The metric returned an error",
  "auto.features.tutorial.lib.demo.data.literal.31": "Writing SQL code",
  "auto.features.tutorial.lib.demo.data.literal.32": "Summarizing articles",
  "auto.features.tutorial.lib.demo.data.literal.33": "Answering questions from documents",
  "auto.features.tutorial.lib.demo.data.literal.34": "User intent labeling",
  "auto.features.tutorial.components.tutorial.menu.literal.1": "Close",
  "auto.features.tutorial.components.tutorial.popover.literal.1": "Pause",
  "auto.features.tutorial.components.tutorial.popover.literal.2": "Autoplay",
  "auto.features.tutorial.components.tutorial.popover.literal.3": "Pause",
  "auto.features.tutorial.components.tutorial.popover.literal.4": "Autoplay",
  "auto.features.tutorial.components.tutorial.popover.literal.5": "Close the guide",
  "auto.features.tutorial.components.tutorial.popover.literal.6": "Finish",
  "auto.features.tutorial.components.tutorial.popover.literal.7": "Next",
  "auto.features.tutorial.lib.demo.data.template.1":
    "{p1} for classifying emails into categories: spam, important, promotional",
  "auto.features.tutorial.lib.demo.data.template.2":
    "Grid search of {p1}: four {p2} {p3}/{p4} on the same {p5}",
  "auto.features.tutorial.lib.demo.data.template.3":
    "Grid search of {p1}: four {p2} {p3}/{p4} on the same {p5}",
  "auto.features.tutorial.lib.demo.data.template.4": "GEPA with a light, fast {p1}",
  "auto.features.tutorial.lib.demo.data.template.5": "GEPA with an advanced {p1}",
  "auto.features.tutorial.lib.steps.template.1": "{p1} table",
  "auto.features.tutorial.lib.steps.template.2":
    "All runs appear here. You can sort by the column headers, filter through the filters, resize columns, and open the details of any {p1}.",
  "auto.features.tutorial.lib.steps.template.3":
    "At the top there is navigation to the dashboard, text labeling, submitting a new {p1}, and semantic search across existing runs. At the bottom, the run history appears grouped by date. Through the ⋯ next to a {p2} you can share, rename, clone, pin, or delete.",
  "auto.features.tutorial.lib.steps.template.4":
    "Interactive charts that show {p1}, efficiency, a {p2} comparison, and a leaderboard table. Clicking a bar in the chart filters the list to the relevant {p3}.",
  "auto.features.tutorial.lib.steps.template.5": "How to compare {p1}",
  "auto.features.tutorial.lib.steps.template.6": "{p1} comparison",
  "auto.features.tutorial.lib.steps.template.7":
    "At the top you see the leading run: the run with the highest {p1}, alongside the improvement percentage, the run duration, and the {p2} it used.",
  "auto.features.tutorial.lib.steps.template.8": "{p1} comparison",
  "auto.features.tutorial.lib.steps.template.9":
    "A bar chart and table showing {p1} against the {p2} for each run.",
  "auto.features.tutorial.lib.steps.template.10":
    "The 'Config' tab shows which {p1}, {p2}, {p3}, and {p4} size were used in each run.",
  "auto.features.tutorial.lib.steps.template.11":
    "The 'Prompts' tab shows side by side the optimized instructions and the {p1} that each run produced. This makes it easy to understand how a change in {p2} or in the config affected the final prompt.",
  "auto.features.tutorial.lib.steps.template.12": "{p1} comparison",
  "auto.features.tutorial.lib.steps.template.13":
    "The '{p1}' tab goes example by example and shows how each run answered and what {p2} it received. You can filter to only the {p3} where the runs disagreed, and see exactly where they differ.",
  "auto.features.tutorial.lib.steps.template.14":
    "These are the submission steps: basic details, {p1}, parameters, code, {p2}, summary, and submit.",
  "auto.features.tutorial.lib.steps.template.15":
    "In the first step you choose a name, description, and {p1} type ({p2} or {p3} that compares several configs in parallel).",
  "auto.features.tutorial.lib.steps.template.16": "Uploading a {p1}",
  "auto.features.tutorial.lib.steps.template.17":
    "Drag a CSV, JSON, or Excel file into the upload area. Each row is one example the system can learn from. The more high-quality {p1} there are, the better the {p2} will be.",
  "auto.features.tutorial.lib.steps.template.18":
    "Mark each column as an input sent to the {p1}, as an output that is the desired answer, or as a column that is not used. Next to each column the content type is also shown: text is sent as plain text, and an image is sent as vision input to a model that supports it. This mapping automatically generates the Signature.",
  "auto.features.tutorial.lib.steps.template.19":
    "Predict sends the input directly to the {p1} and gets a prediction. Chain of Thought adds a reasoning field before the output, so the {p2} lays out its reasoning before the result.",
  "auto.features.tutorial.lib.steps.template.20":
    "{p1}: examples the {p2} uses to build candidates. {p4}: examples that rank the candidates during the run. {p6}: a final measurement on {p7} that were not used to select the prompt.",
  "auto.features.tutorial.lib.steps.template.21":
    "Reflection minibatch size: how many {p1} the {p2} analyzes each round to identify errors. Maximum number of evaluation rounds: usually set by the search level, but you can configure it manually. Candidate merging: combining ideas from several good candidates into a single prompt.",
  "auto.features.tutorial.lib.steps.template.22":
    "The Signature defines what the {p1} receives and what it needs to return. It is generated automatically from the column mapping, but it is important to edit it and add precise descriptions for each field so that it is high quality.",
  "auto.features.tutorial.lib.steps.template.23":
    "A function that returns {p1} between 0 and 1 for each prediction. It defines what counts as a “good answer,” and the {p2} tries to improve this {p3} over the course of the run.",
  "auto.features.tutorial.lib.steps.template.24": "Choosing a {p1}",
  "auto.features.tutorial.lib.steps.template.25":
    "{p1} produces the answers. {p2} analyzes errors and proposes prompt improvements. You can choose a different {p3} for each role.",
  "auto.features.tutorial.lib.steps.template.26":
    "A summary of all the config across five tabs: General, {p1}, {p2}, {p3}, and Code. Check that everything is correct before submitting.",
  "auto.features.tutorial.lib.steps.template.27": "Submitting a {p1}",
  "auto.features.tutorial.lib.steps.template.28":
    "This is the button that starts the run. After clicking, the system validates the input, splits the data, runs {p1}, and then runs the {p2}. You can follow the progress in real time on the results page.",
  "auto.features.tutorial.lib.steps.template.29":
    "After submitting a {p1}, you land on the results page. At the top of the page the {p2} name, description, run status, and elapsed time appear. A clone button creates a new {p3} with the same config. During an active run a cancel button appears, and after it finishes or fails a delete button appears.",
  "auto.features.tutorial.lib.steps.template.30":
    "Every {p1} goes through five stages: input validation, dataset split, a {p2} run before {p3}, the {p4} itself, and final evaluation. Each stage updates in real time.",
  "auto.features.tutorial.lib.steps.template.31": "{p1} cards",
  "auto.features.tutorial.lib.steps.template.32":
    "Three cards: {p1} (before {p2}), {p3} (after), and the improvement percentage between them.",
  "auto.features.tutorial.lib.steps.template.33": "{p1} chart",
  "auto.features.tutorial.lib.steps.template.34":
    "Tracks the {p1} across the {p2}'s attempts. This shows when a better combination was found and how the {p3} changed over the run.",
  "auto.features.tutorial.lib.steps.template.35":
    "Each example from the {p1} is shown with a color-coded {p2} (green = high, red = low), the {p3} prediction, and the split into {p4}, {p5}, and {p6}. You can sort by {p7} to spot patterns.",
  "auto.features.tutorial.lib.steps.template.36":
    "After the run finishes, you can test the optimized prompt in real time. Enter a new input and get an immediate prediction from the {p1}.",
  "auto.features.tutorial.lib.steps.template.37":
    "Real-time logs from the {p1}. They update automatically while the run is active. You can filter by level (info, warning, error, and more), by log source, or by pair in a grid search, search free text, and sort by time.",
  "auto.features.tutorial.lib.steps.template.38":
    "All of the run's config in one place: {p1}, parameters, and dataset split.",
  "auto.features.tutorial.lib.steps.template.39": "Grid search of {p1} (Grid Search)",
  "auto.features.tutorial.lib.steps.template.40":
    "Instead of running a single {p1}, a grid search compares several {p2} of {p3} × {p4} on the same {p5}. Each row is a {p6}: on the right the quality {p7}, and in the middle the average response time. The leading pair is the one that got the highest quality score and is marked with a crown.",
  "auto.features.tutorial.lib.steps.template.41": "{p1} {p2} details",
  "auto.features.tutorial.lib.steps.template.42":
    "Clicking a {p1} opens a detailed view: {p2} before and after, the {p3} in {p4}, the run duration, the number of {p5} calls, and the progress chart.",
  "auto.features.tutorial.lib.steps.template.43":
    "A tool for labeling {p1}s directly in the system. Upload a CSV, JSON, or Excel file, choose a labeling mode, and label row by row. When you're done you can export the results.",
  "auto.features.tutorial.lib.steps.template.44":
    "Now you know Skynet's main capabilities. Submit your first {p1}, follow the {p2} in real time, and test the optimized prompt in the Usage tab.",
  "auto.features.tutorial.lib.steps.template.45":
    "Predict sends the input directly to the {p1} and gets a prediction. Chain of Thought adds a reasoning field before the output, so the {p2} lays out its reasoning before the result. ReAct suits tasks that require tools: the agent combines reasoning with tool calls in a loop until it reaches an answer.",
  "auto.features.tutorial.lib.steps.template.46":
    "A performance chart that shows, for each run, the {p1} as normalized quality, alongside relative speed when time data is available. Below it, a table comparing {p2} against the {p3} for each run.",
  "auto.features.tutorial.lib.steps.template.47":
    "Interactive charts that show {p1}, a breakdown by status and type, the distribution of run times against improvement, a daily timeline, and {p2} usage ranking. Clicking a bar in the chart filters the list to the relevant {p3}.",

  "auto.features.tutorial.components.concepts.guide.literal.1": "Background and concepts",
  "auto.features.tutorial.components.concepts.guide.literal.2": "GEPA: improving prompts with reflection",
  "auto.features.tutorial.components.concepts.guide.literal.3": "Tuning GEPA and run costs",
  "auto.features.tutorial.components.concepts.guide.literal.4": "Task, dataset, and metric",
  "auto.features.tutorial.components.concepts.guide.literal.5": "From submission to using the artifact",
  "auto.features.tutorial.components.concepts.guide.literal.6": "Best practices and common issues",
  "auto.features.tutorial.components.concepts.guide.literal.7": "Glossary",
  "auto.features.tutorial.components.concepts.guide.literal.8": "Skynet · Concepts guide",
  "auto.features.tutorial.components.concepts.guide.literal.9":
    "How Skynet improves prompts with DSPy and GEPA",
  "auto.features.tutorial.components.concepts.guide.literal.10": "Close",
  "auto.features.tutorial.components.concepts.guide.literal.11": "Table of contents",
  "auto.features.tutorial.components.concepts.guide.literal.12": "Copied",
  "auto.features.tutorial.components.concepts.guide.literal.13": "Copy",
  "auto.features.tutorial.components.concepts.guide.literal.14": "Parameter",
  "auto.features.tutorial.components.concepts.guide.literal.15": "Description",
  "auto.features.tutorial.components.concepts.guide.literal.16": "From a language model to a measurable prompt",
  "auto.features.tutorial.components.concepts.guide.literal.17": "1.1 What are large language models (LLMs)?",
  "auto.features.tutorial.components.concepts.guide.literal.18":
    "Large language models are systems that take input in natural language, code, or structured data, and return a response: text, JSON, a classification, code, or a decision. They do not apply human judgment, but they are very good at recognizing patterns and following instructions based on the examples and knowledge they learned during training.",
  "auto.features.tutorial.components.concepts.guide.literal.19":
    "In software products they are used to classify inquiries, summarize documents, extract fields, answer questions, write code, or turn free text into structured output. The quality of the response depends on the model, the quality of the input, the run config, and above all on how the task is framed and measured.",
  "auto.features.tutorial.components.concepts.guide.literal.20": "1.2 What is a prompt?",
  "auto.features.tutorial.components.concepts.guide.literal.21":
    "A prompt is the instruction that steers the model: what the task is, which input fields are available, how the output should look, and what is forbidden or especially important. A small change in wording can change responses, so in production it is not enough to write an instruction that sounds reasonable. You have to test it on real examples.",
  "auto.features.tutorial.components.concepts.guide.literal.22": "Simple prompt:",
  "auto.features.tutorial.components.concepts.guide.literal.23": "\"Summarize the following article.\"",
  "auto.features.tutorial.components.concepts.guide.literal.24": "Detailed prompt:",
  "auto.features.tutorial.components.concepts.guide.literal.25":
    "\"Read the article and return valid JSON with three fields: summary up to 80 words, key_points as a list of three points, and confidence with the values low, medium, or high based on how well the original text supports it.\"",
  "auto.features.tutorial.components.concepts.guide.literal.26": "1.3 The problem: writing a good prompt is hard",
  "auto.features.tutorial.components.concepts.guide.literal.27":
    "A prompt that looks good on a single example can fail silently once you connect it to a real dataset. The problem is usually not a lack of creativity in the wording, but the absence of an orderly process that shows what works, where it fails, and why. Four challenges recur in almost every project:",
  "auto.features.tutorial.components.concepts.guide.literal.28": "Time:",
  "auto.features.tutorial.components.concepts.guide.literal.29":
    "Manually testing dozens of wordings against real examples takes time, and it's hard to reproduce why one version was chosen over another.",
  "auto.features.tutorial.components.concepts.guide.literal.30": "Expertise:",
  "auto.features.tutorial.components.concepts.guide.literal.31":
    "You need to understand when to add structure, when to ask for step-by-step reasoning, when to require JSON, and when a long instruction just adds noise.",
  "auto.features.tutorial.components.concepts.guide.literal.32": "Consistency:",
  "auto.features.tutorial.components.concepts.guide.literal.33":
    "The same prompt can behave differently across models, versions, languages, input lengths, and document types.",
  "auto.features.tutorial.components.concepts.guide.literal.34": "Scalability:",
  "auto.features.tutorial.components.concepts.guide.literal.35":
    "An improvement that looks good on five examples won't necessarily hold up on hundreds of examples, edge cases, and new data.",
  "auto.features.tutorial.components.concepts.guide.literal.36": "Skynet's solution:",
  "auto.features.tutorial.components.concepts.guide.literal.37":
    "you define a task, a dataset, and a metric, and let the optimizer search for a better prompt through a measurable, documented, repeatable process",
  "auto.features.tutorial.components.concepts.guide.literal.38": ".",
  "auto.features.tutorial.components.concepts.guide.literal.39": "1.4 What is DSPy?",
  "auto.features.tutorial.components.concepts.guide.literal.40": " (",
  "auto.features.tutorial.components.concepts.guide.literal.41": "Declarative Self-improving Python",
  "auto.features.tutorial.components.concepts.guide.literal.42":
    ") is a Python library for building programs that drive language models. Instead of holding all the logic in a single prompt string, DSPy separates the shape of the task, the module that runs the model, and the metric that checks success. This separation makes it possible to improve the program systematically rather than just editing text by hand.",
  "auto.features.tutorial.components.concepts.guide.literal.43": "Input",
  "auto.features.tutorial.components.concepts.guide.literal.44":
    ": the information the model receives in each example, for example a question, an email, a customer review, or an image.",
  "auto.features.tutorial.components.concepts.guide.literal.45": "Output",
  "auto.features.tutorial.components.concepts.guide.literal.46":
    ": the response the model needs to return, for example a category, a short answer, a list of fields, or JSON.",
  "auto.features.tutorial.components.concepts.guide.literal.47": "Test",
  "auto.features.tutorial.components.concepts.guide.literal.48":
    ": a metric that returns a score, and in GEPA preferably also textual feedback that explains what was good or wrong.",
  "auto.features.tutorial.components.concepts.guide.literal.49":
    "DSPy's core modules, such as Predict and ChainOfThought, drive the LLM according to the input and output fields you defined. DSPy's optimizers use the examples and the metric to improve instructions, select few-shot examples for the prompt, or tune other components of the program.",
  "auto.features.tutorial.components.concepts.guide.literal.50": "1.5 What is an optimizer?",
  "auto.features.tutorial.components.concepts.guide.literal.51":
    "An optimizer is a controlled search process: it tries different versions of a DSPy program, runs them on examples, computes scores, and uses what it learned to propose a better version. For Skynet users, this replaces manual guessing with a process that has a dataset, logs, scores, and an artifact you can inspect.",
  "auto.features.tutorial.components.concepts.guide.literal.52": "Takes a dataset with inputs and desired answers, not just a gut feeling about what works.",
  "auto.features.tutorial.components.concepts.guide.literal.53": "Runs different candidates of the same program and compares them on consistent examples.",
  "auto.features.tutorial.components.concepts.guide.literal.54": "Computes a score using the metric you defined, so the improvement is aimed at what matters to you.",
  "auto.features.tutorial.components.concepts.guide.literal.55": "Uses successes, failures, and feedback to propose focused prompt changes.",
  "auto.features.tutorial.components.concepts.guide.literal.56": "Returns an optimized program you can run through Skynet or integrate through the API.",
  "auto.features.tutorial.components.concepts.guide.literal.57": "Initialization",
  "auto.features.tutorial.components.concepts.guide.literal.58": "Evaluation",
  "auto.features.tutorial.components.concepts.guide.literal.59": "Reflection",
  "auto.features.tutorial.components.concepts.guide.literal.60": "Improvement",
  "auto.features.tutorial.components.concepts.guide.literal.61": "Pareto update",
  "auto.features.tutorial.components.concepts.guide.literal.62":
    "update candidates and repeat until the run budget is exhausted",
  "auto.features.tutorial.components.concepts.guide.literal.63": "The algorithm that searches for better instructions without retraining the model",
  "auto.features.tutorial.components.concepts.guide.literal.64": "2.1 What is GEPA?",
  "auto.features.tutorial.components.concepts.guide.literal.65":
    "GEPA is a reflective optimizer that improves the text components of a program, mainly the prompt instructions. It does not change the LLM's weights. Instead of settling for a numeric score, it also looks at ",
  "auto.features.tutorial.components.concepts.guide.literal.66": "trajectories",
  "auto.features.tutorial.components.concepts.guide.literal.67":
    ": a record of the input, the execution steps, the prediction, the score, and the feedback for each attempt. The reflection model reads these trajectories like a developer doing a code review: it looks for patterns of success and failure, and then proposes new instructions aimed at the problems it found.",
  "auto.features.tutorial.components.concepts.guide.literal.70":
    "2.2 How GEPA works: the practical loop",
  "auto.features.tutorial.components.concepts.guide.literal.71": "Initialization:",
  "auto.features.tutorial.components.concepts.guide.literal.72":
    " you start from the program you defined and its Signature. GEPA measures the baseline so it can tell whether a new candidate actually improved.",
  "auto.features.tutorial.components.concepts.guide.literal.73": "Evaluation:",
  "auto.features.tutorial.components.concepts.guide.literal.74":
    " candidates are run on examples. The train examples provide trajectories and feedback for reflection; the val examples are used to track candidate scores and to choose between them.",
  "auto.features.tutorial.components.concepts.guide.literal.75": "Reflection:",
  "auto.features.tutorial.components.concepts.guide.literal.76":
    " this is the part that sets GEPA apart from blind search. The reflection model (",
  "auto.features.tutorial.components.concepts.guide.literal.77":
    ") receives trajectories, scores, and feedback, and articulates an explanation of recurring errors, missing conditions, and instructions worth changing.",
  "auto.features.tutorial.components.concepts.guide.literal.78": "Improvement:",
  "auto.features.tutorial.components.concepts.guide.literal.79":
    " GEPA creates a new candidate that applies the insights. In some runs it also combines two strong candidates using a merge (",
  "auto.features.tutorial.components.concepts.guide.literal.80": ") of successful variations.",
  "auto.features.tutorial.components.concepts.guide.literal.81": "Pareto update:",
  "auto.features.tutorial.components.concepts.guide.literal.82":
    " the candidate pool and the ",
  "auto.features.tutorial.components.concepts.guide.literal.83": "Pareto",
  "auto.features.tutorial.components.concepts.guide.literal.84":
    " front are updated based on per-example scores on the val set. A candidate can stay important even if it isn't the best on average, as long as it excels on certain examples and isn't completely dominated by another candidate.",
  "auto.features.tutorial.components.concepts.guide.literal.85":
    "To be precise: the diagram is a useful explanation, not every implementation detail. In practice GEPA samples candidates from the Pareto front, runs a minibatch to collect feedback, evaluates a new candidate, updates the candidate pool, and in the end returns the candidate with the best aggregate score on the val set.",
  "auto.features.tutorial.components.concepts.guide.literal.86": "2.3 Why is reflection so important?",
  "auto.features.tutorial.components.concepts.guide.literal.87": "GEPA with reflection",
  "auto.features.tutorial.components.concepts.guide.literal.88": "Selection by score only",
  "auto.features.tutorial.components.concepts.guide.literal.89": "Candidate A",
  "auto.features.tutorial.components.concepts.guide.literal.90":
    "Score 0.30, feedback: failed on questions that require checking two conditions before classifying.",
  "auto.features.tutorial.components.concepts.guide.literal.91": "Candidate B",
  "auto.features.tutorial.components.concepts.guide.literal.92":
    "Score 0.50, feedback: improved when the prompt asked to return an answer only after validating the format and the category.",
  "auto.features.tutorial.components.concepts.guide.literal.93": "Candidate C",
  "auto.features.tutorial.components.concepts.guide.literal.94": "What do we learn from this?",
  "auto.features.tutorial.components.concepts.guide.literal.95":
    "The problem isn't just “weak wording.” An instruction is missing that forces the model to check conditions before the output, so the next improvement should be focused rather than random.",
  "auto.features.tutorial.components.concepts.guide.literal.96": "The next generation:",
  "auto.features.tutorial.components.concepts.guide.literal.97":
    "a new prompt that incorporates the insight and defines a clearer order of operations.",
  "auto.features.tutorial.components.concepts.guide.literal.98": "What's missing without reflection?",
  "auto.features.tutorial.components.concepts.guide.literal.99":
    "The score tells you who did better, but not why. Without feedback and trajectories, the next change is almost a guess.",
  "auto.features.tutorial.components.concepts.guide.literal.100": "Controlling budget, quality, and cost",
  "auto.features.tutorial.components.concepts.guide.literal.101":
    "3.1 Run budget",
  "auto.features.tutorial.components.concepts.guide.literal.102":
    "GEPA is managed by a budget of calls to the metric and to the models, not by a fixed number of rounds. In DSPy you have to choose exactly one budgeting method: auto, max_full_evals, or max_metric_calls. In Skynet, usually start from a light budget, check that the task, the dataset, and the metric are valid, and only then raise the search level.",
  "auto.features.tutorial.components.concepts.guide.literal.103":
    "Automatic budget. Choose ",
  "auto.features.tutorial.components.concepts.guide.literal.104": "\"light\"",
  "auto.features.tutorial.components.concepts.guide.literal.105":
    " for a fast first run and to check the configuration; ",
  "auto.features.tutorial.components.concepts.guide.literal.106": "\"medium\"",
  "auto.features.tutorial.components.concepts.guide.literal.107":
    " once the dataset and feedback are stable and you want a balance between quality and cost; ",
  "auto.features.tutorial.components.concepts.guide.literal.108": "\"heavy\"",
  "auto.features.tutorial.components.concepts.guide.literal.109":
    " when there is a business reason for a wider search and a suitable time and call budget ",
  "auto.features.tutorial.components.concepts.guide.literal.110": "(recommended only after a small run that succeeded)",
  "auto.features.tutorial.components.concepts.guide.literal.111": "Alternative to ",
  "auto.features.tutorial.components.concepts.guide.literal.112":
    ": set a manual budget when it's important to control cost or to compare runs under identical conditions. max_full_evals translates into a number of full evaluations, and max_metric_calls directly limits the number of calls to the metric.",
  "auto.features.tutorial.components.concepts.guide.literal.113":
    "3.2 Candidate merging",
  "auto.features.tutorial.components.concepts.guide.literal.114":
    "Whether to let GEPA perform a merge between candidates (system-aware merge). The merge isn't a free mix of words: it tries to combine prompt components from two parents that improved in different ways. In DSPy the parameter is called use_merge. Default: ",
  "auto.features.tutorial.components.concepts.guide.literal.115":
    "Maximum number of merge attempts. Increase it carefully, since each attempt can add evaluations and model calls. Default: ",
  "auto.features.tutorial.components.concepts.guide.literal.116":
    "3.3 Evaluation, scores, and logging",
  "auto.features.tutorial.components.concepts.guide.literal.117":
    "The number of evaluation threads running in parallel. A high value will shorten the total run time, but will also load the model provider and may trigger rate limit errors. For example ",
  "auto.features.tutorial.components.concepts.guide.literal.118":
    "The score recorded when evaluating an example fails due to an exception, a format failure, or a runtime error. Default: ",
  "auto.features.tutorial.components.concepts.guide.literal.119": "The score that represents a perfect answer. Default: ",
  "auto.features.tutorial.components.concepts.guide.literal.120": " (also used to identify perfect examples that can be skipped during reflection).",
  "auto.features.tutorial.components.concepts.guide.literal.121":
    "Whether to save the optimization's internal statistics and artifacts. Turn it on when you want to understand why a candidate was chosen, compare candidates, or investigate results; turn it off only if it's important to minimize logging.",
  "auto.features.tutorial.components.concepts.guide.literal.122":
    "3.4 Reflection model",
  "auto.features.tutorial.components.concepts.guide.literal.123": "Important to know",
  "auto.features.tutorial.components.concepts.guide.literal.124":
    " GEPA in DSPy needs a reflection model, unless you supply a custom mechanism for proposing instructions. In Skynet you choose a reflection model, and it is the one that reads trajectories and feedback and proposes how to improve the prompt. It's usually worth choosing a stronger model here than the task model, even if the task model itself is cheaper and faster.",
  "auto.features.tutorial.components.concepts.guide.literal.125": "",
  "auto.features.tutorial.components.concepts.guide.literal.126": ".",
  "auto.features.tutorial.components.concepts.guide.literal.127":
    " The language model that performs the reflection. In GEPA runs in Skynet it is required, because without such a model there is no one to read the failures and propose new instructions.",
  "auto.features.tutorial.components.concepts.guide.literal.128":
    "Signature · dataset · metric",
  "auto.features.tutorial.components.concepts.guide.literal.129": "4.1 Signature",
  "auto.features.tutorial.components.concepts.guide.literal.130":
    "The Signature is Python code that defines the shape of the task for DSPy: which fields come in, which fields go out, and what the overall instruction is. Skynet can generate a baseline from the column mapping, but it's worth editing it so that it clarifies the task, the output limits, and the meaning of each field.",
  "auto.features.tutorial.components.concepts.guide.literal.131": "Input fields:",
  "auto.features.tutorial.components.concepts.guide.literal.132": " what the model receives in each example, for example a question, an email, a review, or an image.",
  "auto.features.tutorial.components.concepts.guide.literal.133": "Output fields:",
  "auto.features.tutorial.components.concepts.guide.literal.134":
    " what the model needs to return, for example a category, a short answer, an explanation, or a JSON object.",
  "auto.features.tutorial.components.concepts.guide.literal.135": "Task instruction:",
  "auto.features.tutorial.components.concepts.guide.literal.136": " a short, precise description of the work the model needs to do, including important format constraints.",
  "auto.features.tutorial.components.concepts.guide.literal.137": "4.2 Dataset",
  "auto.features.tutorial.components.concepts.guide.literal.138":
    "A dataset is a collection of examples that show the system what a good answer is. Each row should include the inputs the model will receive and the desired output for comparison. A small, clean, representative dataset is almost always preferable to a large dataset with conflicting labels or mixed formats.",
  "auto.features.tutorial.components.concepts.guide.literal.139": "Dataset split in Skynet:",
  "auto.features.tutorial.components.concepts.guide.literal.140":
    "Instead of a fixed default, Skynet shows in the 'Recommended split' card a split based on the dataset size: tiny sets, with fewer than 30 examples, are all assigned to train and GEPA uses the same set for val as well; small sets, from 30 to 79 examples, get 80/20/0 to leave enough examples for train; and sets of 80 examples and up get the classic 60/20/20 split. From 300 examples, the same split is kept with caps for val and test. In 'Manual selection' mode you can set the ratios yourself, shuffle the rows, and lock a seed for reproducibility.",
  "auto.features.tutorial.components.concepts.guide.literal.141": "Split",
  "auto.features.tutorial.components.concepts.guide.literal.142": "Purpose",
  "auto.features.tutorial.components.concepts.guide.literal.143": " (train)",
  "auto.features.tutorial.components.concepts.guide.literal.144":
    "Provides examples for trajectories, feedback, and ideas for improving candidates.",
  "auto.features.tutorial.components.concepts.guide.literal.145": " (val)",
  "auto.features.tutorial.components.concepts.guide.literal.146":
    "Used to track candidate scores, update the Pareto front, and select the winning program.",
  "auto.features.tutorial.components.concepts.guide.literal.147": " (test)",
  "auto.features.tutorial.components.concepts.guide.literal.148":
    "Used for the final evaluation on examples that were not used to create the candidates or to choose between them.",
  "auto.features.tutorial.components.concepts.guide.literal.149":
    "4.3 Metric for GEPA",
  "auto.features.tutorial.components.concepts.guide.literal.150": "Important:",
  "auto.features.tutorial.components.concepts.guide.literal.151":
    " a metric for GEPA needs to take five parameters: gold, pred, trace, pred_name, and pred_trace. It can return only a numeric score, but to give GEPA real material for reflection it's better to return ",
  "auto.features.tutorial.components.concepts.guide.literal.152": " with ",
  "auto.features.tutorial.components.concepts.guide.literal.153": " and also ",
  "auto.features.tutorial.components.concepts.guide.literal.154": "Scoring examples:",
  "auto.features.tutorial.components.concepts.guide.literal.155": "Identical answer:",
  "auto.features.tutorial.components.concepts.guide.literal.156": " score ",
  "auto.features.tutorial.components.concepts.guide.literal.157": ", with feedback explaining that the output matches the desired output.",
  "auto.features.tutorial.components.concepts.guide.literal.158": "Partial answer:",
  "auto.features.tutorial.components.concepts.guide.literal.159": ", with feedback detailing what's correct, what's missing, and which rule would have helped.",
  "auto.features.tutorial.components.concepts.guide.literal.160": "Wrong answer:",
  "auto.features.tutorial.components.concepts.guide.literal.161": ", with feedback that points to the source of the error in a way that can become a new instruction.",
  "auto.features.tutorial.components.concepts.guide.literal.162": "Cleaning and checking the dataset",
  "auto.features.tutorial.components.concepts.guide.literal.163": "Writing a Signature",
  "auto.features.tutorial.components.concepts.guide.literal.164": "Writing a metric with feedback",
  "auto.features.tutorial.components.concepts.guide.literal.165": "Assembling the API request",
  "auto.features.tutorial.components.concepts.guide.literal.166": "Submitting to the service (POST /run)",
  "auto.features.tutorial.components.concepts.guide.literal.167":
    "Tracking (GET /optimizations/{id})",
  "auto.features.tutorial.components.concepts.guide.literal.168": "Testing the optimized program",
  "auto.features.tutorial.components.concepts.guide.literal.169": "Using the artifact (POST /serve/{id})",
  "auto.features.tutorial.components.concepts.guide.literal.170": "From defining a task to running it in your own code",
  "auto.features.tutorial.components.concepts.guide.literal.171": "5.1 Process overview",
  "auto.features.tutorial.components.concepts.guide.literal.306": "Preparing the task",
  "auto.features.tutorial.components.concepts.guide.literal.307": "You define an input, output, dataset, and a metric that represents real success, not just nice wording.",
  "auto.features.tutorial.components.concepts.guide.literal.308": "Running the optimization",
  "auto.features.tutorial.components.concepts.guide.literal.309": "You send a request, follow the status and logs, and check whether the scores improve over the course of the run.",
  "auto.features.tutorial.components.concepts.guide.literal.310": "Using the result",
  "auto.features.tutorial.components.concepts.guide.literal.311": "You test the artifact on new inputs, download the program, or run it through the serving endpoint.",
  "auto.features.tutorial.components.concepts.guide.literal.312": "max_full_evals / max_metric_calls",
  "auto.features.tutorial.components.concepts.guide.literal.313": "use_merge",
  "auto.features.tutorial.components.concepts.guide.literal.314": "max_merge_invocations",
  "auto.features.tutorial.components.concepts.guide.literal.315": "num_threads",
  "auto.features.tutorial.components.concepts.guide.literal.316": "failure_score",
  "auto.features.tutorial.components.concepts.guide.literal.317": "perfect_score",
  "auto.features.tutorial.components.concepts.guide.literal.318": "track_stats",
  "auto.features.tutorial.components.concepts.guide.literal.319": "module_name",
  "auto.features.tutorial.components.concepts.guide.literal.320": "optimizer_name",
  "auto.features.tutorial.components.concepts.guide.literal.321": "signature_code",
  "auto.features.tutorial.components.concepts.guide.literal.322": "metric_code",
  "auto.features.tutorial.components.concepts.guide.literal.323": "dataset",
  "auto.features.tutorial.components.concepts.guide.literal.324": "column_mapping",
  "auto.features.tutorial.components.concepts.guide.literal.325": "model_config",
  "auto.features.tutorial.components.concepts.guide.literal.326": "reflection_model_config",
  "auto.features.tutorial.components.concepts.guide.literal.327": "optimizer_kwargs",
  "auto.features.tutorial.components.concepts.guide.literal.328": "split_fractions",
  "auto.features.tutorial.components.concepts.guide.literal.329": "shuffle",
  "auto.features.tutorial.components.concepts.guide.literal.330": "seed",
  "auto.features.tutorial.components.concepts.guide.literal.331": "Language model (LLM)",
  "auto.features.tutorial.components.concepts.guide.literal.332": "Prompt",
  "auto.features.tutorial.components.concepts.guide.literal.333": "Optimizer",
  "auto.features.tutorial.components.concepts.guide.literal.334": "Reflection",
  "auto.features.tutorial.components.concepts.guide.literal.335": "Signature",
  "auto.features.tutorial.components.concepts.guide.literal.336": "Metric",
  "auto.features.tutorial.components.concepts.guide.literal.337": "Dataset",
  "auto.features.tutorial.components.concepts.guide.literal.338": "Train / val / test",
  "auto.features.tutorial.components.concepts.guide.literal.339": "Optimization",
  "auto.features.tutorial.components.concepts.guide.literal.340": "Saved artifact",
  "auto.features.tutorial.components.concepts.guide.literal.341": "Full evaluation",
  "auto.features.tutorial.components.concepts.guide.literal.342": "Trajectory",
  "auto.features.tutorial.components.concepts.guide.literal.343": "Step-by-step reasoning",
  "auto.features.tutorial.components.concepts.guide.literal.344": "ReAct (agent with tools)",
  "auto.features.tutorial.components.concepts.guide.literal.345":
    "A module that combines reasoning with tool calls in a loop: the agent picks a tool, calls it, reads the result, and decides the next step, until it returns the defined output. It suits tasks that require access to an external source or multiple steps.",
  "auto.features.tutorial.components.concepts.guide.literal.346":
    " For tasks that require using external tools you can also choose dspy.ReAct: the agent alternates between reasoning and tool calls in a loop. In a ReAct run you define a tool source — a live MCP server or a snapshot of tools from the dataset — and you can filter the tool list. The dataset columns also map recorded trajectories: the steps that were taken, the allowed tools, the schema signatures, and the form state before and after, all of which feed the score computation by replay.",
  "auto.features.tutorial.components.concepts.guide.literal.347": "Sharing and privacy",
  "auto.features.tutorial.components.concepts.guide.literal.348":
    "You can share an optimization Google Drive-style. The owner decides whether the run is private or public, and whether access via the link is restricted or open to anyone who has it, with a viewer or editor role. You can invite users by username, give each one a viewer or editor role, remove members, and even transfer ownership. The settings are in the share button at the top of the optimization page.",
  "auto.features.tutorial.components.concepts.guide.literal.349":
    "After the run finishes, the Usage tab adapts itself to the module type: for Predict and Chain of Thought runs it shows a form where you enter a single input and get a prediction, and for ReAct runs it shows an interactive chat that talks to the agent in real time, with tool calls, approval cards, and trust mode. In addition, for runs that recorded model activity, a model activity tab appears with a breakdown of the number of calls and the average response time for each stage and each model role.",
  "auto.features.tutorial.components.concepts.guide.literal.172": "5.2 Request structure for the service",
  "auto.features.tutorial.components.concepts.guide.literal.173":
    "These are the key fields worth understanding before submitting through the API. You don't have to learn DSPy in depth, but it is important to know what each field changes in the run:",
  "auto.features.tutorial.components.concepts.guide.literal.174": "The module type. For tasks that require an explanation, condition checking, or several steps, usually choose ",
  "auto.features.tutorial.components.concepts.guide.literal.175": "\"dspy.ChainOfThought\"",
  "auto.features.tutorial.components.concepts.guide.literal.176": " (adds a ",
  "auto.features.tutorial.components.concepts.guide.literal.177":
    " field before the output; for simple, short, fast tasks you can use dspy.Predict).",
  "auto.features.tutorial.components.concepts.guide.literal.178": "The optimizer name (",
  "auto.features.tutorial.components.concepts.guide.literal.179": "\"dspy.GEPA\"",
  "auto.features.tutorial.components.concepts.guide.literal.180": "Python code that defines the Signature, the input fields, the output fields, and the task instruction.",
  "auto.features.tutorial.components.concepts.guide.literal.181": "Python code for the metric. In GEPA make sure it takes five parameters and returns a score, and preferably also feedback.",
  "auto.features.tutorial.components.concepts.guide.literal.182": "A list of the dataset rows in JSON format. Each row should match the field mapping.",
  "auto.features.tutorial.components.concepts.guide.literal.183":
    "A mapping between the field names in the Signature and the column names in the dataset. For example, if the field is ",
  "auto.features.tutorial.components.concepts.guide.literal.184": " and in the dataset the column is called ",
  "auto.features.tutorial.components.concepts.guide.literal.185": "The config of the task model that produces the answers during optimization and at serving time.",
  "auto.features.tutorial.components.concepts.guide.literal.186": "The config of the reflection model (",
  "auto.features.tutorial.components.concepts.guide.literal.187": "required in GEPA",
  "auto.features.tutorial.components.concepts.guide.literal.188": "Parameters for the optimizer (such as ",
  "auto.features.tutorial.components.concepts.guide.literal.189": "auto=\"heavy\"",
  "auto.features.tutorial.components.concepts.guide.literal.190":
    "(optional) the split ratio for train, val, and test. Keep a separate test set when it's important to understand performance on examples that weren't used for selection. For example ",
  "auto.features.tutorial.components.concepts.guide.literal.191": "[0.6, 0.2, 0.2]",
  "auto.features.tutorial.components.concepts.guide.literal.192":
    "(optional) whether to shuffle the examples before the split. Turn it on when the source file is ordered by time, customer, or category.",
  "auto.features.tutorial.components.concepts.guide.literal.193":
    "(optional) a fixed seed that lets you reproduce the same split and the same shuffle across runs.",
  "auto.features.tutorial.components.concepts.guide.literal.194":
    "5.3 Key endpoints",
  "auto.features.tutorial.components.concepts.guide.literal.195":
    "These are the key endpoints for working with the service. In the browser, Skynet uses the same steps, and the API lets you integrate them into your own code:",
  "auto.features.tutorial.components.concepts.guide.literal.196": "URL",
  "auto.features.tutorial.components.concepts.guide.literal.197":
    "Submitting a new optimization run with a dataset, Signature, metric, task model, and reflection model.",
  "auto.features.tutorial.components.concepts.guide.literal.198": "Running a grid search that compares several pairs of task model and reflection model on the same task.",
  "auto.features.tutorial.components.concepts.guide.literal.199":
    "A list of runs, including status, filtering, pagination, and summary data.",
  "auto.features.tutorial.components.concepts.guide.literal.200":
    "The full state and details of a specific run: config, scores, stages, and artifacts.",
  "auto.features.tutorial.components.concepts.guide.literal.201":
    "A condensed summary for display on the dashboard or in lists.",
  "auto.features.tutorial.components.concepts.guide.literal.202": "The run's logs, including filtering by level and source.",
  "auto.features.tutorial.components.concepts.guide.literal.203": "Real-time progress updates over SSE, useful for monitoring screens and an interface that updates in real time.",
  "auto.features.tutorial.components.concepts.guide.literal.204":
    "Downloading the saved artifact of the optimized program for inspection, archiving, or use outside the interface.",
  "auto.features.tutorial.components.concepts.guide.literal.205": "The model grid-search results when the run is a Grid Search.",
  "auto.features.tutorial.components.concepts.guide.literal.206": "Cancelling an active run.",
  "auto.features.tutorial.components.concepts.guide.literal.207":
    "Cloning a run's config as a basis for a new run, for example to swap a model or budget.",
  "auto.features.tutorial.components.concepts.guide.literal.208":
    "Retrying a failed run without rebuilding the request.",
  "auto.features.tutorial.components.concepts.guide.literal.209":
    "Information about the input fields the optimized program expects to receive before you run it.",
  "auto.features.tutorial.components.concepts.guide.literal.210":
    "Running the optimized program on new input and getting a prediction through the API.",
  "auto.features.tutorial.components.concepts.guide.literal.211": "Service health check.",
  "auto.features.tutorial.components.concepts.guide.literal.212": "Overall queue state.",
  "auto.features.tutorial.components.concepts.guide.literal.213": "5.4 Monitoring progress",
  "auto.features.tutorial.components.concepts.guide.literal.214":
    "Common statuses in an optimization run and what to do about them:",
  "auto.features.tutorial.components.concepts.guide.literal.215": ": waiting in the queue.",
  "auto.features.tutorial.components.concepts.guide.literal.216": ": GEPA or the evaluation stages are running now. Follow the logs and intermediate scores.",
  "auto.features.tutorial.components.concepts.guide.literal.217": ": completed successfully and the artifact is available for inspection and use.",
  "auto.features.tutorial.components.concepts.guide.literal.218": ": failed. Open the logs and check the JSON, Python code, column mapping, model permissions, and reflection model.",
  "auto.features.tutorial.components.concepts.guide.literal.219": ": cancelled by the user.",
  "auto.features.tutorial.components.concepts.guide.literal.220": "Stable improvement starts with a good setup",
  "auto.features.tutorial.components.concepts.guide.literal.221": "6.1 Preparing a dataset",
  "auto.features.tutorial.components.concepts.guide.literal.222": "Quality over quantity:",
  "auto.features.tutorial.components.concepts.guide.literal.223":
    " 50 consistent, accurate examples are preferable to 200 noisy ones. GEPA learns from the examples and from the feedback you give it.",
  "auto.features.tutorial.components.concepts.guide.literal.224": "Diversity:",
  "auto.features.tutorial.components.concepts.guide.literal.225":
    " include common cases, edge cases, and examples that are easy to get wrong: text that's too short, long input, similar categories, and imperfect formats.",
  "auto.features.tutorial.components.concepts.guide.literal.226": "Cleanliness:",
  "auto.features.tutorial.components.concepts.guide.literal.227":
    " remove duplicates, conflicting labels, and rows where the desired output isn't clear. If humans don't agree on the answer, the optimization will struggle too.",
  "auto.features.tutorial.components.concepts.guide.literal.228": "Representativeness:",
  "auto.features.tutorial.components.concepts.guide.literal.229":
    " keep a format and distribution similar to the data that will arrive in real use. Don't build a dataset made up of only easy examples.",
  "auto.features.tutorial.components.concepts.guide.literal.230":
    "6.2 Writing a good metric for GEPA",
  "auto.features.tutorial.components.concepts.guide.literal.231": "Detailed feedback:",
  "auto.features.tutorial.components.concepts.guide.literal.232":
    " return feedback that explains exactly what is correct or incorrect, not just that the answer is wrong.",
  "auto.features.tutorial.components.concepts.guide.literal.233": "Useful feedback:",
  "auto.features.tutorial.components.concepts.guide.literal.234":
    " write what's worth changing in the direction of the prompt: a missing format, a condition that wasn't checked, confusion between categories, or a field the model ignored.",
  "auto.features.tutorial.components.concepts.guide.literal.235": "Graded score:",
  "auto.features.tutorial.components.concepts.guide.literal.236":
    " use partial scoring when there's a partial answer. Only 0 or 1 suits simple tasks, but makes it harder for GEPA to learn subtle differences.",
  "auto.features.tutorial.components.concepts.guide.literal.237":
    " give similar scores to similar cases. Consistency helps ",
  "auto.features.tutorial.components.concepts.guide.literal.238": "6.3 Troubleshooting common issues",
  "auto.features.tutorial.components.concepts.guide.literal.239": "Problem",
  "auto.features.tutorial.components.concepts.guide.literal.240": "Solution",
  "auto.features.tutorial.components.concepts.guide.literal.241": "The run fails immediately",
  "auto.features.tutorial.components.concepts.guide.literal.242":
    "Check that the JSON is valid, that the column mapping covers all the fields, that the metric takes five parameters, and that a reflection model was selected.",
  "auto.features.tutorial.components.concepts.guide.literal.243": "Poor results",
  "auto.features.tutorial.components.concepts.guide.literal.244":
    "First check the dataset and the metric's feedback. A higher budget won't fix conflicting labels or a metric that rewards wrong answers.",
  "auto.features.tutorial.components.concepts.guide.literal.245": "Very slow run",
  "auto.features.tutorial.components.concepts.guide.literal.246": "Use ",
  "auto.features.tutorial.components.concepts.guide.literal.247":
    " for a first experiment; if the model provider limits concurrency or returns rate limit errors, lower the ",
  "auto.features.tutorial.components.concepts.guide.literal.248": "The metric always returns 0",
  "auto.features.tutorial.components.concepts.guide.literal.249":
    "Check the comparison logic and add basic normalization: stripping extra whitespace, casing consistency, handling valid JSON, and comparing categories against a closed list.",
  "auto.features.tutorial.components.concepts.guide.literal.250": "Reflection doesn't improve",
  "auto.features.tutorial.components.concepts.guide.literal.251":
    "Check that the feedback explains errors in a way that can be turned into instructions. If it does, try a stronger reflection model, a more diverse train dataset, or a higher budget.",
  "auto.features.tutorial.components.concepts.guide.literal.252": "Quick reference",
  "auto.features.tutorial.components.concepts.guide.literal.253": "Term",
  "auto.features.tutorial.components.concepts.guide.literal.254": "Explanation",
  "auto.features.tutorial.components.concepts.guide.literal.255": "A model that takes input in natural language, code, or structured data and returns a response, summary, classification, or structured output.",
  "auto.features.tutorial.components.concepts.guide.literal.256": "The instruction that steers the model: what to do, in what format to answer, and what to prefer or avoid.",
  "auto.features.tutorial.components.concepts.guide.literal.257":
    "A Python library for building programs that drive language models and improve them based on examples and metrics.",
  "auto.features.tutorial.components.concepts.guide.literal.258":
    "An algorithm that tries to improve a DSPy program based on a dataset, a metric, and a run budget.",
  "auto.features.tutorial.components.concepts.guide.literal.259":
    "A reflective optimizer that improves instructions using trajectories, feedback, minibatches, and Pareto selection.",
  "auto.features.tutorial.components.concepts.guide.literal.260":
    "A stage where the reflection model analyzes successes and failures and proposes a prompt change based on evidence from the run.",
  "auto.features.tutorial.components.concepts.guide.literal.261":
    "The Signature code that defines the input fields, output fields, and task instruction for DSPy.",
  "auto.features.tutorial.components.concepts.guide.literal.262":
    "A function that checks an answer and returns a score. In GEPA it's recommended to also return textual feedback that explains the score.",
  "auto.features.tutorial.components.concepts.guide.literal.263":
    "A collection of input and desired-answer examples used to create candidates, compare them, and test.",
  "auto.features.tutorial.components.concepts.guide.literal.264":
    "A split into examples that produce feedback, examples that select candidates, and examples for the final test.",
  "auto.features.tutorial.components.concepts.guide.literal.265":
    "A process in which Skynet searches for a better DSPy program based on the configuration you provided.",
  "auto.features.tutorial.components.concepts.guide.literal.266": "The optimized program that is saved at the end of the run and available for download or running through the API.",
  "auto.features.tutorial.components.concepts.guide.literal.267":
    "A run of a candidate on a set of examples wide enough to compute a comparative score.",
  "auto.features.tutorial.components.concepts.guide.literal.268":
    "A record of the input, the execution steps, the prediction, the score, and the feedback for a given attempt.",
  "auto.features.tutorial.components.concepts.guide.literal.269":
    "A DSPy module that adds a reasoning field before the output, so it suits tasks that require condition checking, explanation, or a multi-step solution.",
  "auto.features.tutorial.components.concepts.guide.literal.270":
    "The language model that analyzes the trajectories and feedback and proposes prompt improvements.",
  "auto.features.tutorial.components.concepts.guide.literal.271": "DSPy",
  "auto.features.tutorial.components.concepts.guide.literal.272": "GEPA",
  "auto.features.tutorial.components.concepts.guide.literal.273": "/serve/{id}",
  "auto.features.tutorial.components.concepts.guide.literal.274": "reflection_lm",
  "auto.features.tutorial.components.concepts.guide.literal.275": "merge",
  "auto.features.tutorial.components.concepts.guide.literal.276": "auto",
  "auto.features.tutorial.components.concepts.guide.literal.277": "True",
  "auto.features.tutorial.components.concepts.guide.literal.281": "Train",
  "auto.features.tutorial.components.concepts.guide.literal.282": "Validation",
  "auto.features.tutorial.components.concepts.guide.literal.283": "Test",
  "auto.features.tutorial.components.concepts.guide.literal.284": "dspy.Prediction",
  "auto.features.tutorial.components.concepts.guide.literal.285": "score",
  "auto.features.tutorial.components.concepts.guide.literal.286": "feedback",
  "auto.features.tutorial.components.concepts.guide.literal.287": "reasoning",
  "auto.features.tutorial.components.concepts.guide.literal.288": "question",
  "auto.features.tutorial.components.concepts.guide.literal.289": "q",
  "auto.features.tutorial.components.concepts.guide.literal.290": "pending",
  "auto.features.tutorial.components.concepts.guide.literal.291": "running",
  "auto.features.tutorial.components.concepts.guide.literal.292": "success",
  "auto.features.tutorial.components.concepts.guide.literal.293": "failed",
  "auto.features.tutorial.components.concepts.guide.literal.294": "cancelled",
  "auto.features.tutorial.components.concepts.guide.literal.295": 'auto="light"',
  "auto.features.tutorial.components.concepts.guide.literal.296": "num_threads",
  "auto.features.tutorial.components.concepts.guide.literal.297":
    "Skynet is a service for optimizing prompts. Instead of guessing wording and testing it by hand, you upload a dataset, define a metric, choose models, and Skynet searches for a DSPy program that returns better answers. Behind the scenes it uses ",
  "auto.features.tutorial.components.concepts.guide.literal.298": " and the optimizer ",
  "auto.features.tutorial.components.concepts.guide.literal.299": ".",
  "auto.features.tutorial.components.concepts.guide.literal.300": `import dspy

def score_answer(gold, pred, trace, pred_name, pred_trace):
    """Return a score and constructive feedback for the reflection LM."""
    expected = gold.answer.strip().lower()
    actual = (pred.answer or "").strip().lower()

    if expected == actual:
        return dspy.Prediction(score=1.0, feedback="The answer matches the desired output.")
    if expected in actual:
        return dspy.Prediction(
            score=0.5,
            feedback=f"Partial answer. The desired output is {expected}, and we got {actual}.",
        )
    return dspy.Prediction(
        score=0.0,
        feedback=f"The answer does not match. The desired output is {expected}, and we got {actual}.",
    )
`,
  "auto.features.tutorial.components.concepts.guide.literal.301": "validating",
  "auto.features.tutorial.components.concepts.guide.literal.302": ": the input, the code, the column mapping, and the models are checked before the run starts.",
  "auto.features.tutorial.components.concepts.guide.literal.303": "Score 0.30 only, with no explanation of why the candidate failed.",
  "auto.features.tutorial.components.concepts.guide.literal.304": "Score 0.50 only, with no information about the instruction that helped.",
  "auto.features.tutorial.components.concepts.guide.literal.305": "Score 0.40 only, with no way to tell whether it's good for edge cases.",
  "auto.features.tutorial.components.concepts.guide.template.1": "Part {p1}",
};
