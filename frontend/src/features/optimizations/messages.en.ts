// English UI strings for the optimizations slice. Edit directly; keys missing here fall
// back to the Hebrew slice via msg(), so partial translations are safe.

import type { optimizationsMessages } from "./messages";

export const optimizationsMessagesEn: Partial<Record<keyof typeof optimizationsMessages, string>> = {
  "optimization.cancel.sent": "Cancellation request sent",
  "optimization.cancel.failed": "Cancellation failed",
  "optimization.rerun": "Run again",
  "optimization.rerun_tooltip": "Create a new optimization based on this one",
  "optimization.rerun.success": "New optimization created",
  "optimization.rerun.failed": "Couldn't create a new optimization",
  "optimization.resume": "Resume",
  "optimization.resume_tooltip": "Resume the run from where it stopped",
  "optimization.resume.success": "Run resuming",
  "optimization.resume.failed": "Couldn't resume the run",
  "optimization.pause": "Pause",
  "optimization.pause_tooltip": "Pause the run — you can resume it from the same point",
  "optimization.pause.success": "Run paused",
  "optimization.pause.failed": "Couldn't pause the run",
  "optimization.pair.restart": "Restart",
  "optimization.pair.restart_tooltip": "Restart this pair",
  "optimization.pair.restart.success": "Pair restarting",
  "optimization.pair.restart.failed": "Couldn't restart the pair",
  "optimization.pair.resume": "Resume",
  "optimization.pair.resume_tooltip": "Resume the pair from where it stopped",
  "optimization.pair.resume.success": "Pair resuming",
  "optimization.pair.resume.failed": "Couldn't resume the pair",
  "optimization.delete.failed": "Delete failed",
  "optimization.storage_label": "Run storage usage — click to manage",
  "optimization.file.parse_error": "Error parsing the file",
  "optimization.progress.gepa": "GEPA optimization",
  "optimizations.react.optimized_tools": "Optimized tools (ReAct)",
  "optimizations.react.chat_empty_title": "Chat with the agent",
  "optimizations.react.chat_empty_desc":
    "Send a message to start a conversation with the optimized ReAct agent and the tools available to it.",
  "optimizations.react.chat_placeholder": "Write a message to the agent…",
  "optimizations.react.chat_send_aria": "Send message",
  "optimizations.react.chat_stop_aria": "Stop the conversation",
  "optimizations.react.chat_retry": "Try again",
  "optimizations.react.api_title": "Service API",
  "optimizations.logs.verbosity.aria": "Log verbosity level",
  "optimizations.logs.verbosity.quiet": "Quiet",
  "optimizations.logs.verbosity.normal": "Normal",
  "optimizations.logs.verbosity.verbose": "Verbose",
  "optimizations.logs.verbosity.empty_quiet": "No warnings or errors in this run",
  "optimizations.logs.verbosity.empty_filtered": "No logs match the filter",
  "optimizations.datatab.description":
    "The data used in the {term.optimization} — split into {term.splitTrain}, {term.splitVal}, and {term.splitTest}, with results for each example.",
  "optimizations.lmactivity.description":
    "Language-model activity by phase — how many calls there were and how long they took, for the {term.generationModelShort} and the reflection model separately.",
  "optimizations.source_dataset.label": "Data source: {term.dataset} from the library",
  "optimizations.source_dataset.view": "Go to {term.dataset}",
} as const;
