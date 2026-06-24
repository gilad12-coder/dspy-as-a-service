// English UI strings for the trajectory slice. Edit directly; keys missing here fall
// back to the Hebrew slice via msg(), so partial translations are safe.

import type { trajectoryMessages } from "./messages";

export const trajectoryMessagesEn: Partial<Record<keyof typeof trajectoryMessages, string>> = {
  "trajectory.panel.title": "Candidate tree",

  "trajectory.live.new_candidate": "New candidate received",

  "trajectory.node.winning_label": "Winning",

  "trajectory.detail.pareto_title": "Validation examples",
  "trajectory.detail.pareto_title.explain": "Each square is a validation example. Green = correct prediction, red = incorrect prediction. Click to open the example.",
  "trajectory.detail.pareto_example_label": "Example {id}, score {score}",
  "trajectory.detail.pareto_passed": "{count} of {total} passed",
  "trajectory.detail.diff_unchanged": "Unchanged from parent",

  "trajectory.drawer.section.minibatch": "Mini-batch feedback",
  "trajectory.drawer.section.minibatch.explain": "mini-batch = a small subset of examples the proposal was tested on before deciding whether to accept it.",
  "trajectory.drawer.toggle.aria": "Prompt view",
  "trajectory.drawer.toggle.prompt": "Prompt",
  "trajectory.drawer.toggle.diff": "Diff",
  "trajectory.drawer.rejected.prompt_title": "Proposed and rejected prompt",
  "trajectory.drawer.rejected.prompt_title.explain": "The difference between the parent prompt and the prompt the reflection model produced and that was rejected. Green lines were added, red lines were removed.",
  "trajectory.drawer.rejected.prompt_unavailable": "The proposal text was not saved in this run",

  "trajectory.pareto.cell_detail_pending": "Example content has not loaded from the server yet",
  "trajectory.pareto.cell.inputs_label": "Input",
  "trajectory.pareto.cell.outputs_label": "Expected prediction",
  "trajectory.pareto.cell.prediction_label": "Candidate prediction",
  "trajectory.pareto.cell.prediction_unavailable": "Prediction not found",
  "trajectory.pareto.cell.inputs_label.explain": "The data shown to the candidate from the validation example.",
  "trajectory.pareto.cell.prediction_label.explain": "The prediction the candidate made for this input during the run over the validation examples.",
  "trajectory.pareto.cell.outputs_label.explain": "The correct prediction according to the validation example; the score is set by comparing against it.",
  "trajectory.pareto.cell.details_label": "More details",
  "trajectory.pareto.cell.allowed_tools_label": "Available tools",
  "trajectory.minibatch.no_data": "No mini-batch feedback is available at this stage",
  "trajectory.minibatch.score_label": "Score",
  "trajectory.minibatch.score_label.explain": "The score the metric returned for this example alone. Higher = a better prediction according to the metric. Intermediate values indicate partial credit.",
  "trajectory.minibatch.feedback_label": "Reflection feedback",
  "trajectory.minibatch.feedback_label.explain": "The feedback the reflection model wrote after evaluation — explaining why the prediction is correct or incorrect.",
  "trajectory.minibatch.pass_label": "Example that passed validation",
  "trajectory.minibatch.fail_label": "Example that failed validation",

  "trajectory.ghost.legend": "Rejected proposals",

  "trajectory.outline.best": "Best",
  "trajectory.outline.rejected_row": "Proposal {id}",

  "trajectory.node.header.accepted_title": "Candidate {id}",
  "trajectory.node.header.rejected_title": "Rejected proposal",
  "trajectory.node.header.label.iteration": "Iteration",
  "trajectory.node.header.label.score_valset": "Validation score",
  "trajectory.node.header.label.score_minibatch": "Rejected proposal score",
  "trajectory.node.header.label.parent_score": "Parent score",
  "trajectory.node.header.sub.examples": "{n} examples",

  "trajectory.node.section.prompt": "Prompt",
  "trajectory.node.section.prompt.explain": "The agent instructions for this candidate — the text the reflection model changes to improve performance.",
  "trajectory.prompt.react.tools": "Tool descriptions ({n})",
  "trajectory.prompt.react.tools.explain": "The description of each tool and its arguments, as updated during the optimization. Page through the tools to see the full detail.",
  "trajectory.prompt.react.tools_carousel_aria": "Paging through tool descriptions",
  "trajectory.prompt.react.tools.view_aria": "Tool descriptions view",
  "trajectory.prompt.react.tools.view_plain": "Description",
  "trajectory.prompt.react.tools.view_compare": "Compare",
  "trajectory.prompt.react.tool.added": "Added",
  "trajectory.prompt.react.tool.removed": "Removed",
  "trajectory.prompt.react.tool.changed": "Changed",
  "trajectory.json.empty_value": "Empty",
  "trajectory.chat.recorded_label": "Recorded conversation",
  "trajectory.chat.recorded_label.explain": "A saved record of the message exchange shown to the candidate — read-only, not a live conversation.",
  "trajectory.chat.recorded_count": "{n} messages",
  "trajectory.history.turn": "Turn {n}",
  "trajectory.node.section.score_detail.valset": "Scores per validation example",

  "trajectory.explainer.trajectory": "The sequence of candidates accepted over the run",

  "trajectory.controls.zoom_in": "Zoom in",
  "trajectory.controls.zoom_out": "Zoom out",
  "trajectory.controls.zoom_reset": "Reset view",
  "trajectory.controls.fullscreen_enter": "Enter fullscreen",
  "trajectory.controls.fullscreen_exit": "Exit fullscreen",

  "trajectory.scrubber.label": "Filter by generation",
  "trajectory.scrubber.live": "Live",
  "trajectory.scrubber.generation_value": "Generation {gen}",

  "trajectory.a11y.tree_label": "The optimization's candidate tree",
  "trajectory.a11y.node_label": "Candidate {id}, generation {gen}, score {score}",
};
