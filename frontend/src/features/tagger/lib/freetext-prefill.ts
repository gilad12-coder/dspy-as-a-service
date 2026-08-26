import type { Annotation, AssistPrediction } from "./types";

/** Fill empty free-text rows from AI predictions without replacing human text. */
export function prefillFreetextPredictions(
  annotations: Record<string, Annotation>,
  predictions: Record<string, AssistPrediction>,
  rowIds: readonly string[] = Object.keys(predictions),
): Record<string, Annotation> {
  let next = annotations;
  for (const id of rowIds) {
    const current = annotations[id];
    if (typeof current === "string" && current.trim()) continue;
    const predicted = predictions[id]?.value;
    if (typeof predicted !== "string" || !predicted.trim()) continue;
    if (next === annotations) next = { ...annotations };
    next[id] = predicted;
  }
  return next;
}
