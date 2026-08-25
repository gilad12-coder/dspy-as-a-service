/**
 * Builds a default DSPy metric from a column-role map.
 *
 * Compares each output column on the prediction to its labeled value
 * on the gold example. Single-output tasks degenerate to a 0/1 score;
 * multi-output tasks get partial credit (average of matches).
 *
 * String comparison is case-insensitive + whitespace-trimmed so that
 * trivial formatting differences ("Yes " vs "yes") don't tank scores.
 */
export function buildMetricTemplate(
  roles: Record<string, "input" | "output" | "ignore" | string>,
): string {
  const outputs = Object.entries(roles)
    .filter(([, r]) => r === "output")
    .map(([c]) => c);
  const toId = (s: string) => s.replace(/[^a-zA-Z0-9_\u0590-\u05FF]/g, "_").replace(/^(\d)/, "_$1");

  // Fallback when no roles are set yet — mirrors buildSignatureTemplate's
  // fallback so the editor always compiles even before the user maps columns.
  const fields = outputs.length > 0 ? outputs.map(toId) : ["output_field"];
  const fieldsLiteral = `[${fields.map((f) => `"${f}"`).join(", ")}]`;

  return `def metric(gold: dspy.Example, pred: dspy.Prediction, trace: bool = None, pred_name: str = None, pred_trace: list = None) -> dspy.Prediction:
    fields = ${fieldsLiteral}
    total = len(fields)
    correct = 0
    mismatches = []
    field_scores = {}
    for f in fields:
        expected = getattr(gold, f, None)
        actual = getattr(pred, f, None)
        if expected is None:
            continue
        matched = str(actual).strip().lower() == str(expected).strip().lower()
        field_scores[f] = 1.0 if matched else 0.0
        if matched:
            correct += 1
        else:
            mismatches.append(f"{f}: expected {expected!r}, got {actual!r}")
    if len(field_scores) > 1:
        # Surface each field's match rate alongside the averaged score.
        log_metrics(**field_scores)
    score = correct / total if total else 0.0
    feedback = "Matches all outputs." if not mismatches else "Mismatches: " + "; ".join(mismatches)
    return dspy.Prediction(score=score, feedback=feedback)
`;
}
