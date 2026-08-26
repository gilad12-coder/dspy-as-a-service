/** Guard the Hebrew tutorial copy against product and API drift. */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

const TITLE_KEYS = [
  "auto.features.tutorial.lib.steps.template.24",
  "auto.features.tutorial.lib.steps.template.27",
] as const;
const GUIDE_PREFIX = "auto.features.tutorial.components.concepts.guide.";

function readHebrewCatalog(): Record<string, string> {
  return JSON.parse(
    fs.readFileSync(path.join(process.cwd(), "..", "i18n", "locales", "ui", "he.json"), "utf8"),
  ) as Record<string, string>;
}

test("tutorial titles are complete Hebrew phrases", () => {
  const messages = readHebrewCatalog();

  assert.equal(messages[TITLE_KEYS[0]], "בחירת מודלים");
  assert.equal(messages[TITLE_KEYS[1]], "שליחת אופטימיזציה");
  for (const key of TITLE_KEYS) {
    assert.doesNotMatch(messages[key] ?? "", /\{p\d+\}/, `${key} must be a complete phrase`);
  }
});

test("tutorial title call sites do not compose grammar from placeholders", () => {
  const source = fs.readFileSync(
    path.join(process.cwd(), "src", "features", "tutorial", "lib", "steps.ts"),
    "utf8",
  );

  assert.match(source, new RegExp(`title: msg\\("${TITLE_KEYS[0].replaceAll(".", "\\.")}\"\\)`));
  assert.doesNotMatch(source, new RegExp(`title: formatMsg\\("${TITLE_KEYS[0].replaceAll(".", "\\.")}`));
});

test("the concepts guide matches the current optimization surface", () => {
  const source = fs.readFileSync(
    path.join(process.cwd(), "src", "features", "tutorial", "components", "concepts-guide.tsx"),
    "utf8",
  );
  const messages = readHebrewCatalog();
  const referencedKeys = new Set(
    [...source.matchAll(/msg\("(auto\.features\.tutorial\.components\.concepts\.guide\.[^"]+)"\)/g)].map(
      (match) => match[1]!,
    ),
  );
  const guideCopy = [...referencedKeys].map((key) => messages[key]).join("\n");
  const contract = `${source}\n${guideCopy}`;

  for (const key of referencedKeys) {
    assert.ok(messages[key]?.trim(), `${key} must be present and non-empty`);
  }
  for (const currentTerm of [
    "Flex",
    "workflow_definition",
    "POST /workflows/dry-run",
    "reflection_minibatch_size",
    "pxn_parents",
    "pxn_proposals",
    "target_score",
    "token_source",
    "temperature",
    "max_tokens",
    "paused",
  ]) {
    assert.match(contract, new RegExp(currentTerm.replaceAll("/", "\\/")), currentTerm);
  }
  for (const staleTerm of [
    "max_cost_credits",
    "max_merge_invocations",
    "failure_score",
    "perfect_score",
    "track_stats",
    "snapshot of tools from the dataset",
  ]) {
    assert.doesNotMatch(source, new RegExp(staleTerm), staleTerm);
  }
});

test("Hebrew guide copy preserves runtime identifiers and numeric rules", () => {
  const messages = readHebrewCatalog();
  const prefix = `${GUIDE_PREFIX}literal.`;
  const identifierKeys = [
    274, 290, 291, 292, 293, 294, 295, 301, 312, 313, 314, 315, 316, 318, 319, 321, 322, 323,
    324, 325, 326, 327, 328, 331, 332, 344,
  ];

  for (const number of identifierKeys) {
    assert.match(messages[`${prefix}${number}`] ?? "", /[A-Za-z0-9_./{}=" -]+/);
  }
  for (const identifier of ["gold", "pred", "trace", "pred_name", "pred_trace", "dspy.Prediction"]) {
    assert.match(messages[`${prefix}155`], new RegExp(identifier.replace(".", "\\.")));
  }
  for (const number of [30, 79, 80, 60, 20, 300, 200, 500]) {
    assert.match(messages[`${prefix}151`], new RegExp(`\\b${number}\\b`));
  }
});
