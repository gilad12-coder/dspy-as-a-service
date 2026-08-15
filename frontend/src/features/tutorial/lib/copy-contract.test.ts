import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

const TITLE_KEYS = [
  "auto.features.tutorial.lib.steps.template.24",
  "auto.features.tutorial.lib.steps.template.27",
] as const;
const USED_TITLE_KEYS = [TITLE_KEYS[0]] as const;

test("tutorial titles are complete localized phrases", () => {
  const localeDir = path.join(process.cwd(), "..", "i18n", "locales", "ui");
  const catalogs = fs
    .readdirSync(localeDir)
    .filter((name) => name.endsWith(".json"))
    .map((name) => ({
      name,
      messages: JSON.parse(fs.readFileSync(path.join(localeDir, name), "utf8")) as Record<
        string,
        string
      >,
    }));

  for (const { name, messages } of catalogs) {
    for (const key of TITLE_KEYS) {
      const value = messages[key];
      if (value === undefined) continue;
      assert.doesNotMatch(value, /\{p\d+\}/, `${name}:${key} must be a complete phrase`);
    }
  }

  const english = catalogs.find(({ name }) => name === "en.json")?.messages;
  assert.equal(english?.[TITLE_KEYS[0]], "Choosing models");
  assert.equal(english?.[TITLE_KEYS[1]], "Submitting an optimization");
});

test("tutorial title call sites do not compose grammar from term placeholders", () => {
  const source = fs.readFileSync(
    path.join(process.cwd(), "src", "features", "tutorial", "lib", "steps.ts"),
    "utf8",
  );

  for (const key of USED_TITLE_KEYS) {
    assert.match(source, new RegExp(`title: msg\\("${key.replaceAll(".", "\\.")}"\\)`));
    assert.doesNotMatch(source, new RegExp(`title: formatMsg\\("${key.replaceAll(".", "\\.")}"`));
  }
});

const GUIDE_PREFIX = "auto.features.tutorial.components.concepts.guide.";
const FULL_UI_LOCALES = [
  "ar",
  "de",
  "en",
  "es",
  "fa",
  "fr",
  "he",
  "hi",
  "it",
  "ja",
  "ko",
  "pt",
  "ru",
  "tr",
  "uk",
  "zh-Hans",
] as const;

test("the concepts guide matches the current optimization surface", () => {
  const source = fs.readFileSync(
    path.join(process.cwd(), "src", "features", "tutorial", "components", "concepts-guide.tsx"),
    "utf8",
  );
  const english = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), "..", "i18n", "locales", "ui", "en.json"), "utf8"),
  ) as Record<string, string>;
  const guideCopy = Object.entries(english)
    .filter(([key]) => key.startsWith(GUIDE_PREFIX))
    .map(([, value]) => value)
    .join("\n");
  const contract = `${source}\n${guideCopy}`;

  for (const currentTerm of [
    "Flex",
    "workflow_definition",
    "POST /workflows/dry-run",
    "reflection_minibatch_size",
    "pxn_parents",
    "pxn_proposals",
    "target_score",
    "max_cost_credits",
    "token_source",
    "temperature",
    "max_tokens",
    "paused",
  ]) {
    assert.match(contract, new RegExp(currentTerm.replaceAll("/", "\\/")), currentTerm);
  }

  for (const staleTerm of [
    "max_merge_invocations",
    "failure_score",
    "perfect_score",
    "track_stats",
    "snapshot of tools from the dataset",
  ]) {
    assert.doesNotMatch(source, new RegExp(staleTerm), staleTerm);
  }
});

test("the concepts guide avoids canned AI-writing patterns", () => {
  const source = fs.readFileSync(
    path.join(process.cwd(), "src", "features", "tutorial", "components", "concepts-guide.tsx"),
    "utf8",
  );
  const english = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), "..", "i18n", "locales", "ui", "en.json"), "utf8"),
  ) as Record<string, string>;
  const referencedKeys = new Set(
    [
      ...source.matchAll(
        /msg\("(auto\.features\.tutorial\.components\.concepts\.guide\.[^"]+)"\)/g,
      ),
    ].map((match) => match[1]!),
  );
  const guideCopy = [...referencedKeys].map((key) => english[key]).join("\n");

  for (const pattern of [
    /\b(?:delve|utilize|leverage|robust|streamline|harness|tapestry|landscape|paradigm|synergy|ecosystem)\b/i,
    /\b(?:the problem (?:isn't|is not)|what's missing|what do we learn|to be precise)\b/i,
    /\b(?:it is worth noting|it's worth noting|importantly|interestingly|let's unpack|let's explore)\b/i,
    /\bnot\s+[^.]{1,80}\s+but\s+[^.]{1,80}[.!?]/i,
  ]) {
    assert.doesNotMatch(guideCopy, pattern);
  }
});

test("every full locale contains the guide copy used by the component", () => {
  const localeDir = path.join(process.cwd(), "..", "i18n", "locales", "ui");
  const source = fs.readFileSync(
    path.join(process.cwd(), "src", "features", "tutorial", "components", "concepts-guide.tsx"),
    "utf8",
  );
  const referencedKeys = new Set(
    [
      ...source.matchAll(
        /msg\("(auto\.features\.tutorial\.components\.concepts\.guide\.[^"]+)"\)/g,
      ),
    ].map((match) => match[1]!),
  );
  const english = JSON.parse(fs.readFileSync(path.join(localeDir, "en.json"), "utf8")) as Record<
    string,
    string
  >;
  const identifierOnly = /^[A-Za-z0-9_./{}=" -]+$/;

  for (const locale of FULL_UI_LOCALES) {
    const messages = JSON.parse(
      fs.readFileSync(path.join(localeDir, `${locale}.json`), "utf8"),
    ) as Record<string, string>;
    for (const key of referencedKeys) {
      assert.ok(messages[key]?.trim(), `${locale}:${key} must be present and non-empty`);
      if (
        locale !== "en" &&
        english[key].length > 30 &&
        !english[key].includes("\n") &&
        !identifierOnly.test(english[key])
      ) {
        assert.notEqual(messages[key], english[key], `${locale}:${key} must be localized`);
      }
    }
  }
});

test("localized guide copy preserves runtime identifiers and numeric rules", () => {
  const localeDir = path.join(process.cwd(), "..", "i18n", "locales", "ui");
  const prefix = "auto.features.tutorial.components.concepts.guide.literal.";
  const identifierKeys = [
    274, 290, 291, 292, 293, 294, 295, 301, 312, 313, 314, 315, 316, 317, 318, 319, 321, 322, 323,
    324, 325, 326, 327, 328, 331, 332, 344,
  ];
  const english = JSON.parse(fs.readFileSync(path.join(localeDir, "en.json"), "utf8")) as Record<
    string,
    string
  >;

  for (const locale of FULL_UI_LOCALES) {
    const messages = JSON.parse(
      fs.readFileSync(path.join(localeDir, `${locale}.json`), "utf8"),
    ) as Record<string, string>;
    for (const number of identifierKeys) {
      const key = `${prefix}${number}`;
      assert.equal(
        messages[key],
        english[key],
        `${locale}:${key} must preserve the API identifier`,
      );
    }
    for (const identifier of [
      "gold",
      "pred",
      "trace",
      "pred_name",
      "pred_trace",
      "dspy.Prediction",
    ]) {
      assert.match(messages[`${prefix}155`], new RegExp(identifier.replace(".", "\\.")));
    }
    const normalizedSplitCopy = messages[`${prefix}151`]
      .replace(/[۰-۹]/g, (digit) => String("۰۱۲۳۴۵۶۷۸۹".indexOf(digit)))
      .replace(/[٠-٩]/g, (digit) => String("٠١٢٣٤٥٦٧٨٩".indexOf(digit)))
      .replace(/[०-९]/g, (digit) => String("०१२३४५६७८९".indexOf(digit)));
    for (const number of [30, 79, 80, 60, 20, 300, 200, 500]) {
      assert.match(normalizedSplitCopy, new RegExp(`\\b${number}\\b`));
    }
    for (const key of Object.keys(messages).filter((candidate) =>
      candidate.startsWith(GUIDE_PREFIX),
    )) {
      assert.doesNotMatch(messages[key], /(?:ZZZTERM|ЗЗЗТЕРМ)\d+/iu, `${locale}:${key}`);
    }
  }
});
