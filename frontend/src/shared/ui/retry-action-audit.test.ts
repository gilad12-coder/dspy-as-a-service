import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import ts from "typescript";

const RECOVERY_LABEL_KEYS = new Set([
  "auto.features.agent.panel.components.generalistpanel.error_retry",
  "auto.features.dashboard.components.analyticsempty.2",
  "auto.features.submit.components.steps.codeagentpanel.2",
  "billing.wallet.retry",
  "optimizations.react.chat_retry",
  "settings.notifications.retry",
  "shared.agent.regenerate",
  "submit.code.interview.retry",
  "submit.react.mcp_retry",
  "tagger.assist.retry",
  "tagger.session.retry",
]);

function tsxFiles(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) return tsxFiles(fullPath);
    return entry.isFile() && entry.name.endsWith(".tsx") ? [fullPath] : [];
  });
}

function recoveryKeysIn(node: ts.Node): string[] {
  const keys: string[] = [];
  const visit = (child: ts.Node) => {
    if (
      ts.isCallExpression(child) &&
      ts.isIdentifier(child.expression) &&
      child.expression.text === "msg" &&
      child.arguments.length === 1 &&
      ts.isStringLiteral(child.arguments[0]!) &&
      RECOVERY_LABEL_KEYS.has(child.arguments[0]!.text)
    ) {
      keys.push(child.arguments[0]!.text);
    }
    ts.forEachChild(child, visit);
  };
  ts.forEachChild(node, visit);
  return keys;
}

test("recovery actions do not render as visible-text buttons", () => {
  const root = path.join(process.cwd(), "src");
  const violations: string[] = [];

  for (const file of tsxFiles(root)) {
    const sourceText = fs.readFileSync(file, "utf8");
    const source = ts.createSourceFile(
      file,
      sourceText,
      ts.ScriptTarget.Latest,
      true,
      ts.ScriptKind.TSX,
    );
    const visit = (node: ts.Node) => {
      if (ts.isJsxElement(node)) {
        const tag = node.openingElement.tagName.getText(source);
        if (tag === "button" || tag === "Button") {
          for (const key of recoveryKeysIn(node)) {
            const line = source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1;
            violations.push(`${path.relative(process.cwd(), file)}:${line} (${key})`);
          }
        }
      }
      ts.forEachChild(node, visit);
    };
    ts.forEachChild(source, visit);
  }

  assert.deepEqual(
    violations,
    [],
    `Use RetryIconButton for recovery actions:\n${violations.join("\n")}`,
  );
});

test("empty-state recovery actions opt into the icon presentation", () => {
  const cases = [
    {
      file: "src/features/dashboard/components/AnalyticsEmpty.tsx",
      key: "auto.features.dashboard.components.analyticsempty.2",
    },
    {
      file: "src/features/tagger/components/TaggingSessionsPanel.tsx",
      key: "tagger.session.retry",
    },
  ];

  for (const item of cases) {
    const source = fs.readFileSync(path.join(process.cwd(), item.file), "utf8");
    const keyStart = source.indexOf(item.key);
    assert.notEqual(keyStart, -1, `${item.file} no longer contains ${item.key}`);
    assert.match(
      source.slice(keyStart, keyStart + 240),
      /iconOnly:\s*true/,
      `${item.file} must keep ${item.key} icon-only`,
    );
  }
});
