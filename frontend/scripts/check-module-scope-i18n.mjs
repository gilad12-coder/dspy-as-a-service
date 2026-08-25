/**
 * Guard against module-scope i18n resolution.
 *
 * `msg()` / `formatMsg()` / `tip()` / `tI18n()` calls and `TERMS.x` reads that
 * execute while a module evaluates resolve BEFORE the active locale/catalog is
 * pinned: in the browser the catalog shim may not have run yet (the constant
 * freezes raw keys like `auto.features….literal.1` for the whole session), and
 * on the server module scope runs once per process with whichever request's
 * catalog happened to be active (freezing one request's language into every
 * later request). Wrap such constants in `perLocale(() => …)` from
 * `@/shared/lib/per-locale`, or resolve inside a function/component instead.
 *
 * Runs as part of `npm run lint`. Exits 1 and lists offenders when any
 * module-scope resolution exists under `src/` (generated code excluded).
 */

import { createRequire } from "node:module";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const ts = require("typescript");

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "src");
const I18N_CALLS = new Set([
  "msg",
  "formatMsg",
  "tip",
  "tI18n",
  "getActiveLocale",
  "getActiveDir",
  "getActiveIntlLocale",
  "getActiveMessages",
]);

function* walkFiles(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "generated" || entry.name === "node_modules") continue;
      yield* walkFiles(path);
    } else if (/\.(ts|tsx)$/.test(entry.name) && !entry.name.endsWith(".d.ts")) {
      yield path;
    }
  }
}

const isFunctionLike = (node) =>
  ts.isFunctionDeclaration(node) ||
  ts.isFunctionExpression(node) ||
  ts.isArrowFunction(node) ||
  ts.isMethodDeclaration(node) ||
  ts.isGetAccessorDeclaration(node) ||
  ts.isSetAccessorDeclaration(node) ||
  ts.isConstructorDeclaration(node);

const findings = [];
for (const file of walkFiles(ROOT)) {
  const source = ts.createSourceFile(
    file,
    readFileSync(file, "utf8"),
    ts.ScriptTarget.Latest,
    true,
    file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const visit = (node, inFunction) => {
    if (isFunctionLike(node)) inFunction = true;
    if (!inFunction) {
      const isI18nCall =
        ts.isCallExpression(node) &&
        ts.isIdentifier(node.expression) &&
        I18N_CALLS.has(node.expression.text);
      const isTermsRead =
        ts.isPropertyAccessExpression(node) &&
        ts.isIdentifier(node.expression) &&
        node.expression.text === "TERMS";
      if (isI18nCall || isTermsRead) {
        const { line } = source.getLineAndCharacterOfPosition(node.getStart());
        findings.push(`${relative(ROOT, file)}:${line + 1}: ${node.getText().split("\n")[0]}`);
      }
    }
    ts.forEachChild(node, (child) => visit(child, inFunction));
  };
  visit(source, false);
}

if (findings.length > 0) {
  console.error(
    `error: module-scope-i18n: ${findings.length} i18n resolution(s) execute at module scope ` +
      "(freezes the wrong locale / raw keys — wrap in perLocale() or move inside a function):",
  );
  for (const finding of findings) console.error(`  src/${finding}`);
  process.exit(1);
}
console.log("module-scope i18n check: 0 offenders");
